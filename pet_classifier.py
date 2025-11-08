"""Utilities to train and use pet classification models.

This module refactors the original notebook-style helpers so that they can be
imported as a library or executed as a script.  The main behavioural change is
that the input pipelines no longer cache the entire training split in memory.
The Stanford Dogs dataset is large enough that caching every resized image can
exhaust the RAM available on typical notebook runtimes (e.g. Google Colab),
causing training to stall or crash.  Instead, the training pipeline now caches
to disk by default, while the (smaller) validation and test sets keep the
in-memory cache for performance.

Usage
-----
Running this module as a script will train the binary species classifier first
and then the dog-breed classifier.  Both models are saved to
``/content/pet_models_out`` by default and the label lists are exported as JSON
files in the same directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


OUTPUT_DIR = Path("/content/pet_models_out")
TFDS_DATA_DIR = Path("/content/tensorflow_datasets")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def make_augmenter() -> tf.keras.Sequential:
    """Return the data augmentation pipeline used for the training split."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )


MapFn = Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def _cache_dataset(dataset: tf.data.Dataset, cache: str | bool | None) -> tf.data.Dataset:
    """Apply ``Dataset.cache`` if requested.

    ``cache`` mirrors ``tf.data.Dataset.cache`` arguments:

    * ``True``/``None`` caches in memory.
    * ``False`` skips caching entirely.
    * ``str``/``Path`` caches to a file on disk.
    """

    if cache is False:
        return dataset

    if cache is True or cache is None:
        return dataset.cache()

    cache_path = Path(cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return dataset.cache(str(cache_path))


def prepare_ds(
    ds: tf.data.Dataset,
    *,
    training: bool,
    map_fn: MapFn,
    cache: bool | str | os.PathLike[str] | None,
) -> tf.data.Dataset:
    """Prepare the dataset for model training or evaluation.

    Parameters
    ----------
    ds:
        The source dataset from ``tfds``.
    training:
        ``True`` if the dataset will be used for model training.
    map_fn:
        Function that converts the TFDS feature dict into ``(image, label)``.
    cache:
        Passed to :meth:`tf.data.Dataset.cache`.  For large training datasets we
        prefer to cache on disk to avoid memory pressure.  Evaluation splits
        (validation/test) can safely cache in RAM.
    """

    augmenter = make_augmenter()

    def _map(features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image, label = map_fn(features)
        return image, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = _cache_dataset(ds, cache)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_backbone() -> tf.keras.Model:
    """Create a frozen imagenet-pretrained convolutional backbone."""

    input_shape = (*IMAGE_SIZE, 3)
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    except Exception:  # pragma: no cover - best-effort fallback
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model.trainable = False
    base_model.preprocess_input = preprocess  # type: ignore[attr-defined]
    return base_model


def unfreeze_last_layers(base_model: tf.keras.Model, fraction: float = 0.3) -> None:
    """Unfreeze the last ``fraction`` of layers for fine-tuning."""

    if not hasattr(base_model, "layers"):
        return

    total_layers = len(base_model.layers)
    num_to_unfreeze = max(1, int(total_layers * fraction))
    for layer in base_model.layers[-num_to_unfreeze:]:
        layer.trainable = True


def build_callbacks(name: str) -> Tuple[list[tf.keras.callbacks.Callback], Path]:
    """Create callbacks that save the best model checkpoint by validation loss."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"{name}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    return callbacks, checkpoint_path


def _common_map_fn(
    features: dict, preprocess: Callable[[tf.Tensor], tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(features["image"], tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess(image)
    return image, features["label"]


def _prepare_species_map_fn(
    features: dict, preprocess: Callable[[tf.Tensor], tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(features["image"], tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess(image)
    return image, features["species"]


def load_species_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Iterable[str]]:
    """Load the Oxford-IIIT Pet dataset split into train/val/test datasets."""

    (train_raw, test_raw), info = tfds.load(
        "oxford_iiit_pet",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
        data_dir=str(TFDS_DATA_DIR),
    )

    total_train = info.splits["train"].num_examples
    val_count = int(0.1 * total_train)

    val_raw = train_raw.take(val_count)
    train_raw = train_raw.skip(val_count)

    backbone = build_backbone()
    preprocess = backbone.preprocess_input

    train_ds = prepare_ds(
        train_raw,
        training=True,
        map_fn=lambda f: _prepare_species_map_fn(f, preprocess),
        cache=OUTPUT_DIR / "species_train.cache",
    )
    val_ds = prepare_ds(
        val_raw,
        training=False,
        map_fn=lambda f: _prepare_species_map_fn(f, preprocess),
        cache=True,
    )
    test_ds = prepare_ds(
        test_raw,
        training=False,
        map_fn=lambda f: _prepare_species_map_fn(f, preprocess),
        cache=True,
    )

    return train_ds, val_ds, test_ds, info.features["species"].names


def build_species_model(num_classes: int = 2) -> Tuple[tf.keras.Model, tf.keras.Model]:
    base_model = build_backbone()

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_species")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop_species")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="species_out",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="species_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train_species_binary() -> Tuple[tf.keras.Model, Iterable[str]]:
    train_ds, val_ds, test_ds, species_names = load_species_data()

    model, base_model = build_species_model(num_classes=len(species_names))
    callbacks, checkpoint = build_callbacks("species_model")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
    )

    unfreeze_last_layers(base_model, fraction=0.3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(checkpoint)
    loss, accuracy = best_model.evaluate(test_ds, verbose=0)
    print(f"[Species cat/dog] Test accuracy: {accuracy:.3f}")

    labels_path = OUTPUT_DIR / "species_labels.json"
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(list(species_names), fh, ensure_ascii=False, indent=2)

    return best_model, species_names


def load_breed_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Iterable[str]]:
    (train_raw, test_raw), info = tfds.load(
        "stanford_dogs",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
        data_dir=str(TFDS_DATA_DIR),
    )

    total_train = info.splits["train"].num_examples
    val_count = int(0.1 * total_train)

    val_raw = train_raw.take(val_count)
    train_raw = train_raw.skip(val_count)

    backbone = build_backbone()
    preprocess = backbone.preprocess_input

    train_ds = prepare_ds(
        train_raw,
        training=True,
        map_fn=lambda f: _common_map_fn(f, preprocess),
        cache=OUTPUT_DIR / "breed_train.cache",
    )
    val_ds = prepare_ds(
        val_raw,
        training=False,
        map_fn=lambda f: _common_map_fn(f, preprocess),
        cache=True,
    )
    test_ds = prepare_ds(
        test_raw,
        training=False,
        map_fn=lambda f: _common_map_fn(f, preprocess),
        cache=True,
    )

    return train_ds, val_ds, test_ds, info.features["label"].names


def build_breed_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    base_model = build_backbone()

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_breed")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop_breed")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="breed_out",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="breed_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train_breed_model() -> Tuple[tf.keras.Model, Iterable[str]]:
    train_ds, val_ds, test_ds, breed_names = load_breed_data()

    model, base_model = build_breed_model(num_classes=len(breed_names))
    callbacks, checkpoint = build_callbacks("breed_model")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
    )

    unfreeze_last_layers(base_model, fraction=0.3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(checkpoint)
    loss, accuracy = best_model.evaluate(test_ds, verbose=0)
    print(f"[Breed dog-ırk] Test accuracy: {accuracy:.3f}")

    labels_path = OUTPUT_DIR / "breed_labels.json"
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(list(breed_names), fh, ensure_ascii=False, indent=2)

    return best_model, breed_names


def _load_image_for_infer(path: os.PathLike[str] | str, preprocess: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(tf.cast(image, tf.float32), IMAGE_SIZE)
    image = preprocess(image)
    return tf.expand_dims(image, 0)


def infer_image(
    img_path: os.PathLike[str] | str,
    species_model: tf.keras.Model,
    species_labels: Iterable[str],
    breed_model: tf.keras.Model,
    breed_labels: Iterable[str],
    dog_threshold: float = 0.5,
) -> dict:
    species_labels = list(species_labels)
    breed_labels = list(breed_labels)

    preprocess = getattr(species_model.layers[1], "preprocess_input", None)
    if preprocess is None:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    image = _load_image_for_infer(img_path, preprocess)

    species_probs = species_model.predict(image, verbose=0)[0]
    species_idx = int(tf.argmax(species_probs, axis=-1))
    species_name = species_labels[species_idx]

    result = {
        "animal": {
            "label": species_name,
            "probs": {
                species_labels[0]: float(species_probs[0]),
                species_labels[1]: float(species_probs[1]),
            },
        }
    }

    if species_name.lower() == "dog" and float(species_probs[species_idx]) >= dog_threshold:
        preprocess_breed = getattr(breed_model.layers[1], "preprocess_input", preprocess)
        breed_image = _load_image_for_infer(img_path, preprocess_breed)
        breed_probs = breed_model.predict(breed_image, verbose=0)[0]
        top_indices = tf.argsort(breed_probs, direction="DESCENDING")[:5].numpy().tolist()

        top5 = [
            {
                "breed": breed_labels[i],
                "prob": float(breed_probs[i]),
            }
            for i in top_indices
        ]

        result["dog_breed"] = {
            "best": top5[0]["breed"],
            "top5": top5,
        }

    return result


def main() -> None:
    species_model, species_labels = train_species_binary()
    breed_model, breed_labels = train_breed_model()

    print("Training complete. Models saved in", OUTPUT_DIR)


if __name__ == "__main__":
    main()

