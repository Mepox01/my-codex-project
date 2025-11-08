# Pet classification helpers

This repository contains a minimal Python module, `pet_classifier.py`, that
trains two TensorFlow models:

* a binary **cat vs. dog** classifier trained on the `oxford_iiit_pet` dataset.
* a **dog breed** classifier (120 classes) trained on `stanford_dogs`.

The script can be run directly after installing TensorFlow and
`tensorflow-datasets`.  By default it expects the following directories to
exist:

```text
/content/pet_models_out       # checkpoints + exported label lists
/content/tensorflow_datasets  # TFDS data cache (downloaded automatically)
```

The helper functions were written with notebook usage in mind and focus on a
robust, low-memory input pipeline.  Unlike the initial version, the training
datasets are cached to disk rather than in-memory.  This avoids the out-of-memory
errors that occurred in environments with limited RAM (e.g. Google Colab) when
processing the full Stanford Dogs dataset.  Validation and test splits are still
cached in memory for fast evaluation.

See the module docstring in `pet_classifier.py` for additional details.

