"""Utility for generating a sample result summary and photo artifact.

This module creates a small placeholder image that can be used for testing
pipelines expecting a photo output, and it prints a simple textual summary of
results.  It is intentionally lightweight so it can run in environments with no
external dependencies.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Tuple

_PHOTO_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/eqCcF8AAAAASUVORK5CYII="
)


def _write_photo(destination: Path) -> Path:
    """Write the embedded PNG image to ``destination``.

    Args:
        destination: File path where the image should be written.

    Returns:
        The path to the written image file.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(_PHOTO_BASE64))
    return destination


def generate_report(photo_filename: str = "photo_test.png") -> Tuple[str, Path]:
    """Create a sample report summary and persist the test photo.

    Args:
        photo_filename: Name of the PNG image file to be written in the current
            working directory.

    Returns:
        A tuple containing the textual report and the ``Path`` to the generated
        image file.
    """

    photo_path = _write_photo(Path(photo_filename))

    summary_lines = [
        "Result Summary",
        "--------------",
        "* Accuracy: 95%",
        "* Precision: 92%",
        "* Recall: 90%",
        "",
        f"Photo written to: {photo_path.resolve()}",
    ]
    return "\n".join(summary_lines), photo_path


def main() -> None:
    """Generate and display the report when executed as a script."""

    summary, _ = generate_report()
    print(summary)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
