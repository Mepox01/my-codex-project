"""Command line interface for ``my-codex-project``.

The module contains a tiny yet well documented example that demonstrates how
we might structure a reusable CLI utility.  It is intentionally
straightforward so that the project can act as a template for future
experiments.
"""
from __future__ import annotations

import argparse


def generate_greeting(name: str) -> str:
    """Return a friendly greeting for ``name``.

    Parameters
    ----------
    name:
        The name that should appear in the greeting.  Leading and trailing
        whitespace is ignored, and an empty value defaults to ``"there"``.

    Returns
    -------
    str
        A greeting string that can be displayed to a user.
    """

    normalised = name.strip()
    if not normalised:
        normalised = "there"
    return f"Hello, {normalised}! Welcome to my-codex-project."


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by :func:`main`."""

    parser = argparse.ArgumentParser(description="Greet someone from my-codex-project")
    parser.add_argument(
        "name",
        nargs="?",
        default="there",
        help="Name to include in the greeting (defaults to 'there').",
    )
    return parser


def main(args: list[str] | None = None) -> str:
    """Execute the command line interface.

    Parameters
    ----------
    args:
        Optional sequence of command line arguments.  If ``None``, arguments
        are read from :data:`sys.argv`.

    Returns
    -------
    str
        The greeting message that was generated.  Returning the value makes
        the function easier to test.
    """

    parser = build_parser()
    namespace = parser.parse_args(args=args)
    message = generate_greeting(namespace.name)
    print(message)
    return message


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    main()
