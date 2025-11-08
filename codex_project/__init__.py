"""Core functionality for my-codex-project.

This module exposes helper utilities used by the command line interface
implemented in :mod:`codex_project.cli`.
"""

from .cli import generate_greeting, main

__all__ = ["generate_greeting", "main"]
