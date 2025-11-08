"""Unit tests for :mod:`codex_project.cli`."""

from codex_project import cli


def test_generate_greeting_strips_whitespace():
    result = cli.generate_greeting("  Codex  ")
    assert result == "Hello, Codex! Welcome to my-codex-project."


def test_generate_greeting_defaults_to_generic_name():
    result = cli.generate_greeting("   ")
    assert result == "Hello, there! Welcome to my-codex-project."


def test_main_returns_message_and_prints(capsys):
    message = cli.main(["Learner"])
    captured = capsys.readouterr()

    assert message == "Hello, Learner! Welcome to my-codex-project."
    assert captured.out.strip() == message
