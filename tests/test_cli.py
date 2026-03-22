"""Tests for the CLI commands."""

from typer.testing import CliRunner

from zerollm.cli import app

runner = CliRunner()


def test_cli_list():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Hardware" in result.output


def test_cli_no_args():
    result = runner.invoke(app, [])
    # Typer with no_args_is_help returns 0 and shows help
    assert "Usage" in result.output or "zerollm" in result.output or "help" in result.output.lower()
