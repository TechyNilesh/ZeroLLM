"""Tests for the CLI commands."""

from typer.testing import CliRunner

from zerollm.cli import app

runner = CliRunner()


def test_cli_list():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "SmolLM2" in result.output
    assert "TinyLlama" in result.output


def test_cli_list_downloaded():
    result = runner.invoke(app, ["list", "--downloaded"])
    assert result.exit_code == 0


def test_cli_info():
    result = runner.invoke(app, ["info", "Qwen/Qwen3-0.6B"])
    assert result.exit_code == 0
    assert "Qwen3" in result.output
    assert "400MB" in result.output or "400" in result.output


def test_cli_info_invalid_model():
    result = runner.invoke(app, ["info", "nonexistent-model"])
    assert result.exit_code != 0


def test_cli_recommend():
    result = runner.invoke(app, ["recommend"])
    assert result.exit_code == 0
    assert "Detected" in result.output or "Hardware" in result.output


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Hardware" in result.output
    assert "Platform" in result.output or "CPU" in result.output


def test_cli_no_args():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    # Should show help
    assert "Usage" in result.output or "zerollm" in result.output
