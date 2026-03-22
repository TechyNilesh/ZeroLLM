"""Tests for the model resolver."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from zerollm.resolver import resolve, ResolvedModel, DEFAULT_MODEL


def test_default_model():
    assert DEFAULT_MODEL == "Qwen/Qwen3.5-4B"


def test_resolve_local_gguf(tmp_path):
    gguf_file = tmp_path / "my-model.gguf"
    gguf_file.write_bytes(b"fake gguf content")

    resolved = resolve(str(gguf_file))
    assert resolved.source == "local_gguf"
    assert resolved.name == "my-model"
    assert resolved.path == str(gguf_file)
    assert resolved.context_length == 4096


def test_resolve_finetuned_adapter_dir(tmp_path):
    adapter_dir = tmp_path / "my-bot"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        "r": 16,
    }))

    resolved = resolve(str(adapter_dir))
    assert resolved.source == "finetuned"
    assert resolved.name == "my-bot"


def test_resolve_merged_model_dir(tmp_path):
    model_dir = tmp_path / "merged-bot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

    resolved = resolve(str(model_dir))
    assert resolved.source == "finetuned"
    assert resolved.name == "merged-bot"


def test_resolve_dir_with_gguf(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "custom.gguf").write_bytes(b"fake")

    resolved = resolve(str(model_dir))
    assert resolved.source == "local_gguf"
    assert resolved.path.endswith("custom.gguf")


def test_resolve_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="does not contain"):
        resolve(str(empty_dir))


def test_resolve_huggingface_delegates():
    """Test that HF repo names call _resolve_huggingface."""
    with patch("zerollm.resolver._resolve_huggingface") as mock:
        mock.return_value = ResolvedModel(
            name="test", path="/fake.gguf", context_length=4096,
            source="huggingface", supports_tools=True,
        )
        resolved = resolve("some-org/some-model")
        mock.assert_called_once_with("some-org/some-model")


def test_resolved_model_dataclass():
    rm = ResolvedModel(
        name="test",
        path="/fake/path.gguf",
        context_length=4096,
        source="local_gguf",
        supports_tools=False,
    )
    assert rm.name == "test"
    assert rm.context_length == 4096
