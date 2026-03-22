"""Tests for the model resolver."""

import json
from pathlib import Path

import pytest

from zerollm.resolver import resolve, ResolvedModel


def test_resolve_registry_model():
    resolved = resolve("Qwen/Qwen3-0.6B")
    assert resolved.source == "registry"
    assert resolved.name == "Qwen/Qwen3-0.6B"
    assert resolved.context_length == 32768
    assert resolved.supports_tools is True


def test_resolve_registry_model_invalid():
    with pytest.raises(ValueError, match="not found"):
        resolve("nonexistent/model")


def test_resolve_local_gguf(tmp_path):
    # Create a fake GGUF file
    gguf_file = tmp_path / "my-model.gguf"
    gguf_file.write_bytes(b"fake gguf content")

    resolved = resolve(str(gguf_file))
    assert resolved.source == "local_gguf"
    assert resolved.name == "my-model"
    assert resolved.path == str(gguf_file)
    assert resolved.context_length == 4096  # default


def test_resolve_finetuned_adapter_dir(tmp_path):
    # Create a fake adapter directory
    adapter_dir = tmp_path / "my-bot"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
        "r": 16,
    }))

    resolved = resolve(str(adapter_dir))
    assert resolved.source == "finetuned"
    assert resolved.name == "my-bot"
    assert resolved.path == str(adapter_dir)


def test_resolve_merged_model_dir(tmp_path):
    # Create a fake merged model directory
    model_dir = tmp_path / "merged-bot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "llama",
    }))

    resolved = resolve(str(model_dir))
    assert resolved.source == "finetuned"
    assert resolved.name == "merged-bot"


def test_resolve_dir_with_gguf(tmp_path):
    # Directory containing a GGUF file
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "custom.gguf").write_bytes(b"fake")

    resolved = resolve(str(model_dir))
    assert resolved.source == "local_gguf"
    assert resolved.path.endswith("custom.gguf")


def test_resolve_expanduser(tmp_path):
    # Ensure ~ expansion works (won't actually test ~, but tests Path expansion)
    gguf_file = tmp_path / "test.gguf"
    gguf_file.write_bytes(b"fake")

    resolved = resolve(str(gguf_file))
    assert resolved.source == "local_gguf"


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
