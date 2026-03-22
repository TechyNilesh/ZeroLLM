"""Tests for the model resolver."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from zerollm.resolver import resolve, ResolvedModel, DEFAULT_MODEL


def test_default_model():
    assert DEFAULT_MODEL == "Qwen/Qwen3.5-4B"


def test_resolve_local_model_dir(tmp_path):
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "max_position_embeddings": 32768,
    }))

    resolved = resolve(str(model_dir))
    assert resolved.source == "local"
    assert resolved.name == "my-model"
    assert resolved.model_id == str(model_dir)
    assert resolved.context_length == 32768


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


def test_resolve_dir_with_safetensors(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake")

    resolved = resolve(str(model_dir))
    assert resolved.source == "local"
    assert resolved.model_id == str(model_dir)


def test_resolve_huggingface_delegates():
    with patch("zerollm.resolver._resolve_huggingface") as mock:
        mock.return_value = ResolvedModel(
            name="test", model_id="org/model", context_length=4096,
            source="huggingface", supports_tools=True,
        )
        resolved = resolve("org/model")
        mock.assert_called_once_with("org/model")


def test_resolved_model_dataclass():
    rm = ResolvedModel(
        name="test",
        model_id="org/model",
        context_length=4096,
        source="huggingface",
        supports_tools=True,
    )
    assert rm.name == "test"
    assert rm.context_length == 4096
