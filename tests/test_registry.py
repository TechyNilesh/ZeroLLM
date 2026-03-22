"""Tests for the model registry."""

import pytest

from zerollm.registry import lookup, list_models, recommend_for


def test_lookup_valid_model():
    info = lookup("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert info.name == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert info.size_mb == 1100
    assert info.supports_tools is True


def test_lookup_all_models():
    models = list_models()
    assert len(models) == 7
    names = [m.name for m in models]
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in names
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in names
    assert "microsoft/Phi-3-mini-4k-instruct" in names


def test_lookup_invalid_model():
    with pytest.raises(ValueError, match="not found"):
        lookup("nonexistent-model")


def test_recommend_for_16gb():
    models = recommend_for(16.0)
    assert len(models) > 0
    # Should recommend largest fitting model first
    assert models[0].size_mb >= models[-1].size_mb


def test_recommend_for_2gb():
    models = recommend_for(2.0)
    # No models fit under 2GB RAM
    assert len(models) == 0


def test_model_size_label():
    info = lookup("microsoft/Phi-3-mini-4k-instruct")
    assert "GB" in info.size_label

    info = lookup("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert "MB" in info.size_label


def test_hf_base_repo_present():
    info = lookup("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert info.hf_base_repo == "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    info = lookup("microsoft/Phi-3-mini-4k-instruct")
    assert info.hf_base_repo == "microsoft/Phi-3-mini-4k-instruct"
