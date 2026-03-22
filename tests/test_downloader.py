"""Tests for the downloader module."""

from pathlib import Path
from unittest.mock import patch

from zerollm.downloader import (
    _model_dir,
    _model_path,
    is_cached,
    list_downloaded,
    cache_size_mb,
    CACHE_DIR,
)


def test_model_dir_simple_name():
    d = _model_dir("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert "HuggingFaceTB_SmolLM2-1.7B-Instruct" in str(d)


def test_model_dir_colon_name():
    d = _model_dir("Qwen/Qwen2.5-0.5B-Instruct")
    assert "Qwen_Qwen2.5-0.5B-Instruct" in str(d)


def test_model_path():
    p = _model_path("HuggingFaceTB/SmolLM2-1.7B-Instruct", "model.gguf")
    assert p.name == "model.gguf"
    assert "HuggingFaceTB_SmolLM2-1.7B-Instruct" in str(p.parent)


def test_is_cached_false():
    # Model should not be cached in test environment
    assert is_cached("HuggingFaceTB/SmolLM2-1.7B-Instruct") is False


def test_list_downloaded_empty():
    # Should return empty list or list of actually downloaded models
    result = list_downloaded()
    assert isinstance(result, list)


def test_cache_size_mb():
    size = cache_size_mb()
    assert isinstance(size, float)
    assert size >= 0.0
