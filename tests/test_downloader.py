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
    d = _model_dir("Qwen/Qwen3-0.6B")
    assert "Qwen_Qwen3-0.6B" in str(d)


def test_model_dir_with_dot():
    d = _model_dir("Qwen/Qwen3-1.7B")
    assert "Qwen_Qwen3-1.7B" in str(d)


def test_model_path():
    p = _model_path("Qwen/Qwen3-0.6B", "model.gguf")
    assert p.name == "model.gguf"
    assert "Qwen_Qwen3-0.6B" in str(p.parent)


def test_is_cached_false():
    # Model should not be cached in test environment
    assert is_cached("Qwen/Qwen3-0.6B") is False


def test_list_downloaded_empty():
    # Should return empty list or list of actually downloaded models
    result = list_downloaded()
    assert isinstance(result, list)


def test_cache_size_mb():
    size = cache_size_mb()
    assert isinstance(size, float)
    assert size >= 0.0
