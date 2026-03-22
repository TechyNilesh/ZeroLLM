"""Tests for the cache manager (registry.py)."""

import json
from pathlib import Path

from zerollm.registry import (
    CachedModel,
    register_download,
    lookup_cache,
    remove_from_cache,
    list_cached,
    cache_size_mb,
    _load_index,
    _save_index,
    CACHE_INDEX,
)


def test_cached_model_dataclass():
    m = CachedModel(
        hf_repo="Qwen/Qwen3.5-4B-GGUF",
        filename="Qwen3.5-4B-Q4_K_M.gguf",
        local_path="/fake/path.gguf",
        size_mb=2400,
        context_length=32768,
        supports_tools=True,
    )
    assert m.hf_repo == "Qwen/Qwen3.5-4B-GGUF"
    assert m.size_mb == 2400


def test_lookup_cache_miss():
    result = lookup_cache("nonexistent/model")
    assert result is None


def test_list_cached():
    result = list_cached()
    assert isinstance(result, list)


def test_cache_size():
    size = cache_size_mb()
    assert isinstance(size, float)
    assert size >= 0.0
