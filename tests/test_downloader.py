"""Tests for the downloader module."""

from pathlib import Path

from zerollm.downloader import (
    _model_dir,
    _pick_best_gguf,
    list_downloaded,
    cache_size_mb,
)


def test_model_dir():
    d = _model_dir("Qwen/Qwen3.5-4B-GGUF")
    assert "Qwen_Qwen3.5-4B-GGUF" in str(d)


def test_model_dir_slash():
    d = _model_dir("bartowski/Phi-3-mini-4k-instruct-GGUF")
    assert "bartowski_Phi-3-mini-4k-instruct-GGUF" in str(d)


def test_pick_best_gguf_prefers_q4_k_m():
    files = [
        "model-Q2_K.gguf",
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
        "model-f16.gguf",
    ]
    assert _pick_best_gguf(files) == "model-Q4_K_M.gguf"


def test_pick_best_gguf_falls_back_to_q5():
    files = [
        "model-Q5_K_M.gguf",
        "model-Q8_0.gguf",
    ]
    assert _pick_best_gguf(files) == "model-Q5_K_M.gguf"


def test_pick_best_gguf_only_q8():
    files = ["model-Q8_0.gguf"]
    assert _pick_best_gguf(files) == "model-Q8_0.gguf"


def test_pick_best_gguf_unknown_quant():
    files = ["model-weird.gguf"]
    assert _pick_best_gguf(files) == "model-weird.gguf"


def test_list_downloaded():
    result = list_downloaded()
    assert isinstance(result, list)


def test_cache_size():
    size = cache_size_mb()
    assert isinstance(size, float)
    assert size >= 0.0
