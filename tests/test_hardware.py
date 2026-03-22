"""Tests for hardware detection."""

from zerollm.hardware import detect, compute_n_gpu_layers, compute_threads


def test_detect_returns_info():
    hw = detect()
    assert hw.platform in ("darwin", "linux", "windows")
    assert hw.ram_gb > 0
    assert hw.n_threads > 0
    assert hw.cpu != ""


def test_compute_n_gpu_layers():
    assert compute_n_gpu_layers(40, 1.0) == -1  # all layers
    assert compute_n_gpu_layers(40, 0.0) == 0  # no GPU
    assert compute_n_gpu_layers(40, 0.5) == 20  # half


def test_compute_threads():
    threads = compute_threads(1.0)
    assert threads > 0

    threads_half = compute_threads(0.5)
    assert threads_half > 0
    assert threads_half <= threads
