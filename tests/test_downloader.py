"""Tests for the downloader module."""

from zerollm.downloader import list_downloaded, cache_size_mb


def test_list_downloaded():
    result = list_downloaded()
    assert isinstance(result, list)


def test_cache_size():
    size = cache_size_mb()
    assert isinstance(size, float)
    assert size >= 0.0
