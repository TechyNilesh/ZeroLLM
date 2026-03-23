"""Cache manager — tracks downloaded models locally.

No curated model list. Any HuggingFace GGUF model works.
This module only manages the local cache index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "zerollm"
CACHE_INDEX = CACHE_DIR / "cache_index.json"


@dataclass
class CachedModel:
    """Metadata for a locally cached model."""

    hf_repo: str  # e.g. "Qwen/Qwen3.5-4B-GGUF"
    filename: str  # e.g. "Qwen3.5-4B-Q4_K_M.gguf"
    local_path: str  # full path to the GGUF file
    size_mb: int
    context_length: int
    supports_tools: bool


def _load_index() -> dict[str, dict]:
    """Load the cache index."""
    if not CACHE_INDEX.exists():
        return {}
    with open(CACHE_INDEX) as f:
        return json.load(f)


def _save_index(index: dict[str, dict]) -> None:
    """Save the cache index."""
    CACHE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def register_download(model: CachedModel) -> None:
    """Register a downloaded model in the cache index."""
    index = _load_index()
    index[model.hf_repo] = asdict(model)
    _save_index(index)


def lookup_cache(hf_repo: str) -> CachedModel | None:
    """Look up a model in the local cache. Returns None if not cached."""
    index = _load_index()
    entry = index.get(hf_repo)
    if entry is None:
        return None

    # Verify file still exists
    if not Path(entry["local_path"]).exists():
        # File was deleted, remove from index
        del index[hf_repo]
        _save_index(index)
        return None

    return CachedModel(**entry)


def remove_from_cache(hf_repo: str) -> bool:
    """Remove a model from cache index and delete the file."""
    import shutil

    index = _load_index()
    entry = index.get(hf_repo)
    if entry is None:
        return False

    # Delete the file
    local_path = Path(entry["local_path"])
    if local_path.exists():
        local_path.unlink()
    # Clean up empty parent dir
    parent = local_path.parent
    if parent.exists() and not any(parent.iterdir()):
        shutil.rmtree(parent)

    del index[hf_repo]
    _save_index(index)
    return True


def list_cached() -> list[CachedModel]:
    """List all cached models."""
    index = _load_index()
    models = []
    for entry in index.values():
        if Path(entry["local_path"]).exists():
            models.append(CachedModel(**entry))
    return models


def cache_size_mb() -> float:
    """Get total size of cached models in MB."""
    if not CACHE_DIR.exists():
        return 0.0
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.gguf") if f.is_file())
    return total / (1024 * 1024)
