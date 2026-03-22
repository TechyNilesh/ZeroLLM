"""Model registry — maps friendly names to HF repos and GGUF files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    """Metadata for a curated model."""

    name: str
    hf_repo: str  # GGUF repo (e.g. bartowski/SmolLM2-1.7B-Instruct-GGUF)
    hf_base_repo: str  # Original HF repo for fine-tuning (e.g. HuggingFaceTB/SmolLM2-1.7B-Instruct)
    filename: str
    size_mb: int
    min_ram_gb: float
    context_length: int
    chat_template: str
    supports_tools: bool
    license: str

    @property
    def size_label(self) -> str:
        if self.size_mb >= 1024:
            return f"{self.size_mb / 1024:.1f}GB"
        return f"{self.size_mb}MB"


# Default registry file shipped with the package
_REGISTRY_PATH = Path(__file__).parent.parent / "registry.json"

_cache: dict[str, ModelInfo] | None = None


def _load_registry() -> dict[str, ModelInfo]:
    """Load registry from JSON file."""
    global _cache
    if _cache is not None:
        return _cache

    if not _REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry not found at {_REGISTRY_PATH}. "
            "Reinstall zerollm or check your installation."
        )

    with open(_REGISTRY_PATH) as f:
        data = json.load(f)

    _cache = {}
    for name, info in data.items():
        _cache[name] = ModelInfo(
            name=name,
            hf_repo=info["hf_repo"],
            hf_base_repo=info["hf_base_repo"],
            filename=info["filename"],
            size_mb=info["size_mb"],
            min_ram_gb=info["min_ram_gb"],
            context_length=info["context_length"],
            chat_template=info["chat_template"],
            supports_tools=info["supports_tools"],
            license=info["license"],
        )

    return _cache


def lookup(name: str) -> ModelInfo:
    """Look up a model by friendly name."""
    registry = _load_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Model '{name}' not found in registry.\n"
            f"Available models: {available}\n"
            f"Run 'zerollm list' to see all models."
        )
    return registry[name]


def list_models() -> list[ModelInfo]:
    """Return all curated models."""
    registry = _load_registry()
    return list(registry.values())


def recommend_for(ram_gb: float, has_gpu: bool = False) -> list[ModelInfo]:
    """Recommend models that fit the given hardware.

    Returns models sorted by size (largest first) that fit in available RAM.
    """
    registry = _load_registry()
    fits = [m for m in registry.values() if m.min_ram_gb <= ram_gb]
    fits.sort(key=lambda m: m.size_mb, reverse=True)
    return fits
