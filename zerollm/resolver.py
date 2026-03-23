"""Model resolver — resolves a model string to a loadable model.

Handles three cases:
    1. HuggingFace repo:  "Qwen/Qwen3.5-4B" (loaded via transformers)
    2. Local directory:    "/path/to/model/" (local HF model or fine-tuned)
    3. Fine-tuned name:   "my-bot" (checks ~/.cache/zerollm/models/)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResolvedModel:
    """Result of resolving a model string."""

    name: str  # display name
    model_id: str  # HF repo name or local path (passed to transformers)
    context_length: int
    source: str  # "huggingface", "local", "finetuned"
    supports_tools: bool


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def resolve(model: str | None = None) -> ResolvedModel:
    """Resolve a model string to a loadable model.

    Args:
        model: HuggingFace repo name, local model path, or fine-tuned name.

    Returns:
        ResolvedModel with model_id for transformers to load.
    """
    if model is None:
        model = DEFAULT_MODEL

    expanded = str(Path(model).expanduser())

    # Case 1: Local directory with model files
    p = Path(expanded)
    if p.is_dir() and _is_model_dir(p):
        return _resolve_local(p)

    # Case 2: Check ~/.cache/zerollm/models/<name>
    cache_path = Path.home() / ".cache" / "zerollm" / "models" / model
    if cache_path.is_dir() and _is_model_dir(cache_path):
        return _resolve_local(cache_path)

    # Case 3: HuggingFace repo name (default)
    return _resolve_huggingface(model)


def _is_model_dir(path: Path) -> bool:
    """Check if a directory contains a model."""
    return (
        (path / "config.json").exists()
        or (path / "adapter_config.json").exists()
        or any(path.glob("*.safetensors"))
        or any(path.glob("*.bin"))
    )


def _resolve_huggingface(model: str) -> ResolvedModel:
    """Resolve a HuggingFace model — validate and get metadata."""
    from zerollm.downloader import get_path, _detect_context_length

    model_id = get_path(model)
    context_length = _detect_context_length(model)

    return ResolvedModel(
        name=model,
        model_id=model_id,
        context_length=context_length,
        source="huggingface",
        supports_tools=True,
    )


def _resolve_local(path: Path) -> ResolvedModel:
    """Resolve a local model directory."""
    import json

    name = path.name
    context_length = 4096

    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        for key in ["max_position_embeddings", "n_ctx", "max_seq_len"]:
            if key in config:
                context_length = config[key]
                break

    source = "finetuned" if (path / "adapter_config.json").exists() else "local"

    return ResolvedModel(
        name=name,
        model_id=str(path),
        context_length=context_length,
        source=source,
        supports_tools=False,
    )
