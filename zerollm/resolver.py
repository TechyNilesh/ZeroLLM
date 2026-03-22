"""Model resolver — resolves a model string to a loadable path.

Handles three cases:
    1. HuggingFace repo:  "Qwen/Qwen3.5-4B" (any HF GGUF model)
    2. Local GGUF file:   "/path/to/model.gguf"
    3. Fine-tuned dir:    "~/.cache/zerollm/models/my-bot"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResolvedModel:
    """Result of resolving a model string."""

    name: str  # display name
    path: str  # path to GGUF file or adapter directory
    context_length: int
    source: str  # "huggingface", "local_gguf", "finetuned"
    supports_tools: bool


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def resolve(model: str | None = None) -> ResolvedModel:
    """Resolve a model string to a loadable model path.

    Args:
        model: HuggingFace repo name, local GGUF path, or fine-tuned dir.
               If None, uses the default model.

    Returns:
        ResolvedModel with path and metadata.
    """
    if model is None:
        model = DEFAULT_MODEL

    expanded = str(Path(model).expanduser())

    # Case 1: Local GGUF file
    if expanded.endswith(".gguf") and Path(expanded).is_file():
        return ResolvedModel(
            name=Path(expanded).stem,
            path=expanded,
            context_length=4096,
            source="local_gguf",
            supports_tools=False,
        )

    # Case 2: Fine-tuned adapter directory
    p = Path(expanded)
    if p.is_dir():
        return _resolve_local_dir(p)

    # Also check ~/.cache/zerollm/models/<name>
    cache_path = Path.home() / ".cache" / "zerollm" / "models" / model
    if cache_path.is_dir():
        return _resolve_local_dir(cache_path)

    # Case 3: HuggingFace repo name (default path)
    return _resolve_huggingface(model)


def _resolve_huggingface(model: str) -> ResolvedModel:
    """Resolve a HuggingFace model name — download GGUF if needed."""
    from zerollm.downloader import get_path
    from zerollm.registry import lookup_cache

    model_path = get_path(model)

    # Try to get metadata from cache
    cached = lookup_cache(model)
    if cached is None:
        gguf_name = f"{model}-GGUF" if not model.upper().endswith("-GGUF") else model
        cached = lookup_cache(gguf_name)

    context_length = cached.context_length if cached else 4096
    supports_tools = cached.supports_tools if cached else True

    return ResolvedModel(
        name=model,
        path=str(model_path),
        context_length=context_length,
        source="huggingface",
        supports_tools=supports_tools,
    )


def _resolve_local_dir(path: Path) -> ResolvedModel:
    """Resolve a local directory — GGUF file, LoRA adapter, or merged model."""
    import json

    name = path.name

    # Check for GGUF file in the directory
    gguf_files = list(path.glob("*.gguf"))
    if gguf_files:
        return ResolvedModel(
            name=name,
            path=str(gguf_files[0]),
            context_length=4096,
            source="local_gguf",
            supports_tools=False,
        )

    # Check for LoRA adapter
    if (path / "adapter_config.json").exists():
        context_length = 4096
        with open(path / "adapter_config.json") as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path", "")
        from zerollm.registry import lookup_cache
        cached = lookup_cache(base_model)
        if cached:
            context_length = cached.context_length

        return ResolvedModel(
            name=name,
            path=str(path),
            context_length=context_length,
            source="finetuned",
            supports_tools=False,
        )

    # Check for merged HF model
    if (path / "config.json").exists():
        return ResolvedModel(
            name=name,
            path=str(path),
            context_length=4096,
            source="finetuned",
            supports_tools=False,
        )

    raise ValueError(
        f"Directory '{path}' does not contain a GGUF file, LoRA adapter, or HF model.\n"
        f"Expected: .gguf file, adapter_config.json, or config.json"
    )
