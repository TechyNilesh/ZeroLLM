"""Model resolver — resolves a model string to a loadable path.

Handles three cases:
    1. Registry model:  "Qwen/Qwen3-0.6B"
    2. Local GGUF file: "/path/to/model.gguf"
    3. Fine-tuned dir:  "~/.cache/zerollm/models/my-bot" (or any dir with adapter_config.json)
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
    source: str  # "registry", "local_gguf", "finetuned"
    supports_tools: bool


def resolve(model: str, power: float = 1.0) -> ResolvedModel:
    """Resolve a model string to a loadable model path.

    Args:
        model: Registry name, local GGUF path, or fine-tuned adapter dir.
        power: Resource usage (passed through for context).

    Returns:
        ResolvedModel with path and metadata.
    """
    expanded = str(Path(model).expanduser())

    # Case 1: Local GGUF file
    if expanded.endswith(".gguf") and Path(expanded).is_file():
        return ResolvedModel(
            name=Path(expanded).stem,
            path=expanded,
            context_length=4096,  # safe default for unknown models
            source="local_gguf",
            supports_tools=False,
        )

    # Case 2: Fine-tuned adapter directory (has adapter_config.json or config.json)
    p = Path(expanded)
    if p.is_dir():
        # Check for LoRA adapter
        if (p / "adapter_config.json").exists():
            return _resolve_finetuned(p)
        # Check for merged HF model
        if (p / "config.json").exists():
            return _resolve_finetuned(p)
        # Check for GGUF file inside directory
        gguf_files = list(p.glob("*.gguf"))
        if gguf_files:
            return ResolvedModel(
                name=p.name,
                path=str(gguf_files[0]),
                context_length=4096,
                source="local_gguf",
                supports_tools=False,
            )

    # Also check ~/.cache/zerollm/models/<name>
    cache_path = Path.home() / ".cache" / "zerollm" / "models" / model
    if cache_path.is_dir():
        if (cache_path / "adapter_config.json").exists() or (cache_path / "config.json").exists():
            return _resolve_finetuned(cache_path)

    # Case 3: Registry model (default)
    from zerollm.downloader import get_path
    from zerollm.registry import lookup

    info = lookup(model)
    model_path = get_path(model)

    return ResolvedModel(
        name=info.name,
        path=str(model_path),
        context_length=info.context_length,
        source="registry",
        supports_tools=info.supports_tools,
    )


def _resolve_finetuned(path: Path) -> ResolvedModel:
    """Resolve a fine-tuned model directory.

    If it's a LoRA adapter, we need to:
    1. Find the base model from adapter_config.json
    2. Load the base GGUF model
    3. Note that we need to apply the adapter at load time

    For now, if the dir has a GGUF file, use it directly.
    Otherwise, return the dir path — the backend needs to handle adapter loading.
    """
    import json

    name = path.name

    # Check for GGUF file in the directory (merged + converted model)
    gguf_files = list(path.glob("*.gguf"))
    if gguf_files:
        return ResolvedModel(
            name=name,
            path=str(gguf_files[0]),
            context_length=4096,
            source="finetuned",
            supports_tools=False,
        )

    # Read adapter config to find base model
    context_length = 4096
    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path", "")
        # Try to get context length from registry
        try:
            from zerollm.registry import lookup
            base_info = lookup(base_model)
            context_length = base_info.context_length
        except (ValueError, KeyError):
            pass

    return ResolvedModel(
        name=name,
        path=str(path),
        context_length=context_length,
        source="finetuned",
        supports_tools=False,
    )
