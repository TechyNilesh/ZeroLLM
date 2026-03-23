"""Download models from HuggingFace Hub and manage local cache."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from zerollm.registry import (
    CACHE_DIR,
    CachedModel,
    lookup_cache,
    register_download,
)

console = Console()


def get_path(hf_repo: str) -> str:
    """Get a HuggingFace model name, downloading if needed.

    With HF transformers, models are cached by huggingface_hub automatically.
    This function just validates the model exists and tracks it in our cache.

    Returns the model name (transformers loads by name, not by path).
    """
    # Check our cache
    cached = lookup_cache(hf_repo)
    if cached:
        return cached.hf_repo

    # Download / validate the model exists
    return download(hf_repo)


def download(hf_repo: str) -> str:
    """Download a model from HuggingFace Hub.

    Transformers + huggingface_hub handle the actual download and caching.
    We just validate the model exists and register it in our cache index.

    Returns the model name.
    """
    from huggingface_hub import model_info

    console.print(f"[dim]Checking {hf_repo} on HuggingFace...[/dim]")

    # Validate model exists
    try:
        info = model_info(hf_repo)
    except Exception as e:
        raise ValueError(
            f"Model '{hf_repo}' not found on HuggingFace.\n"
            f"Make sure the model name is correct (e.g. 'Qwen/Qwen3.5-4B').\n"
            f"Error: {e}"
        )

    # Detect context length from config
    context_length = _detect_context_length(hf_repo)

    # Register in our cache index
    register_download(CachedModel(
        hf_repo=hf_repo,
        filename="transformers",  # no single file, HF manages cache
        local_path=f"huggingface://{hf_repo}",
        size_mb=0,  # managed by HF cache
        context_length=context_length,
        supports_tools=True,
    ))

    console.print(f"[green]✓[/green] {hf_repo} ready")
    return hf_repo


def _detect_context_length(hf_repo: str) -> int:
    """Detect context length from model config on HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        import json

        config_path = hf_hub_download(repo_id=hf_repo, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        for key in ["max_position_embeddings", "n_ctx", "max_seq_len", "seq_length"]:
            if key in config:
                return config[key]
    except Exception:
        pass

    return 4096


def remove(hf_repo: str) -> bool:
    """Remove a model from local cache."""
    from zerollm.registry import remove_from_cache

    if remove_from_cache(hf_repo):
        console.print(f"[green]✓[/green] Removed {hf_repo} from cache index")
        console.print("[dim]Note: HF cache (~/.cache/huggingface/) managed separately[/dim]")
        return True
    console.print(f"[yellow]![/yellow] {hf_repo} not found in cache")
    return False


def list_downloaded() -> list[str]:
    """List all tracked model names."""
    from zerollm.registry import list_cached
    return [m.hf_repo for m in list_cached()]


def cache_size_mb() -> float:
    """Get total size of ZeroLLM cache in MB."""
    from zerollm.registry import cache_size_mb as _cache_size
    return _cache_size()
