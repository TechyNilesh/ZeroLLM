"""Download GGUF models from any HuggingFace repo and manage local cache."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

from zerollm.registry import (
    CACHE_DIR,
    CachedModel,
    lookup_cache,
    register_download,
)

console = Console()

# Quantization preference order — best balance of quality vs size
_QUANT_PREFERENCE = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q3_K_M",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "Q3_K_L",
    "Q3_K_S",
    "Q2_K",
    "f16",
    "f32",
]


def _model_dir(hf_repo: str) -> Path:
    """Get the cache directory for a specific model."""
    safe_name = hf_repo.replace("/", "_").replace(":", "_")
    return CACHE_DIR / safe_name


def get_path(hf_repo: str) -> Path:
    """Get path to a cached model, downloading if needed.

    Args:
        hf_repo: HuggingFace repo name (e.g. "Qwen/Qwen3.5-4B").
                  If it doesn't end with -GGUF, we try appending it.
    """
    # Check cache first
    cached = lookup_cache(hf_repo)
    if cached and Path(cached.local_path).exists():
        return Path(cached.local_path)

    # Also check with -GGUF suffix
    gguf_repo = hf_repo if hf_repo.upper().endswith("-GGUF") else f"{hf_repo}-GGUF"
    cached = lookup_cache(gguf_repo)
    if cached and Path(cached.local_path).exists():
        return Path(cached.local_path)

    return download(hf_repo)


def download(hf_repo: str) -> Path:
    """Download best GGUF file from a HuggingFace repo.

    Automatically:
    1. Tries repo as-is, then with -GGUF suffix
    2. Lists all .gguf files in the repo
    3. Picks the best quantization based on preference order
    4. Downloads and caches locally

    Returns the local file path.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    # Try the repo name as-is first, then with -GGUF suffix
    actual_repo = _find_gguf_repo(hf_repo)

    # List GGUF files in the repo
    console.print(f"[dim]Scanning {actual_repo} for GGUF files...[/dim]")
    all_files = list_repo_files(actual_repo)
    gguf_files = [f for f in all_files if f.endswith(".gguf")]

    if not gguf_files:
        raise ValueError(
            f"No .gguf files found in {actual_repo}.\n"
            f"Make sure this is a GGUF model repo on HuggingFace."
        )

    # Pick the best quantization
    filename = _pick_best_gguf(gguf_files)
    console.print(f"[dim]Selected: {filename}[/dim]")

    # Download
    dest_dir = _model_dir(actual_repo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        console.print(f"[green]✓[/green] Already downloaded: {filename}")
    else:
        console.print(f"[bold]Downloading {filename}[/bold] from {actual_repo}")
        hf_hub_download(
            repo_id=actual_repo,
            filename=filename,
            local_dir=str(dest_dir),
        )
        console.print(f"[green]✓[/green] Downloaded to {dest_path}")

    # Get model metadata
    context_length = _detect_context_length(actual_repo)
    size_mb = int(dest_path.stat().st_size / (1024 * 1024))

    # Register in cache
    register_download(CachedModel(
        hf_repo=actual_repo,
        filename=filename,
        local_path=str(dest_path),
        size_mb=size_mb,
        context_length=context_length,
        supports_tools=True,
    ))

    return dest_path


def _find_gguf_repo(hf_repo: str) -> str:
    """Find the actual GGUF repo on HuggingFace.

    Tries multiple patterns since GGUF repos are often uploaded by
    different orgs (unsloth, bartowski, lmstudio-community, etc.)
    """
    from huggingface_hub import list_repo_files

    # Extract model name (without org prefix)
    parts = hf_repo.split("/")
    model_name = parts[-1] if len(parts) > 1 else parts[0]
    model_name_clean = re.sub(r"-GGUF$", "", model_name, flags=re.IGNORECASE)

    # Build list of repos to try
    candidates = [
        hf_repo,                                    # as-is
        f"{hf_repo}-GGUF",                         # with -GGUF suffix
    ]

    # If user gave org/model, also try common GGUF providers
    for provider in ["unsloth", "bartowski", "lmstudio-community"]:
        candidates.append(f"{provider}/{model_name_clean}-GGUF")

    # Also try Qwen/org_model-GGUF pattern (bartowski uses underscores)
    if len(parts) > 1:
        candidates.append(f"bartowski/{parts[0]}_{model_name_clean}-GGUF")

    tried = []
    for candidate in candidates:
        tried.append(candidate)
        try:
            files = list_repo_files(candidate)
            if any(f.endswith(".gguf") for f in files):
                console.print(f"[dim]Found GGUF repo: {candidate}[/dim]")
                return candidate
        except Exception:
            continue

    raise ValueError(
        f"Could not find GGUF models for '{hf_repo}'.\n"
        f"Tried: {', '.join(tried)}\n"
        f"You can also pass a direct GGUF repo: Chat('unsloth/{model_name_clean}-GGUF')"
    )


def _pick_best_gguf(gguf_files: list[str]) -> str:
    """Pick the best GGUF file based on quantization preference.

    Prefers Q4_K_M for best quality/size balance.
    Falls back to whatever is available.
    """
    for quant in _QUANT_PREFERENCE:
        for f in gguf_files:
            if quant in f:
                return f

    # If no known quantization found, pick first file
    return gguf_files[0]


def _detect_context_length(hf_repo: str) -> int:
    """Try to detect context length from model config on HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        import json

        # Try to read config.json from the base repo (without -GGUF)
        base_repo = re.sub(r"-GGUF$", "", hf_repo, flags=re.IGNORECASE)

        config_path = hf_hub_download(
            repo_id=base_repo,
            filename="config.json",
            local_dir=str(_model_dir(hf_repo)),
        )
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
        console.print(f"[green]✓[/green] Removed {hf_repo}")
        return True
    console.print(f"[yellow]![/yellow] {hf_repo} not found in cache")
    return False


def list_downloaded() -> list[str]:
    """List all downloaded model repo names."""
    from zerollm.registry import list_cached

    return [m.hf_repo for m in list_cached()]


def cache_size_mb() -> float:
    """Get total size of the model cache in MB."""
    from zerollm.registry import cache_size_mb as _cache_size
    return _cache_size()
