"""Download GGUF models from Hugging Face Hub and manage local cache."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from zerollm.registry import lookup

console = Console()

CACHE_DIR = Path.home() / ".cache" / "zerollm"


def _model_dir(model_name: str) -> Path:
    """Get the cache directory for a specific model."""
    return CACHE_DIR / model_name.replace(":", "_").replace("/", "_")


def _model_path(model_name: str, filename: str) -> Path:
    """Get the full path to a cached model file."""
    return _model_dir(model_name) / filename


def is_cached(model_name: str) -> bool:
    """Check if a model is already downloaded."""
    info = lookup(model_name)
    return _model_path(model_name, info.filename).exists()


def get_path(model_name: str) -> Path:
    """Get path to cached model, downloading if needed."""
    info = lookup(model_name)
    path = _model_path(model_name, info.filename)

    if path.exists():
        return path

    return download(model_name)


def download(model_name: str) -> Path:
    """Download a model from Hugging Face Hub.

    Returns the local file path.
    """
    from huggingface_hub import hf_hub_download

    info = lookup(model_name)
    dest_dir = _model_dir(model_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / info.filename

    if dest_path.exists():
        console.print(f"[green]✓[/green] {model_name} already downloaded")
        return dest_path

    console.print(
        f"[bold]Downloading {model_name}[/bold] ({info.size_label}) from {info.hf_repo}"
    )

    # Download using huggingface_hub (handles progress internally)
    downloaded_path = hf_hub_download(
        repo_id=info.hf_repo,
        filename=info.filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )

    console.print(f"[green]✓[/green] Downloaded to {dest_path}")
    return Path(downloaded_path)


def remove(model_name: str) -> bool:
    """Remove a model from local cache. Returns True if removed."""
    model_dir = _model_dir(model_name)
    if model_dir.exists():
        shutil.rmtree(model_dir)
        console.print(f"[green]✓[/green] Removed {model_name}")
        return True
    console.print(f"[yellow]![/yellow] {model_name} not found in cache")
    return False


def list_downloaded() -> list[str]:
    """List all downloaded model names."""
    if not CACHE_DIR.exists():
        return []

    from zerollm.registry import list_models

    downloaded = []
    for model in list_models():
        if is_cached(model.name):
            downloaded.append(model.name)
    return downloaded


def cache_size_mb() -> float:
    """Get total size of the model cache in MB."""
    if not CACHE_DIR.exists():
        return 0.0
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
    return total / (1024 * 1024)
