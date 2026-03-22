"""CLI — command-line interface for ZeroLLM."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="zerollm",
    help="Zero setup, zero config — local LLMs on any hardware.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def list(
    downloaded: bool = typer.Option(False, "--downloaded", "-d", help="Show only downloaded models"),
):
    """List available models."""
    from zerollm.registry import list_models
    from zerollm.downloader import is_cached

    models = list_models()

    table = Table(title="ZeroLLM Models")
    table.add_column("Name", style="bold")
    table.add_column("Size")
    table.add_column("Min RAM")
    table.add_column("Tools")
    table.add_column("License")
    table.add_column("Cached", justify="center")

    for m in models:
        cached = is_cached(m.name)
        if downloaded and not cached:
            continue
        table.add_row(
            m.name,
            m.size_label,
            f"{m.min_ram_gb}GB",
            "Yes" if m.supports_tools else "No",
            m.license,
            "[green]✓[/green]" if cached else "[dim]—[/dim]",
        )

    console.print(table)


@app.command()
def recommend():
    """Recommend the best model for your hardware."""
    from zerollm.hardware import detect
    from zerollm.registry import recommend_for

    hw = detect()

    console.print(f"\n[bold]Hardware Detected[/bold]")
    console.print(f"  {hw.summary()}\n")

    models = recommend_for(hw.ram_gb, hw.has_gpu)

    if not models:
        console.print("[red]No models fit your hardware.[/red] Need at least 3GB RAM.")
        raise typer.Exit(1)

    console.print("[bold]Recommended models:[/bold]")
    for i, m in enumerate(models[:3]):
        prefix = "[green]★[/green]" if i == 0 else " "
        console.print(f"  {prefix} {m.name} ({m.size_label}, needs {m.min_ram_gb}GB RAM)")

    console.print(f"\n  Quick start: [bold]zerollm chat {models[0].name}[/bold]")


@app.command()
def chat(
    model: str = typer.Argument(..., help="Model name (e.g. smollm2, phi3:mini)"),
    power: float = typer.Option(1.0, "--power", "-p", help="Resource usage 0.0-1.0"),
    memory: bool = typer.Option(False, "--memory", "-m", help="Enable conversation memory"),
    system: str = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Start an interactive chat with a local LLM."""
    from zerollm.chat import Chat

    bot = Chat(model=model, power=power, memory=memory, system_prompt=system)
    bot.chat()


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    power: float = typer.Option(1.0, "--power", help="Resource usage 0.0-1.0"),
):
    """Serve a model as an OpenAI-compatible API."""
    from zerollm.server import Server

    server = Server(model=model, power=power, port=port, host=host)
    server.serve()


@app.command()
def download(
    model: str = typer.Argument(..., help="Model name to download"),
):
    """Download a model to local cache."""
    from zerollm.downloader import download as dl

    dl(model)


@app.command()
def remove(
    model: str = typer.Argument(..., help="Model name to remove"),
):
    """Remove a model from local cache."""
    from zerollm.downloader import remove as rm

    rm(model)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name"),
):
    """Show detailed information about a model."""
    from zerollm.registry import lookup
    from zerollm.downloader import is_cached

    m = lookup(model)
    cached = is_cached(model)

    console.print(f"\n[bold]{m.name}[/bold]")
    console.print(f"  HF Repo:        {m.hf_repo}")
    console.print(f"  File:           {m.filename}")
    console.print(f"  Size:           {m.size_label}")
    console.print(f"  Min RAM:        {m.min_ram_gb}GB")
    console.print(f"  Context Length: {m.context_length}")
    console.print(f"  Chat Template:  {m.chat_template}")
    console.print(f"  Tool Calling:   {'Yes' if m.supports_tools else 'No'}")
    console.print(f"  License:        {m.license}")
    console.print(f"  Cached:         {'[green]Yes[/green]' if cached else '[dim]No[/dim]'}")
    console.print()


@app.command()
def doctor():
    """Diagnose your setup and check if everything works."""
    from zerollm.hardware import detect
    from zerollm.downloader import cache_size_mb, list_downloaded

    console.print("\n[bold]ZeroLLM Doctor[/bold]\n")

    # Hardware
    hw = detect()
    console.print(f"[bold]Hardware:[/bold] {hw.summary()}")
    console.print(f"  Platform:  {hw.platform} ({hw.arch})")
    console.print(f"  CPU:       {hw.cpu}")
    console.print(f"  RAM:       {hw.ram_gb}GB")
    console.print(f"  GPU:       {hw.gpu_type or 'None'} ({hw.gpu_name or 'N/A'})")
    console.print(f"  Threads:   {hw.n_threads}")

    # Backend
    console.print()
    try:
        import llama_cpp
        console.print(f"[green]✓[/green] llama-cpp-python installed ({llama_cpp.__version__})")
    except ImportError:
        console.print("[red]✗[/red] llama-cpp-python not installed")

    # Cache
    downloaded = list_downloaded()
    size = cache_size_mb()
    console.print(f"\n[bold]Cache:[/bold] {len(downloaded)} models ({size:.0f}MB)")
    for name in downloaded:
        console.print(f"  • {name}")

    console.print()


if __name__ == "__main__":
    app()
