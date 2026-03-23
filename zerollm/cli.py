"""CLI — command-line interface for ZeroLLM."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="zerollm",
    help="Zero setup, zero config — local LLMs on any hardware. Use any HuggingFace GGUF model.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def list():
    """List downloaded models in local cache."""
    from zerollm.registry import list_cached

    models = list_cached()

    if not models:
        console.print("[dim]No models downloaded yet.[/dim]")
        console.print("Download one with: [bold]zerollm chat Qwen/Qwen3.5-4B[/bold]")
        return

    table = Table(title="Downloaded Models")
    table.add_column("HF Repo", style="bold")
    table.add_column("File")
    table.add_column("Size")
    table.add_column("Context")

    for m in models:
        table.add_row(
            m.hf_repo,
            m.filename,
            f"{m.size_mb}MB",
            str(m.context_length),
        )

    console.print(table)


@app.command()
def chat(
    model: str = typer.Argument("Qwen/Qwen3.5-4B", help="HuggingFace model name"),
    power: float = typer.Option(1.0, "--power", "-p", help="Resource usage 0.0-1.0"),
    memory: bool = typer.Option(False, "--memory", "-m", help="Enable conversation memory"),
    system: str = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Start an interactive chat with any HuggingFace model."""
    from zerollm.chat import Chat

    bot = Chat(model=model, power=power, memory=memory, system_prompt=system)
    bot.chat()


@app.command()
def serve(
    model: str = typer.Argument("Qwen/Qwen3.5-4B", help="HuggingFace model name"),
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
    model: str = typer.Argument(..., help="HuggingFace model name to download"),
):
    """Download a model from HuggingFace to local cache."""
    from zerollm.downloader import download as dl

    dl(model)


@app.command()
def remove(
    model: str = typer.Argument(..., help="HuggingFace model name to remove"),
):
    """Remove a model from local cache."""
    from zerollm.downloader import remove as rm

    rm(model)


@app.command()
def info(
    model: str = typer.Argument(..., help="HuggingFace model name"),
):
    """Show info about a cached model."""
    from zerollm.registry import lookup_cache

    cached = lookup_cache(model)

    if cached is None:
        console.print(f"[yellow]![/yellow] {model} not in cache. Download it first:")
        console.print(f"  [bold]zerollm download {model}[/bold]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{cached.hf_repo}[/bold]")
    console.print(f"  File:           {cached.filename}")
    console.print(f"  Size:           {cached.size_mb}MB")
    console.print(f"  Context Length: {cached.context_length}")
    console.print(f"  Local Path:     {cached.local_path}")
    console.print()


@app.command()
def doctor():
    """Diagnose your setup and check if everything works."""
    from zerollm.hardware import detect
    from zerollm.downloader import list_downloaded, cache_size_mb

    console.print("\n[bold]ZeroLLM Doctor[/bold]\n")

    hw = detect()
    console.print(f"[bold]Hardware:[/bold] {hw.summary()}")
    console.print(f"  Platform:  {hw.platform} ({hw.arch})")
    console.print(f"  CPU:       {hw.cpu}")
    console.print(f"  RAM:       {hw.ram_gb}GB")
    console.print(f"  GPU:       {hw.gpu_type or 'None'} ({hw.gpu_name or 'N/A'})")
    console.print(f"  Threads:   {hw.n_threads}")

    console.print()
    try:
        import llama_cpp
        console.print(f"[green]✓[/green] llama-cpp-python installed ({llama_cpp.__version__})")
    except ImportError:
        console.print("[red]✗[/red] llama-cpp-python not installed")

    downloaded = list_downloaded()
    size = cache_size_mb()
    console.print(f"\n[bold]Cache:[/bold] {len(downloaded)} models ({size:.0f}MB)")
    for name in downloaded:
        console.print(f"  {name}")

    console.print()


if __name__ == "__main__":
    app()
