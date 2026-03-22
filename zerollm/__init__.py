"""ZeroLLM — Zero setup, zero config local LLMs on any hardware.

Usage:
    from zerollm import Chat
    bot = Chat("Qwen/Qwen3-0.6B")
    print(bot.ask("Hello!"))
"""

__version__ = "0.1.2"

from zerollm.chat import Chat
from zerollm.agent import Agent
from zerollm.server import Server
from zerollm.finetune import FineTuner
from zerollm.rag import RAG
from zerollm.hardware import detect as _detect


def recommend():
    """Recommend the best model for your hardware."""
    from rich.console import Console
    from zerollm.registry import recommend_for

    console = Console()
    hw = _detect()

    console.print(f"\n[bold]Detected:[/bold] {hw.summary()}")

    models = recommend_for(hw.ram_gb, hw.has_gpu)
    if not models:
        console.print("[red]No models fit your hardware.[/red]")
        return

    console.print(f"[green]Recommended:[/green]  {models[0].name} ({models[0].size_label})")
    if len(models) > 1:
        console.print(f"[dim]Also good:[/dim]    {models[1].name} ({models[1].size_label})")
    if len(models) > 2:
        console.print(f"[dim]Minimum:[/dim]      {models[-1].name} ({models[-1].size_label})")
    console.print()


__all__ = [
    "Chat",
    "Agent",
    "Server",
    "FineTuner",
    "RAG",
    "recommend",
]
