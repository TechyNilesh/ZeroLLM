"""Chat class — the main user-facing interface for talking to local LLMs."""

from __future__ import annotations

from typing import Generator

from rich.console import Console

from zerollm.backend import HFBackend
from zerollm.hardware import detect
from zerollm.memory import Memory
from zerollm.resolver import resolve

console = Console()


class Chat:
    """Talk to a local LLM with zero configuration.

    Usage:
        bot = Chat("Qwen/Qwen3.5-4B")
        print(bot.ask("What is the capital of France?"))

        for token in bot.stream("Tell me a joke"):
            print(token, end="", flush=True)

        bot.chat()  # interactive REPL
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-4B",
        power: float = 1.0,
        memory: bool = False,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Initialize Chat.

        Args:
            model: Registry name, local GGUF path, or fine-tuned model directory.
                   Examples:
                     "Qwen/Qwen3.5-4B"  (registry)
                     "/path/to/model.gguf"                    (local GGUF)
                     "my-bot"                                 (fine-tuned, saved with FineTuner)
                     "~/.cache/zerollm/models/my-bot"         (fine-tuned, full path)
            power: Resource usage 0.0-1.0 (GPU layers + CPU threads).
            memory: Enable conversation memory.
            system_prompt: Optional system prompt.
            max_tokens: Default max tokens for generation.
            temperature: Default sampling temperature.
        """
        self.power = power
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Resolve model — handles registry, local GGUF, and fine-tuned models
        console.print(f"[dim]Loading {model}...[/dim]")
        resolved = resolve(model)
        self.model_name = resolved.name
        self.hw = detect()

        # Create backend
        self.backend = HFBackend(
            model_name=resolved.model_id,
            context_length=resolved.context_length,
            power=power,
            hw=self.hw,
        )

        # Set up memory
        self.memory = Memory(persist=memory)
        if system_prompt:
            self.memory.add_system(system_prompt)

        console.print(f"[green]✓[/green] {model} ready ({self.hw.summary()})")

    def ask(self, prompt: str) -> str:
        """Send a prompt and get a response.

        Args:
            prompt: Your message to the model.

        Returns:
            The model's response as a string.
        """
        self.memory.add("user", prompt)
        messages = self.memory.get_context()

        response = self.backend.generate(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )

        self.memory.add("assistant", response)
        self.memory.maybe_summarize(backend=self.backend)
        return response

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Send a prompt and stream the response token by token.

        Args:
            prompt: Your message to the model.

        Yields:
            Response tokens one at a time.
        """
        self.memory.add("user", prompt)
        messages = self.memory.get_context()

        full_response = []
        for token in self.backend.generate(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        ):
            full_response.append(token)
            yield token

        self.memory.add("assistant", "".join(full_response))

    def chat(self) -> None:
        """Start an interactive REPL chat in the terminal."""
        console.print(
            f"\n[bold]ZeroLLM Chat[/bold] — {self.model_name}\n"
            f"Type your message and press Enter. Type [bold]/quit[/bold] to exit.\n"
        )

        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/reset":
                self.memory.clear()
                console.print("[dim]Conversation reset.[/dim]\n")
                continue

            # Stream response
            console.print("[bold green]Bot:[/bold green] ", end="")
            response_parts = []
            for token in self.stream(user_input):
                console.print(token, end="", highlight=False)
                response_parts.append(token)
            console.print("\n")

    def reset(self) -> None:
        """Clear conversation history."""
        self.memory.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self.memory.get_full_history()
