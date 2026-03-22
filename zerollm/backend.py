"""llama-cpp-python backend — single engine for CPU, CUDA, Metal, ROCm."""

from __future__ import annotations

from typing import Generator

from zerollm.hardware import HardwareInfo, compute_n_gpu_layers, compute_threads, detect


class LlamaBackend:
    """Wraps llama-cpp-python for inference."""

    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        power: float = 1.0,
        hw: HardwareInfo | None = None,
    ):
        from llama_cpp import Llama

        if hw is None:
            hw = detect()

        self.hw = hw
        self.power = power
        self.context_length = context_length

        # Calculate GPU layers and threads from power setting
        n_threads = compute_threads(power, hw)

        # For GPU layers, we use -1 (all) at power=1.0, scale down with power
        if hw.has_gpu:
            # Use -1 for full GPU, otherwise estimate ~40 layers for typical small models
            if power >= 1.0:
                n_gpu_layers = -1
            elif power <= 0.0:
                n_gpu_layers = 0
            else:
                n_gpu_layers = compute_n_gpu_layers(40, power)
        else:
            n_gpu_layers = 0

        self.model = Llama(
            model_path=model_path,
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Generate a response from messages.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            stream: If True, returns a generator yielding tokens.

        Returns:
            Full response string, or generator of token strings if stream=True.
        """
        if stream:
            return self._stream(messages, max_tokens, temperature)

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response["choices"][0]["message"]["content"]

    def _stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time."""
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a response with tool-calling support.

        Returns a dict with either:
            {"type": "text", "content": "response text"}
            {"type": "tool_call", "name": "func_name", "arguments": {...}}
        """
        response = self.model.create_chat_completion(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response["choices"][0]["message"]

        # Check for tool calls
        tool_calls = choice.get("tool_calls")
        if tool_calls:
            import json

            call = tool_calls[0]
            func = call["function"]
            try:
                args = json.loads(func["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}

            return {
                "type": "tool_call",
                "name": func["name"],
                "arguments": args,
            }

        return {
            "type": "text",
            "content": choice.get("content", ""),
        }

    @property
    def context_size(self) -> int:
        """Return the context window size."""
        return self.context_length
