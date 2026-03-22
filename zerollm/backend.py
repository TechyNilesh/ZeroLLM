"""llama-cpp-python backend — single engine for CPU, CUDA, Metal, ROCm."""

from __future__ import annotations

import re
from typing import Generator

from zerollm.hardware import HardwareInfo, compute_n_gpu_layers, compute_threads, detect

# Regex to strip reasoning/thinking tags from model output
# Matches: <think>...</think>, <reasoning>...</reasoning>, <thought>...</thought>, etc.
_THINK_PATTERN = re.compile(
    r"<(think|thinking|reasoning|thought|reflection)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_tags(text: str) -> str:
    """Remove reasoning/thinking tags from model output."""
    cleaned = _THINK_PATTERN.sub("", text)
    return cleaned.strip()


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

        n_threads = compute_threads(power, hw)

        if hw.has_gpu:
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

        Automatically strips <think>/<reasoning>/<thought> tags from output.
        """
        if stream:
            return self._stream(messages, max_tokens, temperature)

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response["choices"][0]["message"]["content"] or ""
        return _strip_think_tags(content)

    def _stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time, stripping think tags."""
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        # Buffer to detect and strip think tags during streaming
        buffer = ""
        inside_think = False

        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue

            buffer += content

            # Check if we're entering a think block
            if not inside_think:
                # Look for opening tag
                for tag in ["<think>", "<thinking>", "<reasoning>", "<thought>", "<reflection>"]:
                    if tag in buffer.lower():
                        # Yield everything before the tag
                        idx = buffer.lower().index(tag)
                        before = buffer[:idx]
                        if before.strip():
                            yield before
                        buffer = buffer[idx:]
                        inside_think = True
                        break

                if not inside_think:
                    # No think tag found — yield completed content
                    # Keep last 20 chars in buffer in case a tag is split across chunks
                    if len(buffer) > 20:
                        yield buffer[:-20]
                        buffer = buffer[-20:]

            # Check if think block is closing
            if inside_think:
                for tag in ["</think>", "</thinking>", "</reasoning>", "</thought>", "</reflection>"]:
                    if tag in buffer.lower():
                        idx = buffer.lower().index(tag) + len(tag)
                        buffer = buffer[idx:]
                        inside_think = False
                        break

        # Yield remaining buffer (if not inside think block)
        if not inside_think and buffer.strip():
            yield buffer.strip()

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

        content = choice.get("content", "") or ""
        return {
            "type": "text",
            "content": _strip_think_tags(content),
        }

    @property
    def context_size(self) -> int:
        """Return the context window size."""
        return self.context_length
