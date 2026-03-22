"""HuggingFace transformers backend — one engine for all hardware."""

from __future__ import annotations

import re
from typing import Generator

from zerollm.hardware import HardwareInfo, detect

# Regex to strip reasoning/thinking tags from model output
_THINK_PATTERN = re.compile(
    r"<(think|thinking|reasoning|thought|reflection)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Also match content before a closing tag without opening tag
# (some models/tokenizers strip the opening tag)
_ORPHAN_CLOSE_PATTERN = re.compile(
    r"^.*?</(think|thinking|reasoning|thought|reflection)>\s*",
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_tags(text: str) -> str:
    """Remove reasoning/thinking tags from model output."""
    # First strip matched pairs
    result = _THINK_PATTERN.sub("", text)
    # Then strip orphaned closing tags (content before </think> without <think>)
    result = _ORPHAN_CLOSE_PATTERN.sub("", result)
    return result.strip()


class HFBackend:
    """Wraps HuggingFace transformers for inference.

    Uses bitsandbytes INT4 on GPU, FP32 on CPU.
    Supports streaming via TextIteratorStreamer.
    """

    def __init__(
        self,
        model_name: str,
        context_length: int = 4096,
        power: float = 1.0,
        hw: HardwareInfo | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if hw is None:
            hw = detect()

        self.hw = hw
        self.power = power
        self.context_length = context_length
        self.model_name = model_name

        # Load with INT4 quantization on GPU, FP32 on CPU
        if hw.has_gpu and power > 0:
            try:
                from transformers import BitsAndBytesConfig

                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=qconfig,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except (ImportError, Exception):
                # bitsandbytes failed — load FP16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Generate a response. Strips think tags automatically."""
        if stream:
            return self._stream(messages, max_tokens, temperature)

        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _strip_think_tags(response)

    def _stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream tokens using TextIteratorStreamer."""
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        buffer = ""
        inside_think = False

        for token in streamer:
            if not token:
                continue

            buffer += token

            if not inside_think:
                for tag in ["<think>", "<thinking>", "<reasoning>", "<thought>", "<reflection>"]:
                    if tag in buffer.lower():
                        idx = buffer.lower().index(tag)
                        before = buffer[:idx]
                        if before.strip():
                            yield before
                        buffer = buffer[idx:]
                        inside_think = True
                        break

                if not inside_think and len(buffer) > 20:
                    yield buffer[:-20]
                    buffer = buffer[-20:]

            if inside_think:
                for tag in ["</think>", "</thinking>", "</reasoning>", "</thought>", "</reflection>"]:
                    if tag in buffer.lower():
                        idx = buffer.lower().index(tag) + len(tag)
                        buffer = buffer[idx:]
                        inside_think = False
                        break

        thread.join()

        if not inside_think and buffer.strip():
            yield buffer.strip()

    def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Generate with tool-calling support.

        Returns:
            {"type": "text", "content": "..."} or
            {"type": "tool_call", "name": "...", "arguments": {...}}
        """
        import json

        tool_text = json.dumps(tools, indent=2)
        tool_system = (
            "You have access to the following tools:\n"
            f"{tool_text}\n\n"
            "To call a tool, respond with EXACTLY this JSON format and nothing else:\n"
            '{"tool_call": {"name": "function_name", "arguments": {"arg": "value"}}}\n\n'
            "If you don't need a tool, respond normally with text."
        )

        enhanced = list(messages)
        if enhanced and enhanced[0]["role"] == "system":
            enhanced[0] = {
                "role": "system",
                "content": enhanced[0]["content"] + "\n\n" + tool_system,
            }
        else:
            enhanced.insert(0, {"role": "system", "content": tool_system})

        response = self.generate(enhanced, max_tokens=max_tokens, temperature=temperature)

        # Try to parse tool call
        try:
            parsed = json.loads(response.strip())
            if "tool_call" in parsed:
                call = parsed["tool_call"]
                return {
                    "type": "tool_call",
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Try to find tool call in text
        tool_match = re.search(r'\{"tool_call":\s*\{.*?\}\}', response, re.DOTALL)
        if tool_match:
            try:
                parsed = json.loads(tool_match.group())
                call = parsed["tool_call"]
                return {
                    "type": "tool_call",
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                }
            except (json.JSONDecodeError, KeyError):
                pass

        return {"type": "text", "content": response}

    def generate_json(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> dict:
        """Generate a structured JSON response.

        Args:
            messages: Chat messages.
            schema: Optional JSON schema to validate against.
            max_tokens: Max tokens.
            temperature: Low temperature recommended for structured output.

        Returns:
            Parsed JSON dict.
        """
        import json

        # Add JSON instruction to the prompt
        json_instruction = "Respond with valid JSON only. No markdown, no explanation, just JSON."
        if schema:
            json_instruction += f"\n\nExpected schema:\n{json.dumps(schema, indent=2)}"

        enhanced = list(messages)
        if enhanced and enhanced[0]["role"] == "system":
            enhanced[0] = {
                "role": "system",
                "content": enhanced[0]["content"] + "\n\n" + json_instruction,
            }
        else:
            enhanced.insert(0, {"role": "system", "content": json_instruction})

        # Try up to 3 times to get valid JSON
        for attempt in range(3):
            response = self.generate(enhanced, max_tokens=max_tokens, temperature=temperature)

            # Try to extract JSON from response
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                pass

            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Retry with explicit error message
            enhanced.append({"role": "assistant", "content": response})
            enhanced.append({
                "role": "user",
                "content": "That was not valid JSON. Please respond with ONLY valid JSON, nothing else.",
            })

        # Final fallback
        return {"error": "Failed to generate valid JSON", "raw": response}

    @property
    def context_size(self) -> int:
        return self.context_length
