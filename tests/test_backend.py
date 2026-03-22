"""Tests for the backend module (without loading a real model)."""

import sys
from unittest.mock import MagicMock, patch

from zerollm.backend import _strip_think_tags
from zerollm.hardware import HardwareInfo


def _mock_hw(gpu_type=None):
    return HardwareInfo(
        platform="darwin",
        arch="arm64",
        cpu="Apple M2",
        ram_gb=16.0,
        gpu_type=gpu_type,
        gpu_name="Apple M2" if gpu_type else None,
        gpu_vram_gb=16.0 if gpu_type else None,
        n_threads=8,
        recommended_n_gpu_layers=-1 if gpu_type else 0,
        recommended_threads=6,
    )


def _make_backend(hw=None, **kwargs):
    """Create a LlamaBackend with mocked llama_cpp."""
    if hw is None:
        hw = _mock_hw()

    mock_llama_cls = MagicMock()
    mock_llama_module = MagicMock()
    mock_llama_module.Llama = mock_llama_cls

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_module}):
        from zerollm.backend import LlamaBackend
        backend = LlamaBackend("/fake/model.gguf", hw=hw, **kwargs)

    return backend, mock_llama_cls


# ── Init tests ──

def test_backend_init_cpu():
    backend, mock_cls = _make_backend(_mock_hw(gpu_type=None))
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 0


def test_backend_init_gpu_full_power():
    backend, mock_cls = _make_backend(_mock_hw(gpu_type="metal"), power=1.0)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["n_gpu_layers"] == -1


def test_backend_init_gpu_half_power():
    backend, mock_cls = _make_backend(_mock_hw(gpu_type="cuda"), power=0.5)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 20


def test_backend_init_gpu_zero_power():
    backend, mock_cls = _make_backend(_mock_hw(gpu_type="cuda"), power=0.0)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 0


def test_backend_context_size():
    backend, _ = _make_backend(context_length=8192)
    assert backend.context_size == 8192


# ── Generate tests ──

def test_backend_generate():
    backend, mock_cls = _make_backend()
    backend.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    result = backend.generate(messages=[{"role": "user", "content": "Hi"}])
    assert result == "Hello!"


def test_backend_stream():
    backend, _ = _make_backend()
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {}}]},
    ]
    backend.model.create_chat_completion.return_value = iter(chunks)
    tokens = list(backend.generate(messages=[{"role": "user", "content": "Hi"}], stream=True))
    full = "".join(tokens)
    assert "Hello" in full
    assert "world" in full


# ── Tool calling tests ──

def test_backend_generate_with_tools_text():
    backend, _ = _make_backend()
    backend.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Just text", "tool_calls": None}}]
    }
    result = backend.generate_with_tools(messages=[{"role": "user", "content": "Hi"}], tools=[])
    assert result["type"] == "text"
    assert result["content"] == "Just text"


def test_backend_generate_with_tools_call():
    backend, _ = _make_backend()
    backend.model.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Auckland"}',
                    }
                }],
            }
        }]
    }
    result = backend.generate_with_tools(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert result["type"] == "tool_call"
    assert result["name"] == "get_weather"
    assert result["arguments"] == {"city": "Auckland"}


# ── Think tag stripping tests ──

def test_strip_think_tags():
    assert _strip_think_tags("<think>reasoning</think>Answer") == "Answer"


def test_strip_reasoning_tags():
    assert _strip_think_tags("<reasoning>step 1</reasoning>42") == "42"


def test_strip_thinking_tags():
    assert _strip_think_tags("<thinking>hmm</thinking>yes") == "yes"


def test_strip_thought_tags():
    assert _strip_think_tags("<thought>deep</thought>result") == "result"


def test_strip_reflection_tags():
    assert _strip_think_tags("<reflection>check</reflection>done") == "done"


def test_strip_multiline_think():
    text = "<think>\nLet me think about this...\nStep 1\nStep 2\n</think>\n\n4"
    assert _strip_think_tags(text) == "4"


def test_strip_case_insensitive():
    assert _strip_think_tags("<THINK>CAPS</THINK>ok") == "ok"


def test_no_tags_passthrough():
    assert _strip_think_tags("Just a normal response.") == "Just a normal response."


def test_backend_generate_strips_think():
    backend, _ = _make_backend()
    backend.model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "<think>\nreasoning\n</think>\n\n4"}}]
    }
    result = backend.generate(messages=[{"role": "user", "content": "2+2"}])
    assert result == "4"
    assert "<think>" not in result
