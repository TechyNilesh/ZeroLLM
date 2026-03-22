"""Tests for the backend module (without loading a real model)."""

from unittest.mock import MagicMock, patch

from zerollm.backend import LlamaBackend
from zerollm.hardware import HardwareInfo, compute_n_gpu_layers


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


@patch("zerollm.backend.Llama")
def test_backend_init_cpu(mock_llama):
    hw = _mock_hw(gpu_type=None)
    backend = LlamaBackend(
        model_path="/fake/model.gguf",
        context_length=4096,
        power=1.0,
        hw=hw,
    )
    mock_llama.assert_called_once()
    call_kwargs = mock_llama.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 0
    assert call_kwargs["n_ctx"] == 4096


@patch("zerollm.backend.Llama")
def test_backend_init_gpu_full_power(mock_llama):
    hw = _mock_hw(gpu_type="metal")
    backend = LlamaBackend(
        model_path="/fake/model.gguf",
        context_length=8192,
        power=1.0,
        hw=hw,
    )
    call_kwargs = mock_llama.call_args[1]
    assert call_kwargs["n_gpu_layers"] == -1


@patch("zerollm.backend.Llama")
def test_backend_init_gpu_half_power(mock_llama):
    hw = _mock_hw(gpu_type="cuda")
    backend = LlamaBackend(
        model_path="/fake/model.gguf",
        context_length=4096,
        power=0.5,
        hw=hw,
    )
    call_kwargs = mock_llama.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 20  # 50% of 40


@patch("zerollm.backend.Llama")
def test_backend_init_gpu_zero_power(mock_llama):
    hw = _mock_hw(gpu_type="cuda")
    backend = LlamaBackend(
        model_path="/fake/model.gguf",
        context_length=4096,
        power=0.0,
        hw=hw,
    )
    call_kwargs = mock_llama.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 0


@patch("zerollm.backend.Llama")
def test_backend_generate(mock_llama):
    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    mock_llama.return_value = mock_instance

    hw = _mock_hw()
    backend = LlamaBackend("/fake/model.gguf", hw=hw)

    result = backend.generate(
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
    )
    assert result == "Hello!"


@patch("zerollm.backend.Llama")
def test_backend_stream(mock_llama):
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {}}]},
    ]
    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = iter(chunks)
    mock_llama.return_value = mock_instance

    hw = _mock_hw()
    backend = LlamaBackend("/fake/model.gguf", hw=hw)

    tokens = list(backend.generate(
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    ))
    assert tokens == ["Hello", " world"]


@patch("zerollm.backend.Llama")
def test_backend_generate_with_tools_text(mock_llama):
    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Just text", "tool_calls": None}}]
    }
    mock_llama.return_value = mock_instance

    hw = _mock_hw()
    backend = LlamaBackend("/fake/model.gguf", hw=hw)

    result = backend.generate_with_tools(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
    )
    assert result["type"] == "text"
    assert result["content"] == "Just text"


@patch("zerollm.backend.Llama")
def test_backend_generate_with_tools_call(mock_llama):
    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = {
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
    mock_llama.return_value = mock_instance

    hw = _mock_hw()
    backend = LlamaBackend("/fake/model.gguf", hw=hw)

    result = backend.generate_with_tools(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert result["type"] == "tool_call"
    assert result["name"] == "get_weather"
    assert result["arguments"] == {"city": "Auckland"}


@patch("zerollm.backend.Llama")
def test_backend_context_size(mock_llama):
    hw = _mock_hw()
    backend = LlamaBackend("/fake/model.gguf", context_length=8192, hw=hw)
    assert backend.context_size == 8192
