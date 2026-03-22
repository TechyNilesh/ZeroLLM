"""Tests for the Server class (mocked backend)."""

from unittest.mock import patch, MagicMock

import pytest


def _create_test_app():
    """Create a Server with mocked backend and return the FastAPI app."""
    from zerollm.resolver import ResolvedModel

    mock_resolved = ResolvedModel(
        name="Qwen/Qwen3.5-4B",
        path="/fake/model.gguf",
        context_length=8192,
        source="registry",
        supports_tools=True,
    )

    with patch("zerollm.server.resolve", return_value=mock_resolved), \
         patch("zerollm.server.LlamaBackend") as mock_backend:

        mock_instance = MagicMock()
        mock_instance.generate.return_value = "Hello from ZeroLLM!"
        mock_backend.return_value = mock_instance

        from zerollm.server import Server
        server = Server(model="Qwen/Qwen3.5-4B")
        server._mock_backend = mock_instance
        return server


def test_server_init():
    server = _create_test_app()
    assert server.model_name == "Qwen/Qwen3.5-4B"
    assert server.app is not None


def test_health_endpoint():
    from fastapi.testclient import TestClient

    server = _create_test_app()
    client = TestClient(server.app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "Qwen/Qwen3.5-4B"


def test_list_models_endpoint():
    from fastapi.testclient import TestClient

    server = _create_test_app()
    client = TestClient(server.app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "Qwen/Qwen3.5-4B"


def test_chat_completions_endpoint():
    from fastapi.testclient import TestClient

    server = _create_test_app()
    client = TestClient(server.app)

    response = client.post("/v1/chat/completions", json={
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Hello from ZeroLLM!"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_completions_endpoint():
    from fastapi.testclient import TestClient

    server = _create_test_app()
    client = TestClient(server.app)

    response = client.post("/v1/completions", json={
        "prompt": "Hello",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == "Hello from ZeroLLM!"


def test_chat_completions_streaming():
    from fastapi.testclient import TestClient

    server = _create_test_app()
    server._mock_backend.generate.return_value = iter(["Hello", " world"])

    client = TestClient(server.app)

    response = client.post("/v1/chat/completions", json={
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200
    # Streaming returns text/event-stream
    assert "text/event-stream" in response.headers["content-type"]


def test_chat_completions_custom_params():
    from fastapi.testclient import TestClient

    server = _create_test_app()
    client = TestClient(server.app)

    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50,
        "temperature": 0.1,
    })
    assert response.status_code == 200
    # Verify backend was called with custom params
    server._mock_backend.generate.assert_called_once()
    call_kwargs = server._mock_backend.generate.call_args[1]
    assert call_kwargs["max_tokens"] == 50
    assert call_kwargs["temperature"] == 0.1
