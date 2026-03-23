"""OpenAI-compatible REST API server."""

import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from zerollm.backend import HFBackend
from zerollm.hardware import detect
from zerollm.resolver import resolve


# ── Request models (module-level for FastAPI compatibility) ──


class _Message(BaseModel):
    role: str
    content: str


class _ChatRequest(BaseModel):
    model: str = ""
    messages: list[_Message]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False


class _CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7


class Server:
    """Serve a local LLM as an OpenAI-compatible API.

    Usage:
        Server("Qwen/Qwen3.5-4B", port=8080).serve()

    Endpoints:
        POST /v1/chat/completions  — chat completions
        POST /v1/completions       — text completions
        GET  /v1/models            — list loaded model
        GET  /health               — health check
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-4B",
        power: float = 1.0,
        port: int = 8000,
        host: str = "0.0.0.0",
    ):
        self.port = port
        self.host = host

        # Resolve model — handles registry, local GGUF, and fine-tuned models
        resolved = resolve(model)
        self.model_name = resolved.name
        hw = detect()

        self.backend = HFBackend(
            model_name=resolved.model_id,
            context_length=resolved.context_length,
            power=power,
            hw=hw,
        )
        self.app = self._create_app()

    def _create_app(self):
        """Create FastAPI application with OpenAI-compatible routes."""
        app = FastAPI(title="ZeroLLM API", version="0.1.0")

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": self.model_name}

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "owned_by": "zerollm",
                    }
                ],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(req: _ChatRequest):
            messages = [{"role": m.role, "content": m.content} for m in req.messages]

            if req.stream:
                return StreamingResponse(
                    self._stream_chat(messages, req.max_tokens, req.temperature),
                    media_type="text/event-stream",
                )

            response_text = self.backend.generate(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            }

        @app.post("/v1/completions")
        async def completions(req: _CompletionRequest):
            messages = [{"role": "user", "content": req.prompt}]

            response_text = self.backend.generate(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )

            return {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": response_text,
                        "finish_reason": "stop",
                    }
                ],
            }

        return app

    def _stream_chat(self, messages, max_tokens, temperature):
        """SSE stream generator for chat completions."""
        import json

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        for token in self.backend.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    def serve(self) -> None:
        """Start the API server."""
        import uvicorn
        from rich.console import Console

        console = Console()
        console.print(
            f"\n[bold]ZeroLLM API Server[/bold]\n"
            f"  Model:  {self.model_name}\n"
            f"  URL:    http://{self.host}:{self.port}\n"
            f"  Docs:   http://{self.host}:{self.port}/docs\n"
        )

        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
