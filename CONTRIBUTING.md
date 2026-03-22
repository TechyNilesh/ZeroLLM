# Contributing to ZeroLLM

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/TechyNilesh/ZeroLLM.git
cd ZeroLLM
uv venv && uv sync
```

## Making changes

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `uv run pytest`
5. Commit and push
6. Open a Pull Request

## What to work on

- Check [Issues](https://github.com/TechyNilesh/ZeroLLM/issues) for open tasks
- Add new models to `zerollm/registry.json`
- Improve test coverage
- Fix bugs
- Add examples

## Project structure

```
zerollm/
├── chat.py          # Chat class
├── agent.py         # Agent + sub-agents
├── server.py        # OpenAI-compatible API
├── finetune.py      # LoRA fine-tuning
├── rag.py           # RAG with SQLite + sqlite-vec
├── backend.py       # llama-cpp-python wrapper
├── resolver.py      # Model resolution (registry/local/fine-tuned)
├── registry.py      # Model registry loader
├── registry.json    # Curated model list
├── hardware.py      # Hardware detection
├── memory.py        # Conversation memory
├── dataloader.py    # File reader (CSV/JSONL/PDF/DOCX)
├── cli.py           # CLI commands
└── __init__.py      # Public API
```

## Code style

- Python 3.10+
- Use `ruff` for linting: `uv run ruff check .`
- Keep it simple — avoid over-engineering
- No `from __future__ import annotations` in modules that use FastAPI/Pydantic

## Tests

```bash
uv run pytest           # all tests
uv run pytest tests/test_chat.py  # single file
```

Tests that need a real model are skipped by default. Unit tests use mocked backends.

## Adding a model to the registry

Edit `zerollm/registry.json`:

```json
"org/Model-Name": {
  "hf_repo": "org/Model-Name-GGUF",
  "hf_base_repo": "org/Model-Name",
  "filename": "model-name-Q8_0.gguf",
  "size_mb": 640,
  "min_ram_gb": 3,
  "context_length": 32768,
  "chat_template": "chatml",
  "supports_tools": true,
  "license": "Apache-2.0"
}
```

Verify the GGUF filename exists on HuggingFace before submitting.
