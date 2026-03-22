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
- Improve test coverage
- Fix bugs
- Add examples

## Project structure

```
zerollm/
├── chat.py          # Chat class (ask, stream, REPL)
├── agent.py         # Agent, sub-agents, SharedContext, guardrails, ReAct
├── server.py        # OpenAI-compatible API
├── finetune.py      # LoRA fine-tuning via peft
├── rag.py           # RAG with SQLite + sqlite-vec + reranking
├── backend.py       # HuggingFace transformers backend
├── resolver.py      # Model resolution (HF repo / local dir / fine-tuned)
├── registry.py      # Local cache manager (tracks downloaded models)
├── hardware.py      # Hardware detection (CPU/GPU/RAM)
├── memory.py        # Session + persistent memory with auto-summarization
├── dataloader.py    # File reader (CSV/JSONL/TXT/PDF/DOCX)
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

## Models

ZeroLLM works with any HuggingFace model — no curated registry. Just pass the HF repo name:

```python
Chat("Qwen/Qwen3.5-4B")
Chat("microsoft/Phi-3-mini-4k-instruct")
```

Models are downloaded and cached automatically by HuggingFace Hub.
