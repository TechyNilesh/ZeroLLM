# Changelog

All notable changes to ZeroLLM will be documented in this file.

## [Unreleased]

### Added
- **HuggingFace-first model resolution** — pass any HF repo name, ZeroLLM auto-finds GGUF, picks best quantization, downloads
- **Multi-provider GGUF search** — searches unsloth, bartowski, lmstudio-community repos automatically
- **Think tag stripping** — strips `<think>`, `<reasoning>`, `<thought>`, `<thinking>`, `<reflection>` tags from all reasoning models
- **SharedContext** — shared memory space for multi-agent communication
- **Pipeline** — run agents sequentially with automatic context passing
- **Sub-agent context sharing** — sub-agents receive parent conversation context + shared context
- **Auto context length detection** — reads `config.json` from HF to detect model context window
- **Cache index** — local SQLite-free JSON cache tracking downloaded models

### Changed
- Registry is now a cache manager (no more curated model list)
- Default model: `Qwen/Qwen3.5-4B`
- Resolver tries multiple GGUF repo patterns (org/model-GGUF, unsloth/model-GGUF, etc.)
- Quantization auto-pick prefers Q4_K_M > Q5_K_M > Q8_0

### Removed
- Curated `registry.json` model list
- `recommend()` function

## [0.1.6] - 2026-03-23

### Fixed
- Server 422 error — moved Pydantic models to module level (FastAPI + `from __future__ import annotations` conflict)

## [0.1.5] - 2026-03-23

### Fixed
- Server chat completions validation error with Pydantic defaults

## [0.1.4] - 2026-03-23

### Fixed
- GGUF filenames in registry (Qwen3 official only has Q8_0, not Q4_K_M)
- Removed gated gemma model from registry (401 on download)

## [0.1.3] - 2026-03-23

### Fixed
- CUDA detection crash: `total_mem` → `total_memory` in `torch.cuda.get_device_properties()`

## [0.1.2] - 2026-03-23

### Fixed
- README images not loading on PyPI (switched to absolute GitHub URLs)
- Removed broken BUILD badge

## [0.1.1] - 2026-03-23

### Fixed
- README images for PyPI (relative → absolute URLs)

## [0.1.0] - 2026-03-23

### Added
- **Chat** — `ask()`, `stream()`, interactive REPL
- **Agent** — `@tool` decorator, JSON schema from type hints, multi-turn tool calling
- **Sub-agents** — `add_agent()` for task delegation
- **Server** — OpenAI-compatible REST API (FastAPI + uvicorn)
- **FineTuner** — LoRA fine-tuning via peft + transformers
- **RAG** — SQLite + sqlite-vec hybrid search (70% vector + 30% BM25)
- **DataLoader** — CSV, JSONL, TXT, PDF, DOCX support
- **Model resolver** — registry, local GGUF, fine-tuned model directories
- **Hardware detection** — auto-detect CPU/GPU/RAM with power control
- **CLI** — list, chat, serve, download, remove, info, doctor
- **Memory** — session + SQLite persistent memory
- Single backend: llama-cpp-python (CPU/CUDA/Metal/ROCm)
- 7 curated models in registry
- 13 test files, 19 examples
- MIT license
