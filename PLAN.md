# ZeroLLM - Complete Project Plan

> The zero-config Python library for running, fine-tuning, serving, and building
> agents with local LLMs on any hardware: CPU, NVIDIA GPU, or Apple Silicon.

---

## 1. Mission Statement

**"pip install zerollm - and you're done."**

A single Python package that:
- Auto-detects your hardware (CPU / NVIDIA GPU / Apple Silicon)
- Picks and downloads the right local model for your machine
- Lets you chat, fine-tune, serve as API, and build agents
- Never requires you to touch model files, formats, or backend config
- Feels zero-setup for first-time local LLM users

---

## 2. Naming

**Canonical naming**
- Product name: `ZeroLLM`
- Python package / import: `zerollm`
- CLI command: `zerollm`
- Cache root: `~/.cache/zerollm/`

**One install. Everything included. No extras, no choices.**

```bash
uv pip install zerollm
```

That's it. All features ship in one package.

**Development uses `uv`** — the fast Python package manager and project tool:
```bash
uv init zerollm          # create project
uv add llama-cpp-python   # add dependency
uv run zerollm chat       # run CLI
uv build                  # build for PyPI
uv publish                # publish to PyPI
```

---

## 3. Core Pillars

| Pillar    | Class         | One-liner                          |
|-----------|---------------|------------------------------------|
| Chat      | `Chat()`      | Talk to a local model              |
| Fine-Tune | `FineTuner()` | Train on your own data             |
| Serve     | `Server()`    | Expose as OpenAI-compatible API    |
| Agent     | `Agent()`     | Build tool-calling multi-turn bots |

---

## 4. Product Positioning

ZeroLLM should mean:
- Zero setup
- Local-first
- Beginner-friendly
- Smart defaults over backend complexity

The real value is not just backend wrapping. The value is orchestration:
- Detect hardware automatically
- Choose a model that fits
- Download and cache it safely
- Pick the right runtime
- Expose one simple API and CLI

---

## 5. Supported Hardware

| Platform                    | Backend Used            | Auto-detected? |
|-----------------------------|-------------------------|----------------|
| Any CPU (Windows/Linux/Mac) | llama-cpp-python (CPU)   | Yes            |
| NVIDIA GPU (CUDA)           | llama-cpp-python (CUDA)  | Yes            |
| Apple Silicon (M1/M2/M3/M4) | llama-cpp-python (Metal) | Yes            |
| Intel Mac                   | llama-cpp-python (CPU)   | Yes            |
| AMD GPU (ROCm)              | llama-cpp-python (ROCm)  | Yes            |
| Low-end / Raspberry Pi      | llama-cpp-python (CPU)   | Yes            |

---

## 6. Model Registry

- Models sourced from Hugging Face Hub in GGUF format
- No login required for public models
- Downloaded once, cached at `~/.cache/zerollm/`
- Registry is a versioned JSON file, auto-synced from HF Hub

### Curated Starter Models (v1)

| Friendly Name | Base Model               | Size  | Min RAM | Tool Calling | License    |
|---------------|--------------------------|-------|---------|--------------|------------|
| tinyllama     | TinyLlama-1.1B-Chat      | 637MB | 3 GB    | No           | Apache 2.0 |
| smollm2       | SmolLM2-1.7B-Instruct    | 1.1GB | 4 GB    | Yes          | Apache 2.0 |
| qwen2.5:0.5b  | Qwen2.5-0.5B-Instruct    | 400MB | 3 GB    | Yes          | Apache 2.0 |
| qwen2.5:1.5b  | Qwen2.5-1.5B-Instruct    | 900MB | 4 GB    | Yes          | Apache 2.0 |
| gemma3:1b     | Gemma-3-1B-Instruct      | 500MB | 4 GB    | Yes          | Gemma ToS  |
| phi3:mini     | Phi-3-Mini-3.8B-Instruct | 2.2GB | 6 GB    | Yes          | MIT        |
| deepseek:1.5b | DeepSeek-R1-Distill-1.5B | 1.1GB | 4 GB    | Yes          | MIT        |

---

## 7. API Design

### 7.1 Chat

```python
from zerollm import Chat

bot = Chat("smollm2")                    # auto hardware detect
bot = Chat("phi3:mini", power=0.8)       # use 80% of resources
bot = Chat("llama3:8b", backend="cuda")  # force GPU

print(bot.ask("What is the capital of France?"))

for token in bot.stream("Tell me a joke"):
    print(token, end="", flush=True)

bot.chat()
```

### 7.2 Fine-Tuner

```python
from zerollm import FineTuner

tuner = FineTuner("smollm2", power=0.7)

# Train from a list of prompt/response pairs
tuner.train([
    {"prompt": "What is your refund policy?", "response": "Return within 30 days."},
    {"prompt": "How do I reset my password?", "response": "Click Forgot Password."},
], epochs=3)

# Train from files — dataloader auto-detects format
tuner.train("my_data.csv", epochs=5)        # CSV with prompt,response columns
tuner.train("my_data.jsonl", epochs=5)      # JSONL with prompt/response keys
tuner.train("my_data.txt", epochs=5)        # Plain text (auto-chunked for CLM)
tuner.train("my_data.pdf", epochs=5)        # PDF extracted → chunked → trained

# Save and share
tuner.save("my-support-bot")                # save LoRA adapter
tuner.save("my-support-bot", merge=True)    # merge into full GGUF
tuner.push("username/my-support-bot", token="hf_xxx")  # push to HF Hub
```

### 7.3 Server

```python
from zerollm import Server

Server("smollm2", power=0.8, port=8080).serve()

# GET  http://localhost:8080/v1/models
# POST http://localhost:8080/v1/chat/completions
# POST http://localhost:8080/v1/completions
# GET  http://localhost:8080/health
```

### 7.4 Agent

```python
from zerollm import Agent

agent = Agent("smollm2", power=0.8)

@agent.tool
def get_weather(city: str) -> str:
    return f"22C and sunny in {city}"

@agent.tool
def search_web(query: str) -> str:
    return f"Search results for: {query}"

agent.ask("What is the weather in Auckland?")
agent.chat()
agent.serve(port=8081)
```

---

## 8. Hardware Manager

### Power Control Table

| Setting    | GPU Behaviour             | CPU Behaviour              |
|------------|---------------------------|----------------------------|
| power=1.0  | All layers on GPU         | All threads, max priority  |
| power=0.8  | 80% layers on GPU         | 80% of CPU threads         |
| power=0.5  | Half GPU, half CPU        | Half threads, low priority |
| power=0.0  | CPU only (`n_gpu_layers=0`) | All threads, CPU mode only |

### Auto Recommender

```python
from zerollm import recommend

recommend()

# Detected: Apple M2, 16GB unified memory, Metal GPU
# Recommended: phi3:mini (fits fully, Metal acceleration)
# Also good: smollm2 (faster, lighter)
# Minimum: qwen2.5:0.5b (ultra fast)
```

---

## 9. Memory Layer (v1)

```python
bot = Chat("smollm2", memory=True)
bot.ask("My name is Nilesh")
bot.ask("What is my name?")   # returns: Your name is Nilesh
```

### Memory types

| Type           | Storage                        | Purpose                              |
|----------------|--------------------------------|--------------------------------------|
| Session memory | In-memory list                 | Recent messages kept in prompt context |
| Summary memory | SQLite (`~/.cache/zerollm/memory.db`) | Old turns auto-summarized to save context |
| Agent memory   | SQLite (`~/.cache/zerollm/memory.db`) | Tool results, goals, preferences across turns |

All persistent memory uses the same SQLite database as RAG — one file, zero config.
When context fills up, old messages are summarized by the model and stored in SQLite for retrieval.

---

## 10. RAG Layer (v2)

```python
from zerollm import RAG

rag = RAG("smollm2")
rag.add("company_docs.pdf")
rag.add("faq.txt")
rag.ask("What is the return policy?")
```

### Storage: SQLite + sqlite-vec (inspired by OpenClaw)

No ChromaDB. No external server. Just a single `.db` file.

| Component        | Technology          | Purpose                              |
|------------------|---------------------|--------------------------------------|
| Vector search    | sqlite-vec          | Cosine similarity on embeddings      |
| Keyword search   | SQLite FTS5         | BM25 full-text search (built into SQLite) |
| Hybrid scoring   | 70% vector + 30% BM25 | Best of both — semantic + exact match |
| Embeddings       | sentence-transformers | Local embeddings, no API key needed  |
| Storage          | SQLite              | Single file at `~/.cache/zerollm/rag.db` |
| Document parsing | dataloader.py       | PDF, TXT, DOCX, CSV, JSONL loading   |

### Why SQLite over ChromaDB?

| | ChromaDB | SQLite + sqlite-vec |
|---|---|---|
| Dependencies | Heavy (many transitive deps) | Tiny (`sqlite-vec` only, SQLite is built into Python) |
| Server process | Runs its own server | No server — just a file |
| Install size | Large | Negligible |
| Hybrid search | Vector only | Vector + keyword (BM25) — better results |
| Zero config | Needs setup | Works out of the box |
| Proven | Yes | Yes (OpenClaw uses this, 248K+ GitHub stars) |

### How RAG works internally

```text
rag.add("document.pdf")
        |
        v
dataloader.py -> reads file, extracts text
        |
        v
dataloader.py -> chunks text (~400 tokens, 80-token overlap)
        |
        v
sentence-transformers -> generates embedding per chunk
        |
        v
SQLite -> stores chunk text (FTS5) + embedding vector (sqlite-vec)
        |
        v
rag.ask("question")
        |
        v
Embed question -> vector search (sqlite-vec, top 20)
              -> keyword search (FTS5 BM25, top 20)
              -> blend: 0.7 * vector_score + 0.3 * bm25_score
              -> return top-k chunks
        |
        v
LLM generates answer with retrieved chunks as context
```

---

## 11. Data Loader (`dataloader.py`)

**Shared module used by both FineTuner and RAG.** One file reader for the whole library.

### Supported formats

| Format | Used by     | How it's parsed                          | Library          |
|--------|-------------|------------------------------------------|------------------|
| CSV    | FineTuner   | Reads `prompt`, `response` columns       | Built-in `csv`   |
| JSONL  | FineTuner   | Each line: `{"prompt": "...", "response": "..."}` | Built-in `json` |
| TXT    | Both        | Plain text, auto-chunked                 | Built-in `open()`|
| PDF    | Both        | Extracts text from all pages             | `pymupdf`        |
| DOCX   | Both        | Extracts paragraphs                      | `python-docx`    |

### How it works

```text
dataloader.load("file.pdf")
        |
        v
Detect format from extension (.csv, .jsonl, .txt, .pdf, .docx)
        |
        v
Extract raw text (using appropriate parser)
        |
        v
Return based on caller:
  → FineTuner: list of {"prompt": ..., "response": ...} pairs
  → RAG:       list of text chunks (~400 tokens, 80-token overlap)
```

### API

```python
from zerollm.dataloader import load, chunk

# For FineTuner — returns structured pairs
pairs = load("faq.csv")
# [{"prompt": "What is...", "response": "You can..."}, ...]

pairs = load("data.jsonl")
# [{"prompt": "...", "response": "..."}, ...]

# For RAG — returns text chunks
chunks = chunk("report.pdf", chunk_size=400, overlap=80)
# ["chunk 1 text...", "chunk 2 text...", ...]

# Plain text training (causal language modeling)
chunks = chunk("book.txt", chunk_size=512, overlap=0)

# Also handles directories
pairs = load("training_data/")   # reads all CSV/JSONL files in folder
chunks = chunk("documents/")     # reads all PDF/TXT/DOCX files in folder
```

### Why one shared dataloader?

- FineTuner needs to read CSV/JSONL for training data
- RAG needs to read PDF/TXT/DOCX for document ingestion
- Both need chunking logic
- One module = no duplicate code, one place to add new formats

---

## 12. Model Download Workflow

```text
User calls Chat("smollm2")
        |
        v
Registry lookup -> finds HF repo + GGUF filename
        |
        v
Hardware check -> psutil detects RAM and GPU
        |
        v
hf_hub_download -> checks ~/.cache/zerollm/ first
                -> downloads from HF Hub if not cached
                -> returns local file path
        |
        v
Backend load -> llama-cpp-python (auto: CPU / CUDA / Metal / ROCm)
        |
        v
Model ready -> bot.ask() / bot.stream() / bot.chat()
```

---

## 13. Backend Strategy

**One backend: `llama-cpp-python`. One model format: GGUF. Zero choices for the user.**

llama.cpp supports every platform ZeroLLM targets — CPU, NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (Metal). Metal GPU acceleration is enabled by default on macOS ARM64 builds. There is no need for a second backend.

### How it works at runtime

```text
User calls Chat("model")
        |
   Hardware detection (psutil + llama-cpp-python internals)
        |
   Apple Silicon? --> llama-cpp-python with Metal (n_gpu_layers=-1)
        |
   NVIDIA GPU?    --> llama-cpp-python with CUDA (n_gpu_layers=-1)
        |
   AMD GPU?       --> llama-cpp-python with ROCm (n_gpu_layers=-1)
        |
   CPU only?      --> llama-cpp-python CPU (n_gpu_layers=0)
```

`n_gpu_layers=-1` means "offload all layers to GPU". The `power` parameter controls this:
- `power=1.0` → all layers on GPU
- `power=0.5` → half layers on GPU, half on CPU
- `power=0.0` → CPU only

### Why one backend wins

| Advantage                  | Detail                                              |
|----------------------------|-----------------------------------------------------|
| One model format           | GGUF everywhere — no conversion, no confusion       |
| One dependency             | Simpler install, fewer bugs, easier to maintain      |
| One code path              | No backend switching logic, no format mismatches     |
| Apple Silicon is covered   | llama.cpp Metal support is first-class, on by default|
| Proven at scale            | Most popular local LLM runtime, huge community       |

### Why NOT other backends?

| Backend         | Why excluded                                                  |
|-----------------|---------------------------------------------------------------|
| MLX-LM          | Apple-only, different model format, llama.cpp Metal is enough |
| vLLM            | Too heavy, needs AVX512 for CPU, poor on laptops/low-end HW  |
| Ollama          | Separate app install — breaks "pip install and done" promise  |
| ctransformers   | Abandoned (no updates since 2023)                             |
| ExLlamaV2       | NVIDIA-only, requires CUDA toolkit manual install             |
| ONNX GenAI      | No GGUF support, limited model ecosystem                      |
| HF transformers | 2GB+ install, high memory overhead for small models           |

---

## 14. Tech Stack

| Layer           | Library               | Purpose                              |
|-----------------|-----------------------|--------------------------------------|
| Inference       | llama-cpp-python      | GGUF inference: CPU/CUDA/Metal/ROCm  |
| Model Downloads | huggingface_hub       | Fetch GGUF from HF Hub               |
| Tokenisation    | tokenizers            | Fast text to token conversion        |
| Fine-tuning     | peft + transformers   | LoRA adapters                        |
| Data Loading    | pymupdf + python-docx | PDF/DOCX reading (CSV/JSONL/TXT built-in) |
| Hardware Info   | psutil                | RAM / CPU / core detection           |
| GPU Detection   | torch                 | CUDA / MPS availability check        |
| API Server      | fastapi + uvicorn     | OpenAI-compatible REST server        |
| Vector Store    | sqlite-vec + FTS5     | Hybrid search: vector + keyword (BM25)|
| Embeddings      | sentence-transformers | Local document embeddings            |
| CLI             | typer                 | Command-line interface               |
| Progress UI     | rich                  | Pretty output and progress           |
| Package Manager | uv                    | Fast dependency resolution and builds |

**Everything above is installed with `uv pip install zerollm`. No extras.**

---

## 15. CLI Commands

```bash
zerollm list
zerollm list --downloaded
zerollm recommend
zerollm chat smollm2
zerollm serve smollm2 --port 8080
zerollm download phi3:mini
zerollm remove tinyllama
zerollm info smollm2
```

---

## 16. Project Folder Structure

```text
zerollm/
├── zerollm/
│   ├── __init__.py        # public API: Chat, Agent, Server, FineTuner, RAG, recommend
│   ├── chat.py            # Chat class — ask, stream, chat
│   ├── agent.py           # Agent class — tool calling, multi-turn
│   ├── server.py          # Server class — OpenAI-compatible REST API
│   ├── finetune.py        # FineTuner class — LoRA training
│   ├── rag.py             # RAG class — add docs, ask questions
│   ├── backend.py         # llama-cpp-python wrapper (CPU/CUDA/Metal/ROCm)
│   ├── memory.py          # Session + persistent memory (SQLite)
│   ├── hardware.py        # Auto-detect CPU/GPU/RAM
│   ├── registry.py        # Model registry — name → HF repo + GGUF
│   ├── downloader.py      # Download models from HF Hub + cache
│   ├── dataloader.py      # Load & parse: CSV, JSONL, TXT, PDF, DOCX
│   └── cli.py             # CLI commands via Typer
├── tests/
├── examples/
├── registry.json              # curated model list
├── pyproject.toml
├── uv.lock                    # deterministic lock file (committed to git)
├── .python-version            # pinned Python version for uv
├── README.md
└── PLAN.md
```

**Flat structure — no subdirectories inside `zerollm/`.** Every module is one file, easy to find and read.

---

## 17. pyproject.toml

```toml
[project]
name = "zerollm"
version = "0.1.0"
description = "The zero-config Python library for local LLMs on any hardware"
requires-python = ">=3.10"

dependencies = [
    # Inference (single backend — covers CPU, CUDA, Metal, ROCm)
    "llama-cpp-python",
    "torch",
    # Model downloads
    "huggingface_hub",
    "tokenizers",
    # Hardware detection
    "psutil",
    # API server
    "fastapi",
    "uvicorn",
    # Fine-tuning
    "peft",
    "transformers",
    "datasets",
    "accelerate",
    # Data loading (shared by FineTuner + RAG)
    "pymupdf",
    "python-docx",
    # RAG (SQLite-based, inspired by OpenClaw)
    "sqlite-vec",
    "sentence-transformers",
    # CLI & UI
    "rich",
    "typer",
]

[project.scripts]
zerollm = "zerollm.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest",
    "pytest-cov",
    "ruff",
]
```

### Development workflow with uv

```bash
# Setup
uv venv                   # create virtual environment
uv sync                   # install all dependencies from pyproject.toml

# Day-to-day
uv run zerollm chat smollm2   # run CLI directly
uv run pytest                  # run tests
uv run ruff check .            # lint

# Add new dependency
uv add <package>               # adds to pyproject.toml + uv.lock

# Build & publish
uv build                       # creates wheel + sdist in dist/
uv publish                     # publish to PyPI
```

### Why uv over pip/poetry/setuptools?

| | uv | pip | poetry |
|---|---|---|---|
| Install speed | 10-100x faster | Baseline | 2-5x faster |
| Lock file | `uv.lock` (deterministic) | No | `poetry.lock` |
| Resolver | Fast SAT solver | Slow backtracking | Slow |
| Python management | Built-in (`uv python install`) | No | No |
| Build + publish | Built-in | Needs twine | Built-in |
| Single binary | Yes | No | No |

---

## 18. Release Roadmap

### v0.1 - Core (Month 1-2)
- [ ] Model registry with 7 curated models
- [ ] `Chat()` with auto hardware detect and power control
- [ ] Model download and local cache via `huggingface_hub`
- [ ] llama-cpp-python backend (CPU + CUDA + Metal + ROCm)
- [ ] Session memory
- [ ] CLI: `list`, `chat`, `recommend`, `download`, `info`
- [ ] Zero-config first-run onboarding and clear error messages
- [ ] `uv build` + `uv publish` to PyPI

### v0.2 - Serve (Month 2-3)
- [ ] `Server()` with OpenAI-compatible REST API
- [ ] Streaming support
- [ ] Auto-detect best acceleration (CUDA / Metal / ROCm / CPU)
- [ ] Model registry auto-sync from HF Hub

### v0.3 - Fine-Tune (Month 3-4)
- [ ] `dataloader.py` — shared file reader (CSV, JSONL, TXT, PDF, DOCX)
- [ ] `FineTuner()` with LoRA on CPU/GPU
- [ ] Train from files via dataloader (auto-detect format)
- [ ] Auto train/eval split and validation
- [ ] Save adapter and merge to GGUF
- [ ] Push fine-tuned model to HF Hub

### v0.4 - Agents (Month 4-5)
- [ ] `Agent()` with `@agent.tool` decorator
- [ ] Auto JSON schema from Python type hints
- [ ] Multi-turn memory for agents
- [ ] Agent serve as REST API

### v0.5 - RAG (Month 5-6)
- [ ] RAG module using dataloader for document ingestion
- [ ] SQLite + sqlite-vec vector store + FTS5 keyword search
- [ ] Hybrid scoring (70% vector + 30% BM25)
- [ ] RAG combined with Agent

---

## 19. Positioning: Real Gap This Fills

| Feature                | ZeroLLM | llama-cpp-python | local-llm | llama-cpp-agent | Ollama  |
|------------------------|---------|------------------|-----------|-----------------|---------|
| pip install only       | Yes     | Yes              | Yes       | Yes             | No (app) |
| Curated model registry | Yes     | No               | No        | No              | Yes     |
| Auto hardware detect   | Yes     | No               | Yes       | No              | Yes     |
| Power % control        | Yes     | No               | No        | No              | No      |
| Apple Silicon Metal    | Yes     | No               | No        | No              | Yes     |
| Fine-tuning built-in   | Yes     | No               | No        | No              | No      |
| Agents + tool calling  | Yes     | No               | No        | Yes             | No      |
| Session memory         | Yes     | No               | No        | Yes             | No      |
| Beginner-first API     | Yes     | No               | Partial   | No              | Yes     |

---

## 20. Immediate Execution Plan

### Phase 1: Foundation
- [ ] `uv init zerollm` — scaffold project with uv
- [ ] Reserve `zerollm` naming across package, import path, and CLI
- [ ] Create `pyproject.toml`, package skeleton, and Typer entrypoint
- [ ] `uv sync` — lock all dependencies in `uv.lock`
- [ ] Implement hardware detection for CPU, CUDA, and Apple Silicon
- [ ] Define `registry.json` schema and seed curated starter models

### Phase 2: First Working Loop
- [ ] Implement downloader with cache management under `~/.cache/zerollm/`
- [ ] Implement `Chat()` on llama.cpp CPU first
- [ ] Ship `zerollm recommend`, `zerollm list`, and `zerollm chat`
- [ ] Add integration tests for first-run download and basic prompt execution

### Phase 3: Differentiation
- [ ] Add CUDA, Metal, and ROCm GPU acceleration paths
- [ ] Add session memory and streaming
- [ ] Add OpenAI-compatible server
- [ ] Document the zero-config quickstart in `README.md`

### Phase 4: Expansion
- [ ] Add fine-tuning
- [ ] Add agents and tool calling
- [ ] Add RAG with SQLite + sqlite-vec hybrid search
- [ ] Prepare PyPI release, versioning, and benchmark docs

---

*Generated for ZeroLLM - Nilesh Verma, Auckland NZ, March 2026*
