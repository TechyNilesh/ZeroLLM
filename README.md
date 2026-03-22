<p align="center">
  <img src="assets/zerollm-logo-main.png" alt="ZeroLLM" width="400">
</p>

<p align="center">
  <strong>Zero setup. Zero config. Local LLMs on any hardware.</strong>
</p>

<p align="center">
  <a href="https://github.com/TechyNilesh/ZeroLLM/actions"><img src="https://img.shields.io/github/actions/workflow/status/TechyNilesh/ZeroLLM/ci.yml?branch=main&style=for-the-badge" alt="CI"></a>
  <a href="https://pypi.org/project/zerollm/"><img src="https://img.shields.io/pypi/v/zerollm?style=for-the-badge" alt="PyPI"></a>
  <a href="https://github.com/TechyNilesh/ZeroLLM/releases"><img src="https://img.shields.io/github/v/release/TechyNilesh/ZeroLLM?include_prereleases&style=for-the-badge" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
</p>

---

## What is ZeroLLM?

One `pip install`. Auto-detects your hardware. Downloads the right model. You're chatting in 3 lines of Python.

```python
from zerollm import Chat

bot = Chat("HuggingFaceTB/SmolLM2-1.7B-Instruct")
print(bot.ask("What is the capital of France?"))
```

That's it. No config files, no model format headaches, no GPU drivers to manage.

## Install

```bash
pip install zerollm
```

## Quick Start

### Chat

```python
from zerollm import Chat

bot = Chat("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Ask
print(bot.ask("Explain quantum computing in one sentence"))

# Stream
for token in bot.stream("Write a haiku about code"):
    print(token, end="", flush=True)

# Interactive REPL
bot.chat()
```

### Agent with Tools

```python
from zerollm import Agent

agent = Agent("HuggingFaceTB/SmolLM2-1.7B-Instruct")

@agent.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C and sunny in {city}"

print(agent.ask("What's the weather in Auckland?"))
```

### Sub-Agents

```python
researcher = Agent("Qwen/Qwen2.5-1.5B-Instruct", name="researcher")

@researcher.tool
def search(query: str) -> str:
    return f"Results for: {query}"

main = Agent("HuggingFaceTB/SmolLM2-1.7B-Instruct")
main.add_agent("researcher", researcher, "Research any topic")

main.ask("Research the latest AI trends")
```

### Serve as API

```python
from zerollm import Server

Server("HuggingFaceTB/SmolLM2-1.7B-Instruct", port=8080).serve()
```

OpenAI-compatible. Works with any client that speaks the OpenAI API.

### Fine-Tune

```python
from zerollm import FineTuner

tuner = FineTuner("HuggingFaceTB/SmolLM2-1.7B-Instruct")
tuner.train("my_data.csv", epochs=3)
tuner.save("my-bot")
```

Then serve your fine-tuned model:

```python
from zerollm import Chat, Server

Chat("my-bot").ask("Hello!")        # chat with it
Server("my-bot", port=8080).serve() # or serve it
```

### RAG

```python
from zerollm import RAG

rag = RAG("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rag.add("docs.pdf")
print(rag.ask("What is the refund policy?"))
```

Powered by SQLite + sqlite-vec. No external database needed.

## CLI

```bash
zerollm recommend                                          # best model for your hardware
zerollm chat HuggingFaceTB/SmolLM2-1.7B-Instruct          # interactive chat
zerollm serve HuggingFaceTB/SmolLM2-1.7B-Instruct         # start API server
zerollm list                                               # all available models
zerollm doctor                                             # diagnose setup
```

## Supported Hardware

| Platform | Acceleration | Auto-detected |
|----------|-------------|---------------|
| Any CPU | llama.cpp | Yes |
| NVIDIA GPU | CUDA | Yes |
| Apple Silicon | Metal | Yes |
| AMD GPU | ROCm | Yes |
| Raspberry Pi | CPU | Yes |

## Models

Works with any GGUF model from Hugging Face. Pass the full HF model name or a local `.gguf` file:

```python
Chat("HuggingFaceTB/SmolLM2-1.7B-Instruct")  # from registry
Chat("/path/to/any-model.gguf")                # local file
Chat("my-finetuned-bot")                       # your fine-tuned model
```

Run `zerollm list` to see curated models, or `zerollm recommend` to find the best one for your hardware.

## Architecture

<p align="center">
  <img src="assets/zerollm-architecture.png" alt="ZeroLLM Architecture" width="700">
</p>

## License

[MIT](LICENSE)

## Author

Built by [Nilesh Verma](https://nileshverma.com/)
