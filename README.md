<p align="center">
  <img src="https://raw.githubusercontent.com/TechyNilesh/ZeroLLM/main/assets/zerollm-logo-main.png" alt="ZeroLLM" width="400">
</p>

<p align="center">
  <strong>Zero setup. Zero config. Local LLMs on any hardware.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/zerollm-kit/"><img src="https://img.shields.io/pypi/v/zerollm-kit?style=for-the-badge" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pepy.tech/project/zerollm-kit"><img src="https://img.shields.io/pepy/dt/zerollm-kit?style=for-the-badge&label=Downloads" alt="Downloads"></a>
</p>

---

## What is ZeroLLM?

One `pip install`. Auto-detects your hardware. Downloads the right model. You're chatting in 3 lines of Python.

```python
from zerollm import Chat

bot = Chat("Qwen/Qwen3.5-4B")
print(bot.ask("What is the capital of France?"))
```

That's it. No config files, no model format headaches, no GPU drivers to manage.

## Install

```bash
pip install zerollm-kit
```

## Quick Start

### Chat

```python
from zerollm import Chat

bot = Chat("Qwen/Qwen3.5-4B")

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

agent = Agent("Qwen/Qwen3.5-4B")

@agent.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C and sunny in {city}"

print(agent.ask("What's the weather in Auckland?"))
```

### Sub-Agents

```python
researcher = Agent("Qwen/Qwen3-1.7B", name="researcher")

@researcher.tool
def search(query: str) -> str:
    return f"Results for: {query}"

main = Agent("Qwen/Qwen3.5-4B")
main.add_agent("researcher", researcher, "Research any topic")

main.ask("Research the latest AI trends")
```

### Serve as API

```python
from zerollm import Server

Server("Qwen/Qwen3.5-4B", port=8080).serve()
```

OpenAI-compatible. Works with any client that speaks the OpenAI API.

### Fine-Tune

```python
from zerollm import FineTuner

tuner = FineTuner("Qwen/Qwen3.5-4B")
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

rag = RAG("Qwen/Qwen3.5-4B")
rag.add("docs.pdf")
print(rag.ask("What is the refund policy?"))
```

Powered by SQLite + sqlite-vec. No external database needed.

## CLI

```bash
zerollm recommend                                          # best model for your hardware
zerollm chat Qwen/Qwen3.5-4B          # interactive chat
zerollm serve Qwen/Qwen3.5-4B         # start API server
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

Works with **any GGUF model from HuggingFace**. No curated list. Just pass the HF repo name:

```python
Chat("Qwen/Qwen3.5-4B")                        # auto-finds GGUF, picks best quantization
Chat("bartowski/Phi-3-mini-4k-instruct-GGUF")   # direct GGUF repo
Chat("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")  # any GGUF repo works
Chat("/path/to/any-model.gguf")                  # local file
Chat("my-finetuned-bot")                         # fine-tuned model
```

Run `zerollm list` to see downloaded models, or `zerollm doctor` to check your setup.

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/TechyNilesh/ZeroLLM/main/assets/zerollm-architecture.png" alt="ZeroLLM Architecture" width="700">
</p>

> **Note:** ZeroLLM is in early alpha. Things will break, APIs may change, and some models might not load depending on your `llama-cpp-python` version. That's expected — we're iterating fast. If you hit an issue or have ideas, [open an issue](https://github.com/TechyNilesh/ZeroLLM/issues). Your feedback shapes what this becomes.

## Star History

<a href="https://www.star-history.com/?repos=TechyNilesh%2FZeroLLM&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TechyNilesh/ZeroLLM&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TechyNilesh/ZeroLLM&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TechyNilesh/ZeroLLM&type=date&legend=top-left" />
 </picture>
</a>

## License

[MIT](LICENSE)

## Core Contributor

[Nilesh Verma](https://nileshverma.com/)
