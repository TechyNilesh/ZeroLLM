"""ZeroLLM — Zero setup, zero config local LLMs on any hardware.

Works with any GGUF model from HuggingFace.

Usage:
    from zerollm import Chat
    bot = Chat("Qwen/Qwen3.5-4B")
    print(bot.ask("Hello!"))
"""

__version__ = "0.1.7"

from zerollm.chat import Chat
from zerollm.agent import Agent, SharedContext, Pipeline
from zerollm.server import Server
from zerollm.finetune import FineTuner
from zerollm.rag import RAG

__all__ = [
    "Chat",
    "Agent",
    "SharedContext",
    "Pipeline",
    "Server",
    "FineTuner",
    "RAG",
]
