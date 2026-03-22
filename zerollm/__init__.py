"""ZeroLLM — Zero setup, zero config local LLMs on any hardware.

Works with any model from HuggingFace. Powered by HF transformers.

Usage:
    from zerollm import Chat
    bot = Chat("Qwen/Qwen3.5-4B")
    print(bot.ask("Hello!"))
"""

__version__ = "0.1.8"

from zerollm.chat import Chat
from zerollm.agent import Agent, SharedContext
from zerollm.server import Server
from zerollm.finetune import FineTuner
from zerollm.rag import RAG

__all__ = [
    "Chat",
    "Agent",
    "SharedContext",
    "Server",
    "FineTuner",
    "RAG",
]
