"""Basic chat example — simplest way to use ZeroLLM."""

from zerollm import Chat

# One line to start chatting
bot = Chat("Qwen/Qwen3-0.6B")

# Ask a question
print(bot.ask("What is the capital of France?"))

# Stream a response
for token in bot.stream("Tell me a joke"):
    print(token, end="", flush=True)
print()

# Interactive chat (REPL)
# bot.chat()
