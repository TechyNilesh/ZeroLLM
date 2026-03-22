"""Streaming example — see tokens appear one by one."""

from zerollm import Chat

bot = Chat("Qwen/Qwen3.5-4B")

# Stream tokens as they are generated
print("Streaming response:")
for token in bot.stream("Write a haiku about programming"):
    print(token, end="", flush=True)
print("\n")

# Compare with non-streaming
print("Non-streaming response:")
print(bot.ask("Write a haiku about coffee"))
