"""Chat with memory — the bot remembers your conversation."""

from zerollm import Chat

# Enable memory to persist conversation across turns
bot = Chat("Qwen/Qwen3-0.6B", memory=True)

# The bot remembers what you said
print(bot.ask("My name is Nilesh and I live in Auckland"))
print(bot.ask("What is my name?"))  # Should remember: Nilesh
print(bot.ask("Where do I live?"))  # Should remember: Auckland

# View full conversation history
for msg in bot.history:
    print(f"  [{msg['role']}]: {msg['content'][:80]}...")

# Reset conversation
bot.reset()
print(bot.ask("What is my name?"))  # Should NOT remember
