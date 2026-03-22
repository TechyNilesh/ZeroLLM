"""Chat with a custom system prompt — give the bot a personality."""

from zerollm import Chat

# Make the bot speak like a pirate
bot = Chat(
    "Qwen/Qwen3.5-4B",
    system_prompt="You are a friendly pirate. Always speak in pirate language.",
    temperature=0.9,
)

print(bot.ask("What is the capital of France?"))
print(bot.ask("Tell me about the ocean"))
