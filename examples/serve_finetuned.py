"""Serve a fine-tuned model — the full fine-tune → serve workflow."""

from zerollm import FineTuner, Chat, Server

# Step 1: Fine-tune
tuner = FineTuner("Qwen/Qwen3-0.6B")
tuner.train([
    {"prompt": "What is your refund policy?", "response": "Return within 30 days for a full refund."},
    {"prompt": "How do I contact support?", "response": "Email support@example.com or call 0800-123-456."},
    {"prompt": "What are your hours?", "response": "Monday to Friday, 9am to 5pm NZST."},
], epochs=3)

# Step 2: Save (merge into full model for GGUF conversion)
tuner.save("my-support-bot", merge=True)

# Step 3: Chat with your fine-tuned model
bot = Chat("my-support-bot")
print(bot.ask("What is your refund policy?"))

# Step 4: Serve as API
# Server("my-support-bot", port=8080).serve()

# ── Alternative: Load from a local GGUF file ──
# bot = Chat("/path/to/my-custom-model.gguf")
# Server("/path/to/my-custom-model.gguf", port=8080).serve()

# ── Alternative: Load from full path ──
# bot = Chat("~/.cache/zerollm/models/my-support-bot")
