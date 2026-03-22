"""Fine-tuning example — train a model on your own data."""

from zerollm import FineTuner

tuner = FineTuner("Qwen/Qwen3-0.6B")

# Train from a list
tuner.train([
    {"prompt": "What is your refund policy?", "response": "Return within 30 days."},
    {"prompt": "How do I reset my password?", "response": "Click Forgot Password."},
    {"prompt": "What are your hours?", "response": "Mon-Fri 9am-5pm."},
], epochs=3)

# Or train from a CSV file
# tuner.train("my_data.csv", epochs=5)

# Save the fine-tuned model
tuner.save("my-support-bot")
