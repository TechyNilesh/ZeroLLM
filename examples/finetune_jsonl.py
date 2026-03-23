"""Fine-tune from a JSONL file — each line is a prompt/response pair."""

from zerollm import FineTuner

tuner = FineTuner("Qwen/Qwen3.5-4B", power=0.7)

# Train from JSONL (one JSON object per line)
# File format:
#   {"prompt": "What is AI?", "response": "Artificial Intelligence is..."}
#   {"prompt": "Explain ML", "response": "Machine Learning is..."}
tuner.train("training_data.jsonl", epochs=3)

# Save as LoRA adapter (small file, ~10-50MB)
tuner.save("my-custom-model")

# Or merge into full model (larger file, full model size)
tuner.save("my-custom-model-merged", merge=True)

# Push to Hugging Face Hub to share
# tuner.push("your-username/my-custom-model", token="hf_xxx")
