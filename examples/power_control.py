"""Power control — manage how much of your hardware ZeroLLM uses.

power=1.0 → Use all GPU layers + max CPU threads (fastest, uses most resources)
power=0.5 → Use half GPU layers + half threads (balanced)
power=0.0 → CPU only, no GPU (lightest, laptop-friendly)
"""

from zerollm import Chat

# Full power — fastest inference
bot_fast = Chat("HuggingFaceTB/SmolLM2-1.7B-Instruct", power=1.0)
print(bot_fast.ask("Hello!"))

# Half power — good for running alongside other apps
bot_balanced = Chat("Qwen/Qwen2.5-0.5B-Instruct", power=0.5)
print(bot_balanced.ask("Hello!"))

# CPU only — no GPU at all
bot_light = Chat("Qwen/Qwen2.5-0.5B-Instruct", power=0.0)
print(bot_light.ask("Hello!"))
