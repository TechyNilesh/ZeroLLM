"""Recommend the best model for your hardware."""

from zerollm import recommend

# Auto-detects your hardware and suggests the best model
recommend()

# You can also use the registry directly
from zerollm.registry import list_models

print("\nAll available models:")
for model in list_models():
    print(f"  {model.name} — {model.size_label}, needs {model.min_ram_gb}GB RAM")
