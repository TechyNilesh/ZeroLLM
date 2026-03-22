"""Tests for the FineTuner class (mocked — no actual training)."""

import re
from unittest.mock import patch

from zerollm.finetune import FineTuner


def _mock_finetuner(**kwargs):
    with patch("zerollm.finetune.console"):
        return FineTuner(**kwargs)


def test_finetuner_init():
    tuner = _mock_finetuner()
    assert tuner.model_name == "Qwen/Qwen3.5-4B"
    assert tuner.lora_r == 16
    assert tuner.lora_alpha == 32


def test_finetuner_init_custom():
    tuner = _mock_finetuner(
        model="microsoft/Phi-3-mini-4k-instruct",
        power=0.5,
        lora_r=8,
        lora_alpha=16,
    )
    assert tuner.model_name == "microsoft/Phi-3-mini-4k-instruct"
    assert tuner.power == 0.5
    assert tuner.lora_r == 8


def test_finetuner_strips_gguf_suffix():
    model = "Qwen/Qwen3.5-4B-GGUF"
    base = re.sub(r"-GGUF$", "", model, flags=re.IGNORECASE)
    assert base == "Qwen/Qwen3.5-4B"


def test_finetuner_no_gguf_unchanged():
    model = "Qwen/Qwen3.5-4B"
    base = re.sub(r"-GGUF$", "", model, flags=re.IGNORECASE)
    assert base == "Qwen/Qwen3.5-4B"
