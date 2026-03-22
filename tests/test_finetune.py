"""Tests for the FineTuner class (mocked — no actual training)."""

from unittest.mock import patch, MagicMock

from zerollm.finetune import FineTuner


def _mock_finetuner(**kwargs):
    """Create a FineTuner with mocked console."""
    with patch("zerollm.finetune.console"):
        tuner = FineTuner(**kwargs)
        return tuner


def test_finetuner_init():
    tuner = _mock_finetuner()
    assert tuner.model_name == "Qwen/Qwen3-0.6B"
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


def test_finetuner_resolve_base_repo():
    """FineTuner should use hf_base_repo from registry, not hardcoded mapping."""
    tuner = _mock_finetuner()
    from zerollm.registry import lookup
    info = lookup("Qwen/Qwen3-0.6B")
    assert info.hf_base_repo == "Qwen/Qwen3-0.6B"


def test_finetuner_all_models_have_base_repo():
    """Every model in registry must have hf_base_repo for fine-tuning."""
    from zerollm.registry import list_models
    for model in list_models():
        assert model.hf_base_repo, f"{model.name} missing hf_base_repo"
        assert "/" in model.hf_base_repo, f"{model.name} hf_base_repo looks invalid: {model.hf_base_repo}"
