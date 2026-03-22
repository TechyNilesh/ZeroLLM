"""FineTuner — LoRA fine-tuning for local LLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from zerollm.dataloader import load
from zerollm.hardware import detect

console = Console()


class FineTuner:
    """Fine-tune a local LLM on your own data using LoRA.

    This is fine-tuning (adapting a pre-trained model), NOT training from scratch.
    Works on CPU and GPU. Uses peft + transformers under the hood.

    Usage:
        tuner = FineTuner("HuggingFaceTB/SmolLM2-1.7B-Instruct")
        tuner.train("my_data.csv", epochs=3)
        tuner.save("my-bot")
    """

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        power: float = 0.7,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        """Initialize FineTuner.

        Args:
            model: Model name from registry.
            power: Resource usage 0.0-1.0.
            lora_r: LoRA rank (higher = more capacity, more memory).
            lora_alpha: LoRA alpha scaling factor.
            lora_dropout: Dropout for LoRA layers.
        """
        self.model_name = model
        self.power = power
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.hw = detect()
        self._model = None
        self._tokenizer = None
        self._trainer = None

        console.print(f"[dim]FineTuner initialized for {model}[/dim]")

    def _load_base_model(self) -> None:
        """Lazy-load the base model and tokenizer for fine-tuning."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        from zerollm.registry import lookup

        info = lookup(self.model_name)
        hf_repo = info.hf_base_repo

        console.print(f"[dim]Loading base model from {hf_repo}...[/dim]")

        self._tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with appropriate device
        device_map = "auto"
        if not self.hw.has_gpu:
            device_map = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            hf_repo,
            trust_remote_code=True,
            device_map=device_map,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        self._model = get_peft_model(self._model, lora_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        console.print(
            f"[green]✓[/green] Model loaded. "
            f"Trainable: {trainable:,} / {total:,} params "
            f"({100 * trainable / total:.1f}%)"
        )

    def train(
        self,
        data: str | Path | list[dict],
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        eval_split: float = 0.1,
    ) -> dict[str, Any]:
        """Train on your data.

        Args:
            data: Path to CSV/JSONL file, or list of {"prompt": ..., "response": ...}.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            eval_split: Fraction of data to use for evaluation.

        Returns:
            Training metrics dict.
        """
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

        self._load_base_model()

        # Load data using shared dataloader
        pairs = load(data)
        if not pairs:
            raise ValueError("No training data found. Check your file format.")

        console.print(f"[dim]Loaded {len(pairs)} training examples[/dim]")

        # Format as chat/instruction text
        texts = []
        for pair in pairs:
            text = f"### Instruction:\n{pair['prompt']}\n\n### Response:\n{pair['response']}"
            texts.append(text)

        # Tokenize
        def tokenize(examples):
            return self._tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        dataset = Dataset.from_dict({"text": texts})

        # Train/eval split
        if eval_split > 0 and len(dataset) > 10:
            split = dataset.train_test_split(test_size=eval_split, seed=42)
            train_dataset = split["train"].map(tokenize, batched=True)
            eval_dataset = split["test"].map(tokenize, batched=True)
        else:
            train_dataset = dataset.map(tokenize, batched=True)
            eval_dataset = None

        # Training arguments
        output_dir = Path.home() / ".cache" / "zerollm" / "finetune" / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        console.print(f"[bold]Training {self.model_name}[/bold] — {epochs} epochs, {len(pairs)} examples")
        result = trainer.train()
        self._trainer = trainer

        metrics = {
            "train_loss": result.training_loss,
            "epochs": epochs,
            "examples": len(pairs),
        }

        console.print(f"[green]✓[/green] Training complete. Loss: {result.training_loss:.4f}")
        return metrics

    def save(self, name: str, merge: bool = False) -> Path:
        """Save the fine-tuned model.

        Args:
            name: Name for the saved model.
            merge: If True, merge LoRA weights into base model.

        Returns:
            Path to saved model directory.
        """
        save_dir = Path.home() / ".cache" / "zerollm" / "models" / name

        if merge:
            console.print("[dim]Merging LoRA weights into base model...[/dim]")
            merged = self._model.merge_and_unload()
            merged.save_pretrained(str(save_dir))
            self._tokenizer.save_pretrained(str(save_dir))
            console.print(f"[green]✓[/green] Merged model saved to {save_dir}")
        else:
            self._model.save_pretrained(str(save_dir))
            self._tokenizer.save_pretrained(str(save_dir))
            console.print(f"[green]✓[/green] LoRA adapter saved to {save_dir}")

        return save_dir

    def push(self, repo_id: str, token: str | None = None) -> None:
        """Push fine-tuned model to Hugging Face Hub.

        Args:
            repo_id: HF repo ID (e.g. "username/my-model").
            token: Hugging Face API token.
        """
        self._model.push_to_hub(repo_id, token=token)
        self._tokenizer.push_to_hub(repo_id, token=token)
        console.print(f"[green]✓[/green] Pushed to https://huggingface.co/{repo_id}")
