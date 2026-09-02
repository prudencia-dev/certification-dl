"""Tokenisation et fine-tuning de JuriBERT."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig
from .dataset import PreparedDataset
from .metrics import compute_metrics


class TextDataset(torch.utils.data.Dataset):
    """Adapte les textes tokenisés au format attendu par Hugging Face."""

    def __init__(self, encodings: dict[str, Any], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            name: torch.tensor(values[index])
            for name, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def set_random_seeds(seed: int) -> None:
    """Fixe les graines utilisées par Python, NumPy et PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    prepared: PreparedDataset,
    config: TrainingConfig,
    output_dir: Path,
) -> tuple[Trainer, AutoTokenizer]:
    """Tokenise les textes, entraîne JuriBERT et renvoie le meilleur modèle."""
    set_random_seeds(config.random_state)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def to_torch_dataset(frame: Any) -> TextDataset:
        encodings = tokenizer(
            frame[config.text_column].tolist(),
            truncation=True,
            max_length=config.max_length,
        )
        return TextDataset(encodings, frame["labels"].astype(int).tolist())

    train_dataset = to_torch_dataset(prepared.train)
    validation_dataset = to_torch_dataset(prepared.validation)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(prepared.id2label),
        id2label=prepared.id2label,
        label2id=prepared.label2id,
        ignore_mismatched_sizes=True,
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=config.random_state,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer, tokenizer
