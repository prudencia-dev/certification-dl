"""Évaluation finale de JuriBERT sur le jeu de test réservé."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from transformers import DataCollatorWithPadding, Trainer

from .config import TrainingConfig
from .dataset import PreparedDataset
from .metrics import compute_metrics
from .training import TextDataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def evaluate_model(
    trainer: Trainer,
    tokenizer: object,
    prepared: PreparedDataset,
    config: TrainingConfig,
    reports_dir: Path,
) -> dict[str, float]:
    """Évalue le modèle sur des données jamais utilisées pour l'apprentissage."""
    encodings = tokenizer(
        prepared.test[config.text_column].tolist(),
        truncation=True,
        max_length=config.max_length,
    )
    test_dataset = TextDataset(
        encodings, prepared.test["labels"].astype(int).tolist()
    )
    trainer.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output = trainer.predict(test_dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    metrics = compute_metrics((output.predictions, labels))
    names = [prepared.id2label[index] for index in range(len(prepared.id2label))]

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        reports_dir / "classification_report.csv"
    )
    display = ConfusionMatrixDisplay.from_predictions(
        labels, predictions, display_labels=names, xticks_rotation=25, cmap="Blues"
    )
    display.figure_.tight_layout()
    display.figure_.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
    return metrics
