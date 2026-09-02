"""Métriques communes à l'entraînement et à l'évaluation."""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(evaluation: Any) -> dict[str, float]:
    """Transforme les logits en classes puis calcule deux mesures lisibles."""
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    return {
        "exactitude": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }
