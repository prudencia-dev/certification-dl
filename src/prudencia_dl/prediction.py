"""Prédiction d'un nouveau cas avec le modèle JuriBERT sauvegardé."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MODEL_DIR, TrainingConfig


def predict_risk(
    text: str,
    model_dir: Path = MODEL_DIR,
    max_length: int = TrainingConfig.max_length,
) -> dict[str, object]:
    """Renvoie la classe de risque et les probabilités associées."""
    if not text.strip():
        raise ValueError("Le texte à analyser ne peut pas être vide.")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Modèle introuvable : {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    predicted_id = int(torch.argmax(probabilities))
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return {
        "niveau_risque": id2label[predicted_id],
        "probabilites": {
            id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }
