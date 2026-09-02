"""Orchestration lisible du pipeline Prudencia DL niveau 2."""

import logging

from .config import (
    DATASET_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    TRAINING_CONFIG,
)
from .dataset import prepare_dataset
from .evaluation import evaluate_model
from .training import train_model

LOGGER = logging.getLogger(__name__)


def run_pipeline() -> dict[str, float]:
    """Enchaîne préparation, entraînement, évaluation et sauvegarde."""
    LOGGER.info("1/4 — Préparation du dataset Prudencia")
    prepared = prepare_dataset(DATASET_PATH, TRAINING_CONFIG)
    LOGGER.info(
        "Partitions : %s entraînement, %s validation, %s test",
        len(prepared.train),
        len(prepared.validation),
        len(prepared.test),
    )

    LOGGER.info("2/4 — Tokenisation et fine-tuning de JuriBERT")
    trainer, tokenizer = train_model(prepared, TRAINING_CONFIG, OUTPUT_DIR)

    LOGGER.info("3/4 — Évaluation sur le jeu de test réservé")
    metrics = evaluate_model(
        trainer, tokenizer, prepared, TRAINING_CONFIG, REPORTS_DIR
    )

    LOGGER.info("4/4 — Sauvegarde du modèle final")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    return metrics
