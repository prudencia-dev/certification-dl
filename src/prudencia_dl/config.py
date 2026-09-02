"""Chemins et paramètres communs du projet Prudencia DL niveau 2."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data/dl_juribert_training_cases_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models/juribert_prudencia"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class TrainingConfig:
    """Regroupe les hyperparamètres afin de les modifier à un seul endroit."""

    model_name: str = "dascim/juribert-base"
    text_column: str = "Q3"
    label_column: str = "risk_level_aiact"
    validation_size: float = 0.20
    test_size: float = 0.20
    random_state: int = 42
    max_length: int = 256
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5


TRAINING_CONFIG = TrainingConfig()
