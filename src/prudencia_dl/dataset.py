"""Chargement, nettoyage et séparation du dataset commun à Prudencia."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import TrainingConfig


@dataclass
class PreparedDataset:
    """Contient les trois partitions et la correspondance des classes."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    id2label: dict[int, str]
    label2id: dict[str, int]


def prepare_dataset(path: Path, config: TrainingConfig) -> PreparedDataset:
    """Nettoie le CSV, encode la cible et crée trois partitions stratifiées."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    data = pd.read_csv(path, sep=";", encoding="utf-8")
    required = {config.text_column, config.label_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Colonnes absentes : {sorted(missing)}")

    data = data[[config.text_column, config.label_column]].dropna().copy()
    data[config.text_column] = data[config.text_column].astype(str).str.strip()
    data[config.label_column] = data[config.label_column].astype(str).str.strip()
    data = data[
        (data[config.text_column] != "") & (data[config.label_column] != "")
    ].drop_duplicates()

    encoder = LabelEncoder()
    data["labels"] = encoder.fit_transform(data[config.label_column])
    id2label = {index: str(label) for index, label in enumerate(encoder.classes_)}
    label2id = {label: index for index, label in id2label.items()}

    development, test = train_test_split(
        data,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data["labels"],
    )
    relative_validation = config.validation_size / (1 - config.test_size)
    train, validation = train_test_split(
        development,
        test_size=relative_validation,
        random_state=config.random_state,
        stratify=development["labels"],
    )
    return PreparedDataset(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        id2label=id2label,
        label2id=label2id,
    )
