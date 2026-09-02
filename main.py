"""Point d'entrée du projet Prudencia Deep Learning niveau 2."""

import logging

from src.prudencia_dl.pipeline import run_pipeline


def main() -> None:
    """Lance tout le pipeline depuis le bouton « Run Python File » de VS Code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    metrics = run_pipeline()
    logging.info("Pipeline terminé — métriques : %s", metrics)


if __name__ == "__main__":
    main()
