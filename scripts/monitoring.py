"""
Script de monitoramento e detecção de data drift.

Este script utiliza Evidently para:
- Detectar mudanças na distribuição dos dados
- Gerar relatórios HTML interativos
- Comparar dados de referência (treino) vs dados atuais (produção)
"""

import logging
import os
import sys
from importlib import import_module
from typing import Any

from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("monitoring")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_cleaning import (
    clean_data,
    create_target,
    handle_missing_values,
    load_data,
)
from src.feature_engineering import create_features, select_features


def _load_evidently_classes() -> tuple[Any, Any]:
    """Carrega classes do Evidently com fallback entre versões.

    Returns:
        tuple[Any, Any]: Tupla contendo (Report, DataDriftPreset).

    Raises:
        ImportError: Se o Evidently não estiver instalado ou incompatível.
    """
    try:
        report_module = import_module("evidently.report")
        metric_module = import_module("evidently.metric_preset")
        Report = getattr(report_module, "Report")
        DataDriftPreset = getattr(metric_module, "DataDriftPreset")

        return Report, DataDriftPreset
    except ImportError:
        try:
            evidently_module = import_module("evidently")
            presets_module = import_module("evidently.presets")
            Report = getattr(evidently_module, "Report")
            DataDriftPreset = getattr(presets_module, "DataDriftPreset")

            return Report, DataDriftPreset
        except ImportError as exc:
            raise ImportError(
                "Não foi possível importar Evidently. "
                "Verifique a instalação e a versão do pacote 'evidently'."
            ) from exc


def generate_drift_report() -> None:
    """
    Gera relatório de data drift comparando dados de referência vs atuais.

    Utiliza Evidently DataDriftPreset para detectar mudanças significativas
    na distribuição das features entre treino e produção.

    Raises:
        FileNotFoundError: Se arquivo de dados não for encontrado.
        ValueError: Se dados processados estiverem vazios.
    """
    DATA_PATH = "data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    REPORT_PATH = "models/artifacts/data_drift_report.html"

    logger.info("Loading and preparing data for monitoring...")
    try:
        Report, DataDriftPreset = _load_evidently_classes()

        df = load_data(DATA_PATH)
        df = clean_data(df)
        df = create_target(df)
        df = handle_missing_values(df)
        df = create_features(df)
        X, y = select_features(df)

        # Merge X and y for full analysis
        full_data = X.copy()
        full_data["target"] = y

        # Split into Reference (Train) and Current (Test)
        # In a real scenario, Reference would be your training data snapshot
        # and Current would be the new batch of data from production.
        reference_data, current_data = train_test_split(full_data, test_size=0.2, random_state=42)

        logger.info("Reference data shape: %s", reference_data.shape)
        logger.info("Current data shape: %s", current_data.shape)

        # Generate Report
        logger.info("Calculating drift metrics...")
        report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        # In recent versions, run() returns a Snapshot which has save_html
        snapshot = report.run(reference_data=reference_data, current_data=current_data)

        # Save
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        # Try both ways to be safe across versions, or rely on debug result
        if hasattr(snapshot, "save_html"):
            snapshot.save_html(REPORT_PATH)
        elif hasattr(report, "save_html"):
            report.save_html(REPORT_PATH)
        else:
            logger.error("Could not find save_html method.")

        logger.info("Report saved to %s", REPORT_PATH)

    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError):
        logger.exception("Error generating report")
        raise


if __name__ == "__main__":
    generate_drift_report()
