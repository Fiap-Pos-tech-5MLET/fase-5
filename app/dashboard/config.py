"""
Configurações e paths do dashboard.
"""

from __future__ import annotations

import os
from typing import Final

import streamlit as st


def _pick_path(candidates: list[str]) -> str:
    """Seleciona o primeiro caminho existente; fallback para o primeiro candidato válido."""
    sanitized = [path for path in candidates if path]
    for path in sanitized:
        if os.path.exists(path):
            return path
    return sanitized[0]


DASHBOARD_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
APP_DIR: Final[str] = os.path.dirname(DASHBOARD_DIR)
WORKSPACE_DIR: Final[str] = os.path.dirname(APP_DIR)

PROJECT_ROOT: Final[str] = APP_DIR

MODEL_PATH: Final[str] = _pick_path(
    [
        os.getenv("DASHBOARD_MODEL_PATH", ""),
        os.getenv("MODEL_PATH", ""),
        os.path.join(APP_DIR, "models", "model.pkl"),
        os.path.join(WORKSPACE_DIR, "models", "model.pkl"),
    ]
)

DATA_PATH: Final[str] = _pick_path(
    [
        os.getenv("DASHBOARD_DATA_PATH", ""),
        os.getenv("DATASET_PATH", ""),
        os.path.join(WORKSPACE_DIR, "data", "raw", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"),
        os.path.join(APP_DIR, "data", "raw", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"),
    ]
)

ARTIFACTS_DIR: Final[str] = _pick_path(
    [
        os.getenv("ARTIFACTS_DIR", ""),
        os.path.join(APP_DIR, "models", "artifacts"),
        os.path.join(WORKSPACE_DIR, "models", "artifacts"),
        os.path.join(APP_DIR, "artifacts"),
    ]
)

DRIFT_REPORT_PATH: Final[str] = os.path.join(ARTIFACTS_DIR, "data_drift_report.html")
ROC_CURVE_PATH: Final[str] = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
FEATURE_IMP_PATH: Final[str] = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
CLASS_REPORT_PATH: Final[str] = os.path.join(ARTIFACTS_DIR, "classification_report.png")

# API URL explícita por ambiente (Render/local)
API_URL: Final[str] = os.environ.get("API_URL", "http://127.0.0.1:8000")


def configure_page() -> None:
    """
    Configura metadados do app Streamlit.

    Returns:
        None
    """
    st.set_page_config(
        page_title="Passos Mágicos — Predição de Risco",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
