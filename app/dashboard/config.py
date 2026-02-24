"""
Configurações e paths do dashboard.
"""

from __future__ import annotations

import os
from typing import Final

import streamlit as st

PROJECT_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
DATA_PATH: Final[str] = os.path.join(
    PROJECT_ROOT, "data", "raw", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
)
DRIFT_REPORT_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "models", "artifacts", "data_drift_report.html"
)
ROC_CURVE_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "models", "artifacts", "roc_curve.png"
)
FEATURE_IMP_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "models", "artifacts", "feature_importance.png"
)
CLASS_REPORT_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "models", "artifacts", "classification_report.png"
)

# API URL - usar localhost no container único, senão nome do serviço
API_URL: Final[str] = os.environ.get(
    "API_URL",
    "http://127.0.0.1:8000" if os.getenv("ENVIRONMENT") == "production" else "http://api:8000",
)


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
