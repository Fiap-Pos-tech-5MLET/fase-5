"""
Dashboard Streamlit — Datathon Passos Mágicos.

Painel interativo para predição de risco de defasagem escolar,
monitoramento de drift e visualização de métricas do modelo.
"""

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard.config import (
    API_URL,
    CLASS_REPORT_PATH,
    DRIFT_REPORT_PATH,
    FEATURE_IMP_PATH,
    ROC_CURVE_PATH,
    configure_page,
)
from app.dashboard.data import get_model_metrics, load_dataset, load_model, predict_via_api
from app.dashboard.pages.about import render_about_page
from app.dashboard.pages.drift import render_drift_page
from app.dashboard.pages.metrics import render_metrics_page
from app.dashboard.pages.prediction import render_prediction_page
from app.dashboard.pages.retrain import render_retrain_page
from app.dashboard.sidebar import render_sidebar
from app.dashboard.styles import apply_custom_css

configure_page()
apply_custom_css()

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.now().strftime("%d/%m/%Y %H:%M")


model = load_model()
page = render_sidebar(model, load_model, load_dataset)

if page == "🔮 Predição":
    render_prediction_page(model, predict_via_api)
elif page == "📊 Métricas do Modelo":
    render_metrics_page(
        model,
        API_URL,
        load_dataset,
        get_model_metrics,
        ROC_CURVE_PATH,
        FEATURE_IMP_PATH,
        CLASS_REPORT_PATH,
    )
elif page == "🔄 Monitoramento de Drift":
    render_drift_page(DRIFT_REPORT_PATH)
elif page == "⚙️ Retreinamento":
    render_retrain_page(model, API_URL, load_dataset, get_model_metrics, load_model)
elif page == "ℹ️ Sobre o Projeto":
    render_about_page()
