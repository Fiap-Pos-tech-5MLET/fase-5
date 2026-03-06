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
from app.dashboard.data import (
    get_model_cache_buster,
    get_model_metrics,
    load_dataset,
    load_model,
    predict_via_api,
)
from app.dashboard.health import check_api_health, get_api_status
from app.dashboard.pages.about import render_about_page
from app.dashboard.pages.drift import render_drift_page
from app.dashboard.pages.metrics import render_metrics_page
from app.dashboard.pages.prediction import render_prediction_page
from app.dashboard.pages.retrain import render_retrain_page
from app.dashboard.sidebar import render_sidebar
from app.dashboard.styles import apply_custom_css
from app.config import RAW_DATA_PATH

configure_page()
apply_custom_css()

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.now().strftime("%d/%m/%Y %H:%M")

if "api_healthy" not in st.session_state:
    st.session_state["api_healthy"] = check_api_health(API_URL)


# Carregamento de modelo adaptável:
# - Para Predição e Drift: usa apenas health check (zero filesystem access)
# - Para Métricas e Retreinamento: carrega modelo localmente (melhoria futura: migrar para API)
page = render_sidebar(None, load_model, load_dataset, api_healthy=st.session_state["api_healthy"])

model = None
if page in ["📊 Métricas do Modelo", "⚙️ Retreinamento"]:
    # Essas páginas ainda usam métricas locais (melhoria futura: endpoint /metrics/evaluation)
    model = load_model(get_model_cache_buster())

if page == "🔮 Predição":
    render_prediction_page(None, predict_via_api, api_healthy=st.session_state["api_healthy"])
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
    render_drift_page(RAW_DATA_PATH, DRIFT_REPORT_PATH, api_healthy=st.session_state["api_healthy"])
elif page == "⚙️ Retreinamento":
    render_retrain_page(model, API_URL, load_dataset, get_model_metrics, load_model)
elif page == "ℹ️ Sobre o Projeto":
    render_about_page()
