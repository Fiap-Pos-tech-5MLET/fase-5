from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

import app.dashboard.sidebar as dashboard_sidebar
from tests.utils_streamlit import make_streamlit_mock


def test_render_sidebar_model_none(monkeypatch) -> None:
    """Deve renderizar sidebar e retornar página selecionada."""
    st = make_streamlit_mock()
    st.radio.return_value = "🔮 Predição"
    monkeypatch.setattr(dashboard_sidebar, "st", st)

    page = dashboard_sidebar.render_sidebar(None, MagicMock(), MagicMock())

    assert page == "🔮 Predição"
    st.error.assert_called()


def test_render_sidebar_with_model_and_reload(monkeypatch) -> None:
    """Deve exibir informações do modelo e limpar cache ao recarregar."""
    st = make_streamlit_mock()
    st.radio.return_value = "📊 Métricas do Modelo"
    st.button.return_value = True
    st.session_state["last_refresh"] = "01/01/2025"
    monkeypatch.setattr(dashboard_sidebar, "st", st)

    model = MagicMock()
    model.feature_names_in_ = ["A", "B"]
    model.named_steps = {"classifier": MagicMock()}

    load_model_func = MagicMock()
    load_model_func.clear = MagicMock()
    load_dataset_func = MagicMock()
    load_dataset_func.clear = MagicMock()

    page = dashboard_sidebar.render_sidebar(model, load_model_func, load_dataset_func)

    assert page == "📊 Métricas do Modelo"
    load_model_func.clear.assert_called_once()
    load_dataset_func.clear.assert_called_once()
    st.rerun.assert_called_once()
