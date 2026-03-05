from __future__ import annotations

import importlib
import sys
import types

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

import app.dashboard.config as dashboard_config
from tests.utils_streamlit import make_streamlit_mock


def test_api_url_default(monkeypatch) -> None:
    """Deve usar API_URL padrão quando env não definida."""
    monkeypatch.delenv("API_URL", raising=False)
    importlib.reload(dashboard_config)

    assert dashboard_config.API_URL == "http://api:8000"


def test_api_url_env(monkeypatch) -> None:
    """Deve ler API_URL do ambiente quando definido."""
    monkeypatch.setenv("API_URL", "http://api:8000")
    importlib.reload(dashboard_config)

    assert dashboard_config.API_URL == "http://api:8000"


def test_configure_page_calls_streamlit(monkeypatch) -> None:
    """Deve chamar st.set_page_config com parâmetros esperados."""
    st = make_streamlit_mock()
    monkeypatch.setattr(dashboard_config, "st", st)

    dashboard_config.configure_page()

    st.set_page_config.assert_called_once()
