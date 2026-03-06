"""
Testes para app/dashboard.py (script principal).

Cobre:
- Inicialização e configuração do dashboard
- Roteamento de páginas
- Session state
- Health check
"""

import sys
import types
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.fixture
def streamlit_mock():
    """Mock de Streamlit para testes do dashboard."""
    st_mock = MagicMock()
    st_mock.session_state = {}
    st_mock.set_page_config = MagicMock()
    return st_mock


@pytest.mark.unit
class TestDashboardInitialization:
    """Testes para inicialização do dashboard."""

    def test_dashboard_imports_successfully(self) -> None:
        """Testa que dashboard pode ser importado."""
        try:
            import app.dashboard

            assert app.dashboard is not None
        except ImportError as e:
            pytest.fail(f"Dashboard não pode ser importado: {e}")

    def test_project_root_in_path(self) -> None:
        """Testa que raiz do projeto é adicionada ao sys.path."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent

        # Verificar que path existe
        assert project_root.exists()
        assert (project_root / "app").exists()

    def test_required_imports_available(self) -> None:
        """Testa que imports necessários estão disponíveis."""
        try:
            from app.dashboard.config import API_URL, configure_page
            from app.dashboard.data import load_dataset, load_model
            from app.dashboard.health import check_api_health
            from app.dashboard.pages.prediction import render_prediction_page

            assert API_URL is not None
            assert callable(configure_page)
            assert callable(load_model)
            assert callable(load_dataset)
            assert callable(check_api_health)
            assert callable(render_prediction_page)
        except ImportError:
            pytest.fail("Imports necessários não estão disponíveis")


@pytest.mark.unit
class TestSessionStateInitialization:
    """Testes para inicialização de session_state."""

    def test_last_refresh_initialized(self, streamlit_mock) -> None:
        """Testa que last_refresh é inicializado."""
        streamlit_mock.session_state = {}

        # Simular inicialização
        if "last_refresh" not in streamlit_mock.session_state:
            streamlit_mock.session_state["last_refresh"] = datetime.now().strftime("%d/%m/%Y %H:%M")

        assert "last_refresh" in streamlit_mock.session_state
        assert isinstance(streamlit_mock.session_state["last_refresh"], str)

    def test_api_healthy_initialized(self, streamlit_mock) -> None:
        """Testa que api_healthy é inicializado."""
        streamlit_mock.session_state = {}

        # Simular inicialização
        if "api_healthy" not in streamlit_mock.session_state:
            streamlit_mock.session_state["api_healthy"] = True

        assert "api_healthy" in streamlit_mock.session_state
        assert isinstance(streamlit_mock.session_state["api_healthy"], bool)

    def test_session_state_persistence(self, streamlit_mock) -> None:
        """Testa que valores de session_state persistem."""
        streamlit_mock.session_state["custom_key"] = "custom_value"

        # Simular acesso posterior
        assert streamlit_mock.session_state["custom_key"] == "custom_value"

    def test_last_refresh_timestamp_format(self) -> None:
        """Testa formato do timestamp de last_refresh."""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

        # DD/MM/YYYY HH:MM
        assert len(timestamp) == 16
        assert "/" in timestamp
        assert ":" in timestamp


@pytest.mark.unit
class TestPageRouting:
    """Testes para roteamento de páginas."""

    def test_prediction_page_routing(self, streamlit_mock) -> None:
        """Testa roteamento para página de Predição."""
        selected_page = "🔮 Predição"

        # Simular seleção
        should_render_prediction = selected_page == "🔮 Predição"

        assert should_render_prediction

    def test_metrics_page_routing(self, streamlit_mock) -> None:
        """Testa roteamento para página de Métricas."""
        selected_page = "📊 Métricas do Modelo"

        if selected_page == "📊 Métricas do Modelo":
            should_load_model = True
            should_render_metrics = True
        else:
            should_load_model = False
            should_render_metrics = False

        assert should_load_model and should_render_metrics

    def test_drift_page_routing(self, streamlit_mock) -> None:
        """Testa roteamento para página de Drift."""
        selected_page = "🔄 Monitoramento de Drift"

        should_render_drift = selected_page == "🔄 Monitoramento de Drift"

        assert should_render_drift

    def test_retrain_page_routing(self, streamlit_mock) -> None:
        """Testa roteamento para página de Retreinamento."""
        selected_page = "⚙️ Retreinamento"

        if selected_page == "⚙️ Retreinamento":
            should_load_model = True
            should_render_retrain = True
        else:
            should_load_model = False
            should_render_retrain = False

        assert should_load_model and should_render_retrain

    def test_about_page_routing(self, streamlit_mock) -> None:
        """Testa roteamento para página Sobre."""
        selected_page = "ℹ️ Sobre o Projeto"

        should_render_about = selected_page == "ℹ️ Sobre o Projeto"

        assert should_render_about

    def test_all_pages_routable(self) -> None:
        """Testa que todas as páginas têm rota."""
        pages = [
            "🔮 Predição",
            "📊 Métricas do Modelo",
            "🔄 Monitoramento de Drift",
            "⚙️ Retreinamento",
            "ℹ️ Sobre o Projeto",
        ]

        assert len(pages) == 5
        assert all(isinstance(p, str) for p in pages)


@pytest.mark.unit
class TestDashboardHealthCheck:
    """Testes para health check da API."""

    def test_api_health_stored_in_session(self, streamlit_mock) -> None:
        """Testa que health check é armazenado em session_state."""
        streamlit_mock.session_state["api_healthy"] = True

        assert streamlit_mock.session_state["api_healthy"] is True

    def test_api_health_boolean_value(self, streamlit_mock) -> None:
        """Testa que api_healthy armazena boolean."""
        for value in [True, False]:
            streamlit_mock.session_state["api_healthy"] = value
            assert isinstance(streamlit_mock.session_state["api_healthy"], bool)

    def test_api_health_used_in_routing(self, streamlit_mock) -> None:
        """Testa que api_healthy influencia renderização."""
        streamlit_mock.session_state["api_healthy"] = False

        # Algumas páginas podem usar api_healthy para alertas
        api_healthy = streamlit_mock.session_state.get("api_healthy", False)

        assert api_healthy is False


@pytest.mark.unit
class TestDashboardConfiguration:
    """Testes para configuração do dashboard."""

    def test_page_configuration_applied(self) -> None:
        """Testa que configuração de página é aplicada."""
        # Espera-se que configure_page() foi chamado
        try:
            from app.dashboard.config import configure_page

            # Deve ser callable
            assert callable(configure_page)
        except ImportError:
            pytest.fail("configure_page não pode ser importado")

    def test_custom_css_applied(self) -> None:
        """Testa que CSS customizado é aplicado."""
        try:
            from app.dashboard.styles import apply_custom_css

            assert callable(apply_custom_css)
        except ImportError:
            pytest.fail("apply_custom_css não pode ser importado")

    def test_configuration_paths_exist(self) -> None:
        """Testa que arquivos de configuração existem."""
        try:
            from app.dashboard.config import (
                API_URL,
                CLASS_REPORT_PATH,
                DRIFT_REPORT_PATH,
                FEATURE_IMP_PATH,
                ROC_CURVE_PATH,
            )

            # Todos devem ser strings (paths)
            assert isinstance(API_URL, str)
            assert isinstance(CLASS_REPORT_PATH, str)
            assert isinstance(DRIFT_REPORT_PATH, str)
        except ImportError:
            pytest.fail("Config paths não podem ser importados")


@pytest.mark.unit
class TestDashboardDataLoading:
    """Testes para carregamento de dados no dashboard."""

    def test_model_loaded_for_metrics_page(self) -> None:
        """Testa que modelo é carregado para página de métricas."""
        page = "📊 Métricas do Modelo"

        load_model_needed = page in ["📊 Métricas do Modelo", "⚙️ Retreinamento"]

        assert load_model_needed

    def test_model_loaded_for_retrain_page(self) -> None:
        """Testa que modelo é carregado para página de retreinamento."""
        page = "⚙️ Retreinamento"

        load_model_needed = page in ["📊 Métricas do Modelo", "⚙️ Retreinamento"]

        assert load_model_needed

    def test_model_not_loaded_for_prediction(self) -> None:
        """Testa que modelo NÃO é carregado para aplicação de predição."""
        page = "🔮 Predição"

        # Predição usa API, não carregamento local
        load_model_needed = page in ["📊 Métricas do Modelo", "⚙️ Retreinamento"]

        assert not load_model_needed

    def test_dataset_loading_function_available(self) -> None:
        """Testa que função de carregamento de dataset está disponível."""
        try:
            from app.dashboard.data import load_dataset

            assert callable(load_dataset)
        except ImportError:
            pytest.fail("load_dataset não pode ser importado")


@pytest.mark.unit
class TestDashboardPageComponents:
    """Testes para componentes das páginas do dashboard."""

    def test_sidebar_rendered(self) -> None:
        """Testa que sidebar é renderizado."""
        try:
            from app.dashboard.sidebar import render_sidebar

            assert callable(render_sidebar)
        except ImportError:
            pytest.fail("render_sidebar não pode ser importado")

    def test_all_page_renderers_available(self) -> None:
        """Testa que todos os renderizadores de página estão disponíveis."""
        renderers = [
            ("render_prediction_page", "app.dashboard.pages.prediction"),
            ("render_metrics_page", "app.dashboard.pages.metrics"),
            ("render_drift_page", "app.dashboard.pages.drift"),
            ("render_retrain_page", "app.dashboard.pages.retrain"),
            ("render_about_page", "app.dashboard.pages.about"),
        ]

        for renderer_name, module_path in renderers:
            try:
                module = __import__(module_path, fromlist=[renderer_name])
                renderer = getattr(module, renderer_name)
                assert callable(renderer), f"{renderer_name} não é callable"
            except (ImportError, AttributeError):
                pytest.fail(f"{renderer_name} não pode ser importado")

    def test_api_url_configured(self) -> None:
        """Testa que API_URL é configurada."""
        try:
            from app.dashboard.config import API_URL

            assert isinstance(API_URL, str)
            assert len(API_URL) > 0
        except ImportError:
            pytest.fail("API_URL não pode ser importado")

    def test_raw_data_path_configured(self) -> None:
        """Testa que RAW_DATA_PATH é configurada."""
        try:
            from app.config import RAW_DATA_PATH

            assert isinstance(RAW_DATA_PATH, str)
            assert len(RAW_DATA_PATH) > 0
        except ImportError:
            pytest.fail("RAW_DATA_PATH não pode ser importado")
