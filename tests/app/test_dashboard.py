"""
Test suite para o dashboard Streamlit.

Organização:
- TestDashboardConfig: Testes de configuração
- TestDashboardData: Testes de carregamento de dados
- TestDashboardPages: Testes das páginas
"""

import pytest


@pytest.mark.unit
@pytest.mark.dashboard
class TestDashboardConfig:
    """Testes para configuração do dashboard."""

    def test_dashboard_imports(self):
        """Testa que dashboard pode ser importado."""
        try:
            import app.dashboard

            assert True
        except ImportError:
            pytest.fail("Dashboard não pode ser importado")

    def test_dashboard_config_module_exists(self):
        """Testa que módulo de configuração existe."""
        try:
            from app.dashboard import config

            assert config is not None
        except ImportError:
            pytest.fail("Dashboard config não pode ser importado")

    def test_dashboard_data_module_exists(self):
        """Testa que módulo de dados existe."""
        try:
            from app.dashboard import data

            assert data is not None
        except ImportError:
            pytest.fail("Dashboard data não pode ser importado")

    def test_dashboard_sidebar_module_exists(self):
        """Testa que módulo sidebar existe."""
        try:
            from app.dashboard import sidebar

            assert sidebar is not None
        except ImportError:
            pytest.fail("Dashboard sidebar não pode ser importado")

    def test_dashboard_styles_module_exists(self):
        """Testa que módulo styles existe."""
        try:
            from app.dashboard import styles

            assert styles is not None
        except ImportError:
            pytest.fail("Dashboard styles não pode ser importado")


@pytest.mark.unit
@pytest.mark.dashboard
class TestDashboardData:
    """Testes de carregamento de dados do dashboard."""

    def test_load_model_function_exists(self):
        """Testa que função de carregamento de modelo existe."""
        try:
            from app.dashboard.data import load_model

            assert callable(load_model)
        except ImportError:
            pytest.fail("load_model não pode ser importada")

    def test_load_dataset_function_exists(self):
        """Testa que função de carregamento de dataset existe."""
        try:
            from app.dashboard.data import load_dataset

            assert callable(load_dataset)
        except ImportError:
            pytest.fail("load_dataset não pode ser importada")

    def test_get_model_metrics_function_exists(self):
        """Testa que função de métricas existe."""
        try:
            from app.dashboard.data import get_model_metrics

            assert callable(get_model_metrics)
        except ImportError:
            pytest.fail("get_model_metrics não pode ser importada")

    def test_predict_via_api_function_exists(self):
        """Testa que função de predição via API existe."""
        try:
            from app.dashboard.data import predict_via_api

            assert callable(predict_via_api)
        except ImportError:
            pytest.fail("predict_via_api não pode ser importada")


@pytest.mark.unit
@pytest.mark.dashboard
class TestDashboardPages:
    """Testes para páginas do dashboard."""

    def test_prediction_page_exists(self):
        """Testa que página de predição existe."""
        try:
            from app.dashboard.pages import prediction

            assert prediction is not None
        except ImportError:
            pytest.fail("Prediction page não pode ser importada")

    def test_metrics_page_exists(self):
        """Testa que página de métricas existe."""
        try:
            from app.dashboard.pages import metrics

            assert metrics is not None
        except ImportError:
            pytest.fail("Metrics page não pode ser importada")

    def test_drift_page_exists(self):
        """Testa que página de drift existe."""
        try:
            from app.dashboard.pages import drift

            assert drift is not None
        except ImportError:
            pytest.fail("Drift page não pode ser importada")

    def test_retrain_page_exists(self):
        """Testa que página de retrain existe."""
        try:
            from app.dashboard.pages import retrain

            assert retrain is not None
        except ImportError:
            pytest.fail("Retrain page não pode ser importada")

    def test_about_page_exists(self):
        """Testa que página de about existe."""
        try:
            from app.dashboard.pages import about

            assert about is not None
        except ImportError:
            pytest.fail("About page não pode ser importada")

    def test_sidebar_navigation_exists(self):
        """Testa que sidebar de navegação existe."""
        try:
            from app.dashboard import sidebar

            assert sidebar is not None
        except ImportError:
            pytest.fail("Sidebar navigation não pode ser importada")
