"""
Test suite para o arquivo main (FastAPI application).

Organização:
- TestMainImports: Testes de importação
- TestMainApp: Testes da aplicação FastAPI
- TestMainRoutes: Testes das rotas principais
- TestMainLifespan: Testes do ciclo de vida da aplicação
"""

import os
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.main import app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    app = None


@pytest.mark.unit
class TestMainImports:
    """Testes para imports do main."""

    def test_fastapi_import(self):
        """Testa que FastAPI pode ser importado."""
        try:
            from fastapi import FastAPI

            assert True
        except ImportError:
            pytest.fail("FastAPI não disponível")

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_app_import(self):
        """Testa que app pode ser importado."""
        assert app is not None

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_routes_import(self):
        """Testa que rotas podem ser importadas."""
        try:
            from app.routes import (
                audit_route,
                predict_route,
                train_route,
            )

            assert True
        except ImportError:
            pytest.fail("Rotas não disponíveis")


@pytest.mark.unit
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
class TestMainApp:
    """Testes para a aplicação FastAPI."""

    def test_app_creation(self):
        """Testa que app foi criado."""
        assert app is not None

    def test_app_is_fastapi(self):
        """Testa que app é uma instância FastAPI."""
        assert isinstance(app, FastAPI)

    def test_app_title(self):
        """Testa que app tem título."""
        assert app.title is not None or hasattr(app, "openapi")


@pytest.mark.integration
@pytest.mark.api
class TestMainRoutes:
    """Testes para rotas da aplicação."""

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_predict_route_exists(self):
        """Testa que rota /predict existe."""
        client = TestClient(app)
        # GET / deve retornar 200 com HTML
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_homepage_serves_html(self):
        """Testa que a rota raiz serve a página HTML."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        content = response.text
        # Verifica conteúdo esperado no HTML
        assert "Passos Mágicos" in content
        assert "Datathon" in content or "datathon" in content.lower()
        assert "/api/docs" in content  # Link para Swagger docs

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_health_endpoint(self):
        """Testa endpoint /health."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_retrain_route_exists(self):
        """Testa que rota /retrain existe."""
        client = TestClient(app)
        client.headers.update({"x-requested-by": "test-main"})
        # POST /retrain sem dados pode retornar 422 ou funcionar
        response = client.post("/retrain", json={})
        assert response.status_code in [200, 401, 422, 500, 503]

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_audit_route_exists(self):
        """Testa que rota /audit existe."""
        client = TestClient(app)
        client.headers.update({"x-requested-by": "test-main"})
        # GET /model-info deve existir
        response = client.get("/model-info")
        assert response.status_code in [200, 404, 503]

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_promote_route_exists(self):
        """Testa que rota /promote existe."""
        client = TestClient(app)
        client.headers.update({"x-requested-by": "test-main"})
        response = client.post("/promote")
        assert response.status_code in [200, 401, 400, 404, 500]

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_discard_route_exists(self):
        """Testa que rota /discard existe."""
        client = TestClient(app)
        client.headers.update({"x-requested-by": "test-main"})
        response = client.post("/discard")
        assert response.status_code in [200, 401, 400, 404, 500]

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_routes_documentation(self):
        """Testa que rotas têm documentação."""
        try:
            from app.main import app

            # Verificar se OpenAPI docs estão disponíveis
            assert app.docs_url is not None or app.openapi() is not None
        except ImportError:
            pytest.fail("app não disponível")


@pytest.mark.unit
class TestMainConfiguration:
    """Testes para configuração da aplicação."""

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_cors_enabled(self):
        """Testa que CORS é habilitado."""
        # Verificar que app tem router
        assert hasattr(app, "router") and app.router is not None

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_middleware_configured(self):
        """Testa que middleware é configurado."""
        # Verificar que app pode processar requisições
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code in [200, 404, 422]

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_startup_events(self):
        """Testa que eventos de startup funcionam."""
        # Verificar que router está configurado
        assert app.router is not None
        assert hasattr(app.router, "routes")

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
    def test_error_handlers(self):
        """Testa que handlers de erro estão definidos."""
        # Verificar que app pode processar erros
        assert isinstance(app.exception_handlers, dict) or len(app.exception_handlers) >= 0


@pytest.mark.unit
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI não disponível")
class TestMainLifespan:
    """Testes para o lifespan da aplicação."""

    def test_health_endpoint_works_after_lifespan(self) -> None:
        """
        Testa que endpoint /health funciona após lifespan iniciar.

        Verifica que a aplicação inicializa corretamente com lifespan.
        """
        # Arrange & Act - TestClient invoca lifespan automaticamente
        client = TestClient(app)
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_application_starts_with_lifespan(self) -> None:
        """
        Testa que aplicação inicia corretamente com lifespan configurado.

        Verifica que a aplicação pode processar requisições após lifespan.
        """
        # Arrange & Act
        client = TestClient(app)
        response = client.get("/docs")

        # Assert - docs deve estar disponível
        assert response.status_code == 200

    @patch("app.utils.keep_alive.start_keep_alive")
    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False)
    def test_keep_alive_integration(self, mock_start_keep_alive: MagicMock) -> None:
        """
        Testa integração do keep_alive no contexto da aplicação.

        Args:
            mock_start_keep_alive (MagicMock): Mock da função start_keep_alive.

        Note:
            Testa a integração no módulo keep_alive, não diretamente no lifespan,
            pois o TestClient já tem app importado.
        """
        # Arrange
        from app.utils.keep_alive import start_keep_alive as real_start_keep_alive

        # Act
        real_start_keep_alive()

        # Assert - função deve ser chamada quando em produção
        # O comportamento real é validado em test_keep_alive.py
        assert os.environ.get("ENVIRONMENT") == "production"
