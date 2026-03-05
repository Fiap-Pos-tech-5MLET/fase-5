"""
Testes para health check da API (app/dashboard/health.py).

Valida funções de verificação de saúde da API e status do modelo.
"""

from __future__ import annotations

import sys
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Mock streamlit primeiro
if "streamlit" not in sys.modules:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_data = lambda ttl=None: lambda func: func
    sys.modules["streamlit"] = streamlit_stub

from app.dashboard.health import check_api_health, get_api_status


class TestCheckAPIHealth:
    """Testes para check_api_health()."""

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_success(self, mock_get: MagicMock) -> None:
        """Deve retornar True quando API responde com status 200."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        result = check_api_health("http://127.0.0.1:8000/api", timeout=5)
        
        # Assert
        assert result is True
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "/model-info" in call_args[0][0]
        assert call_args[1]["timeout"] == 5

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_failure_status_code(self, mock_get: MagicMock) -> None:
        """Deve retornar False quando API responde com status diferente de 200."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        # Act
        result = check_api_health("http://127.0.0.1:8000/api")
        
        # Assert
        assert result is False

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_timeout(self, mock_get: MagicMock) -> None:
        """Deve retornar False quando ocorre timeout de conexão."""
        # Arrange
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        # Act
        result = check_api_health("http://127.0.0.1:8000/api", timeout=1)
        
        # Assert
        assert result is False

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_connection_error(self, mock_get: MagicMock) -> None:
        """Deve retornar False quando ocorre erro de conexão."""
        # Arrange
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        # Act
        result = check_api_health("http://127.0.0.1:8000/api")
        
        # Assert
        assert result is False

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_generic_error(self, mock_get: MagicMock) -> None:
        """Deve retornar False para qualquer exceção RequestException."""
        # Arrange
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Generic error")
        
        # Act
        result = check_api_health("http://127.0.0.1:8000/api")
        
        # Assert
        assert result is False

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_custom_timeout(self, mock_get: MagicMock) -> None:
        """Deve usar timeout customizado."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        check_api_health("http://127.0.0.1:8000/api", timeout=15)
        
        # Assert
        call_args = mock_get.call_args
        assert call_args[1]["timeout"] == 15

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_url_construction(self, mock_get: MagicMock) -> None:
        """Deve construir URL correta com /model-info."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        api_url = "https://example.com/api"
        
        # Act
        check_api_health(api_url)
        
        # Assert
        called_url = mock_get.call_args[0][0]
        assert called_url == f"{api_url}/model-info"

    @patch("app.dashboard.health.requests.get")
    def test_check_api_health_headers(self, mock_get: MagicMock) -> None:
        """Deve incluir header Accept: application/json."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        check_api_health("http://127.0.0.1:8000/api")
        
        # Assert
        call_kwargs = mock_get.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["Accept"] == "application/json"


class TestGetAPIStatus:
    """Testes para get_api_status()."""

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_success(self, mock_get: MagicMock) -> None:
        """Deve retornar status do modelo quando API responde com 200."""
        # Arrange
        expected_status = {
            "model_id": "rf-2024-03-05",
            "version": "1.0.0",
            "loaded": True,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_status
        mock_get.return_value = mock_response
        
        # Act
        result = get_api_status("http://127.0.0.1:8000/api")
        
        # Assert
        assert result == expected_status
        assert result["model_id"] == "rf-2024-03-05"

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_failure(self, mock_get: MagicMock) -> None:
        """Deve retornar None quando API responde com status diferente de 200."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        # Act
        result = get_api_status("http://127.0.0.1:8000/api")
        
        # Assert
        assert result is None

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_timeout(self, mock_get: MagicMock) -> None:
        """Deve retornar None quando ocorre timeout."""
        # Arrange
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        # Act
        result = get_api_status("http://127.0.0.1:8000/api", timeout=2)
        
        # Assert
        assert result is None

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_connection_error(self, mock_get: MagicMock) -> None:
        """Deve retornar None quando ocorre erro de conexão."""
        # Arrange
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        # Act
        result = get_api_status("http://127.0.0.1:8000/api")
        
        # Assert
        assert result is None

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_custom_timeout(self, mock_get: MagicMock) -> None:
        """Deve usar timeout customizado."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        # Act
        get_api_status("http://127.0.0.1:8000/api", timeout=10)
        
        # Assert
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 10

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_json_parsing(self, mock_get: MagicMock) -> None:
        """Deve parsear JSON corretamente."""
        # Arrange
        status_data = {
            "model_id": "model-001",
            "version": "2.0.0",
            "features": 25,
            "loaded": True,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = status_data
        mock_get.return_value = mock_response
        
        # Act
        result = get_api_status("http://127.0.0.1:8000/api")
        
        # Assert
        assert isinstance(result, dict)
        assert result["features"] == 25
        assert result["version"] == "2.0.0"

    @patch("app.dashboard.health.requests.get")
    def test_get_api_status_url_construction(self, mock_get: MagicMock) -> None:
        """Deve construir URL correta."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        api_url = "https://prod.example.com/api"
        
        # Act
        get_api_status(api_url)
        
        # Assert
        called_url = mock_get.call_args[0][0]
        assert called_url == f"{api_url}/model-info"
