"""Testes para o módulo keep_alive."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.utils.keep_alive import ping_self, start_keep_alive


class TestKeepAlive:
    """Testes para o sistema de keep-alive."""

    @patch("app.utils.keep_alive.requests.get")
    @patch("app.utils.keep_alive.time.sleep")
    def test_ping_self_success(
        self,
        mock_sleep: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """
        Testa ping_self com resposta bem-sucedida.

        Args:
            mock_sleep (MagicMock): Mock da função time.sleep.
            mock_get (MagicMock): Mock da função requests.get.
        """
        # Arrange - simula apenas 1 iteração do loop
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Configurar para parar após 1 iteração
        def stop_after_one_call(*args, **kwargs):
            mock_sleep.side_effect = KeyboardInterrupt
            return None

        mock_sleep.side_effect = stop_after_one_call

        # Act & Assert
        with pytest.raises(KeyboardInterrupt):
            ping_self()

        # Verificar chamada ao requests.get
        expected_url = os.getenv("RENDER_APP_URL", "http://localhost:8080")
        mock_get.assert_called_once_with(
            f"{expected_url}/health",
            timeout=5,
            allow_redirects=False,
        )

    @patch("app.utils.keep_alive.requests.get")
    @patch("app.utils.keep_alive.time.sleep")
    def test_ping_self_non_200_status(
        self,
        mock_sleep: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """
        Testa ping_self com resposta status code != 200.

        Args:
            mock_sleep (MagicMock): Mock da função time.sleep.
            mock_get (MagicMock): Mock da função requests.get.
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        def stop_after_one_call(*args, **kwargs):
            mock_sleep.side_effect = KeyboardInterrupt
            return None

        mock_sleep.side_effect = stop_after_one_call

        # Act & Assert
        with pytest.raises(KeyboardInterrupt):
            ping_self()

        assert mock_get.called

    @patch("app.utils.keep_alive.requests.get")
    @patch("app.utils.keep_alive.time.sleep")
    def test_ping_self_request_exception(
        self,
        mock_sleep: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """
        Testa ping_self com exceção de requests.

        Args:
            mock_sleep (MagicMock): Mock da função time.sleep.
            mock_get (MagicMock): Mock da função requests.get.
        """
        # Arrange - simula timeout
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")

        def stop_after_one_call(*args, **kwargs):
            mock_sleep.side_effect = KeyboardInterrupt
            return None

        mock_sleep.side_effect = stop_after_one_call

        # Act & Assert - não deve levantar exceção
        with pytest.raises(KeyboardInterrupt):
            ping_self()

        assert mock_get.called

    @patch("app.utils.keep_alive.requests.get")
    @patch("app.utils.keep_alive.time.sleep")
    def test_ping_self_generic_exception(
        self,
        mock_sleep: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """
        Testa ping_self com exceção genérica.

        Args:
            mock_sleep (MagicMock): Mock da função time.sleep.
            mock_get (MagicMock): Mock da função requests.get.
        """
        # Arrange
        mock_get.side_effect = ValueError("Unexpected error")

        def stop_after_one_call(*args, **kwargs):
            mock_sleep.side_effect = KeyboardInterrupt
            return None

        mock_sleep.side_effect = stop_after_one_call

        # Act & Assert
        with pytest.raises(KeyboardInterrupt):
            ping_self()

        assert mock_get.called

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False)
    @patch("app.utils.keep_alive.threading.Thread")
    def test_start_keep_alive_production(self, mock_thread_class: MagicMock) -> None:
        """
        Testa start_keep_alive em ambiente de produção.

        Args:
            mock_thread_class (MagicMock): Mock da classe threading.Thread.
        """
        # Arrange
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        # Act
        result = start_keep_alive()

        # Assert
        mock_thread_class.assert_called_once_with(target=ping_self, daemon=True)
        mock_thread_instance.start.assert_called_once()
        assert result == mock_thread_instance

    @patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False)
    def test_start_keep_alive_non_production(self) -> None:
        """Testa start_keep_alive em ambiente não-produção."""
        # Act
        result = start_keep_alive()

        # Assert
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_start_keep_alive_no_environment(self) -> None:
        """Testa start_keep_alive sem variável ENVIRONMENT."""
        # Act
        result = start_keep_alive()

        # Assert
        assert result is None

    @patch.dict(
        os.environ,
        {
            "RENDER_APP_URL": "https://test.onrender.com",
            "KEEP_ALIVE_INTERVAL": "300",
        },
        clear=False,
    )
    @patch("app.utils.keep_alive.requests.get")
    @patch("app.utils.keep_alive.time.sleep")
    def test_ping_self_custom_environment_vars(
        self,
        mock_sleep: MagicMock,
        mock_get: MagicMock,
    ) -> None:
        """
        Testa ping_self com variáveis de ambiente customizadas.

        Args:
            mock_sleep (MagicMock): Mock da função time.sleep.
            mock_get (MagicMock): Mock da função requests.get.
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        def stop_after_one_call(*args, **kwargs):
            mock_sleep.side_effect = KeyboardInterrupt
            return None

        mock_sleep.side_effect = stop_after_one_call

        # Act & Assert
        with pytest.raises(KeyboardInterrupt):
            ping_self()

        # Verificar que usou as variáveis customizadas
        mock_get.assert_called_once_with(
            "https://test.onrender.com/health",
            timeout=5,
            allow_redirects=False,
        )
        # Sleep deve ter sido chamado com 300 segundos
        mock_sleep.assert_called_with(300)
