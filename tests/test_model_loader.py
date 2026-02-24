"""
Test suite para model_loader - Carregamento de modelos em produção.

Organização:
- TestModelLoaderBasic: Testes básicos de carregamento
- TestModelLoaderInfo: Testes de metadados do modelo
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestModelLoaderBasic:
    """Testes básicos para carregamento de modelos."""

    @patch("app.utils.model_loader.joblib.load")
    def test_load_model_returns_none_when_missing(self, mock_joblib_load) -> None:
        """Testa que load_model retorna (None, None) se arquivo não existe."""
        mock_joblib_load.side_effect = FileNotFoundError("Model not found")

        from app.utils.model_loader import load_model

        model, loaded_at = load_model()
        assert model is None
        assert loaded_at is None

    @patch("app.utils.model_loader.joblib.load")
    def test_load_model_returns_model_when_present(self, mock_joblib_load) -> None:
        """Testa que load_model retorna modelo e timestamp quando existe."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        from app.utils.model_loader import load_model

        model, loaded_at = load_model()
        assert model is mock_model
        assert loaded_at is not None

    @patch("app.utils.model_loader.joblib.load")
    def test_reload_model_calls_load(self, mock_joblib_load) -> None:
        """Testa que reload_model utiliza load_model internamente."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        from app.utils.model_loader import reload_model

        model, loaded_at = reload_model()
        assert model is mock_model
        assert loaded_at is not None

    def test_get_model_paths_returns_three_paths(self) -> None:
        """Testa que get_model_paths retorna três paths válidos."""
        from app.utils.model_loader import get_model_paths

        models_dir, model_path, candidate_path = get_model_paths()
        assert isinstance(models_dir, str)
        assert model_path.endswith("model.pkl")
        assert candidate_path.endswith("model_candidate.pkl")


@pytest.mark.unit
class TestModelLoaderInfo:
    """Testes de metadados do modelo carregado."""

    @patch("app.utils.model_loader.joblib.load")
    def test_get_current_model_after_load(self, mock_joblib_load) -> None:
        """Testa get_current_model após load_model."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        from app.utils.model_loader import get_current_model, load_model

        load_model()
        assert get_current_model() is mock_model

    @patch("app.utils.model_loader.joblib.load")
    def test_get_model_info_after_load(self, mock_joblib_load) -> None:
        """Testa get_model_info após load_model."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        from app.utils.model_loader import get_model_info, load_model

        load_model()
        info = get_model_info()

        assert "model_loaded" in info
        assert "model_path" in info
        assert "loaded_at" in info
