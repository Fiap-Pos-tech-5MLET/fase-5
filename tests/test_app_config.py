"""
Testes para app.config.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.unit
class TestAppConfig:
    """Testes de configuração da aplicação."""

    def test_paths_are_defined(self) -> None:
        """Testa que os paths principais são definidos."""
        from app import config

        assert config.BASE_DIR is not None
        assert config.DATA_DIR is not None
        assert config.MODEL_DIR is not None
        assert config.ARTIFACTS_DIR is not None

    def test_model_params_defaults(self) -> None:
        """Testa parâmetros padrão do modelo."""
        from app import config

        assert config.MODEL_PARAMS["n_estimators"] == 100
        assert "random_state" in config.MODEL_PARAMS

    def test_train_params_defaults(self) -> None:
        """Testa parâmetros padrão de treinamento."""
        from app import config

        assert config.TRAIN_PARAMS["test_size"] == 0.2

    def test_get_mlflow_tracking_uri_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Testa URI do MLflow padrão quando env não definida."""
        from app import config

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert config.get_mlflow_tracking_uri() == "file:./mlruns"

    def test_get_mlflow_tracking_uri_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Testa URI do MLflow via variável de ambiente."""
        from app import config

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        assert config.get_mlflow_tracking_uri() == "http://mlflow:5000"

    def test_directories_created(self) -> None:
        """Testa que diretórios são criados."""
        from app import config

        assert os.path.isdir(config.MODEL_DIR)
        assert os.path.isdir(config.ARTIFACTS_DIR)
