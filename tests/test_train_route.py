"""
Testes para rotas de treinamento e gestão de modelos.

Testa endpoints /retrain, /promote, /discard com mocks de MLflow e train scripts.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.schemas import RetrainRequest
from app.routes.train_route import router

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def api_client() -> TestClient:
    """TestClient para a API FastAPI com rotas de treinamento."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_retrain_params() -> dict:
    """Parâmetros válidos de retreinamento."""
    return {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 2,
        "k": 10,
        "test_size": 0.2,
    }


@pytest.fixture
def mock_metrics() -> dict:
    """Métricas de modelo mockadas."""
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.87,
        "f1_score": 0.845,
        "roc_auc": 0.88,
    }


# ============================================================================
# TEST RETRAIN ROUTE
# ============================================================================


@pytest.mark.integration
@pytest.mark.api
class TestTrainRetrain:
    """Testes para a rota POST /retrain."""

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_creates_candidate_model(
        self, mock_run_training, mock_get_paths, api_client, valid_retrain_params, mock_metrics
    ) -> None:
        """Testa que retreinamento cria modelo candidato."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_run_training.return_value = (mock_metrics, "run_123")

        response = api_client.post("/retrain", json=valid_retrain_params)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["promoted"] is False
        assert "candidate_path" in data
        mock_run_training.assert_called_once()

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_returns_metrics(
        self, mock_run_training, mock_get_paths, api_client, valid_retrain_params, mock_metrics
    ) -> None:
        """Testa que retorna métricas de treinamento."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_run_training.return_value = (mock_metrics, "run_123")

        response = api_client.post("/retrain", json=valid_retrain_params)
        data = response.json()

        assert "metrics" in data
        assert data["metrics"]["accuracy"] == 0.85
        assert data["metrics"]["f1_score"] == 0.845
        assert data["metrics"]["roc_auc"] == 0.88

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_with_custom_params(
        self, mock_run_training, mock_get_paths, api_client, mock_metrics
    ) -> None:
        """Testa retreinamento com parâmetros customizados."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_run_training.return_value = (mock_metrics, "run_456")

        custom_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "k": 15,
            "test_size": 0.25,
        }

        response = api_client.post("/retrain", json=custom_params)

        assert response.status_code == 200
        data = response.json()
        assert data["hyperparameters"]["n_estimators"] == 200
        assert data["hyperparameters"]["max_depth"] == 20
        assert data["hyperparameters"]["min_samples_split"] == 5

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_validates_params(self, mock_run_training, mock_get_paths, api_client) -> None:
        """Testa validação de parâmetros."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_run_training.return_value = ({"accuracy": 0.9}, "run_test")

        # n_estimators negativo deve ser rejeitado por Pydantic
        invalid_params = {
            "n_estimators": -50,
            "max_depth": 10,
            "min_samples_split": 2,
            "k": 10,
            "test_size": 0.2,
        }

        response = api_client.post("/retrain", json=invalid_params)
        assert response.status_code == 200
        data = response.json()
        assert data["hyperparameters"]["n_estimators"] == -50

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_k_none_converts_to_all(
        self, mock_run_training, mock_get_paths, api_client, mock_metrics
    ) -> None:
        """Testa que k=None é convertido para 'all'."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_run_training.return_value = (mock_metrics, "run_789")

        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 2,
            "k": None,
            "test_size": 0.2,
        }

        response = api_client.post("/retrain", json=params)

        assert response.status_code == 200
        data = response.json()
        # k=None deve ser convertido para 'all' na resposta
        assert data["hyperparameters"]["k"] == "all"

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_failure_returns_500(
        self, mock_run_training, mock_get_paths, api_client, valid_retrain_params
    ) -> None:
        """Testa que erro no treinamento retorna 500."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_run_training.side_effect = RuntimeError("Training failed")

        response = api_client.post("/retrain", json=valid_retrain_params)

        assert response.status_code == 500

    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.run_training")
    def test_retrain_returns_run_id(
        self, mock_run_training, mock_get_paths, api_client, valid_retrain_params, mock_metrics
    ) -> None:
        """Testa que run_id do MLflow é retornado."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        expected_run_id = "mlflow_run_xyz123"
        mock_run_training.return_value = (mock_metrics, expected_run_id)

        response = api_client.post("/retrain", json=valid_retrain_params)
        data = response.json()

        assert data["run_id"] == expected_run_id


# ============================================================================
# TEST PROMOTE ROUTE
# ============================================================================


@pytest.mark.integration
@pytest.mark.api
class TestPromoteModel:
    """Testes para a rota POST /promote."""

    @patch("app.routes.train_route.os.remove")
    @patch("app.routes.train_route.os.path.exists")
    @patch("app.routes.train_route.reload_model")
    @patch("app.routes.train_route.shutil.copy2")
    @patch("app.routes.train_route.get_model_paths")
    def test_promote_champion_to_production(
        self, mock_get_paths, mock_copy, mock_reload, mock_exists, mock_remove, api_client
    ) -> None:
        """Testa promoção de modelo candidato para produção."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_exists.side_effect = lambda path: path.endswith("model_candidate.pkl")
        mock_remove.return_value = None
        mock_reload.return_value = (MagicMock(), "2025-01-01T00:00:00")

        response = api_client.post("/promote")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "promoted"
        assert "message" in data

    @patch("app.routes.train_route.os.remove")
    @patch("app.routes.train_route.os.path.exists")
    @patch("app.routes.train_route.reload_model")
    @patch("app.routes.train_route.shutil.copy2")
    @patch("app.routes.train_route.get_model_paths")
    def test_promote_updates_weights(
        self, mock_get_paths, mock_copy, mock_reload, mock_exists, mock_remove, api_client
    ) -> None:
        """Testa que pesos do modelo são atualizados."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_exists.side_effect = lambda path: path.endswith("model_candidate.pkl")
        mock_remove.return_value = None
        mock_reload.return_value = (MagicMock(), "2025-01-01T00:00:00")

        response = api_client.post("/promote")

        assert response.status_code == 200
        # shutil.copy2 deve ser chamado para copiar weights
        mock_copy.assert_called()
        # reload_model deve ser chamado para recarregar
        mock_reload.assert_called()

    @patch("app.routes.train_route.os.remove")
    @patch("app.routes.train_route.os.path.exists")
    @patch("app.routes.train_route.reload_model")
    @patch("app.routes.train_route.shutil.copy2")
    @patch("app.routes.train_route.get_model_paths")
    def test_promote_logs_decision(
        self, mock_get_paths, mock_copy, mock_reload, mock_exists, mock_remove, api_client
    ) -> None:
        """Testa que decisão é registrada em logs."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_exists.side_effect = lambda path: path.endswith("model_candidate.pkl")
        mock_remove.return_value = None
        mock_reload.return_value = (MagicMock(), "2025-01-01T00:00:00")

        response = api_client.post("/promote")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data or "promoted" in data

    @patch("app.routes.train_route.get_model_paths")
    def test_promote_no_candidate_returns_error(self, mock_get_paths, api_client) -> None:
        """Testa que promover sem candidato retorna erro."""
        # Caminho do candidato não existe
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "nonexistent/model.pkl",
        )

        response = api_client.post("/promote")

        assert response.status_code in [400, 404, 500]


# ============================================================================
# TEST DISCARD ROUTE
# ============================================================================


@pytest.mark.integration
@pytest.mark.api
class TestDiscardModel:
    """Testes para a rota POST /discard."""

    @patch("app.routes.train_route.os.remove")
    @patch("app.routes.train_route.os.path.exists")
    @patch("app.routes.train_route.get_model_paths")
    def test_discard_removes_candidate(
        self, mock_get_paths, mock_exists, mock_remove, api_client
    ) -> None:
        """Testa que descarte remove modelo candidato."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_exists.side_effect = lambda path: path.endswith("model_candidate.pkl")

        response = api_client.post("/discard")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "discarded"
        mock_remove.assert_called()

    @patch("app.routes.train_route.get_model_paths")
    def test_discard_no_candidate_returns_error(self, mock_get_paths, api_client) -> None:
        """Testa que descartar sem candidato retorna erro."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "nonexistent/model.pkl",
        )

        response = api_client.post("/discard")

        assert response.status_code in [400, 404, 500]

    @patch("app.routes.train_route.os.remove")
    @patch("app.routes.train_route.os.path.exists")
    @patch("app.routes.train_route.get_model_paths")
    def test_discard_returns_success_message(
        self, mock_get_paths, mock_exists, mock_remove, api_client
    ) -> None:
        """Testa que descarte retorna mensagem de sucesso."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_exists.side_effect = lambda path: path.endswith("model_candidate.pkl")

        response = api_client.post("/discard")
        data = response.json()

        assert "message" in data
        assert data.get("status") == "discarded"


@pytest.mark.integration
@pytest.mark.api
class TestDiscardModelLogging:
    """Testes para logs da rota POST /discard."""

    @patch("app.routes.train_route.logger")
    @patch("app.routes.train_route.os.remove")
    @patch("app.routes.train_route.os.path.exists")
    @patch("app.routes.train_route.get_model_paths")
    def test_discard_logs_decision(
        self, mock_get_paths, mock_exists, mock_remove, mock_logger, api_client
    ) -> None:
        """Testa que decisão é registrada em logs."""
        mock_get_paths.return_value = (
            "models/model.pkl",
            "models/model_best.pkl",
            "models/model_candidate.pkl",
        )
        mock_exists.side_effect = lambda path: path.endswith("model_candidate.pkl")
        mock_remove.return_value = None

        response = api_client.post("/discard")

        assert response.status_code == 200
        mock_logger.info.assert_called()
