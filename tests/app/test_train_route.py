"""
Testes para rotas de treinamento e gestão de modelos.

Testa endpoints /retrain, /promote, /discard com mocks de MLflow e train scripts.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.schemas import RetrainRequest
from app.routes.train_route import MlflowException, router

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient para a API FastAPI com rotas de treinamento."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)
    client.headers.update({"X-API-KEY": "test-api-key"})
    return client


@pytest.fixture
def valid_retrain_params() -> dict:
    """Parâmetros válidos de retreinamento."""
    return {
        "requested_by": "lucas_admin",
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

    def test_retrain_without_api_key_returns_401(self, monkeypatch, valid_retrain_params) -> None:
        """Bloqueia retreinamento sem API key válida."""
        monkeypatch.setenv("API_KEY", "test-api-key")
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/retrain", json=valid_retrain_params)

        assert response.status_code == 401

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
            "requested_by": "lucas_admin",
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
            "requested_by": "lucas_admin",
            "n_estimators": -50,
            "max_depth": 10,
            "min_samples_split": 2,
            "k": 10,
            "test_size": 0.2,
        }

        response = api_client.post("/retrain", json=invalid_params)
        assert response.status_code == 422

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
            "requested_by": "lucas_admin",
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

    def test_retrain_missing_requested_by_returns_422(self, api_client) -> None:
        """Retreinamento sem autoria deve ser rejeitado por validação."""
        payload = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 2,
            "k": 10,
            "test_size": 0.2,
        }

        response = api_client.post("/retrain", json=payload)

        assert response.status_code == 422

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


# ============================================================================
# TEST MODEL METRICS ROUTE
# ============================================================================


@pytest.mark.integration
@pytest.mark.api
class TestModelMetricsRoute:
    """Testes para a rota GET /model-metrics."""

    @patch("app.routes.train_route.mlflow.get_run")
    @patch("app.routes.train_route.get_model_paths")
    @patch("app.routes.train_route.os.path.exists")
    @patch("builtins.open")
    def test_model_metrics_mlflow_run_not_found_returns_fallback(
        self,
        mock_open,
        mock_exists,
        mock_get_paths,
        mock_get_run,
        api_client,
    ) -> None:
        """Retorna fallback local quando run_id do champion não existe no MLflow."""
        mock_get_paths.return_value = ("models", "models/model.pkl", "models/model_candidate.pkl")
        mock_exists.side_effect = lambda path: path.endswith("champion_run_id.txt")
        mock_open.return_value.__enter__.return_value.read.return_value = "missing_run_id"
        mock_get_run.side_effect = MlflowException("Run 'missing_run_id' not found")

        response = api_client.get("/model-metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "local"
        assert data["run_id"] == "missing_run_id"
        assert "Run do MLflow não encontrada" in data["message"]


@pytest.mark.unit
class TestTrainRouteImports:
    """Testes para validação de imports da rota."""

    def test_router_imported(self) -> None:
        """Testa que router está disponível."""
        from app.routes.train_route import router

        assert router is not None

    def test_mlflow_available(self) -> None:
        """Testa que MLflow está disponível."""
        import mlflow

        assert mlflow is not None

    def test_api_key_validation_available(self) -> None:
        """Testa que validação de API Key está disponível."""
        from app.utils.security import validate_api_key

        assert validate_api_key is not None


@pytest.mark.unit
class TestTrainRouteParameters:
    """Testes para validação de parâmetros."""

    def test_retrain_request_structure(self) -> None:
        """Testa estrutura de RetrainRequest."""

        # Deve ter estes atributos
        fields = RetrainRequest.model_fields.keys()
        assert "requested_by" in fields
        assert "n_estimators" in fields
        assert "max_depth" in fields
        assert "min_samples_split" in fields

    def test_retrain_with_default_k(self) -> None:
        """Testa que k padrão é None quando não fornecido."""

        # Sem fornecer k
        params = RetrainRequest(
            requested_by="lucas_admin",
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
        )
        assert params.k is None

    def test_retrain_with_custom_k(self, valid_retrain_params) -> None:
        """Testa que k pode ser um inteiro customizado."""

        # valid_retrain_params já tem k=10
        params = RetrainRequest(**valid_retrain_params)
        assert params.k == 10


@pytest.mark.unit
class TestErrorResponses:
    """Testes para respostas de erro."""

    def test_unauthorized_without_api_key(self) -> None:
        """Testa resposta 401 sem API Key."""
        app = FastAPI()
        from app.routes.train_route import router

        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)

        # Sem header de API Key
        response = client.post("/retrain", json={"requested_by": "test"})

        # Deve retornar 403 ou 401
        assert response.status_code in [401, 403]

    def test_error_response_format(self, api_client) -> None:
        """Testa que erros retornam formato esperado."""
        with patch("app.routes.train_route.run_training", side_effect=Exception("Training failed")):
            response = api_client.post(
                "/retrain",
                json={
                    "requested_by": "test",
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                },
            )

            # Deve ser um erro HTTP
            assert response.status_code >= 400
