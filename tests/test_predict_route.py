"""
Testes para rotas da API da predição.

Testa endpoints /predict com mocks de modelo e preprocessamento.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.schemas import PredictionResponse, StudentData
from app.routes.predict_route import router

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def api_client() -> TestClient:
    """TestClient para a API FastAPI com rota de predição."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def valid_student_data() -> dict:
    """Dados válidos de estudante para predição."""
    return {
        "data": {
            "FASE": "5A",
            "IDADE": 13.5,
            "INDE_22": 75.0,
            "INDE_23": 78.5,
            "PEDRA_23": 85.0,
            "CG": 4.0,
            "GENERO": "M",
        }
    }


@pytest.fixture
def mock_model():
    """Mock de um modelo sklearn treinado."""
    model = MagicMock()
    model.predict = Mock(return_value=np.array([1]))
    model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
    model.feature_names_in_ = np.array(["IDADE", "INDE_23", "FASE"])
    model.named_steps = {"preprocessor": MagicMock()}
    return model


# ============================================================================
# TEST PREDICT ROUTE
# ============================================================================


@pytest.mark.integration
@pytest.mark.api
class TestPredictRoute:
    """Testes para a rota POST /predict."""

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_valid_input_returns_prediction(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """POST /predict com dados válidos retorna predição."""
        # Setup mocks
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df.copy()
        mock_missing.return_value = test_df.copy()
        mock_features.return_value = test_df.copy()
        mock_get_model.return_value = mock_model

        # Request
        response = api_client.post("/predict", json=valid_student_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "risk_prediction" in data
        assert "risk_probability" in data
        assert data["risk_prediction"] in [0, 1]
        assert 0.0 <= data["risk_probability"] <= 1.0

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_risk_prediction_is_binary(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """risk_prediction deve ser 0 ou 1."""
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_model.predict.return_value = np.array([1])
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=valid_student_data)
        data = response.json()

        assert data["risk_prediction"] in [0, 1]

    @patch("app.routes.predict_route.get_current_model")
    def test_predict_no_model_loaded_returns_503(
        self, mock_get_model, api_client, valid_student_data
    ) -> None:
        """POST /predict sem modelo retorna 503."""
        mock_get_model.return_value = None

        response = api_client.post("/predict", json=valid_student_data)

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_empty_input_processes(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        mock_model,
    ) -> None:
        """POST /predict com data={} (vazio) processa."""
        empty_data = {"data": {}}
        test_df = pd.DataFrame(empty_data["data"], index=[0])

        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=empty_data)

        assert response.status_code == 200
        assert "risk_prediction" in response.json()

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_probability_in_valid_range(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """risk_probability está sempre em [0.0, 1.0]."""
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=valid_student_data)
        prob = response.json()["risk_probability"]

        assert 0.0 <= prob <= 1.0

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_various_student_profiles(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        mock_model,
    ) -> None:
        """Predição funciona com diferentes perfis de alunos."""
        student_profiles = [
            {"data": {"IDADE": 10, "INDE_23": 50}},  # Aluno jovem, índice baixo
            {"data": {"IDADE": 18, "INDE_23": 95}},  # Aluno mais velho, índice alto
            {"data": {"FASE": "1A"}},  # Apenas fase
        ]

        mock_get_model.return_value = mock_model

        for profile in student_profiles:
            test_df = pd.DataFrame(profile["data"], index=[0])
            mock_clean.return_value = test_df
            mock_missing.return_value = test_df
            mock_features.return_value = test_df

            response = api_client.post("/predict", json=profile)

            assert response.status_code == 200
            assert "risk_prediction" in response.json()

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_calls_preprocessing_pipeline(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """Predição chama pipeline de preprocessamento."""
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        api_client.post("/predict", json=valid_student_data)

        # Verify pipeline was called
        mock_clean.assert_called_once()
        mock_missing.assert_called_once()
        mock_features.assert_called_once()

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_model_prediction_called(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """Predição chama model.predict() e model.predict_proba()."""
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        api_client.post("/predict", json=valid_student_data)

        # Verify model methods were called
        mock_model.predict.assert_called()
        mock_model.predict_proba.assert_called()

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_preprocessing_error_returns_400(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """Erro em preprocessamento retorna 400."""
        mock_clean.side_effect = ValueError("Invalid data")
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=valid_student_data)

        assert response.status_code == 400
        assert "Invalid data" in response.json()["detail"]

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_single_vs_multiple_predictions(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """Predição retorna sempre uma única predição (não lista)."""
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=valid_student_data)
        data = response.json()

        # Não deve ser lista, deve ser objeto único
        assert isinstance(data, dict)
        assert not isinstance(data["risk_prediction"], list)
        assert not isinstance(data["risk_probability"], list)

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_missing_fields(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        mock_model,
    ) -> None:
        """Testa predição com campos faltantes."""
        partial_data = {"data": {"IDADE": 15}}  # Apenas um campo
        test_df = pd.DataFrame(partial_data["data"], index=[0])

        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=partial_data)

        assert response.status_code == 200
        data = response.json()
        assert "risk_prediction" in data
        assert "risk_probability" in data

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_invalid_data_type(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        mock_model,
    ) -> None:
        """Testa predição com tipos de dados inválidos."""
        # API deve coercionar ou rejeitar dados inválidos
        invalid_data = {"data": {"IDADE": "not_a_number", "INDE_23": "invalid"}}
        test_df = pd.DataFrame({"IDADE": [0], "INDE_23": [0]}, index=[0])  # Após coerção

        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        # Pode ser 200 (com coerção) ou 400 (com rejeição)
        response = api_client.post("/predict", json=invalid_data)

        assert response.status_code in [200, 400]

    @patch("app.routes.predict_route.get_current_model")
    @patch("app.routes.predict_route.create_features")
    @patch("app.routes.predict_route.handle_missing_values")
    @patch("app.routes.predict_route.clean_data")
    def test_predict_response_format(
        self,
        mock_clean,
        mock_missing,
        mock_features,
        mock_get_model,
        api_client,
        valid_student_data,
        mock_model,
    ) -> None:
        """Testa formato da resposta."""
        test_df = pd.DataFrame(valid_student_data["data"], index=[0])
        mock_clean.return_value = test_df
        mock_missing.return_value = test_df
        mock_features.return_value = test_df
        mock_get_model.return_value = mock_model

        response = api_client.post("/predict", json=valid_student_data)
        data = response.json()

        # Validar estrutura da resposta
        assert isinstance(data, dict)
        assert set(data.keys()) == {"risk_prediction", "risk_probability"}
        assert isinstance(data["risk_prediction"], int)
        assert isinstance(data["risk_probability"], (int, float))
