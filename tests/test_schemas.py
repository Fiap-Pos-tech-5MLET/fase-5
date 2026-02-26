"""
Testes para schemas Pydantic (app/models/schemas/).

Valida contrato de dados, type hints, campos obrigatórios/opcionais,
aliases e transformações de dados.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas.model_info_response import ModelInfoResponse
from app.models.schemas.model_metrics_response import ModelMetricsResponse
from app.models.schemas.prediction_response import FeatureContribution, PredictionResponse
from app.models.schemas.promote_response import PromoteResponse
from app.models.schemas.retrain_request import RetrainRequest
from app.models.schemas.discard_response import DiscardResponse
from app.models.schemas.student_data import StudentData
from app.models.schemas.student_input import StudentInput

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def valid_student_input_dict():
    """Dicionário com dados válidos de StudentInput."""
    return {
        "FASE": "5A",
        "IDADE": 13.5,
        "INDE_22": 75.0,
        "INDE_23": 78.5,
        "PEDRA_23": 85.0,
        "CG": 4.0,
        "GENERO": "M",
        "ATIVO_INATIVO": "ATIVO",
    }


@pytest.fixture
def minimal_student_input_dict():
    """Dicionário mínimo para StudentInput (todos campos opcionais)."""
    return {}


@pytest.fixture
def valid_student_data_dict():
    """Dicionário com dados válidos de StudentData."""
    return {
        "data": {
            "FASE": "5A",
            "IDADE": 13.5,
            "INDE_23": 78.5,
        }
    }


# ============================================================================
# TEST STUDENT INPUT SCHEMA
# ============================================================================


@pytest.mark.unit
@pytest.mark.schemas
class TestStudentInputSchema:
    """Testes para StudentInput Pydantic model."""

    def test_student_input_creates_with_valid_data(self, valid_student_input_dict) -> None:
        """StudentInput deve aceitar dados válidos e criar instância."""
        student = StudentInput(**valid_student_input_dict)
        assert student.FASE == "5A"
        assert student.IDADE == 13.5
        assert student.INDE_22 == 75.0

    def test_student_input_all_fields_optional(self, minimal_student_input_dict) -> None:
        """StudentInput deve aceitar dict vazio (todos campos opcionais)."""
        student = StudentInput(**minimal_student_input_dict)
        assert student.FASE is None
        assert student.IDADE is None
        assert student.INDE_22 is None

    def test_student_input_accepts_genero_alias(self) -> None:
        """StudentInput deve aceitar alias 'GÊNERO' para 'GENERO'."""
        student = StudentInput(**{"GÊNERO": "F"})
        assert student.GENERO == "F"

    def test_student_input_accepts_genero_field(self) -> None:
        """StudentInput deve aceitar campo 'GENERO' diretamente."""
        student = StudentInput(**{"GENERO": "M"})
        assert student.GENERO == "M"

    def test_student_input_accepts_instituicao_alias(self) -> None:
        """StudentInput aceita alias 'INSTITUIÇÃO_DE_ENSINO'."""
        student = StudentInput(**{"INSTITUIÇÃO_DE_ENSINO": "Escola A"})
        assert student.INSTITUICAO_DE_ENSINO == "Escola A"

    def test_student_input_accepts_numer_av_alias(self) -> None:
        """StudentInput aceita alias 'Nº_AV' para 'N_AV'."""
        student = StudentInput(**{"Nº_AV": 3.0})
        assert student.N_AV == 3.0

    def test_student_input_accepts_ativo_inativo_alias(self) -> None:
        """StudentInput aceita alias 'ATIVO/_INATIVO'."""
        student = StudentInput(**{"ATIVO/_INATIVO": "ATIVO"})
        assert student.ATIVO_INATIVO == "ATIVO"

    def test_student_input_float_conversion(self) -> None:
        """StudentInput deve converter strings numéricas para float."""
        student = StudentInput(**{"IDADE": "13.5", "INDE_22": "75"})
        assert isinstance(student.IDADE, float)
        assert student.IDADE == 13.5
        assert student.INDE_22 == 75.0

    def test_student_input_invalid_float_raises_error(self) -> None:
        """StudentInput deve rejeitar string não-numérica para campos float."""
        with pytest.raises(ValidationError):
            StudentInput(**{"IDADE": "abc"})

    def test_student_input_rejects_out_of_range_age(self) -> None:
        """StudentInput deve rejeitar idade fora de faixa."""
        with pytest.raises(ValidationError):
            StudentInput(**{"IDADE": 150})

    def test_student_input_rejects_out_of_range_inde(self) -> None:
        """StudentInput deve rejeitar INDE fora de faixa."""
        with pytest.raises(ValidationError):
            StudentInput(**{"INDE_22": 120})

    def test_student_input_multiple_pedra_fields(self) -> None:
        """StudentInput deve aceitar PEDRA_20, PEDRA_21, PEDRA_22, PEDRA_23."""
        student = StudentInput(
            **{
                "PEDRA_20": 70.0,
                "PEDRA_21": 72.0,
                "PEDRA_22": 74.0,
                "PEDRA_23": 76.0,
            }
        )
        assert student.PEDRA_20 == 70.0
        assert student.PEDRA_23 == 76.0

    def test_student_input_multiple_avaliador_fields(self) -> None:
        """StudentInput deve aceitar até 6 avaliadores."""
        student = StudentInput(
            **{
                "AVALIADOR1": "Prof A",
                "AVALIADOR2": "Prof B",
                "AVALIADOR3": "Prof C",
                "AVALIADOR4": "Prof D",
                "AVALIADOR5": "Prof E",
                "AVALIADOR6": "Prof F",
            }
        )
        assert student.AVALIADOR1 == "Prof A"
        assert student.AVALIADOR6 == "Prof F"

    def test_student_input_model_config_populate_by_name(self, valid_student_input_dict) -> None:
        """StudentInput deve ter populate_by_name=True para aliases."""
        # Se temos um alias funcionando, isso já foi testado acima
        # Esse teste apenas valida que a config está ativa
        student = StudentInput(**valid_student_input_dict)
        assert StudentInput.model_config.get("populate_by_name") is True

    def test_student_input_fields_with_none(self) -> None:
        """StudentInput deve aceitar None explicitamente para campos opcionais."""
        student = StudentInput(**{"FASE": None, "IDADE": None, "INDE_22": None})
        assert student.FASE is None
        assert student.IDADE is None
        assert student.INDE_22 is None

    def test_student_input_inde_indices(self) -> None:
        """StudentInput deve aceitar INDE_22 e INDE_23."""
        student = StudentInput(**{"INDE_22": 50.0, "INDE_23": 60.0})
        assert student.INDE_22 == 50.0
        assert student.INDE_23 == 60.0


# ============================================================================
# TEST STUDENT DATA SCHEMA
# ============================================================================


@pytest.mark.unit
@pytest.mark.schemas
class TestStudentDataSchema:
    """Testes para StudentData Pydantic model."""

    def test_student_data_creates_with_valid_data(self, valid_student_data_dict) -> None:
        """StudentData deve aceitar dicionário de dados válidos."""
        student_data = StudentData(**valid_student_data_dict)
        assert student_data.data == {
            "FASE": "5A",
            "IDADE": 13.5,
            "INDE_23": 78.5,
        }

    def test_student_data_data_field_required(self) -> None:
        """StudentData exige campo 'data'."""
        with pytest.raises(ValidationError):
            StudentData()

    def test_student_data_accepts_empty_dict(self) -> None:
        """StudentData deve aceitar data={} (dict vazio)."""
        student_data = StudentData(**{"data": {}})
        assert student_data.data == {}

    def test_student_data_accepts_nested_student_input(self, valid_student_input_dict) -> None:
        """StudentData pode envolver StudentInput data."""
        student_data = StudentData(**{"data": valid_student_input_dict})
        assert student_data.data["FASE"] == "5A"
        assert student_data.data["IDADE"] == 13.5

    def test_student_data_flexible_format(self) -> None:
        """StudentData aceita qualquer estrutura dict dentro de 'data'."""
        student_data = StudentData(**{"data": {"custom_field": "value", "number": 42}})
        assert student_data.data["custom_field"] == "value"
        assert student_data.data["number"] == 42

    def test_student_data_preserves_dict_structure(self) -> None:
        """StudentData preserva estrutura do dicionário."""
        input_dict = {"data": {"level1": {"level2": {"value": "nested"}}}}
        student_data = StudentData(**input_dict)
        assert student_data.data["level1"]["level2"]["value"] == "nested"


# ============================================================================
# TEST PREDICTION RESPONSE SCHEMA
# ============================================================================


@pytest.mark.unit
@pytest.mark.schemas
class TestPredictionResponseSchema:
    """Testes para PredictionResponse Pydantic model."""

    def test_prediction_response_creates_with_valid_data(self) -> None:
        """PredictionResponse deve criar com risk_prediction e risk_probability."""
        prediction = PredictionResponse(
            **{
                "risk_prediction": 1,
                "risk_probability": 0.85,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert prediction.risk_prediction == 1
        assert prediction.risk_probability == 0.85

    def test_prediction_response_requires_risk_prediction(self) -> None:
        """PredictionResponse exige risk_prediction."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                **{
                    "risk_probability": 0.85,
                    "explanation_method": "shap",
                    "top_features": [],
                }
            )

    def test_prediction_response_requires_risk_probability(self) -> None:
        """PredictionResponse exige risk_probability."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                **{
                    "risk_prediction": 1,
                    "explanation_method": "shap",
                    "top_features": [],
                }
            )

    def test_prediction_response_risk_prediction_binary(self) -> None:
        """risk_prediction deve ser 0 ou 1."""
        pred0 = PredictionResponse(
            **{
                "risk_prediction": 0,
                "risk_probability": 0.1,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        pred1 = PredictionResponse(
            **{
                "risk_prediction": 1,
                "risk_probability": 0.9,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert pred0.risk_prediction == 0
        assert pred1.risk_prediction == 1

        with pytest.raises(ValidationError):
            PredictionResponse(
                **{
                    "risk_prediction": 2,
                    "risk_probability": 0.9,
                    "explanation_method": "shap",
                    "top_features": [],
                }
            )

    def test_prediction_response_risk_probability_range(self) -> None:
        """risk_probability deve estar em [0.0, 1.0]."""
        pred = PredictionResponse(
            **{
                "risk_prediction": 1,
                "risk_probability": 0.5,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert 0.0 <= pred.risk_probability <= 1.0

        with pytest.raises(ValidationError):
            PredictionResponse(
                **{
                    "risk_prediction": 1,
                    "risk_probability": 1.5,
                    "explanation_method": "shap",
                    "top_features": [],
                }
            )

    def test_prediction_response_probability_zero(self) -> None:
        """PredictionResponse deve aceitar risk_probability=0.0 (sem risco)."""
        prediction = PredictionResponse(
            **{
                "risk_prediction": 0,
                "risk_probability": 0.0,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert prediction.risk_probability == 0.0

    def test_prediction_response_probability_one(self) -> None:
        """PredictionResponse deve aceitar risk_probability=1.0 (risco certo)."""
        prediction = PredictionResponse(
            **{
                "risk_prediction": 1,
                "risk_probability": 1.0,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert prediction.risk_probability == 1.0

    def test_prediction_response_float_conversion(self) -> None:
        """risk_probability deve converter de string para float."""
        prediction = PredictionResponse(
            **{
                "risk_prediction": 1,
                "risk_probability": "0.75",
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert isinstance(prediction.risk_probability, float)
        assert prediction.risk_probability == 0.75

    def test_prediction_response_int_conversion(self) -> None:
        """risk_prediction deve converter de string para int."""
        prediction = PredictionResponse(
            **{
                "risk_prediction": "1",
                "risk_probability": 0.75,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        assert isinstance(prediction.risk_prediction, int)
        assert prediction.risk_prediction == 1

    def test_feature_contribution_validates_direction(self) -> None:
        """FeatureContribution deve validar direção permitida."""
        valid = FeatureContribution(
            feature_name="INDE_23",
            feature_value=75.0,
            contribution=0.2,
            direction="aumenta_risco",
        )
        assert valid.direction == "aumenta_risco"

        with pytest.raises(ValidationError):
            FeatureContribution(
                feature_name="INDE_23",
                feature_value=75.0,
                contribution=0.2,
                direction="indefinido",
            )


@pytest.mark.unit
@pytest.mark.schemas
class TestRetrainRequestSchema:
    """Testes para validação de atributos no schema de retreinamento."""

    def test_retrain_request_requires_requested_by(self) -> None:
        """requested_by deve ser obrigatório para governança."""
        with pytest.raises(ValidationError):
            RetrainRequest(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                k=10,
                test_size=0.2,
            )

    def test_retrain_request_rejects_invalid_n_estimators(self) -> None:
        """n_estimators negativo deve falhar."""
        with pytest.raises(ValidationError):
            RetrainRequest(
                requested_by="lucas_admin",
                n_estimators=-10,
                max_depth=10,
                min_samples_split=2,
                k=10,
                test_size=0.2,
            )


# ============================================================================
# TEST MODEL INFO RESPONSE SCHEMA
# ============================================================================


@pytest.mark.unit
@pytest.mark.schemas
class TestModelInfoResponseSchema:
    """Testes para ModelInfoResponse Pydantic model."""

    def test_model_info_response_creates_with_required_fields(self) -> None:
        """ModelInfoResponse exige model_loaded, model_path e retraining_strategy."""
        model_info = ModelInfoResponse(
            **{
                "model_loaded": True,
                "model_path": "/path/to/model.pkl",
                "loaded_at": None,
                "model_type": None,
                "n_features": None,
                "features": None,
                "retraining_strategy": {"frequency": "monthly"},
                "production_scenarios": {"scenario1": "data"},
            }
        )
        assert model_info.model_loaded is True
        assert model_info.model_path == "/path/to/model.pkl"

    def test_model_info_response_requires_model_loaded(self) -> None:
        """ModelInfoResponse exige model_loaded."""
        with pytest.raises(ValidationError):
            ModelInfoResponse(
                **{
                    "model_path": "/path/to/model.pkl",
                    "retraining_strategy": {},
                    "production_scenarios": {},
                    "loaded_at": None,
                    "model_type": None,
                    "n_features": None,
                    "features": None,
                }
            )

    def test_model_info_response_requires_model_path(self) -> None:
        """ModelInfoResponse exige model_path."""
        with pytest.raises(ValidationError):
            ModelInfoResponse(
                **{
                    "model_loaded": True,
                    "retraining_strategy": {},
                    "production_scenarios": {},
                    "loaded_at": None,
                    "model_type": None,
                    "n_features": None,
                    "features": None,
                }
            )

    def test_model_info_response_requires_retraining_strategy(self) -> None:
        """ModelInfoResponse exige retraining_strategy."""
        with pytest.raises(ValidationError):
            ModelInfoResponse(
                **{
                    "model_loaded": True,
                    "model_path": "/path/to/model.pkl",
                    "production_scenarios": {},
                    "loaded_at": None,
                    "model_type": None,
                    "n_features": None,
                    "features": None,
                }
            )

    def test_model_info_response_requires_production_scenarios(self) -> None:
        """ModelInfoResponse exige production_scenarios."""
        with pytest.raises(ValidationError):
            ModelInfoResponse(
                **{
                    "model_loaded": True,
                    "model_path": "/path/to/model.pkl",
                    "retraining_strategy": {},
                    "loaded_at": None,
                    "model_type": None,
                    "n_features": None,
                    "features": None,
                }
            )

    def test_model_info_response_optional_fields(self) -> None:
        """ModelInfoResponse tem campos opcionais: loaded_at, model_type, etc."""
        model_info = ModelInfoResponse(
            **{
                "model_loaded": True,
                "model_path": "/path/to/model.pkl",
                "loaded_at": "2024-01-15 10:30:00",
                "model_type": "RandomForest",
                "n_features": 25,
                "features": ["INDE_23", "IDADE", "FASE"],
                "retraining_strategy": {"frequency": "monthly"},
                "production_scenarios": {"scenario1": "data"},
            }
        )
        assert model_info.loaded_at == "2024-01-15 10:30:00"
        assert model_info.model_type == "RandomForest"
        assert model_info.n_features == 25

    def test_model_info_response_features_list(self) -> None:
        """features deve ser lista de strings."""
        model_info = ModelInfoResponse(
            **{
                "model_loaded": True,
                "model_path": "/path/to/model.pkl",
                "features": ["feat1", "feat2", "feat3"],
                "retraining_strategy": {},
                "production_scenarios": {},
                "loaded_at": None,
                "model_type": None,
                "n_features": None,
            }
        )
        assert isinstance(model_info.features, list)
        assert len(model_info.features) == 3

    def test_model_info_response_dict_fields(self) -> None:
        """retraining_strategy e production_scenarios devem ser dicts."""
        model_info = ModelInfoResponse(
            **{
                "model_loaded": True,
                "model_path": "/path/to/model.pkl",
                "retraining_strategy": {
                    "frequency": "monthly",
                    "min_samples": 500,
                },
                "production_scenarios": {
                    "scenario_a": {"threshold": 0.5},
                    "scenario_b": {"threshold": 0.7},
                },
                "loaded_at": None,
                "model_type": None,
                "n_features": None,
                "features": None,
            }
        )
        assert isinstance(model_info.retraining_strategy, dict)
        assert model_info.retraining_strategy["frequency"] == "monthly"

    def test_model_info_response_rejects_negative_features(self) -> None:
        """ModelInfoResponse deve rejeitar n_features negativo."""
        with pytest.raises(ValidationError):
            ModelInfoResponse(
                **{
                    "model_loaded": True,
                    "model_path": "/path/to/model.pkl",
                    "retraining_strategy": {"frequency": "monthly"},
                    "production_scenarios": {"scenario1": "data"},
                    "loaded_at": None,
                    "model_type": None,
                    "n_features": -1,
                    "features": None,
                }
            )


# ============================================================================
# TEST MODEL METRICS RESPONSE SCHEMA
# ============================================================================


@pytest.mark.unit
@pytest.mark.schemas
class TestModelMetricsResponseSchema:
    """Testes para ModelMetricsResponse Pydantic model."""

    def test_model_metrics_response_creates_with_valid_data(self) -> None:
        """ModelMetricsResponse deve criar com campos obrigatórios."""
        metrics = ModelMetricsResponse(
            **{
                "source": "mlflow",
                "run_id": "abc123",
                "metrics": {"accuracy": 0.85},
                "params": {"epochs": 50},
                "artifacts": ["model.pkl"],
            }
        )
        assert metrics.source == "mlflow"

    def test_model_metrics_response_requires_source(self) -> None:
        """ModelMetricsResponse exige source."""
        with pytest.raises(ValidationError):
            ModelMetricsResponse(
                **{
                    "metrics": {"accuracy": 0.85},
                    "params": {"epochs": 50},
                    "artifacts": ["model.pkl"],
                }
            )

    def test_model_metrics_response_requires_run_id(self) -> None:
        """ModelMetricsResponse exige run_id."""
        with pytest.raises(ValidationError):
            ModelMetricsResponse(
                **{
                    "source": "mlflow",
                    "metrics": {"accuracy": 0.85},
                    "params": {"epochs": 50},
                    "artifacts": ["model.pkl"],
                }
            )

    def test_model_metrics_response_optional_run_id(self) -> None:
        """ModelMetricsResponse tem run_id opcional."""
        metrics = ModelMetricsResponse(
            **{
                "source": "mlflow",
                "run_id": "abc123",
                "metrics": {"accuracy": 0.85},
                "params": {"epochs": 50},
                "artifacts": ["model.pkl"],
            }
        )
        assert metrics.run_id == "abc123"

    def test_model_metrics_response_optional_message(self) -> None:
        """ModelMetricsResponse tem message opcional."""
        metrics = ModelMetricsResponse(
            **{
                "source": "mlflow",
                "run_id": "abc123",
                "message": "Training completed",
                "metrics": {"accuracy": 0.85},
                "params": {"epochs": 50},
                "artifacts": ["model.pkl"],
            }
        )
        assert metrics.message == "Training completed"

    def test_model_metrics_response_metrics_dict(self) -> None:
        """metrics deve ser dicionário opcional."""
        metrics = ModelMetricsResponse(
            **{
                "source": "mlflow",
                "run_id": "abc123",
                "metrics": {"accuracy": 0.95, "f1": 0.92},
                "params": {"epochs": 50},
                "artifacts": ["model.pkl"],
            }
        )
        assert isinstance(metrics.metrics, dict)
        assert metrics.metrics["accuracy"] == 0.95

    def test_model_metrics_response_params_dict(self) -> None:
        """params deve ser dicionário opcional."""
        metrics = ModelMetricsResponse(
            **{
                "source": "mlflow",
                "run_id": "abc123",
                "metrics": {},
                "params": {"epochs": 50, "lr": 0.001},
                "artifacts": ["model.pkl"],
            }
        )
        assert isinstance(metrics.params, dict)
        assert metrics.params["epochs"] == 50

    def test_model_metrics_response_artifacts_list(self) -> None:
        """artifacts deve ser lista obrigatória."""
        metrics = ModelMetricsResponse(
            **{
                "source": "mlflow",
                "run_id": "abc123",
                "metrics": {},
                "params": {},
                "artifacts": ["model.pkl", "metrics.json", "data.csv"],
            }
        )
        assert isinstance(metrics.artifacts, list)
        assert len(metrics.artifacts) == 3

    def test_model_metrics_response_rejects_invalid_source(self) -> None:
        """ModelMetricsResponse deve rejeitar source fora do enum."""
        with pytest.raises(ValidationError):
            ModelMetricsResponse(
                **{
                    "source": "external",
                    "run_id": "abc123",
                    "metrics": {},
                    "params": {},
                    "artifacts": [],
                }
            )


# ============================================================================
# INTEGRATION TESTS (Pydantic Validation)
# ============================================================================


@pytest.mark.integration
@pytest.mark.schemas
class TestSchemaIntegration:
    """Testes de integração entre schemas."""

    def test_student_input_to_dict(self, valid_student_input_dict) -> None:
        """StudentInput deve ser serializável para dict."""
        student = StudentInput(**valid_student_input_dict)
        serialized = student.model_dump()
        assert isinstance(serialized, dict)
        assert serialized["FASE"] == "5A"

    def test_student_input_to_json(self, valid_student_input_dict) -> None:
        """StudentInput deve ser serializável para JSON."""
        student = StudentInput(**valid_student_input_dict)
        json_str = student.model_dump_json()
        assert isinstance(json_str, str)
        assert "5A" in json_str

    def test_student_data_wraps_student_input(self, valid_student_input_dict) -> None:
        """StudentData pode envolver dados de StudentInput."""
        student = StudentInput(**valid_student_input_dict)
        student_data = StudentData(**{"data": student.model_dump()})
        assert student_data.data["FASE"] == "5A"

    def test_prediction_response_json_serializable(self) -> None:
        """PredictionResponse deve ser JSON-serializável."""
        prediction = PredictionResponse(
            **{
                "risk_prediction": 1,
                "risk_probability": 0.85,
                "explanation_method": "shap",
                "top_features": [],
            }
        )
        json_str = prediction.model_dump_json()
        assert "1" in json_str
        assert "0.85" in json_str

    def test_model_info_response_json_serializable(self) -> None:
        """ModelInfoResponse deve ser JSON-serializável."""
        model_info = ModelInfoResponse(
            **{
                "model_loaded": True,
                "model_path": "/path/to/model.pkl",
                "retraining_strategy": {"frequency": "monthly"},
                "production_scenarios": {"scenario1": "data"},
                "loaded_at": None,
                "model_type": None,
                "n_features": None,
                "features": None,
            }
        )
        json_str = model_info.model_dump_json()
        assert "true" in json_str or "True" in json_str
        assert "/path/to/model.pkl" in json_str


@pytest.mark.unit
@pytest.mark.schemas
class TestPromoteAndDiscardResponseSchema:
    """Testes para schemas de promote/discard."""

    def test_promote_response_accepts_valid_payload(self) -> None:
        """PromoteResponse deve aceitar status promoted."""
        response = PromoteResponse(
            status="promoted",
            message="Modelo promovido",
            loaded_at="2026-02-26T15:00:00+00:00",
            champion_run_id="run-123",
        )
        assert response.status == "promoted"

    def test_promote_response_rejects_invalid_status(self) -> None:
        """PromoteResponse deve rejeitar status fora do permitido."""
        with pytest.raises(ValidationError):
            PromoteResponse(
                status="done",
                message="ok",
                loaded_at="2026-02-26T15:00:00+00:00",
                champion_run_id="run-123",
            )

    def test_discard_response_rejects_invalid_status(self) -> None:
        """DiscardResponse deve rejeitar status fora do permitido."""
        with pytest.raises(ValidationError):
            DiscardResponse(status="removed", message="ok")
