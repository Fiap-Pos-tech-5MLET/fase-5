"""
Testes para schemas Pydantic (app/models/schemas/).

Valida contrato de dados, type hints, campos obrigatórios/opcionais,
aliases e transformações de dados.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas.discard_response import DiscardResponse
from app.models.schemas.model_info_response import ModelInfoResponse
from app.models.schemas.model_metrics_response import ModelMetricsResponse
from app.models.schemas.prediction_response import FeatureContribution, PredictionResponse
from app.models.schemas.promote_response import PromoteResponse
from app.models.schemas.retrain_request import RetrainRequest
from app.models.schemas.student_data import StudentData
from app.models.schemas.student_input import StudentInput

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def valid_student_input_dict():
    """Dicionário com dados válidos de StudentInput."""
    return {
        "nivel_de_defasagem": 0.0,
        "idade": 13.5,
        "genero": 1.0,
        "ano_de_ingresso": 2021,
        "veterano": 1.0,
        "em_fase": 1.0,
        "qtde_aval_realizadas": 8,
        "instituicao_ensino": 2.0,
        "iaa": 6.5,
        "ieg": 7.1,
        "ips": 6.8,
        "ida": 7.3,
        "ipv": 6.9,
        "ian": 6.7,
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
            "idade": 13.5,
            "iaa": 6.8,
            "ieg": 7.0,
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
        assert student.idade == 13.5
        assert student.iaa == 6.5
        assert student.nivel_de_defasagem == 0.0

    def test_student_input_all_fields_optional(self, minimal_student_input_dict) -> None:
        """StudentInput deve aceitar dict vazio (todos campos opcionais)."""
        student = StudentInput(**minimal_student_input_dict)
        assert student.idade is None
        assert student.iaa is None
        assert student.ano_de_ingresso is None

    def test_student_input_float_conversion(self) -> None:
        """StudentInput deve converter strings numéricas para float."""
        student = StudentInput(**{"idade": "13.5", "iaa": "7.2"})
        assert isinstance(student.idade, float)
        assert student.idade == 13.5
        assert student.iaa == 7.2

    def test_student_input_invalid_float_raises_error(self) -> None:
        """StudentInput deve rejeitar string não-numérica para campos float."""
        with pytest.raises(ValidationError):
            StudentInput(**{"idade": "abc"})

    def test_student_input_rejects_out_of_range_age(self) -> None:
        """StudentInput deve rejeitar idade fora de faixa."""
        with pytest.raises(ValidationError):
            StudentInput(**{"idade": 150})

    def test_student_input_rejects_out_of_range_iaa(self) -> None:
        """StudentInput deve rejeitar IAA fora de faixa."""
        with pytest.raises(ValidationError):
            StudentInput(**{"iaa": 12})

    def test_student_input_accepts_binary_flags(self) -> None:
        """StudentInput deve aceitar variáveis binárias do contrato atual."""
        student = StudentInput(**{"veterano": 1, "em_fase": 0, "genero": 1})
        assert student.veterano == 1
        assert student.em_fase == 0
        assert student.genero == 1

    def test_student_input_fields_with_none(self) -> None:
        """StudentInput deve aceitar None explicitamente para campos opcionais."""
        student = StudentInput(**{"idade": None, "iaa": None, "ipv": None})
        assert student.idade is None
        assert student.iaa is None
        assert student.ipv is None

    def test_student_input_indices_range(self) -> None:
        """StudentInput deve aceitar os índices de 0 a 10."""
        student = StudentInput(
            **{"iaa": 5.0, "ieg": 6.0, "ips": 7.0, "ida": 8.0, "ipv": 9.0, "ian": 10.0}
        )
        assert student.iaa == 5.0
        assert student.ian == 10.0


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
            "idade": 13.5,
            "iaa": 6.8,
            "ieg": 7.0,
        }

    def test_student_data_data_field_required(self) -> None:
        """StudentData aplica payload default quando 'data' não é enviado."""
        student_data = StudentData()
        assert "idade" in student_data.data
        assert student_data.data["idade"] == 12.0

    def test_student_data_accepts_empty_dict(self) -> None:
        """StudentData deve aceitar data={} (dict vazio)."""
        student_data = StudentData(**{"data": {}})
        assert student_data.data == {}

    def test_student_data_accepts_nested_student_input(self, valid_student_input_dict) -> None:
        """StudentData pode envolver StudentInput data."""
        student_data = StudentData(**{"data": valid_student_input_dict})
        assert student_data.data["idade"] == 13.5
        assert student_data.data["iaa"] == 6.5

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
                learning_rate=0.1,
                num_leaves=31,
                test_size=0.2,
            )

    def test_retrain_request_rejects_invalid_n_estimators(self) -> None:
        """n_estimators negativo deve falhar."""
        with pytest.raises(ValidationError):
            RetrainRequest(
                requested_by="lucas_admin",
                n_estimators=-10,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
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
        assert serialized["idade"] == 13.5

    def test_student_input_to_json(self, valid_student_input_dict) -> None:
        """StudentInput deve ser serializável para JSON."""
        student = StudentInput(**valid_student_input_dict)
        json_str = student.model_dump_json()
        assert isinstance(json_str, str)
        assert "idade" in json_str

    def test_student_data_wraps_student_input(self, valid_student_input_dict) -> None:
        """StudentData pode envolver dados de StudentInput."""
        student = StudentInput(**valid_student_input_dict)
        student_data = StudentData(**{"data": student.model_dump()})
        assert student_data.data["idade"] == 13.5

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
