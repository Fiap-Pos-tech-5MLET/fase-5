"""
Schemas Pydantic para validação de dados da API.

Centraliza todos os contratos de entrada e saída,
garantindo type safety e documentação automática via OpenAPI.
"""

from .discard_response import DiscardResponse
from .model_info_response import ModelInfoResponse
from .model_metrics_response import ModelMetricsResponse
from .prediction_response import PredictionResponse
from .promote_response import PromoteResponse
from .retrain_request import RetrainRequest
from .student_data import StudentData
from .student_input import StudentInput

__all__ = [
    "DiscardResponse",
    "ModelInfoResponse",
    "ModelMetricsResponse",
    "PredictionResponse",
    "PromoteResponse",
    "RetrainRequest",
    "StudentData",
    "StudentInput",
]
