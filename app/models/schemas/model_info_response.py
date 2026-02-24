"""
Schema para informações do modelo.

Define o formato de saída do endpoint /model-info.
"""

from typing import Optional

from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    """Informações sobre o modelo em produção."""

    model_loaded: bool
    model_path: str
    loaded_at: Optional[str]
    model_type: Optional[str]
    n_features: Optional[int]
    features: Optional[list]
    retraining_strategy: dict
    production_scenarios: dict
