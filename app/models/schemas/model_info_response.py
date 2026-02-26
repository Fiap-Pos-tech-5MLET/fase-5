"""
Schema para informações do modelo.

Define o formato de saída do endpoint /model-info.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Informações sobre o modelo em produção."""

    model_loaded: bool
    model_path: str = Field(min_length=1, description="Caminho do modelo em produção")
    loaded_at: Optional[str] = Field(default=None, min_length=1)
    model_type: Optional[str] = Field(default=None, min_length=1)
    n_features: Optional[int] = Field(default=None, ge=0)
    features: Optional[list[str]] = None
    retraining_strategy: dict[str, Any]
    production_scenarios: dict[str, Any]
