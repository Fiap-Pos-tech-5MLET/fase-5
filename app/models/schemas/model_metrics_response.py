"""
Schema para resposta de métricas do modelo.

Define o formato de saída do endpoint /model-metrics.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ModelMetricsResponse(BaseModel):
    """Métricas do modelo champion."""

    source: Literal["local", "mlflow", "error"]
    run_id: Optional[str] = Field(min_length=1)
    message: Optional[str] = Field(default=None, min_length=1, max_length=500)
    run_name: Optional[str] = Field(default=None, min_length=1)
    start_time: Optional[int] = Field(default=None, ge=0)
    end_time: Optional[int] = Field(default=None, ge=0)
    status: Optional[Literal["RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"]] = None
    metrics: Optional[dict[str, Any]] = None
    params: Optional[dict[str, Any]] = None
    artifacts: list[str]
