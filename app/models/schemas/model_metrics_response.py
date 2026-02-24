"""
Schema para resposta de métricas do modelo.

Define o formato de saída do endpoint /model-metrics.
"""

from typing import Optional

from pydantic import BaseModel


class ModelMetricsResponse(BaseModel):
    """Métricas do modelo champion."""

    source: str
    run_id: Optional[str]
    message: Optional[str] = None
    run_name: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    status: Optional[str] = None
    metrics: Optional[dict]
    params: Optional[dict]
    artifacts: list
