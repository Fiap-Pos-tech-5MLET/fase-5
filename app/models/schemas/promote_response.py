"""
Schema para resposta de promoção de modelo.

Define o formato de saída do endpoint /promote.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PromoteResponse(BaseModel):
    """Resposta da promoção de modelo."""

    status: Literal["promoted"] = Field(description="Status fixo da promoção")
    message: str = Field(min_length=3, max_length=500, description="Mensagem de retorno")
    loaded_at: str = Field(min_length=1, description="Timestamp de carregamento do modelo")
    champion_run_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Run ID do modelo champion no MLflow",
    )
