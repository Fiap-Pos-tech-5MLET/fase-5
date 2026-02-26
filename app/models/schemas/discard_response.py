"""
Schema para resposta de descarte de modelo candidato.

Define o formato de saída do endpoint /discard.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DiscardResponse(BaseModel):
    """Resposta do descarte de modelo candidato."""

    status: Literal["discarded"] = Field(description="Status fixo da operação de descarte")
    message: str = Field(min_length=3, max_length=500, description="Mensagem de retorno")
