"""
Schema para resposta de descarte de modelo candidato.

Define o formato de saída do endpoint /discard.
"""

from pydantic import BaseModel


class DiscardResponse(BaseModel):
    """Resposta do descarte de modelo candidato."""

    status: str
    message: str
