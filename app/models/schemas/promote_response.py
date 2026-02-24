"""
Schema para resposta de promoção de modelo.

Define o formato de saída do endpoint /promote.
"""

from typing import Optional

from pydantic import BaseModel


class PromoteResponse(BaseModel):
    """Resposta da promoção de modelo."""

    status: str
    message: str
    loaded_at: str
    champion_run_id: Optional[str]
