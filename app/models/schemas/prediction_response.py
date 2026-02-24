"""
Schema para resposta de predição.

Define o formato de saída do endpoint /predict.
"""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Resposta da predição com explicação dos campos."""

    risk_prediction: int = Field(description="0 = Sem risco, 1 = Em risco de defasagem")
    risk_probability: float = Field(description="Probabilidade de risco (0.0 a 1.0)")
