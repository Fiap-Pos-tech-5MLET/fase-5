"""
Schema para resposta de predição.

Define o formato de saída do endpoint /predict.
"""

from typing import Any

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Representa a contribuição de uma feature para a predição."""

    feature_name: str = Field(min_length=1, description="Nome da feature analisada")
    feature_value: Any = Field(description="Valor observado da feature para o aluno")
    contribution: float = Field(description="Contribuição local da feature para o risco")
    direction: str = Field(
        pattern="^(aumenta_risco|reduz_risco)$",
        description="Direção do efeito: aumenta_risco ou reduz_risco",
    )


class PredictionResponse(BaseModel):
    """Resposta da predição com explicação dos campos."""

    risk_prediction: int = Field(ge=0, le=1, description="0 = Sem risco, 1 = Em risco de defasagem")
    risk_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidade de risco (0.0 a 1.0)",
    )
    explanation_method: str = Field(
        min_length=1,
        description="Método de explicação aplicado na inferência",
    )
    top_features: list[FeatureContribution] = Field(
        default_factory=list,
        description="Principais features que influenciaram a decisão do modelo",
    )
