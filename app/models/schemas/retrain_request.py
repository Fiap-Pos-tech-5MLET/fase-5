"""
Schema para requisição de retreinamento.

Define parâmetros aceitos no endpoint /retrain.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, Field, StringConstraints

RequestedBy = Annotated[str, StringConstraints(min_length=3, max_length=120, strip_whitespace=True)]


class RetrainRequest(BaseModel):
    """Parâmetros opcionais para retreinamento do modelo."""

    requested_by: RequestedBy = Field(description="Usuário responsável por solicitar o retreinamento")
    n_estimators: int = Field(
        100,
        ge=10,
        le=1000,
        description="Número de árvores no Random Forest",
    )
    max_depth: Optional[int] = Field(
        None,
        ge=2,
        le=100,
        description="Profundidade máxima (None = ilimitada)",
    )
    min_samples_split: int = Field(2, ge=2, le=50, description="Mínimo de amostras para split")
    k: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Features selecionadas via SelectKBest (None = todas)",
    )
    test_size: float = Field(
        0.2,
        gt=0.0,
        lt=0.5,
        description="Proporção de dados para teste (0.1 a 0.4)",
    )
