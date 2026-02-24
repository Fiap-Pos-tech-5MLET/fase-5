"""
Schema para requisição de retreinamento.

Define parâmetros aceitos no endpoint /retrain.
"""

from typing import Optional

from pydantic import BaseModel, Field


class RetrainRequest(BaseModel):
    """Parâmetros opcionais para retreinamento do modelo."""

    n_estimators: int = Field(100, description="Número de árvores no Random Forest")
    max_depth: Optional[int] = Field(None, description="Profundidade máxima (None = ilimitada)")
    min_samples_split: int = Field(2, description="Mínimo de amostras para split")
    k: Optional[int] = Field(
        None, description="Features selecionadas via SelectKBest (None = todas)"
    )
    test_size: float = Field(0.2, description="Proporção de dados para teste (0.1 a 0.4)")
