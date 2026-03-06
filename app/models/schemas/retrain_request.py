"""
Schema para requisição de retreinamento.

Define parâmetros aceitos no endpoint /retrain.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, Field, StringConstraints

RequestedBy = Annotated[str, StringConstraints(min_length=3, max_length=120, strip_whitespace=True)]


class RetrainRequest(BaseModel):
    """Parâmetros opcionais para retreinamento do modelo."""

    requested_by: RequestedBy = Field(
        description="Usuário responsável por solicitar o retreinamento"
    )
    n_estimators: int = Field(
        100,
        ge=10,
        le=1000,
        description="Número de árvores (boosting rounds) no LightGBM",
    )
    max_depth: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Profundidade máxima das árvores (None = sem limite explícito)",
    )
    learning_rate: float = Field(
        0.1,
        gt=0.0,
        le=1.0,
        description="Taxa de aprendizado do boosting",
    )
    num_leaves: int = Field(
        31,
        ge=2,
        le=255,
        description="Número máximo de folhas por árvore",
    )
    subsample: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description="Fraçao de linhas usadas por iteração",
    )
    colsample_bytree: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description="Fraçao de colunas usadas por árvore",
    )
    test_size: float = Field(
        0.2,
        gt=0.0,
        lt=0.5,
        description="Proporção de dados para teste (0.1 a 0.4)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "requested_by": "dashboard-teste",
                    "n_estimators": 100,
                    "max_depth": None,
                    "learning_rate": 0.1,
                    "num_leaves": 31,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "test_size": 0.2,
                }
            ]
        }
    }
