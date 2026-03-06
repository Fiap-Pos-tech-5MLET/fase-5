"""
Schema para wrapper de dados do aluno.

Aceita formato flexível para retrocompatibilidade.
"""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_STUDENT_DATA: dict[str, float] = {
    "nivel_de_defasagem": 0.0,
    "idade": 12.0,
    "genero": 0.0,
    "ano_de_ingresso": 2022.0,
    "veterano": 0.0,
    "em_fase": 1.0,
    "qtde_aval_realizadas": 4.0,
    "iaa": 6.0,
    "ieg": 6.0,
    "ips": 6.0,
    "ida": 6.0,
    "ipv": 6.0,
    "ian": 6.0,
}


class StudentData(BaseModel):
    """
    Aceita input flexível (dict) OU campos tipados.
    Mantém retrocompatibilidade com o formato anterior {"data": {...}}.
    """

    data: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_STUDENT_DATA.copy(),
        description="Dicionário de atributos de entrada do aluno",
        examples=[DEFAULT_STUDENT_DATA],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"data": cast(dict[str, Any], DEFAULT_STUDENT_DATA)},
                {
                    "data": {
                        "nivel_de_defasagem": 1.0,
                        "idade": 14.0,
                        "genero": 1.0,
                        "ano_de_ingresso": 2021.0,
                        "veterano": 1.0,
                        "em_fase": 0.0,
                        "qtde_aval_realizadas": 3.0,
                        "iaa": 5.1,
                        "ieg": 5.4,
                        "ips": 4.8,
                        "ida": 5.0,
                        "ipv": 4.6,
                        "ian": 5.2,
                    }
                },
            ]
        }
    )
