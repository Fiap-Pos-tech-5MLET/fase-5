"""
Schema para wrapper de dados do aluno.

Aceita formato flexível para retrocompatibilidade.
"""

from typing import Any

from pydantic import BaseModel, Field


class StudentData(BaseModel):
    """
    Aceita input flexível (dict) OU campos tipados.
    Mantém retrocompatibilidade com o formato anterior {"data": {...}}.
    """

    data: dict[str, Any] = Field(description="Dicionário de atributos de entrada do aluno")
