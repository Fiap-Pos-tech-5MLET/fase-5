"""
Schema para wrapper de dados do aluno.

Aceita formato flexível para retrocompatibilidade.
"""

from pydantic import BaseModel


class StudentData(BaseModel):
    """
    Aceita input flexível (dict) OU campos tipados.
    Mantém retrocompatibilidade com o formato anterior {"data": {...}}.
    """

    data: dict
