"""
Schema para dados de entrada do aluno.

Define o contrato das 14 variáveis do modelo atual de predição.
"""

from typing import Optional

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """Contrato tipado para predição de risco de defasagem."""

    nivel_de_defasagem: Optional[float] = Field(
        None,
        description=("Diferença entre a série atual e a série ideal para a idade do aluno."),
    )
    idade: Optional[float] = Field(None, ge=0, le=120, description="Idade do aluno no ano-base")
    genero: Optional[float] = Field(
        None,
        description="Gênero em formato binário numérico (0/1) para o modelo",
    )
    ano_de_ingresso: Optional[float] = Field(
        None,
        ge=1990,
        le=2100,
        description="Ano em que o aluno ingressou na instituição",
    )
    veterano: Optional[float] = Field(
        None,
        description="Flag binária (0/1) indicando ingresso anterior ao ano de referência",
    )
    em_fase: Optional[float] = Field(
        None,
        description="Flag binária (0/1) indicando se está na fase ideal",
    )
    qtde_aval_realizadas: Optional[float] = Field(
        None,
        ge=0,
        description="Quantidade total de avaliações realizadas no período",
    )
    instituicao_ensino_mapped: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Tipo de instituição mapeada para binário: 0=Pública, 1=Privada",
    )
    iaa: Optional[float] = Field(None, ge=0, le=10, description="Índice de Autoavaliação")
    ieg: Optional[float] = Field(None, ge=0, le=10, description="Índice de Engajamento")
    ips: Optional[float] = Field(None, ge=0, le=10, description="Índice Psicossocial")
    ida: Optional[float] = Field(None, ge=0, le=10, description="Índice de Aprendizagem")
    ipv: Optional[float] = Field(None, ge=0, le=10, description="Índice de Ponto de Virada")
    ian: Optional[float] = Field(None, ge=0, le=10, description="Índice de Adequação Nivelar")
