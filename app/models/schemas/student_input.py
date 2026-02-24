"""
Schema para dados de entrada do aluno.

Define o contrato completo de dados do aluno para predição.
"""

from typing import Optional

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """
    Contrato de dados tipado para predição de risco de defasagem.

    Todos os campos são opcionais — a API preenche campos faltantes com
    valores padrão seguros (0 para numéricos, 'UNKNOWN' para categóricos).

    Campos mais relevantes para a predição: INDE_22, INDE_23, FASE, IDADE.
    """

    # --- Campos Demográficos ---
    FASE: Optional[str] = Field(None, description="Fase do aluno (ex: '1A', '2B', '7')")
    TURMA: Optional[str] = Field(None, description="Turma do aluno (ex: 'A', 'B')")
    IDADE: Optional[float] = Field(None, description="Idade do aluno em anos")
    GENERO: Optional[str] = Field(None, alias="GÊNERO", description="Gênero do aluno")
    DATA_DE_NASC: Optional[str] = Field(None, description="Data de nascimento")
    ANO_INGRESSO: Optional[float] = Field(None, description="Ano de ingresso na associação")
    INSTITUICAO_DE_ENSINO: Optional[str] = Field(
        None, alias="INSTITUIÇÃO_DE_ENSINO", description="Escola do aluno"
    )

    # --- Histórico de Performance (INDE) ---
    INDE_22: Optional[float] = Field(None, description="Índice de Desenvolvimento Educacional 2022")
    INDE_23: Optional[float] = Field(None, description="Índice de Desenvolvimento Educacional 2023")

    # --- Histórico de Pedra ---
    PEDRA_20: Optional[float] = Field(None, description="Pedra classificatória 2020")
    PEDRA_21: Optional[float] = Field(None, description="Pedra classificatória 2021")
    PEDRA_22: Optional[float] = Field(None, description="Pedra classificatória 2022")
    PEDRA_23: Optional[float] = Field(None, description="Pedra classificatória 2023")

    # --- Avaliações ---
    CG: Optional[float] = Field(None, description="Conceito Geral")
    CF: Optional[float] = Field(None, description="Conceito Final")
    CT: Optional[float] = Field(None, description="Conceito Total")
    N_AV: Optional[float] = Field(None, alias="Nº_AV", description="Número de avaliações")
    AVALIADOR1: Optional[str] = Field(None, description="Avaliador 1")
    AVALIADOR2: Optional[str] = Field(None, description="Avaliador 2")
    AVALIADOR3: Optional[str] = Field(None, description="Avaliador 3")
    AVALIADOR4: Optional[str] = Field(None, description="Avaliador 4")
    AVALIADOR5: Optional[str] = Field(None, description="Avaliador 5")
    AVALIADOR6: Optional[str] = Field(None, description="Avaliador 6")

    # --- Situação Escolar ---
    FASE_IDEAL: Optional[str] = Field(None, description="Fase ideal para a idade")
    ESCOLA: Optional[str] = Field(None, description="Nome da escola")
    ATIVO_INATIVO: Optional[str] = Field(
        None, alias="ATIVO/_INATIVO", description="Status ativo/inativo"
    )

    model_config = {"populate_by_name": True}
