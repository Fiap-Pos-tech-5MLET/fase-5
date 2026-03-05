"""
Transformações e criação de features específicas para o projeto Passos Mágicos.

Este módulo contém funções de feature engineering específicas do domínio educacional,
incluindo normalização de fases, extração de turmas, e criação de flags de condição
escolar (veterano, em_fase, etc).

Funções principais:
- Normalização de FASE (ALFA → Fase 0, etc)
- Extração de TURMA (8A → A, etc)
- Criação de flags de condição (veterano, em_fase)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def obter_nova_turma(fase: str) -> str:
    """
    Extrai a letra da turma a partir do código da fase.

    Args:
        fase (str): Código da fase (ex: '8A', '1B', 'ALFA').

    Returns:
        str: Letra da turma ('A', 'B', etc) ou 'NÃO SE APLICA' para fases sem turma.

    Examples:
        >>> obter_nova_turma('8A')
        'A'
        >>> obter_nova_turma('ALFA')
        'NÃO SE APLICA'
        >>> obter_nova_turma('9')
        'NÃO SE APLICA'
    """
    if pd.isna(fase) or fase == "9":
        return "NÃO SE APLICA"

    fase_str = str(fase).strip().upper()

    # Casos especiais sem turma
    if fase_str in ["ALFA", "9"]:
        return "NÃO SE APLICA"

    # Extrai última letra se existir
    if len(fase_str) >= 2 and fase_str[-1].isalpha():
        return fase_str[-1]

    return "NÃO SE APLICA"


def obter_nova_fase(fase: str) -> str:
    """
    Normaliza o código da fase para formato padrão 'Fase X (descrição)'.

    Mapeamento 2022/2023:
    - ALFA → Fase 0 (1° e 2° ano)
    - 1A/1B → Fase 1 (3° e 4° ano)
    - 2A/2B → Fase 2 (5° e 6° ano)
    - 3A/3B → Fase 3 (7° e 8° ano)
    - 4A/4B → Fase 4 (9° ano)
    - 5A/5B → Fase 5 (1° EM)
    - 6A → Fase 6 (2° EM)
    - 7A → Fase 7 (3° EM)
    - 8A/8B → Fase 8 (Universitários)
    - 9 → NÃO SE APLICA (descartado)

    Args:
        fase (str): Código da fase original.

    Returns:
        str: Fase normalizada ou 'NÃO SE APLICA'.

    Examples:
        >>> obter_nova_fase('ALFA')
        'Fase 0 (1° e 2° ano)'
        >>> obter_nova_fase('8A')
        'Fase 8 (Universitários)'
        >>> obter_nova_fase('9')
        'NÃO SE APLICA'
    """
    if pd.isna(fase) or fase == "9":
        return "NÃO SE APLICA"

    fase_str = str(fase).strip().upper()

    # Mapeamento completo
    mapeamento = {
        "ALFA": "Fase 0 (1° e 2° ano)",
        "1A": "Fase 1 (3° e 4° ano)",
        "1B": "Fase 1 (3° e 4° ano)",
        "2A": "Fase 2 (5° e 6° ano)",
        "2B": "Fase 2 (5° e 6° ano)",
        "3A": "Fase 3 (7° e 8° ano)",
        "3B": "Fase 3 (7° e 8° ano)",
        "4A": "Fase 4 (9° ano)",
        "4B": "Fase 4 (9° ano)",
        "5A": "Fase 5 (1° EM)",
        "5B": "Fase 5 (1° EM)",
        "6A": "Fase 6 (2° EM)",
        "7A": "Fase 7 (3° EM)",
        "8A": "Fase 8 (Universitários)",
        "8B": "Fase 8 (Universitários)",
        "9": "NÃO SE APLICA",
    }

    return mapeamento.get(fase_str, "NÃO SE APLICA")


def obter_nova_fase_24(fase: str) -> str:
    """
    Normaliza o código da fase para formato padrão no ano 2024.

    Diferenças em relação a 2022/2023:
    - '9' é uma fase válida em 2024 (não mais descartado)
    - Mantém mesmo mapeamento geral

    Args:
        fase (str): Código da fase original.

    Returns:
        str: Fase normalizada ou 'NÃO SE APLICA'.

    Examples:
        >>> obter_nova_fase_24('9')
        'Fase 4 (9° ano)'
        >>> obter_nova_fase_24('ALFA')
        'Fase 0 (1° e 2° ano)'
    """
    if pd.isna(fase):
        return "NÃO SE APLICA"

    fase_str = str(fase).strip().upper()

    mapeamento_24 = {
        "ALFA": "Fase 0 (1° e 2° ano)",
        "1A": "Fase 1 (3° e 4° ano)",
        "1B": "Fase 1 (3° e 4° ano)",
        "2A": "Fase 2 (5° e 6° ano)",
        "2B": "Fase 2 (5° e 6° ano)",
        "3A": "Fase 3 (7° e 8° ano)",
        "3B": "Fase 3 (7° e 8° ano)",
        "4A": "Fase 4 (9° ano)",
        "4B": "Fase 4 (9° ano)",
        "5A": "Fase 5 (1° EM)",
        "5B": "Fase 5 (1° EM)",
        "6A": "Fase 6 (2° EM)",
        "7A": "Fase 7 (3° EM)",
        "8A": "Fase 8 (Universitários)",
        "8B": "Fase 8 (Universitários)",
        "9": "Fase 4 (9° ano)",  # Diferença: '9' é válido em 2024
    }

    return mapeamento_24.get(fase_str, "NÃO SE APLICA")


def obter_nova_turma_24(fase: str) -> str:
    """
    Extrai a letra da turma para o ano 2024 (mesmo comportamento de `obter_nova_turma`).

    Args:
        fase (str): Código da fase.

    Returns:
        str: Letra da turma ou 'NÃO SE APLICA'.

    Examples:
        >>> obter_nova_turma_24('8A')
        'A'
        >>> obter_nova_turma_24('9')
        'NÃO SE APLICA'
    """
    # Em 2024, o comportamento é o mesmo da função original
    return obter_nova_turma(fase)


def criar_coluna_veterano(df: pd.DataFrame, col_ano_ingresso: str, ano_corte: int = 2022) -> pd.DataFrame:
    """
    Cria coluna binária 'VETERANO' indicando se o aluno ingressou antes do ano de corte.

    Args:
        df (pd.DataFrame): DataFrame com dados dos alunos.
        col_ano_ingresso (str): Nome da coluna contendo o ano de ingresso.
        ano_corte (int): Ano de referência para definir veterano (padrão: 2022).

    Returns:
        pd.DataFrame: DataFrame com nova coluna 'VETERANO' (1=veterano, 0=novato).

    Examples:
        >>> df = pd.DataFrame({"ANO_INGRESSO": [2020, 2022, 2023]})
        >>> criar_coluna_veterano(df, "ANO_INGRESSO", 2022)
           ANO_INGRESSO  VETERANO
        0          2020         1
        1          2022         0
        2          2023         0
    """
    df_out = df.copy()
    df_out["VETERANO"] = (df_out[col_ano_ingresso] < ano_corte).astype(int)
    return df_out


def criar_coluna_em_fase(
    df: pd.DataFrame, col_fase_atual: str, col_fase_ideal: str
) -> pd.DataFrame:
    """
    Cria coluna binária 'EM_FASE' indicando se o aluno está na fase ideal para sua idade.

    Args:
        df (pd.DataFrame): DataFrame com dados dos alunos.
        col_fase_atual (str): Nome da coluna com a fase atual do aluno.
        col_fase_ideal (str): Nome da coluna com a fase ideal esperada.

    Returns:
        pd.DataFrame: DataFrame com nova coluna 'EM_FASE' (1=em fase, 0=defasado).

    Examples:
        >>> df = pd.DataFrame({
        ...     "FASE_ATUAL": ["Fase 2", "Fase 1", "Fase 3"],
        ...     "FASE_IDEAL": ["Fase 2", "Fase 2", "Fase 3"]
        ... })
        >>> criar_coluna_em_fase(df, "FASE_ATUAL", "FASE_IDEAL")
          FASE_ATUAL FASE_IDEAL  EM_FASE
        0     Fase 2     Fase 2        1
        1     Fase 1     Fase 2        0
        2     Fase 3     Fase 3        1
    """
    df_out = df.copy()
    df_out["EM_FASE"] = (df_out[col_fase_atual] == df_out[col_fase_ideal]).astype(int)
    return df_out


def aplicar_transformacoes_fase_turma(
    df: pd.DataFrame,
    col_fase: str,
    ano: int = 2023,
    criar_turma: bool = True,
) -> pd.DataFrame:
    """
    Aplica transformações de FASE e TURMA em um DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.
        col_fase (str): Nome da coluna contendo os códigos de fase.
        ano (int): Ano de referência (2022/2023 ou 2024) para escolher função correta.
        criar_turma (bool): Se True, cria também a coluna de TURMA.

    Returns:
        pd.DataFrame: DataFrame com novas colunas 'FASE_PADRONIZADA' e opcionalmente 'TURMA'.

    Examples:
        >>> df = pd.DataFrame({"FASE_ORIGINAL": ["8A", "ALFA", "9"]})
        >>> aplicar_transformacoes_fase_turma(df, "FASE_ORIGINAL", ano=2023)
          FASE_ORIGINAL           FASE_PADRONIZADA           TURMA
        0            8A  Fase 8 (Universitários)               A
        1          ALFA    Fase 0 (1° e 2° ano)  NÃO SE APLICA
        2             9            NÃO SE APLICA  NÃO SE APLICA
    """
    df_out = df.copy()

    # Escolhe função correta baseada no ano
    if ano >= 2024:
        df_out["FASE_PADRONIZADA"] = df_out[col_fase].apply(obter_nova_fase_24)
        if criar_turma:
            df_out["TURMA"] = df_out[col_fase].apply(obter_nova_turma_24)
    else:
        df_out["FASE_PADRONIZADA"] = df_out[col_fase].apply(obter_nova_fase)
        if criar_turma:
            df_out["TURMA"] = df_out[col_fase].apply(obter_nova_turma)

    return df_out
