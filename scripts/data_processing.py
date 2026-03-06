"""
Funções de processamento e limpeza de dados para o projeto Passos Mágicos.

Este módulo contém funções utilitárias para ETL (Extract, Transform, Load) específicas
do pipeline de análise educacional da Associação Passos Mágicos, consolidando dados
de múltiplos anos (2022-2024) em estrutura uniforme para modelagem.

Funções principais:
- Padronização de colunas por ano
- Análise diagnóstica de qualidade (nulos, tipos)
- Cálculo de idade a partir de datas
- Operações de conjuntos entre DataFrames
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


def padronizar_colunas_ano(
    df: pd.DataFrame, ano: int, ignorar_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas para snake_case, adiciona o sufixo do ano
    e converte para maiúsculas no final do processo.

    Args:
        df (pd.DataFrame): O DataFrame original.
        ano (int): O ano de referência (ex: 2022, 2023, 2024).
        ignorar_cols (Optional[List[str]]): Lista de colunas a ignorar na padronização.
            Essas colunas serão apenas convertidas para maiúsculas.

    Returns:
        pd.DataFrame: Novo DataFrame com colunas padronizadas.

    Examples:
        >>> df = pd.DataFrame({"Nome Aluno": ["Ana"], "Idade": [12]})
        >>> padronizar_colunas_ano(df, 2022)
           NOME_ALUNO_22  IDADE_22
        0            Ana        12
    """
    df_out = df.copy()

    if ignorar_cols is None:
        ignorar_cols = []

    ano_str = str(ano)
    ano_curto = ano_str[-2:]

    novas_colunas = []

    for col in df_out.columns:
        if col in ignorar_cols:
            novas_colunas.append(col.upper())
            continue

        # 1. Remove espaços e substitui por underline
        col_nova = col.strip().replace(" ", "_")

        # 2. Adiciona sufixo se necessário
        if not (col_nova.endswith(ano_str) or col_nova.endswith(ano_curto)):
            col_nova = f"{col_nova}_{ano_curto}"

        # 3. Converte para maiúsculas
        col_nova = col_nova.upper()

        novas_colunas.append(col_nova)

    df_out.columns = novas_colunas
    return df_out


def adicionar_colunas_vazias(df: pd.DataFrame, novas_colunas: List[str]) -> pd.DataFrame:
    """
    Adiciona colunas preenchidas com NaN ao DataFrame caso elas não existam.

    Args:
        df (pd.DataFrame): O DataFrame original.
        novas_colunas (List[str]): Lista de strings com os nomes das colunas a adicionar.

    Returns:
        pd.DataFrame: Novo DataFrame com as colunas adicionadas.

    Examples:
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> adicionar_colunas_vazias(df, ["B", "C"])
           A   B   C
        0  1 NaN NaN
        1  2 NaN NaN
    """
    df_out = df.copy()

    for col in novas_colunas:
        if col not in df_out.columns:
            df_out[col] = np.nan

    return df_out


def analise_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um relatório contendo a quantidade e a porcentagem de valores nulos por coluna.

    Args:
        df (pd.DataFrame): O DataFrame a ser analisado.

    Returns:
        pd.DataFrame: DataFrame com colunas:
            - 'coluna': nome da coluna
            - 'qtd_nulos': quantidade absoluta de nulos
            - 'perc_nulos': porcentagem de nulos

    Examples:
        >>> df = pd.DataFrame({"A": [1, None, 3], "B": [1, 2, 3]})
        >>> analise_nulos(df)
          coluna  qtd_nulos  perc_nulos
        0      A          1       33.33
        1      B          0        0.00
    """
    nulos = df.isnull().sum()
    perc_nulos = (nulos / len(df)) * 100

    resultado = pd.DataFrame(
        {"coluna": nulos.index, "qtd_nulos": nulos.values, "perc_nulos": perc_nulos.values}
    )

    resultado = resultado[resultado["qtd_nulos"] > 0].sort_values(
        by="perc_nulos", ascending=False
    )

    return resultado.reset_index(drop=True)


def calcular_idade_2023(data_nascimento: str) -> Optional[int]:
    """
    Calcula a idade em 2023 a partir de uma string de data no formato MM/DD/YYYY.

    Args:
        data_nascimento (str): Data de nascimento no formato 'MM/DD/YYYY'.

    Returns:
        Optional[int]: Idade em anos completos ou None se a data for inválida.

    Examples:
        >>> calcular_idade_2023("05/15/2010")
        13
        >>> calcular_idade_2023("invalid")
        None
    """
    try:
        data_nasc = datetime.strptime(str(data_nascimento), "%m/%d/%Y")
        ano_referencia = datetime(2023, 1, 1)
        idade = ano_referencia.year - data_nasc.year
        if (ano_referencia.month, ano_referencia.day) < (data_nasc.month, data_nasc.day):
            idade -= 1
        return idade
    except (ValueError, TypeError):
        return None


def obter_elementos_comuns(df1: pd.DataFrame, df2: pd.DataFrame, coluna: str) -> List:
    """
    Obtém a interseção de valores únicos de uma coluna entre dois DataFrames.

    Args:
        df1 (pd.DataFrame): Primeiro DataFrame.
        df2 (pd.DataFrame): Segundo DataFrame.
        coluna (str): Nome da coluna para comparar.

    Returns:
        List: Lista de valores que aparecem em ambos os DataFrames.

    Examples:
        >>> df1 = pd.DataFrame({"ID": [1, 2, 3]})
        >>> df2 = pd.DataFrame({"ID": [2, 3, 4]})
        >>> obter_elementos_comuns(df1, df2, "ID")
        [2, 3]
    """
    set1 = set(df1[coluna].unique())
    set2 = set(df2[coluna].unique())
    return sorted(set1.intersection(set2))


def renomear_colunas_ano(df: pd.DataFrame, mapeamento: dict) -> pd.DataFrame:
    """
    Renomeia colunas específicas de um DataFrame usando um dicionário de mapeamento.

    Args:
        df (pd.DataFrame): DataFrame original.
        mapeamento (dict): Dicionário {nome_antigo: nome_novo}.

    Returns:
        pd.DataFrame: Novo DataFrame com colunas renomeadas.

    Examples:
        >>> df = pd.DataFrame({"col_antiga": [1, 2]})
        >>> renomear_colunas_ano(df, {"col_antiga": "COL_NOVA"})
           COL_NOVA
        0         1
        1         2
    """
    df_out = df.copy()
    return df_out.rename(columns=mapeamento)


def filtrar_colunas_relevantes(df: pd.DataFrame, colunas_manter: List[str]) -> pd.DataFrame:
    """
    Filtra DataFrame mantendo apenas as colunas especificadas.

    Args:
        df (pd.DataFrame): DataFrame original.
        colunas_manter (List[str]): Lista de nomes de colunas a manter.

    Returns:
        pd.DataFrame: DataFrame filtrado contendo apenas as colunas especificadas.

    Examples:
        >>> df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        >>> filtrar_colunas_relevantes(df, ["A", "C"])
           A  C
        0  1  3
    """
    colunas_existentes = [col for col in colunas_manter if col in df.columns]
    return df[colunas_existentes].copy()


def consolidar_dataframes(
    dfs: List[pd.DataFrame], id_col: str = "NOME", sufixos: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Consolida múltiplos DataFrames em um único DataFrame usando merge progressivo.

    Args:
        dfs (List[pd.DataFrame]): Lista de DataFrames a consolidar.
        id_col (str): Nome da coluna identificadora para merge (padrão: "NOME").
        sufixos (Optional[List[str]]): Lista de sufixos para diferenciar colunas
            duplicadas. Se None, usa padrão ["_x", "_y", "_z", ...].

    Returns:
        pd.DataFrame: DataFrame consolidado com todas as informações.

    Examples:
        >>> df1 = pd.DataFrame({"NOME": ["Ana"], "NOTA_22": [8.5]})
        >>> df2 = pd.DataFrame({"NOME": ["Ana"], "NOTA_23": [9.0]})
        >>> consolidar_dataframes([df1, df2], "NOME")
          NOME  NOTA_22  NOTA_23
        0  Ana      8.5      9.0
    """
    if not dfs:
        return pd.DataFrame()

    if sufixos is None:
        sufixos = [f"_{chr(120 + i)}" for i in range(len(dfs))]  # _x, _y, _z, ...

    df_consolidado = dfs[0].copy()

    for i, df in enumerate(dfs[1:], start=1):
        sufixo_esq = sufixos[i - 1] if i - 1 < len(sufixos) else f"_{i-1}"
        sufixo_dir = sufixos[i] if i < len(sufixos) else f"_{i}"

        df_consolidado = df_consolidado.merge(
            df, on=id_col, how="outer", suffixes=(sufixo_esq, sufixo_dir)
        )

    return df_consolidado
