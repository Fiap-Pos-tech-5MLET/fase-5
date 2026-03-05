"""
Funções de limpeza e preparação de dados para análise temporal do Datathon Passos Mágicos.

Module: scripts.datathon_cleaning
Author: GitHub Copilot
Date: 2025-03-05

Functions:
    - filter_columns(): Remove colunas contendo padrões específicos
    - cleaning_dataset(): Remove registros com valores nulos
    - create_annual_datasets(): Criar datasets separados por ano
    - analyze_student_continuity(): Analisar continuidade de alunos entre anos
"""

import pandas as pd
from typing import List, Dict, Set, Tuple


def filter_columns(df: pd.DataFrame, filters: List[str]) -> pd.DataFrame:
    """
    Remove colunas que contêm qualquer padrão da lista de filtros.
    
    Exemplo:
        filter_columns(df, ['2021', '2022'])  -> Remove colunas com _2021 ou _2022
    
    Args:
        df: DataFrame de entrada
        filters: Lista de strings para filtrar (ex: ['2021', '2022'])
    
    Returns:
        DataFrame com colunas filtradas
    
    Raises:
        TypeError: Se df não for DataFrame ou filters não for lista
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame")
    if not isinstance(filters, list):
        raise TypeError("filters deve ser uma lista")
    
    selected = [True] * len(df.columns)
    for idx, col in enumerate(df.columns):
        if any(f in col for f in filters):
            selected[idx] = False
    
    return df[df.columns[selected]].copy()


def cleaning_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas onde todas as colunas (exceto NOME) são NaN.
    
    Justificativa:
        Dataset temporal tem registros incompletos em alguns anos.
        Precisamos remover alunos que não têm dados em nenhuma coluna exceto nome.
    
    Args:
        df: DataFrame de entrada
    
    Returns:
        DataFrame com registros vazios removidos
    
    Raises:
        TypeError: Se df não for DataFrame
        ValueError: Se coluna NOME não existir mas há colunas
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame")
    
    # Remover linhas onde todas as colunas (exceto NOME) são NaN
    cols_to_check = df.columns.difference(['NOME'])
    if len(cols_to_check) == 0:
        return df.copy()
    
    # Step 1: Remover onde todas as colunas exceto NOME são NaN
    _df = df.dropna(subset=cols_to_check, how='all')
    
    # Step 2: Remover linhas completamente vazias
    _df = _df[~_df.isna().all(axis=1)]
    
    return _df.copy()


def create_annual_datasets(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Criar datasets separados por ano (2020, 2021, 2022).
    
    Lógica:
        1. Para cada ano, filtram colunas dos outros anos
        2. Remove registros vazios
        3. Padroniza nomes de colunas removendo sufixos de ano
    
    Args:
        df: DataFrame com dados de múltiplos anos (sufixos _YYYY)
    
    Returns:
        Dicionário {ano: DataFrame}
            - Chaves: 2020, 2021, 2022
            - Valores: DataFrames limpos para cada ano
    
    Raises:
        TypeError: Se df não for DataFrame
    
    Example:
        >>> datasets = create_annual_datasets(df_raw)
        >>> df_2020 = datasets[2020]
        >>> len(df_2020)
        150
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame")
    
    datasets = {}
    
    # Dataset 2020
    df_2020 = filter_columns(df, ['2021', '2022'])
    df_2020 = cleaning_dataset(df_2020)
    datasets[2020] = df_2020
    
    # Dataset 2021
    df_2021 = filter_columns(df, ['2020', '2022'])
    df_2021 = cleaning_dataset(df_2021)
    df_2021.columns = df_2021.columns.str.replace('_2021', '', regex=False)
    datasets[2021] = df_2021
    
    # Dataset 2022
    df_2022 = filter_columns(df, ['2020', '2021'])
    df_2022 = cleaning_dataset(df_2022)
    df_2022.columns = df_2022.columns.str.replace('_2022', '', regex=False)
    datasets[2022] = df_2022
    
    return datasets


def analyze_student_continuity(
    df_2020: pd.DataFrame,
    df_2021: pd.DataFrame,
    df_2022: pd.DataFrame,
    nome_col: str = 'NOME'
) -> Dict[str, any]:
    """
    Analisar continuidade de alunos entre anos.
    
    Calcula:
        - Alunos que continuaram de 2020→2021
        - Alunos que continuaram de 2021→2022
        - Alunos novos ingressados em 2022
        - Taxa de evasão/continuidade
    
    Args:
        df_2020: Dataset de 2020
        df_2021: Dataset de 2021
        df_2022: Dataset de 2022
        nome_col: Nome da coluna de identificação (default: 'NOME')
    
    Returns:
        Dict com análises:
            {
                'alunos_2020': int,
                'alunos_2021': int,
                'alunos_2022': int,
                'continuidade_2020_2021': int,
                'taxa_2020_2021': float,
                'continuidade_2021_2022': int,
                'taxa_2021_2022': float,
                'novos_2022': int,
                'alunos_2020_set': Set[str],
                'alunos_2021_set': Set[str],
                'alunos_2022_set': Set[str],
            }
    
    Raises:
        TypeError: Se dfs não forem DataFrames
        KeyError: Se coluna nome_col não existir
    
    Example:
        >>> resultado = analyze_student_continuity(df_2020, df_2021, df_2022)
        >>> print(f"Taxa 2020→2021: {resultado['taxa_2020_2021']:.1f}%")
    """
    # Validação de entrada
    for df, year in [(df_2020, 2020), (df_2021, 2021), (df_2022, 2022)]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df_{year} deve ser um pandas.DataFrame")
        if nome_col not in df.columns:
            raise KeyError(f"Coluna '{nome_col}' não encontrada em df_{year}")
    
    # Extrair conjuntos de alunos únicos
    alunos_2020 = set(df_2020[nome_col].dropna().str.strip().unique())
    alunos_2021 = set(df_2021[nome_col].dropna().str.strip().unique())
    alunos_2022 = set(df_2022[nome_col].dropna().str.strip().unique())
    
    # Calcular continuidade
    continuidade_2020_2021 = len(alunos_2020 & alunos_2021)
    taxa_2020_2021 = continuidade_2020_2021 / len(alunos_2020) * 100 if len(alunos_2020) > 0 else 0
    
    continuidade_2021_2022 = len(alunos_2021 & alunos_2022)
    taxa_2021_2022 = continuidade_2021_2022 / len(alunos_2021) * 100 if len(alunos_2021) > 0 else 0
    
    novos_2022 = len(alunos_2022 - alunos_2021)
    
    return {
        'alunos_2020': len(alunos_2020),
        'alunos_2021': len(alunos_2021),
        'alunos_2022': len(alunos_2022),
        'continuidade_2020_2021': continuidade_2020_2021,
        'taxa_2020_2021': taxa_2020_2021,
        'continuidade_2021_2022': continuidade_2021_2022,
        'taxa_2021_2022': taxa_2021_2022,
        'novos_2022': novos_2022,
        'alunos_2020_set': alunos_2020,
        'alunos_2021_set': alunos_2021,
        'alunos_2022_set': alunos_2022,
    }
