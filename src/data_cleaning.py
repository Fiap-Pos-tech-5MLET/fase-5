"""
Módulo de limpeza e pré-processamento de dados educacionais.

Este módulo fornece funções para carregar, limpar e preparar
os dados da Associação Passos Mágicos para análise e modelagem.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(file_path: str, sheet_name: str = "PEDE2024") -> pd.DataFrame:
    """
    Carrega o dataset do arquivo Excel especificado.

    Lê os dados educacionais da Associação Passos Mágicos a partir
    de um arquivo Excel, validando a existência do arquivo e do sheet.

    Args:
        file_path (str): Caminho completo para o arquivo Excel.
        sheet_name (str): Nome da planilha a ser carregada. Padrão: 'PEDE2024'.

    Returns:
        pd.DataFrame: DataFrame contendo os dados carregados.

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho especificado.
        ValueError: Se o sheet especificado não existir no arquivo.

    Examples:
        >>> df = load_data('data/raw/PEDE2024.xlsx')
        >>> print(df.shape)
        (500, 50)
    """
    if not os.path.exists(file_path):
        error_msg = f"Arquivo não encontrado: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not file_path.endswith((".xlsx", ".xls")):
        error_msg = f"Formato de arquivo inválido: {file_path}. Esperado .xlsx ou .xls"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(
            "Dataset carregado com sucesso: %d linhas, %d colunas",
            df.shape[0],
            df.shape[1],
        )
        return df
    except ValueError as e:
        error_msg = f"Sheet '{sheet_name}' não encontrado no arquivo {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Erro ao carregar arquivo: {e!s}"
        logger.error(error_msg)
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza básica dos dados.

    Aplica transformações padronizadas incluindo:
    - Padronização de nomes de colunas (uppercase, sem espaços)
    - Remoção de colunas identificadoras que não agregam valor preditivo
    - Conversão de colunas numéricas que podem conter strings

    Args:
        df (pd.DataFrame): DataFrame bruto a ser limpo.

    Returns:
        pd.DataFrame: DataFrame limpo com transformações aplicadas.

    Raises:
        ValueError: Se o DataFrame estiver vazio.
        TypeError: Se o input não for um DataFrame.

    Examples:
        >>> df_raw = pd.DataFrame({'Nome Anonimizado': ['A', 'B'], 'inde 2024': [10, 20]})
        >>> df_clean = clean_data(df_raw)
        >>> print(df_clean.columns.tolist())
        ['INDE_2024']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    if df.empty:
        logger.warning("DataFrame vazio recebido em clean_data")
        raise ValueError("DataFrame não pode estar vazio")

    df = df.copy()
    original_shape = df.shape

    # Padronizar nomes de colunas
    df.columns = [c.strip().replace(" ", "_").upper() for c in df.columns]
    logger.debug("Nomes de colunas padronizados: %d colunas", len(df.columns))

    # Remover colunas identificadoras
    cols_to_drop = ["NOME_ANONIMIZADO"]
    cols_dropped = [c for c in cols_to_drop if c in df.columns]
    if cols_dropped:
        df = df.drop(columns=cols_dropped, errors="ignore")
        logger.debug("Colunas removidas: %s", cols_dropped)

    # Converter colunas numéricas
    numeric_candidates = [
        c
        for c in df.columns
        if any(
            keyword in c for keyword in ["INDE", "PEDRA", "NOTA", "IAA", "IEG", "IPS", "IPP", "IDA"]
        )
    ]

    for col in numeric_candidates:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if original_dtype != df[col].dtype:
                logger.debug(
                    "Coluna %s convertida de %s para %s",
                    col,
                    original_dtype,
                    df[col].dtype,
                )

    logger.info("Limpeza concluída: %s -> %s", original_shape, df.shape)
    return df


def create_target(df: pd.DataFrame, target_column: str = "DEFASAGEM") -> pd.DataFrame:
    """
    Cria a variável target binária para predição de risco.

    Gera a coluna 'TARGET' baseada na defasagem escolar:
    - TARGET = 1: Aluno em risco (DEFASAGEM < 0)
    - TARGET = 0: Aluno sem risco (DEFASAGEM >= 0)

    Args:
        df (pd.DataFrame): DataFrame com os dados educacionais.
        target_column (str): Nome da coluna de defasagem. Padrão: 'DEFASAGEM'.

    Returns:
        pd.DataFrame: DataFrame com a coluna 'TARGET' adicionada.

    Raises:
        ValueError: Se a coluna de defasagem não existir no DataFrame.
        TypeError: Se o input não for um DataFrame.

    Examples:
        >>> df = pd.DataFrame({'DEFASAGEM': [-1, 0, 1, -2]})
        >>> df_target = create_target(df)
        >>> print(df_target['TARGET'].tolist())
        [1, 0, 0, 1]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    if target_column not in df.columns:
        error_msg = (
            f"Coluna '{target_column}' não encontrada no dataset. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    df = df.copy()

    # Remover linhas sem valor de defasagem
    rows_before = len(df)
    df = df.dropna(subset=[target_column])
    rows_after = len(df)

    if rows_before > rows_after:
        logger.warning("%d linhas removidas por NaN em %s", rows_before - rows_after, target_column)

    if df.empty:
        raise ValueError(f"Nenhuma linha válida após remover NaN de {target_column}")

    # Criar target binário
    df["TARGET"] = (df[target_column] < 0).astype(int)

    # Estatísticas da distribuição
    target_dist = df["TARGET"].value_counts()
    logger.info(
        "Target criado: %d sem risco (0), %d em risco (1)",
        target_dist[0],
        target_dist[1],
    )
    logger.info("Proporção de risco: %.1f%%", target_dist[1] / len(df) * 100)

    return df


def handle_missing_values(df: pd.DataFrame, numeric_strategy: str = "zero") -> pd.DataFrame:
    """
    Trata valores ausentes em colunas numéricas e categóricas.

    Aplica estratégias diferentes para cada tipo de dado:
    - Numéricos: Preenche com 0 (padrão) ou mediana
    - Categóricos: Preenche com 'UNKNOWN' (padrão)

    Args:
        df (pd.DataFrame): DataFrame com valores ausentes.
        numeric_strategy (str): Estratégia para numéricos ('zero' ou 'median'). Padrão: 'zero'.

    Returns:
        pd.DataFrame: DataFrame com valores ausentes tratados.

    Raises:
        ValueError: Se a estratégia for inválida ou DataFrame vazio.
        TypeError: Se o input não for um DataFrame.

    Examples:
        >>> df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': ['x', None, 'z']})
        >>> df_filled = handle_missing_values(df)
        >>> print(df_filled['A'].tolist())
        [1.0, 0.0, 3.0]
        >>> print(df_filled['B'].tolist())
        ['x', 'UNKNOWN', 'z']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    if df.empty:
        logger.warning("DataFrame vazio recebido em handle_missing_values")
        raise ValueError("DataFrame não pode estar vazio")

    if numeric_strategy not in ["zero", "median"]:
        raise ValueError(
            f"Estratégia numérica inválida: {numeric_strategy}. Use 'zero' ou 'median'"
        )

    df = df.copy()

    # Identificar tipos de colunas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Contar valores ausentes antes
    missing_before = df.isnull().sum().sum()

    # Tratar numéricos
    if numeric_cols:
        if numeric_strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0)
            logger.debug("%d colunas numéricas preenchidas com 0", len(numeric_cols))
        elif numeric_strategy == "median":
            for col in numeric_cols:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            logger.debug("%d colunas numéricas preenchidas com mediana", len(numeric_cols))

    # Tratar categóricos
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna("UNKNOWN")
        # Garantir tipo string
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        logger.debug("%d colunas categóricas preenchidas com 'UNKNOWN'", len(categorical_cols))

    # Contar valores ausentes depois
    missing_after = df.isnull().sum().sum()

    logger.info("Valores ausentes tratados: %d -> %d", missing_before, missing_after)

    if missing_after > 0:
        logger.warning("Ainda existem %d valores ausentes após tratamento", missing_after)

    return df
