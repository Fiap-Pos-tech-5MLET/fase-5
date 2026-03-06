"""
Módulo de limpeza e pré-processamento de dados educacionais.

Este módulo fornece funções para carregar, limpar e preparar
os dados da Associação Passos Mágicos para análise e modelagem.
"""

import logging
import os
from typing import Any

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


def map_instituicao_ensino(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia a coluna 'Instituição de ensino' para variável binária.

    Converte instituição de ensino categórica em binário:
    - 'Escola Pública' → 0 (Pública)
    - 'Rede Decisão', 'Escola JP II' → 1 (Privada)

    Cria a coluna 'INSTITUICAO_ENSINO_MAPPED' com valores binários
    adequados para modelagem.

    Args:
        df (pd.DataFrame): DataFrame com dados educacionais.

    Returns:
        pd.DataFrame: DataFrame com coluna 'INSTITUICAO_ENSINO_MAPPED' adicionada.

    Raises:
        TypeError: Se o input não for um DataFrame.
        ValueError: Se a coluna de instituição não for encontrada após normalização.

    Examples:
        >>> df = pd.DataFrame({'Instituição de ensino': ['Escola Pública', 'Rede Decisão']})
        >>> df_mapped = map_instituicao_ensino(df)
        >>> print(df_mapped['INSTITUICAO_ENSINO_MAPPED'].tolist())
        [0, 1]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    df = df.copy()

    # Tentar encontrar coluna de instituição com diferentes variações de nome
    possible_names = [
        "INSTITUICAO_DE_ENSINO",
        "INSTITUIÇÃO_DE_ENSINO",
        "INSTITUICAO_ENSINO",
        "INSTITUIÇÃO_ENSINO",
    ]

    instituicao_col = None
    for col_name in possible_names:
        if col_name in df.columns:
            instituicao_col = col_name
            break

    if instituicao_col is None:
        logger.warning(
            "Coluna de instituição de ensino não encontrada. "
            "Usando valor padrão 0 (pública) para INSTITUICAO_ENSINO_MAPPED"
        )
        df["INSTITUICAO_ENSINO_MAPPED"] = 0
        return df

    # Mapear valores categóricos para binário
    # Pública = 0, Privada = 1
    mapping = {
        "Escola Pública": 0,
        "ESCOLA PÚBLICA": 0,
        "escola pública": 0,
        "Rede Decisão": 1,
        "REDE DECISÃO": 1,
        "rede decisão": 1,
        "Escola JP II": 1,
        "ESCOLA JP II": 1,
        "escola jp ii": 1,
    }

    # Aplicar mapeamento
    df["INSTITUICAO_ENSINO_MAPPED"] = df[instituicao_col].map(mapping)

    # Verificar valores não mapeados
    unmapped_count = df["INSTITUICAO_ENSINO_MAPPED"].isna().sum()
    if unmapped_count > 0:
        logger.warning(
            "%d valores de instituição não mapeados. Usando valor padrão 0 (pública)",
            unmapped_count,
        )
        df["INSTITUICAO_ENSINO_MAPPED"] = df["INSTITUICAO_ENSINO_MAPPED"].fillna(0)

    # Garantir tipo inteiro
    df["INSTITUICAO_ENSINO_MAPPED"] = df["INSTITUICAO_ENSINO_MAPPED"].astype(int)

    # Estatísticas
    value_counts = df["INSTITUICAO_ENSINO_MAPPED"].value_counts()
    logger.info(
        "Instituição mapeada: %d públicas (0), %d privadas (1)",
        value_counts.get(0, 0),
        value_counts.get(1, 0),
    )

    # Remover coluna categórica original se ainda existir
    if instituicao_col in df.columns:
        df = df.drop(columns=[instituicao_col])
        logger.debug("Coluna categórica original '%s' removida", instituicao_col)

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


def validate_data_quality(
    df: pd.DataFrame,
    inde_column: str = "INDE_2024",
    defasagem_column: str = "DEFASAGEM",
    defasagem_max_null_ratio: float = 0.25,
) -> None:
    """Executa validações de qualidade de dados com Great Expectations.

    Regras aplicadas antes do treino:
    - `INDE_2024` não pode ter valores negativos.
    - Taxa de nulos em `DEFASAGEM` não pode exceder o limite configurado.

    Args:
        df (pd.DataFrame): Dataset de entrada para validação.
        inde_column (str): Nome da coluna INDE a validar.
        defasagem_column (str): Nome da coluna de defasagem.
        defasagem_max_null_ratio (float): Limite máximo aceitável de nulos em DEFASAGEM.

    Raises:
        ValueError: Quando uma regra de qualidade falha.
        ImportError: Quando Great Expectations não está disponível.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    if df.empty:
        raise ValueError("DataFrame não pode estar vazio para validação de qualidade")

    try:
        import great_expectations as ge
    except ImportError as exc:
        raise ImportError(
            "Great Expectations não está instalado. "
            "Instale a dependência para validar qualidade de dados."
        ) from exc

    ge_df: Any = ge.from_pandas(df)

    if inde_column in df.columns:
        inde_expectation = ge_df.expect_column_values_to_be_between(
            inde_column,
            min_value=0,
            max_value=None,
            allow_cross_type_comparisons=True,
            mostly=1.0,
        )
        if not inde_expectation["success"]:
            raise ValueError(
                f"Data quality check falhou: coluna {inde_column} contém valores negativos."
            )

    if defasagem_column in df.columns:
        defasagem_expectation = ge_df.expect_column_values_to_not_be_null(
            defasagem_column,
            mostly=1.0 - defasagem_max_null_ratio,
        )
        if not defasagem_expectation["success"]:
            null_ratio = float(df[defasagem_column].isna().mean())
            raise ValueError(
                "Data quality check falhou: taxa de nulos em "
                f"{defasagem_column} ({null_ratio:.2%}) acima do limite "
                f"de {defasagem_max_null_ratio:.2%}."
            )

    logger.info("Data quality checks aprovados com Great Expectations")
