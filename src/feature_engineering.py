"""
Módulo de engenharia de features para análise educacional.

Este módulo fornece funções para criar features derivadas e selecionar
as variáveis mais relevantes para predição de risco de defasagem escolar.
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


MODEL_FEATURE_COLUMNS: list[str] = [
    "nivel_de_defasagem",
    "idade",
    "genero",
    "ano_de_ingresso",
    "veterano",
    "em_fase",
    "qtde_aval_realizadas",
    "instituicao_ensino_mapped",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "ian",
]

_COLUMN_ALIASES: dict[str, str] = {
    "NIVEL_DE_DEFASAGEM": "nivel_de_defasagem",
    "DEFASAGEM": "nivel_de_defasagem",
    "IDADE": "idade",
    "GENERO": "genero",
    "GÊNERO": "genero",
    "ANO_DE_INGRESSO": "ano_de_ingresso",
    "ANO_INGRESSO": "ano_de_ingresso",
    "VETERANO": "veterano",
    "EM_FASE": "em_fase",
    "QTDE_AVAL_REALIZADAS": "qtde_aval_realizadas",
    "Nº_AV": "qtde_aval_realizadas",
    "N_AV": "qtde_aval_realizadas",
    "INSTITUICAO_ENSINO_MAPPED": "instituicao_ensino_mapped",
    "INSTITUICAO_ENSINO": "instituicao_ensino_mapped",
    "INSTITUICAO_DE_ENSINO": "instituicao_ensino_mapped",
    "IAA": "iaa",
    "IEG": "ieg",
    "IPS": "ips",
    "IDA": "ida",
    "IPV": "ipv",
    "IAN": "ian",
}


def _normalize_new_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes/tipos para o contrato de features do modelo atual."""
    normalized = df.copy()

    current_cols_upper = {str(col).strip().upper() for col in normalized.columns}
    canonical_present = any(col in normalized.columns for col in MODEL_FEATURE_COLUMNS)
    new_contract_signals = {
        "NIVEL_DE_DEFASAGEM",
        "ANO_DE_INGRESSO",
        "ANO_INGRESSO",
        "QTDE_AVAL_REALIZADAS",
        "INSTITUICAO_ENSINO_MAPPED",
        "INSTITUICAO_ENSINO",
        "INSTITUICAO_DE_ENSINO",
        "IAA",
        "IEG",
        "IPS",
        "IDA",
        "IPV",
        "IAN",
        "EM_FASE",
        "VETERANO",
    }

    should_map_aliases = canonical_present or bool(
        current_cols_upper.intersection(new_contract_signals)
    )
    if not should_map_aliases:
        return normalized

    for col in normalized.columns:
        upper_col = str(col).strip().upper()
        if upper_col in _COLUMN_ALIASES:
            canonical_col = _COLUMN_ALIASES[upper_col]
            if canonical_col not in normalized.columns:
                normalized[canonical_col] = normalized[col]

    if "veterano" not in normalized.columns and "ano_de_ingresso" in normalized.columns:
        normalized["veterano"] = (
            pd.to_numeric(normalized["ano_de_ingresso"], errors="coerce") < 2024
        ).astype(int)

    if "em_fase" not in normalized.columns and "nivel_de_defasagem" in normalized.columns:
        normalized["em_fase"] = (
            pd.to_numeric(normalized["nivel_de_defasagem"], errors="coerce") == 0
        ).astype(int)

    if "genero" in normalized.columns:
        genero_map = {
            "M": 1,
            "MASCULINO": 1,
            "HOMEM": 1,
            "F": 0,
            "FEMININO": 0,
            "MULHER": 0,
        }
        normalized["genero"] = (
            normalized["genero"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(genero_map)
            .fillna(pd.to_numeric(normalized["genero"], errors="coerce"))
            .fillna(0)
        )

    for col in MODEL_FEATURE_COLUMNS:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0)

    return normalized


def build_feature_matrix_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói matriz no mesmo schema de features esperado pelo modelo atual."""
    normalized = _normalize_new_feature_columns(df)
    feature_matrix = pd.DataFrame(index=normalized.index)
    for col in MODEL_FEATURE_COLUMNS:
        if col in normalized.columns:
            feature_matrix[col] = normalized[col]
        else:
            feature_matrix[col] = 0.0
    return feature_matrix


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features derivadas para o modelo preditivo.

    Gera features baseadas em:
    - Evolução temporal do INDE (crescimento)
    - Disponibilidade de histórico educacional
    - Indicadores de desempenho relativo

    Args:
        df (pd.DataFrame): DataFrame com dados educacionais brutos.

    Returns:
        pd.DataFrame: DataFrame com features derivadas adicionadas.

    Raises:
        TypeError: Se o input não for um DataFrame.
        ValueError: Se o DataFrame estiver vazio.

    Examples:
        >>> df = pd.DataFrame({
        ...     'INDE_2024': [10.0, 8.0, 12.0],
        ...     'INDE_23': [8.0, 0.0, 10.0]
        ... })
        >>> df_features = create_features(df)
        >>> print(df_features['INDE_GROWTH'].tolist())
        [2.0, 8.0, 2.0]
        >>> print(df_features['HAS_HISTORY_23'].tolist())
        [1, 0, 1]

    Notes:
        - INDE_GROWTH: Diferença entre INDE_2024 e INDE_23
        - HAS_HISTORY_23: Flag binária indicando histórico disponível
        - Features categóricas são mantidas para codificação posterior
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    if df.empty:
        logger.warning("DataFrame vazio recebido em create_features")
        raise ValueError("DataFrame não pode estar vazio")

    df = df.copy()
    features_created = []

    # Normalização para o schema atual (modelo lgbm)
    df = _normalize_new_feature_columns(df)

    # Feature 1: Crescimento do INDE (evolução temporal)
    if "INDE_2024" in df.columns and "INDE_23" in df.columns:
        # Flag para indicar se aluno tem histórico em 2023
        df["HAS_HISTORY_23"] = (df["INDE_23"] > 0).astype(int)
        features_created.append("HAS_HISTORY_23")

        # Crescimento absoluto do INDE
        df["INDE_GROWTH"] = df["INDE_2024"] - df["INDE_23"]
        features_created.append("INDE_GROWTH")

        logger.debug("Features de evolução criadas: %s", features_created)
        logger.debug(
            "Alunos com histórico: %d/%d",
            df["HAS_HISTORY_23"].sum(),
            len(df),
        )
    else:
        logger.warning(
            "Colunas INDE_2024 ou INDE_23 não encontradas. Features de evolução não criadas."
        )

    # Feature 2: Features categóricas mantidas para codificação posterior
    categorical_features = ["FASE", "TURMA", "GENERO", "INSTITUICAO_DE_ENSINO"]
    existing_categorical = [col for col in categorical_features if col in df.columns]

    if existing_categorical:
        logger.debug("Features categóricas mantidas: %s", existing_categorical)

    if features_created:
        logger.info("Total de %d features derivadas criadas", len(features_created))
    else:
        logger.info("Nenhuma feature derivada foi criada")

    return df


def select_features(
    df: pd.DataFrame, target_col: str = "TARGET", remove_leakage: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona features relevantes e separa features do target (y).

    Remove colunas que podem causar vazamento de informação (data leakage)
    e separa o conjunto de features do target para treinamento.

    Args:
        df (pd.DataFrame): DataFrame completo com features e target.
        target_col (str): Nome da coluna target. Padrão: 'TARGET'.
        remove_leakage (bool): Se True, remove colunas com potencial data leakage. Padrão: True.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tupla contendo:
            - feature_matrix: DataFrame com features selecionadas
            - target_series: Series com valores target

    Raises:
        ValueError: Se a coluna target não existir ou DataFrame vazio.
        TypeError: Se o input não for um DataFrame.

    Examples:
        >>> df = pd.DataFrame({
        ...     'TARGET': [0, 1, 0],
        ...     'INDE_23': [8.0, 6.0, 9.0],
        ...     'INDE_2024': [10.0, 7.0, 11.0],
        ...     'DEFASAGEM': [0, -1, 1]
        ... })
        >>> feature_matrix, target_series = select_features(df)
        >>> print(feature_matrix.columns.tolist())
        ['INDE_23']
        >>> print(target_series.tolist())
        [0, 1, 0]

    Notes:
        - Remove automaticamente DEFASAGEM, RA, identificadores
        - Remove features de 2024 que podem causar data leakage
        - Mantém apenas features disponíveis em momento de predição
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(df)}")

    if df.empty:
        logger.error("DataFrame vazio recebido em select_features")
        raise ValueError("DataFrame não pode estar vazio")

    if target_col not in df.columns:
        error_msg = (
            f"Coluna target '{target_col}' não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()[:10]}..."
        )
        logger.error(error_msg)
        raise ValueError("Target column not found. Run create_target first.")

    df = df.copy()
    df = _normalize_new_feature_columns(df)

    # Caminho principal do projeto atual: usar as 13 features canônicas
    canonical_hits = sum(1 for col in MODEL_FEATURE_COLUMNS if col in df.columns)
    if canonical_hits >= 4:
        target_series = df[target_col]
        feature_matrix = build_feature_matrix_for_model(df)
        logger.info(
            "Features canônicas selecionadas: %d colunas, %d samples",
            feature_matrix.shape[1],
            feature_matrix.shape[0],
        )
        return feature_matrix, target_series

    # Separar target
    target_series = df[target_col]
    logger.debug(
        "Target extraído: %d samples, distribuição: %s",
        len(target_series),
        target_series.value_counts().to_dict(),
    )

    # Definir colunas a serem removidas
    drop_cols = [target_col]

    # 1. Remover colunas de metadados e identificadores
    metadata_cols = ["RA", "NOME_ANONIMIZADO", "DATA_DE_NASC"]
    drop_cols.extend([c for c in metadata_cols if c in df.columns])

    # 2. Remover coluna usada para criar o target (data leakage óbvio)
    if "DEFASAGEM" in df.columns:
        drop_cols.append("DEFASAGEM")

    # 3. Remover features de 2024 (data leakage - não disponíveis em tempo de predição)
    if remove_leakage:
        leakage_cols = [
            "INDE_2024",
            "PEDRA_2024",
            "IAA",
            "IEG",
            "IPS",
            "IPP",
            "IDA",  # Componentes do INDE 2024
            "MAT",
            "POR",
            "ING",  # Notas específicas de 2024
            "IPV",
            "IAN",  # Outros indicadores de 2024
            "DESTAQUE_IEG",
            "DESTAQUE_IDA",
            "DESTAQUE_IPV",  # Indicadores de destaque 2024
            "ATINGIU_PV",
            "INDICADO",  # Resultados de 2024
            "REC_AV1",
            "REC_AV2",
            "REC_PSICOLOGIA",  # Recomendações baseadas em 2024
        ]

        # Remover INDE_GROWTH se presente (usa dados de 2024)
        if "INDE_GROWTH" in df.columns:
            leakage_cols.append("INDE_GROWTH")

        drop_cols.extend([c for c in leakage_cols if c in df.columns])
        logger.debug(
            "Colunas de data leakage removidas: %s",
            [c for c in leakage_cols if c in df.columns],
        )

    # Remover colunas duplicadas da lista
    drop_cols = list(set(drop_cols))

    # Criar DataFrame feature_matrix com features selecionadas
    feature_matrix = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Validar resultado
    if feature_matrix.empty or feature_matrix.shape[1] == 0:
        logger.error("Nenhuma feature restou após seleção")
        raise ValueError(
            "Nenhuma feature disponível após seleção. Verifique o DataFrame de entrada."
        )

    logger.info(
        "Features selecionadas: %d colunas, %d samples",
        feature_matrix.shape[1],
        feature_matrix.shape[0],
    )
    logger.info("Features finais: %s", feature_matrix.columns.tolist())

    # Verificar valores ausentes
    missing_count = feature_matrix.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(
            "%d valores ausentes encontrados em features_df após seleção",
            missing_count,
        )

    return feature_matrix, target_series


def get_feature_names(feature_matrix: pd.DataFrame) -> list[str]:
    """
    Retorna os nomes das features após seleção.

    Função auxiliar para obter lista de nomes de features,
    útil para logging e análise de importância.

    Args:
        feature_matrix (pd.DataFrame): DataFrame com features selecionadas.

    Returns:
        list[str]: Lista com nomes das colunas de features.

    Raises:
        TypeError: Se o input não for um DataFrame.

    Examples:
        >>> feature_matrix = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> names = get_feature_names(feature_matrix)
        >>> print(names)
        ['A', 'B']
    """
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError(f"Esperado pd.DataFrame, recebido {type(feature_matrix)}")

    return feature_matrix.columns.tolist()  # type: ignore[no-any-return]
