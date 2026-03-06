"""
Módulo de criação e treinamento de modelos de Machine Learning.

Este módulo fornece funções para criar pipelines de ML com pré-processamento,
treinar modelos LightGBM e avaliar performance.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def create_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    k: Union[int, str] = "all",
    random_state: int = 42,
) -> Pipeline:
    """
    Cria pipeline de ML com pré-processamento e modelo LightGBM.

    Constrói pipeline completo incluindo:
    - Normalização de features numéricas (StandardScaler)
    - Codificação one-hot de features categóricas
    - Seleção de features (SelectKBest)
    - Classificador LightGBM

    Args:
        numeric_features (List[str]): Lista de nomes de colunas numéricas.
        categorical_features (List[str]): Lista de nomes de colunas categóricas.
        n_estimators (int): Número de árvores no boosting. Padrão: 100.
        max_depth (Optional[int]): Profundidade máxima das árvores. None = ilimitado.
            Padrão: None.
        learning_rate (float): Taxa de aprendizado. Padrão: 0.1.
        num_leaves (int): Número máximo de folhas no LightGBM. Padrão: 31.
        subsample (float): Fração de linhas por árvore. Padrão: 1.0.
        colsample_bytree (float): Fração de colunas por árvore. Padrão: 1.0.
        k (Union[int, str]): Número de features a selecionar com SelectKBest. 'all' mantém todas.
            Padrão: 'all'.
        random_state (int): Seed para reprodutibilidade. Padrão: 42.

    Returns:
        Pipeline: Pipeline sklearn completo pronto para fit.

    Raises:
        ValueError: Se listas de features forem vazias ou parâmetros inválidos.
        TypeError: Se tipos de parâmetros estiverem incorretos.

    Examples:
        >>> numeric_features = ['idade', 'nota']
        >>> categorical_features = ['fase', 'genero']
        >>> pipeline = create_pipeline(numeric_features, categorical_features, n_estimators=50)
        >>> print(type(pipeline))
        <class 'sklearn.pipeline.Pipeline'>

    Notes:
        - StandardScaler: Normaliza features numéricas para média 0 e desvio 1
        - OneHotEncoder: Converte categóricas em variáveis dummy
        - SelectKBest: Seleciona k melhores features por ANOVA F-value
        - RandomForestClassifier: Ensemble de árvores de decisão
    """
    # Validações de entrada
    if not isinstance(numeric_features, list) or not isinstance(categorical_features, list):
        raise TypeError("numeric_features e categorical_features devem ser listas")

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise ValueError(
            "Pelo menos um tipo de feature (numérica ou categórica) deve ser fornecido"
        )

    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError(f"n_estimators deve ser inteiro positivo, recebido: {n_estimators}")

    if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
        raise ValueError(f"max_depth deve ser inteiro positivo ou None, recebido: {max_depth}")

    if not isinstance(min_samples_split, int) or min_samples_split < 2:
        raise ValueError(f"min_samples_split deve ser >= 2, recebido: {min_samples_split}")

    if not isinstance(learning_rate, int | float) or learning_rate <= 0:
        raise ValueError(f"learning_rate deve ser > 0, recebido: {learning_rate}")

    if not isinstance(num_leaves, int) or num_leaves < 2:
        raise ValueError(f"num_leaves deve ser >= 2, recebido: {num_leaves}")

    if not isinstance(subsample, int | float) or subsample <= 0 or subsample > 1:
        raise ValueError(f"subsample deve estar em (0, 1], recebido: {subsample}")

    if (
        not isinstance(colsample_bytree, int | float)
        or colsample_bytree <= 0
        or colsample_bytree > 1
    ):
        raise ValueError(f"colsample_bytree deve estar em (0, 1], recebido: {colsample_bytree}")

    if k != "all" and (not isinstance(k, int) or k <= 0):
        raise ValueError(f"k deve ser inteiro positivo ou 'all', recebido: {k}")

    logger.info(
        "Criando pipeline: %d numéricas, %d categóricas",
        len(numeric_features),
        len(categorical_features),
    )
    logger.debug(
        "Parâmetros: n_estimators=%d, max_depth=%s, learning_rate=%.3f, k=%s",
        n_estimators,
        max_depth,
        learning_rate,
        k,
    )

    # Transformadores de pré-processamento
    transformers = []

    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        transformers.append(("num", numeric_transformer, numeric_features))
        logger.debug(f"Features numéricas: {numeric_features}")

    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))
        logger.debug(f"Features categóricas: {categorical_features}")

    preprocessor = ColumnTransformer(transformers=transformers)

    if LGBMClassifier is not None:
        classifier = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=-1 if max_depth is None else max_depth,
            learning_rate=float(learning_rate),
            num_leaves=num_leaves,
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        logger.warning("lightgbm não encontrado; usando fallback GradientBoostingClassifier")
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=float(learning_rate),
            random_state=random_state,
        )

    # Pipeline completo
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_classif, k=k)),
            ("classifier", classifier),
        ]
    )

    logger.info("Pipeline criado com sucesso")
    return clf


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    k: Union[int, str] = "all",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, pd.Series]:
    """
    Treina modelo LightGBM e retorna métricas de avaliação.

    Divide dados em treino/teste, cria pipeline, treina modelo e
    calcula métricas detalhadas de performance.

    Args:
        X (pd.DataFrame): DataFrame com features.
        y (pd.Series): Series com target binário.
        n_estimators (int): Número de árvores no boosting. Padrão: 100.
        max_depth (Optional[int]): Profundidade máxima das árvores. Padrão: None.
        learning_rate (float): Taxa de aprendizado. Padrão: 0.1.
        num_leaves (int): Número máximo de folhas. Padrão: 31.
        subsample (float): Fração de linhas por árvore. Padrão: 1.0.
        colsample_bytree (float): Fração de colunas por árvore. Padrão: 1.0.
        k (Union[int, str]): Número de features a selecionar. Padrão: 'all'.
        test_size (float): Proporção dos dados para teste (0.0-1.0). Padrão: 0.2.
        random_state (int): Seed para reprodutibilidade. Padrão: 42.

    Returns:
        Tuple[Pipeline, Dict[str, float], pd.DataFrame, pd.Series]: Tupla contendo:
            - clf: Modelo treinado (Pipeline)
            - metrics: Dicionário com métricas de avaliação
            - X_test: Features do conjunto de teste
            - y_test: Target do conjunto de teste

    Raises:
        ValueError: Se dados forem inválidos ou test_size fora do range.
        TypeError: Se tipos de parâmetros estiverem incorretos.

    Examples:
        >>> X = pd.DataFrame({'idade': [10, 11, 12], 'nota': [7, 8, 9]})
        >>> y = pd.Series([0, 1, 0])
        >>> model, metrics, X_test, y_test = train_model(X, y, n_estimators=50)
        >>> print(f"AUC: {metrics['roc_auc']:.3f}")
        AUC: 0.875

    Notes:
        Métricas retornadas:
        - classification_report: Relatório completo (precision, recall, f1)
        - roc_auc: Area Under ROC Curve
        - accuracy: Acurácia geral
        - f1_score: F1-Score ponderado
        - precision: Precisão ponderada
        - recall: Recall ponderado
    """
    # Validações de entrada
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X deve ser pd.DataFrame, recebido: {type(X)}")

    if not isinstance(y, pd.Series | np.ndarray):
        raise TypeError(f"y deve ser pd.Series ou np.ndarray, recebido: {type(y)}")

    if X.empty or len(y) == 0:
        raise ValueError("X e y não podem estar vazios")

    if len(X) != len(y):
        raise ValueError(f"X e y devem ter mesmo comprimento. X: {len(X)}, y: {len(y)}")

    if not isinstance(test_size, int | float) or test_size <= 0 or test_size >= 1:
        raise ValueError(f"test_size deve estar entre 0 e 1, recebido: {test_size}")

    logger.info(f"Iniciando treinamento: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Distribuição do target: {pd.Series(y).value_counts().to_dict()}")

    # Identificar tipos de features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    logger.debug(
        f"{len(numeric_features)} features numéricas, {len(categorical_features)} categóricas"
    )

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise ValueError("Nenhuma feature válida encontrada em X")

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Manter proporção de classes
    )

    logger.info(f"Split: {len(X_train)} treino, {len(X_test)} teste (test_size={test_size})")

    # Criar e treinar pipeline
    clf = create_pipeline(
        numeric_features,
        categorical_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        k=k,
        random_state=random_state,
    )

    logger.info("Iniciando fit do modelo...")
    clf.fit(X_train, y_train)
    logger.info("Modelo treinado com sucesso")

    # Avaliar modelo
    logger.info("Calculando métricas de avaliação...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Calcular métricas detalhadas
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)

    metrics = {
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "roc_auc": float(auc_score),
        "accuracy": float(report_dict["accuracy"]),
        "f1_score": float(report_dict["weighted avg"]["f1-score"]),
        "precision": float(report_dict["weighted avg"]["precision"]),
        "recall": float(report_dict["weighted avg"]["recall"]),
    }

    logger.info(
        "Métricas calculadas - AUC: %.4f, F1: %.4f",
        metrics["roc_auc"],
        metrics["f1_score"],
    )
    logger.info(
        "Accuracy: %.4f, Precision: %.4f, Recall: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
    )

    return clf, metrics, X_test, y_test


def save_model(model: Pipeline, filepath: str) -> None:
    """
    Salva modelo treinado em disco usando joblib.

    Serializa o pipeline completo (pré-processamento + modelo) para
    arquivo usando joblib, permitindo carregamento posterior.

    Args:
        model (Pipeline): Pipeline treinado a ser salvo.
        filepath (str): Caminho completo do arquivo de destino.

    Raises:
        TypeError: Se model não for um Pipeline válido.
        ValueError: Se filepath for vazio ou inválido.
        IOError: Se houver erro ao salvar o arquivo.

    Examples:
        >>> from sklearn.pipeline import Pipeline
        >>> model = Pipeline([('clf', RandomForestClassifier())])
        >>> save_model(model, 'models/model.pkl')
        >>> # Modelo salvo com sucesso

    Notes:
        - Usa joblib para serialização eficiente
        - Salva pipeline completo (preprocessor + model)
        - Compatível com sklearn >= 1.0
    """
    if not isinstance(model, Pipeline):
        raise TypeError(f"model deve ser sklearn.pipeline.Pipeline, recebido: {type(model)}")

    if not filepath or not isinstance(filepath, str):
        raise ValueError("filepath deve ser uma string não vazia")

    if not filepath.endswith(".pkl"):
        logger.warning(f"filepath não termina com .pkl: {filepath}")

    try:
        joblib.dump(model, filepath)
        logger.info(f"Modelo salvo com sucesso em: {filepath}")
    except Exception as e:
        error_msg = f"Erro ao salvar modelo em {filepath}: {e!s}"
        logger.error(error_msg)
        raise OSError(error_msg) from e


def load_model(filepath: str) -> Pipeline:
    """
    Carrega modelo treinado do disco.

    Deserializa pipeline completo de arquivo joblib.

    Args:
        filepath (str): Caminho completo do arquivo do modelo.

    Returns:
        Pipeline: Pipeline carregado pronto para predição.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se filepath for inválido.
        IOError: Se houver erro ao carregar o arquivo.

    Examples:
        >>> model = load_model('models/model.pkl')
        >>> predictions = model.predict(X_new)
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("filepath deve ser uma string não vazia")

    if not os.path.exists(filepath):
        error_msg = f"Arquivo não encontrado: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        model = joblib.load(filepath)
        logger.info(f"Modelo carregado com sucesso de: {filepath}")
        return model
    except Exception as e:
        error_msg = f"Erro ao carregar modelo de {filepath}: {e!s}"
        logger.error(error_msg)
        raise OSError(error_msg) from e
