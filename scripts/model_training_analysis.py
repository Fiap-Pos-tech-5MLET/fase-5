"""Utilitários de modelagem para o notebook de treinamento.

Este módulo concentra funções reutilizáveis para o fluxo de modelagem com
LightGBM (ou fallback compatível), mantendo notebooks enxutos e orientados
à orquestração.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

LGBMClassifier: Any

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier  # pyright: ignore[reportMissingImports]

    LGBMClassifier = _LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - fallback apenas para ambientes sem lightgbm
    LGBMClassifier = RandomForestClassifier
    _HAS_LIGHTGBM = False


MetricasFold = Dict[str, Dict[str, List[float]]]


def preparar_dados_modelagem(
    df: pd.DataFrame,
    target_coluna: str = "evadiu",
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Prepara split estratificado para classificação binária.

    Args:
        df (pd.DataFrame): DataFrame com target e features.
        target_coluna (str): Nome da coluna target binária.
        feature_columns (Optional[List[str]]): Features a usar. Se None, usa numéricas.
        test_size (float): Proporção de dados de teste.
        random_state (int): Seed para reprodutibilidade.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
            X_train, X_test, y_train, y_test, lista final de features.

    Raises:
        ValueError: Quando target não existe ou não há features válidas.
    """
    if target_coluna not in df.columns:
        raise ValueError(f"Coluna target '{target_coluna}' não encontrada.")

    if feature_columns is None:
        colunas = (
            df.select_dtypes(include=[np.number])
            .drop(columns=[target_coluna], errors="ignore")
            .columns.tolist()
        )
    else:
        colunas = [col for col in feature_columns if col in df.columns and col != target_coluna]

    if not colunas:
        raise ValueError("Nenhuma feature válida encontrada para modelagem.")

    X = df[colunas].copy()
    y = df[target_coluna].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, colunas


def criar_modelo_lgbm(
    parametros: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """Cria classificador com defaults alinhados ao projeto.

    Args:
        parametros (Optional[Dict[str, Any]]): Hiperparâmetros para sobrescrever defaults.
        random_state (int): Seed para reprodutibilidade.

    Returns:
        Any: Instância de classificador compatível com API sklearn.
    """
    params_base: Dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 4,
        "num_leaves": 31,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }

    if parametros:
        params_base.update(parametros)

    if _HAS_LIGHTGBM:
        return LGBMClassifier(**params_base)

    # Fallback para ambientes sem LightGBM: preserva hiperparâmetros equivalentes.
    rf_params = {
        "n_estimators": int(params_base.get("n_estimators", 100)),
        "max_depth": params_base.get("max_depth"),
        "class_weight": params_base.get("class_weight", "balanced"),
        "random_state": int(params_base.get("random_state", random_state)),
        "n_jobs": int(params_base.get("n_jobs", -1)),
    }
    return RandomForestClassifier(**rf_params)


def _calcular_metricas_binarias(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Calcula métricas binárias com proteção contra cenários degenerados."""
    y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba_pos)),
        "pr_auc": float(average_precision_score(y_true, y_proba_pos)),
    }


def executar_validacao_cruzada_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modelo: Any,
    n_splits: int = 5,
    random_state: int = 42,
) -> MetricasFold:
    """Executa validação cruzada estratificada com métricas de treino e validação.

    Args:
        X_train (pd.DataFrame): Features de treino.
        y_train (pd.Series): Target de treino.
        modelo (Any): Modelo com métodos fit/predict/predict_proba.
        n_splits (int): Número de folds.
        random_state (int): Seed para embaralhamento dos folds.

    Returns:
        MetricasFold: Dicionário no formato results[dataset][metrica] -> lista por fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    resultados: MetricasFold = {"train": {}, "validation": {}}

    metricas = ["accuracy", "f1_score", "recall", "precision", "log_loss", "roc_auc", "pr_auc"]
    for dataset in resultados:
        for metrica in metricas:
            resultados[dataset][metrica] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        model_fold = clone(modelo)
        model_fold.fit(X_train_scaled, y_train_fold)

        for nome, (X_fold, y_fold) in {
            "train": (X_train_scaled, y_train_fold),
            "validation": (X_val_scaled, y_val_fold),
        }.items():
            y_pred = model_fold.predict(X_fold)
            y_proba = model_fold.predict_proba(X_fold)
            fold_metrics = _calcular_metricas_binarias(y_fold, y_pred, y_proba)
            for metrica, valor in fold_metrics.items():
                resultados[nome][metrica].append(valor)

    return resultados


def resumir_resultados_cv(resultados_cv: MetricasFold) -> pd.DataFrame:
    """Consolida métricas de validação cruzada em DataFrame de leitura rápida.

    Args:
        resultados_cv (MetricasFold): Saída de executar_validacao_cruzada_lgbm.

    Returns:
        pd.DataFrame: Tabela com média e desvio padrão para treino/validação.
    """
    linhas: List[Dict[str, Any]] = []

    for metrica in resultados_cv["validation"]:
        valores_train = resultados_cv["train"][metrica]
        valores_val = resultados_cv["validation"][metrica]
        linhas.append(
            {
                "metrica": metrica,
                "treino_media": np.mean(valores_train),
                "treino_std": np.std(valores_train),
                "validacao_media": np.mean(valores_val),
                "validacao_std": np.std(valores_val),
            }
        )

    return pd.DataFrame(linhas).sort_values("metrica").reset_index(drop=True)


def treinar_e_avaliar_lgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    modelo: Any,
) -> Dict[str, Any]:
    """Treina modelo final e retorna predições, probabilidades e métricas.

    Args:
        X_train (pd.DataFrame): Features de treino.
        X_test (pd.DataFrame): Features de teste.
        y_train (pd.Series): Target de treino.
        y_test (pd.Series): Target de teste.
        modelo (Any): Modelo não treinado.

    Returns:
        Dict[str, Any]: Artefatos do experimento com modelo, scaler e métricas.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo_final = clone(modelo)
    modelo_final.fit(X_train_scaled, y_train)

    y_pred = modelo_final.predict(X_test_scaled)
    y_proba = modelo_final.predict_proba(X_test_scaled)

    metricas_teste = _calcular_metricas_binarias(y_test, y_pred, y_proba)

    return {
        "model": modelo_final,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "test_metrics": metricas_teste,
    }
