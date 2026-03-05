"""
Funções de análise estatística e EDA para o projeto Passos Mágicos.

Este módulo contém funções para análise exploratória de dados (EDA), incluindo
testes de normalidade, transformações estatísticas e validação cruzada com
métricas completas.

Funções principais:
- Testes de normalidade (Shapiro-Wilk, D'Agostino K²)
- Transformações logarítmicas para distribuições skewed
- Pipeline de validação cruzada estratificada
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def verificar_normalidade(
    dados: pd.Series, alpha: float = 0.05, verbose: bool = True
) -> Dict[str, Any]:
    """
    Verifica normalidade de uma distribuição usando testes estatísticos.

    Aplica dois testes:
    - Shapiro-Wilk (recomendado para n < 5000)
    - D'Agostino K² (alternativa para amostras maiores)

    Args:
        dados (pd.Series): Série de dados numéricos a testar.
        alpha (float): Nível de significância (padrão: 0.05).
        verbose (bool): Se True, imprime resultados.

    Returns:
        Dict[str, Any]: Dicionário com resultados:
            - 'normal': bool indicando se distribuição é normal
            - 'shapiro_stat': estatística do teste Shapiro-Wilk
            - 'shapiro_p': p-value do Shapiro-Wilk
            - 'dagostino_stat': estatística do teste D'Agostino
            - 'dagostino_p': p-value do D'Agostino

    Examples:
        >>> dados_normais = pd.Series(np.random.normal(0, 1, 1000))
        >>> resultado = verificar_normalidade(dados_normais, verbose=False)
        >>> resultado['normal']
        True
    """
    # Remove NaN
    dados_limpos = dados.dropna()

    # Shapiro-Wilk
    shapiro_stat, shapiro_p = stats.shapiro(dados_limpos)

    # D'Agostino K²
    dagostino_stat, dagostino_p = stats.normaltest(dados_limpos)

    # Considera normal se ambos os testes aceitam H0
    normal = (shapiro_p > alpha) and (dagostino_p > alpha)

    if verbose:
        print(f"Shapiro-Wilk: estatística={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        print(f"D'Agostino K²: estatística={dagostino_stat:.4f}, p-value={dagostino_p:.4f}")
        if normal:
            print("✓ Distribuição NORMAL (falhamos em rejeitar H0)")
        else:
            print("✗ Distribuição NÃO NORMAL (rejeitamos H0)")

    return {
        "normal": normal,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "dagostino_stat": dagostino_stat,
        "dagostino_p": dagostino_p,
    }


def aplicar_transformacao_log(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Aplica transformação logarítmica (log1p) em colunas especificadas.

    A transformação log1p(x) = log(1 + x) é útil para:
    - Reduzir skewness em distribuições assimétricas
    - Estabilizar variância
    - Lidar com valores zero (evita log(0) = -inf)

    Args:
        df (pd.DataFrame): DataFrame original.
        colunas (List[str]): Lista de colunas a transformar.

    Returns:
        pd.DataFrame: DataFrame com colunas transformadas (sufixo '_log').

    Examples:
        >>> df = pd.DataFrame({"A": [0, 1, 10, 100]})
        >>> aplicar_transformacao_log(df, ["A"])
           A     A_log
        0  0  0.000000
        1  1  0.693147
        2 10  2.397895
        3 100 4.615120
    """
    df_out = df.copy()

    for col in colunas:
        if col in df_out.columns:
            col_transformada = f"{col}_log"
            df_out[col_transformada] = np.log1p(df_out[col])

    return df_out


def perform_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executa validação cruzada estratificada com métricas completas.

    Args:
        model (Any): Modelo sklearn compatível (não treinado, será clonado).
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        n_folds (int): Número de folds para cross-validation (padrão: 5).
        random_state (int): Seed para reprodutibilidade.
        verbose (bool): Se True, imprime progresso.

    Returns:
        Dict[str, Any]: Dicionário com resultados:
            - 'accuracy': lista de acurácias por fold
            - 'precision': lista de precisões por fold
            - 'recall': lista de recalls por fold
            - 'f1': lista de F1-scores por fold
            - 'roc_auc': lista de ROC-AUC por fold
            - 'confusion_matrices': lista de matrizes de confusão
            - 'mean_accuracy': média da acurácia
            - 'std_accuracy': desvio padrão da acurácia
            - 'mean_f1': média do F1
            - 'std_f1': desvio padrão do F1

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, random_state=42)
        >>> model = RandomForestClassifier(random_state=42)
        >>> resultado = perform_cross_validation(model, pd.DataFrame(X), pd.Series(y), n_folds=3, verbose=False)
        >>> resultado['mean_accuracy'] > 0.7
        True
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Armazenamento de métricas
    metricas: Dict[str, List[Any]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "confusion_matrices": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if verbose:
            print(f"Fold {fold}/{n_folds}...")

        # Divisão train/val
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone e treina modelo
        model_fold = clone(model)
        model_fold.fit(X_train, y_train)

        # Predições
        y_pred = model_fold.predict(X_val)
        y_proba = model_fold.predict_proba(X_val)[:, 1]

        # Métricas
        metricas["accuracy"].append(accuracy_score(y_val, y_pred))
        metricas["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metricas["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        metricas["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        metricas["roc_auc"].append(roc_auc_score(y_val, y_proba))
        metricas["confusion_matrices"].append(confusion_matrix(y_val, y_pred))

        if verbose:
            print(f"  Accuracy: {metricas['accuracy'][-1]:.4f}")
            print(f"  F1-Score: {metricas['f1'][-1]:.4f}")
            print(f"  ROC-AUC:  {metricas['roc_auc'][-1]:.4f}\n")

    # Estatísticas agregadas
    metricas["mean_accuracy"] = np.mean(metricas["accuracy"])
    metricas["std_accuracy"] = np.std(metricas["accuracy"])
    metricas["mean_precision"] = np.mean(metricas["precision"])
    metricas["std_precision"] = np.std(metricas["precision"])
    metricas["mean_recall"] = np.mean(metricas["recall"])
    metricas["std_recall"] = np.std(metricas["recall"])
    metricas["mean_f1"] = np.mean(metricas["f1"])
    metricas["std_f1"] = np.std(metricas["f1"])
    metricas["mean_roc_auc"] = np.mean(metricas["roc_auc"])
    metricas["std_roc_auc"] = np.std(metricas["roc_auc"])

    if verbose:
        print("=" * 50)
        print(f"Accuracy: {metricas['mean_accuracy']:.4f} ± {metricas['std_accuracy']:.4f}")
        print(f"F1-Score: {metricas['mean_f1']:.4f} ± {metricas['std_f1']:.4f}")
        print(f"ROC-AUC:  {metricas['mean_roc_auc']:.4f} ± {metricas['std_roc_auc']:.4f}")
        print("=" * 50)

    return metricas


def calcular_coeficiente_variacao(df: pd.DataFrame, colunas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calcula o coeficiente de variação (CV = std/mean * 100%) para colunas numéricas.

    CV é útil para comparar variabilidade relativa entre features com escalas diferentes.

    Args:
        df (pd.DataFrame): DataFrame original.
        colunas (Optional[List[str]]): Lista de colunas a analisar. Se None, usa todas numéricas.

    Returns:
        pd.DataFrame: DataFrame com colunas 'coluna', 'media', 'desvio_padrao', 'cv_percent'.

    Examples:
        >>> df = pd.DataFrame({"A": [10, 20, 30], "B": [100, 200, 300]})
        >>> calcular_coeficiente_variacao(df)
          coluna  media  desvio_padrao  cv_percent
        0      A   20.0      10.000000       50.00
        1      B  200.0     100.000000       50.00
    """
    if colunas is None:
        colunas = df.select_dtypes(include=[np.number]).columns.tolist()

    resultados = []

    for col in colunas:
        if col in df.columns:
            media = df[col].mean()
            desvio = df[col].std()
            cv = (desvio / media * 100) if media != 0 else np.nan

            resultados.append(
                {"coluna": col, "media": media, "desvio_padrao": desvio, "cv_percent": cv}
            )

    return pd.DataFrame(resultados)
