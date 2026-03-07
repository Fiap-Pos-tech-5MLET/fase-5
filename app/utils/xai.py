"""Utilitários para explicabilidade local (XAI) em predições.

Este módulo calcula as principais features que influenciaram uma predição
individual. Prioriza SHAP para modelos baseados em árvore e oferece fallback
determinístico baseado em importância global do modelo.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_dense(matrix: Any) -> np.ndarray:
    """Converte matriz esparsa/densa para ndarray.

    Args:
        matrix (Any): Matriz de entrada.

    Returns:
        np.ndarray: Matriz densa em formato NumPy.
    """
    if hasattr(matrix, "toarray"):
        return cast(np.ndarray, matrix.toarray())
    return cast(np.ndarray, np.asarray(matrix))


def _resolve_feature_names(model: Any, transformed_size: int) -> list[str]:
    """Obtém nomes das features após preprocessamento e seleção.

    Args:
        model (Pipeline): Pipeline treinado.
        transformed_size (int): Quantidade final de features no classificador.

    Returns:
        list[str]: Nomes das features alinhadas ao classificador.
    """
    preprocessor = model.named_steps.get("preprocessor")
    selector = model.named_steps.get("feature_selection")

    feature_names: list[str] = []

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [str(name) for name in raw_names]

    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(transformed_size)]

    if selector is not None and hasattr(selector, "get_support") and feature_names:
        support_mask = selector.get_support()
        if len(support_mask) == len(feature_names):
            feature_names = [
                name for name, keep in zip(feature_names, support_mask, strict=False) if keep
            ]

    if len(feature_names) != transformed_size:
        feature_names = [f"feature_{idx}" for idx in range(transformed_size)]

    return feature_names


def _build_contributions(
    feature_names: list[str],
    feature_values: np.ndarray,
    contributions: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    """Gera lista ordenada das principais contribuições locais.

    Args:
        feature_names (list[str]): Nomes das features.
        feature_values (np.ndarray): Valores da observação transformada.
        contributions (np.ndarray): Vetor de contribuição por feature.
        top_n (int): Limite de features no retorno.

    Returns:
        list[dict[str, Any]]: Features mais relevantes para a decisão.
    """
    if contributions.size == 0:
        return []

    safe_top_n = min(top_n, contributions.size)
    top_indices = np.argsort(np.abs(contributions))[::-1][:safe_top_n]

    result: list[dict[str, Any]] = []
    for idx in top_indices:
        direction = "aumenta_risco" if float(contributions[idx]) >= 0 else "reduz_risco"
        feature_value = float(feature_values[idx]) if idx < feature_values.size else 0.0
        result.append(
            {
                "feature_name": feature_names[idx],
                "feature_value": feature_value,
                "contribution": float(contributions[idx]),
                "direction": direction,
            }
        )

    return result


def explain_prediction(
    model: Any,
    feature_matrix: pd.DataFrame,
    top_n: int = 10,
) -> tuple[list[dict[str, Any]], str]:
    """Explica localmente uma predição com SHAP ou fallback determinístico.

    Args:
        model (Pipeline): Pipeline treinado em produção.
        feature_matrix (pd.DataFrame): Matriz de features já alinhada para inferência.
        top_n (int): Número de features explicativas no retorno.

    Returns:
        tuple[list[dict[str, Any]], str]:
            - Lista das principais features e contribuições
            - Método utilizado na explicação
    """
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        preprocessor = named_steps.get("preprocessor")
        classifier = named_steps.get("classifier")
        selector = named_steps.get("feature_selection")
    else:
        preprocessor = None
        classifier = model
        selector = None

    if classifier is None:
        return [], "unavailable"

    if preprocessor is None:
        if hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_)
            if feature_matrix.empty or importances.size == 0:
                return [], "unavailable"

            row_values_series = pd.to_numeric(feature_matrix.iloc[0], errors="coerce").fillna(0.0)
            row_values = row_values_series.to_numpy(dtype=float)
            feature_names = [str(col) for col in feature_matrix.columns]
            size = min(importances.size, row_values.size, len(feature_names))
            if size == 0:
                return [], "unavailable"

            proxy_contrib = row_values[:size] * importances[:size]
            return (
                _build_contributions(feature_names[:size], row_values[:size], proxy_contrib, top_n),
                "feature_importance_proxy",
            )

        return [], "unavailable"

    transformed = preprocessor.transform(feature_matrix)
    transformed_dense = _to_dense(transformed)

    if selector is not None:
        transformed_dense = _to_dense(selector.transform(transformed_dense))

    if transformed_dense.shape[0] == 0:
        return [], "unavailable"

    feature_names = _resolve_feature_names(model, transformed_dense.shape[1])
    row_values = transformed_dense[0]

    try:
        import shap

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_dense)

        if isinstance(shap_values, list):
            class_index = 1 if len(shap_values) > 1 else 0
            row_contrib = np.asarray(shap_values[class_index][0])
        else:
            row_contrib = np.asarray(shap_values[0])

        return _build_contributions(feature_names, row_values, row_contrib, top_n), "shap"
    except ImportError:  # pragma: no cover
        logger.info("SHAP não instalado; usando fallback por importância global.")
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        IndexError,
    ) as exc:  # pragma: no cover
        logger.warning("Falha ao gerar explicação SHAP: %s", exc)

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)
        size = min(importances.size, row_values.size)
        if size == 0:
            return [], "unavailable"

        proxy_contrib = row_values[:size] * importances[:size]
        return (
            _build_contributions(feature_names[:size], row_values[:size], proxy_contrib, top_n),
            "feature_importance_proxy",
        )

    return [], "unavailable"
