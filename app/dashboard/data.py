"""
Carregamento de dados e integração com API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from app.dashboard.config import API_URL, DASHBOARD_REQUESTED_BY, DATA_PATH, MODEL_PATH
from src.data_cleaning import clean_data, create_target, handle_missing_values, load_data
from src.feature_engineering import create_features, select_features


def get_model_cache_buster() -> int:
    """
    Retorna versão do arquivo de modelo para invalidar cache automaticamente.

    Returns:
        int: Timestamp em nanossegundos quando o arquivo existe; -1 caso contrário.
    """
    try:
        return os.stat(MODEL_PATH).st_mtime_ns
    except OSError:
        return -1


@st.cache_resource
def load_model(_cache_buster: int = -1) -> Optional[Any]:
    """
    Carrega o modelo treinado do disco.

    Args:
        _cache_buster (int): Chave para invalidar cache quando o arquivo muda.

    Returns:
        Optional[Any]: Modelo carregado ou None se não encontrado.
    """
    try:
        return joblib.load(MODEL_PATH)
    except (FileNotFoundError, OSError, ValueError, EOFError):
        return None


@st.cache_data
def load_dataset() -> Optional[pd.DataFrame]:
    """
    Carrega e prepara o dataset completo para visualizações.

    Returns:
        Optional[pd.DataFrame]: Dataset tratado ou None em caso de falha.
    """
    try:
        df = load_data(DATA_PATH)
        df = clean_data(df)
        df = create_target(df)
        df = handle_missing_values(df)
        df = create_features(df)
        return df
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        st.error(f"Erro ao carregar dataset: {exc}")
        return None


def get_model_metrics(model: Any, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Calcula métricas do modelo usando o dataset informado.

    Args:
        model (Any): Modelo treinado.
        df (Optional[pd.DataFrame]): Dataset já preparado.

    Returns:
        Optional[Dict[str, Any]]: Métricas e dados auxiliares ou None.
    """
    if model is None or df is None:
        return None
    try:
        X, y = select_features(df)
        _X_train, X_test, _y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Remover features extras que o modelo não espera (compatibilidade 13 vs 14 features)
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            extra_cols = set(X_test.columns) - set(expected_cols)
            if extra_cols:
                X_test = X_test.drop(columns=list(extra_cols))
            X_test = X_test[expected_cols]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "report": report,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    except (AttributeError, KeyError, ValueError, RuntimeError) as exc:
        st.error(f"Erro ao calcular métricas: {exc}")
        return None


def predict_via_api(student_data: Dict[str, Any]) -> Tuple[int, float, list[Dict[str, Any]]]:
    """
    Chama o endpoint /predict da API com explicabilidade.

    Args:
        student_data (Dict[str, Any]): Dados do aluno.

    Returns:
        Tuple[int, float, list[Dict[str, Any]]]: Classe prevista, probabilidade e explicações das features.
                                                  Cada explicação contém: feature_name, contribution, direction, feature_value.

    Raises:
        ConnectionError: Se a API não estiver disponível.
        RuntimeError: Se a API retornar erro.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"data": student_data},
            headers={"x-requested-by": DASHBOARD_REQUESTED_BY},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        explanations = result.get("top_features", [])
        explanation_method = result.get("explanation_method", "unavailable")

        if not explanations:
            st.warning(f"⚠️ Explicabilidade indisponível (método: {explanation_method})")

        return int(result["risk_prediction"]), float(result["risk_probability"]), explanations
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Não foi possível conectar à API em {API_URL}. "
            "Certifique-se de que a API está rodando (uvicorn app.main:app)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise RuntimeError(f"Erro na API: {detail}") from exc
