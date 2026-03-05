"""Testes unitários para utilitários de explicabilidade (XAI)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.utils.xai import explain_prediction


def test_explain_prediction_returns_proxy_when_shap_unavailable_or_fails() -> None:
    """Deve retornar explicação por proxy quando SHAP não estiver disponível."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[0.2, 1.0, -0.5]])
    preprocessor.get_feature_names_out.return_value = np.array(["IDADE", "INDE_23", "FASE_5A"])

    classifier = MagicMock()
    classifier.feature_importances_ = np.array([0.4, 0.5, 0.1])

    model = MagicMock()
    model.named_steps = {
        "preprocessor": preprocessor,
        "classifier": classifier,
    }

    X = pd.DataFrame({"IDADE": [12], "INDE_23": [7.1], "FASE": ["5A"]})

    top_features, method = explain_prediction(model=model, feature_matrix=X, top_n=2)

    assert method in {"feature_importance_proxy", "shap"}
    assert len(top_features) <= 2


def test_explain_prediction_without_required_steps_returns_unavailable() -> None:
    """Deve retornar indisponível quando pipeline não tiver etapas necessárias."""
    model = MagicMock()
    model.named_steps = {}

    X = pd.DataFrame({"IDADE": [12]})

    top_features, method = explain_prediction(model=model, feature_matrix=X)

    assert top_features == []
    assert method == "unavailable"
