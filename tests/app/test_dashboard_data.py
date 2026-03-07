from __future__ import annotations

import sys
import types
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

if "streamlit" not in sys.modules:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_resource = lambda func: func
    streamlit_stub.cache_data = lambda func: func
    sys.modules["streamlit"] = streamlit_stub

try:
    import pandas as _pandas
except ImportError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    pandas_stub.Series = object
    sys.modules["pandas"] = pandas_stub

try:
    import requests  # type: ignore[import-not-found]
except ImportError:
    requests_stub = types.ModuleType("requests")

    class RequestError(Exception):
        """Exceção base stub."""

    class ConnectionError(RequestError):
        """Exceção de conexão stub."""

    class HTTPError(RequestError):
        """Exceção HTTP stub."""

        def __init__(self, response: object | None = None) -> None:
            super().__init__("HTTP error")
            self.response = response

    requests_stub.exceptions = types.SimpleNamespace(
        RequestException=RequestError,
        ConnectionError=ConnectionError,
        HTTPError=HTTPError,
    )
    sys.modules["requests"] = requests_stub
    import requests  # type: ignore[import-not-found]

import importlib


def _load_dashboard_data_module() -> tuple[object, types.ModuleType]:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_resource = lambda func: func
    streamlit_stub.cache_data = lambda func: func
    streamlit_stub.error = MagicMock()
    sys.modules["streamlit"] = streamlit_stub

    sys.modules.pop("app.dashboard.data", None)
    module = importlib.import_module("app.dashboard.data")
    return importlib.reload(module), streamlit_stub


class DummyResponse:
    """Resposta HTTP simulada."""

    def __init__(self, status_code: int = 200, json_data: Dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = b"image-bytes"

    def json(self) -> Dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


class DummyDataFrame:
    """DataFrame mínimo para testes sem pandas."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns


def test_load_model_file_not_found(monkeypatch) -> None:
    """Deve retornar None se o arquivo do modelo não existir."""
    dashboard_data, _st = _load_dashboard_data_module()

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(dashboard_data.joblib, "load", _raise)

    assert dashboard_data.load_model() is None


def test_load_dataset_success(monkeypatch) -> None:
    """Deve carregar dataset quando pipeline não falha."""
    dashboard_data, _st = _load_dashboard_data_module()

    df = DummyDataFrame(["A", "TARGET"])
    monkeypatch.setattr(dashboard_data, "load_data", lambda *_args: df)
    monkeypatch.setattr(dashboard_data, "clean_data", lambda d: d)
    monkeypatch.setattr(dashboard_data, "create_target", lambda d: d)
    monkeypatch.setattr(dashboard_data, "handle_missing_values", lambda d: d)
    monkeypatch.setattr(dashboard_data, "create_features", lambda d: d)

    result = dashboard_data.load_dataset()

    assert result is not None
    assert "TARGET" in result.columns


def test_load_dataset_failure(monkeypatch) -> None:
    """Deve retornar None e registrar erro quando pipeline falha."""
    dashboard_data, st = _load_dashboard_data_module()

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("invalid")

    monkeypatch.setattr(dashboard_data, "load_data", _raise)

    result = dashboard_data.load_dataset()

    assert result is None
    st.error.assert_called()


def test_get_model_metrics_none_inputs(monkeypatch) -> None:
    """Deve retornar None se model ou df são None."""
    dashboard_data, _st = _load_dashboard_data_module()

    assert dashboard_data.get_model_metrics(None, None) is None


def test_get_model_metrics_success(monkeypatch) -> None:
    """Deve retornar métricas calculadas."""
    dashboard_data, _st = _load_dashboard_data_module()

    df = DummyDataFrame(["x1", "x2", "TARGET"])
    X = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    y = np.array([0, 1])

    monkeypatch.setattr(dashboard_data, "select_features", lambda _df: (X, y))
    monkeypatch.setattr(
        dashboard_data,
        "train_test_split",
        lambda _X, _y, **_kwargs: (None, X, None, y),
    )

    model = MagicMock()
    model.predict.return_value = np.array([0, 1])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
    model.feature_names_in_ = ["x1", "x2"]

    metrics = dashboard_data.get_model_metrics(model, df)

    assert metrics is not None
    assert set(metrics.keys()) >= {"accuracy", "roc_auc", "report"}


def test_get_model_metrics_error(monkeypatch) -> None:
    """Deve retornar None e registrar erro quando cálculo falha."""
    dashboard_data, st = _load_dashboard_data_module()

    df = DummyDataFrame(["x", "TARGET"])
    X = ["x1", "x2"]
    y = np.array([0, 1])

    monkeypatch.setattr(dashboard_data, "select_features", lambda _df: (X, y))
    monkeypatch.setattr(
        dashboard_data,
        "train_test_split",
        lambda _X, _y, **_kwargs: (None, X, None, y),
    )

    model = MagicMock()
    model.predict.side_effect = ValueError("fail")

    metrics = dashboard_data.get_model_metrics(model, df)

    assert metrics is None
    st.error.assert_called_once()


def test_predict_via_api_success(monkeypatch) -> None:
    """Deve retornar classe, probabilidade e explicações quando a API responde OK."""
    dashboard_data, _st = _load_dashboard_data_module()
    response = DummyResponse(
        status_code=200,
        json_data={
            "risk_prediction": 1,
            "risk_probability": 0.7,
            "top_features": [
                {
                    "feature_name": "ida",
                    "feature_value": 6.0,
                    "contribution": 0.15,
                    "direction": "aumenta_risco",
                },
            ],
        },
    )
    monkeypatch.setattr(
        dashboard_data.requests, "post", lambda *_args, **_kwargs: response, raising=False
    )

    pred, prob, expl = dashboard_data.predict_via_api({"IDADE": 10})

    assert pred == 1
    assert prob == 0.7
    assert isinstance(expl, list)
    assert len(expl) == 1


def test_predict_via_api_connection_error(monkeypatch) -> None:
    """Deve levantar ConnectionError quando API não responde."""
    dashboard_data, _st = _load_dashboard_data_module()

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise requests.exceptions.ConnectionError

    monkeypatch.setattr(dashboard_data.requests, "post", _raise, raising=False)

    import builtins

    with pytest.raises(builtins.ConnectionError):
        dashboard_data.predict_via_api({"IDADE": 10})


def test_predict_via_api_http_error(monkeypatch) -> None:
    """Deve levantar RuntimeError quando API retorna erro HTTP."""
    dashboard_data, _st = _load_dashboard_data_module()
    response = DummyResponse(status_code=400, json_data={"detail": "bad"})

    def _post(*_args: Any, **_kwargs: Any) -> DummyResponse:
        return response

    monkeypatch.setattr(dashboard_data.requests, "post", _post, raising=False)

    with pytest.raises(RuntimeError):
        dashboard_data.predict_via_api({"IDADE": 10})


def test_predict_via_api_http_error_without_detail(monkeypatch) -> None:
    """Deve usar texto padrão quando JSON falha."""
    dashboard_data, _st = _load_dashboard_data_module()

    class BadJsonResponse(DummyResponse):
        def json(self) -> Dict[str, Any]:
            raise ValueError("invalid json")

    response = BadJsonResponse(status_code=500)

    def _post(*_args: Any, **_kwargs: Any) -> DummyResponse:
        return response

    monkeypatch.setattr(dashboard_data.requests, "post", _post, raising=False)

    with pytest.raises(RuntimeError):
        dashboard_data.predict_via_api({"IDADE": 10})
