from __future__ import annotations

import sys
import types
from typing import Any, ClassVar, Dict
from unittest.mock import MagicMock

import pytest

from tests.utils_streamlit import make_streamlit_mock

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

try:
    import pandas as _pandas

    if getattr(_pandas, "DataFrame", None) is object:
        _pandas.DataFrame = MagicMock
        _pandas.Series = object
except ImportError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = MagicMock
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

import pandas as pd

import app.dashboard.pages.about as about_page
import app.dashboard.pages.drift as drift_page
import app.dashboard.pages.metrics as metrics_page
import app.dashboard.pages.prediction as prediction_page
import app.dashboard.pages.retrain as retrain_page


class DummyResponse:
    """Resposta HTTP simulada."""

    def __init__(self, status_code: int = 200, json_data: Dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = b"image-bytes"
        self.text = "error"

    def json(self) -> Dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


class FakeFigure:
    """Figura fake para substituir Plotly."""

    def update_layout(self, **_kwargs: Any) -> FakeFigure:
        return self


class FakeColumn:
    """Coluna fake para mapear labels."""

    def map(self, _mapping: Dict[int, str]) -> list[str]:
        return ["Sem Risco", "Em Risco"]


class FakeCounts:
    """DataFrame fake para value_counts().reset_index()."""

    columns: ClassVar[list[str]] = []

    def reset_index(self) -> FakeCounts:
        return self

    def __getitem__(self, _key: str) -> FakeColumn:
        return FakeColumn()

    def __setitem__(self, _key: str, _value: object) -> None:
        return None


class FakeSeries:
    """Series fake com value_counts/reset_index."""

    def value_counts(self) -> FakeCounts:
        return FakeCounts()


class FakeDataFrame:
    """DataFrame fake com TARGET e IDADE."""

    def __init__(self) -> None:
        self.columns = ["TARGET", "IDADE"]

    def __getitem__(self, _key: str) -> FakeSeries:
        return FakeSeries()


def test_render_about_page(monkeypatch) -> None:
    """Deve renderizar a página sobre o projeto."""
    st = make_streamlit_mock()
    monkeypatch.setattr(about_page, "st", st)

    about_page.render_about_page()

    st.markdown.assert_called()


def test_render_drift_page_with_existing_report(tmp_path, monkeypatch) -> None:
    """Deve renderizar relatório de drift quando existe."""
    st = make_streamlit_mock()
    st.button.return_value = True
    monkeypatch.setattr(drift_page, "st", st)

    data_path = tmp_path / "data.csv"
    data_path.write_text("col1,col2\n1,2\n", encoding="utf-8")
    
    report_path = tmp_path / "report.html"
    report_path.write_text("<html>ok</html>", encoding="utf-8")

    generate_mock = MagicMock()
    monkeypatch.setattr(drift_page, "generate_drift_report", generate_mock)

    drift_page.render_drift_page(str(data_path), str(report_path))

    generate_mock.assert_called_once()
    st.components.v1.html.assert_called_once()


def test_render_drift_page_missing_report(tmp_path, monkeypatch) -> None:
    """Deve permitir gerar relatório quando não existe."""
    st = make_streamlit_mock()
    st.button.return_value = True
    monkeypatch.setattr(drift_page, "st", st)

    data_path = tmp_path / "data.csv"
    data_path.write_text("col1,col2\n1,2\n", encoding="utf-8")
    
    missing_path = tmp_path / "missing.html"
    generate_mock = MagicMock()
    monkeypatch.setattr(drift_page, "generate_drift_report", generate_mock)

    drift_page.render_drift_page(str(data_path), str(missing_path))

    generate_mock.assert_called_once()
    st.warning.assert_called()


def test_render_drift_page_missing_report_error(tmp_path, monkeypatch) -> None:
    """Deve exibir erro quando geração falha sem relatório prévio."""
    st = make_streamlit_mock()
    monkeypatch.setattr(drift_page, "st", st)
    
    data_path = tmp_path / "data.csv"
    data_path.write_text("col1,col2\n1,2\n", encoding="utf-8")
    
    missing_path = tmp_path / "missing.html"

    def _raise(data_path: str, report_path: str) -> None:
        raise RuntimeError("fail")

    monkeypatch.setattr(drift_page, "generate_drift_report", _raise)

    drift_page.render_drift_page(str(data_path), str(missing_path))

    st.error.assert_called()
    st.exception.assert_called_once()


def test_render_drift_page_generate_error(tmp_path, monkeypatch) -> None:
    """Deve exibir erro quando geração falha."""
    st = make_streamlit_mock()
    st.button.return_value = True
    monkeypatch.setattr(drift_page, "st", st)

    data_path = tmp_path / "data.csv"
    data_path.write_text("col1,col2\n1,2\n", encoding="utf-8")
    
    report_path = tmp_path / "report.html"
    report_path.write_text("<html>ok</html>", encoding="utf-8")

    def _raise(data_path: str, report_path: str) -> None:
        raise RuntimeError("fail")

    monkeypatch.setattr(drift_page, "generate_drift_report", _raise)

    drift_page.render_drift_page(str(data_path), str(report_path))

    st.error.assert_called()
    st.exception.assert_called_once()


def test_render_metrics_page_mlflow(monkeypatch) -> None:
    """Deve renderizar métricas quando API retorna dados do MLflow."""
    st = make_streamlit_mock()
    monkeypatch.setattr(metrics_page, "st", st)

    def _get(url: str, *_args: Any, **_kwargs: Any) -> DummyResponse:
        if url.endswith("/model-metrics"):
            return DummyResponse(
                status_code=200,
                json_data={
                    "source": "mlflow",
                    "metrics": {"accuracy": 0.9, "roc_auc": 0.95},
                    "params": {"model_type": "RF", "n_estimators": 100, "max_depth": 5},
                    "run_id": "abc123",
                },
            )
        return DummyResponse(status_code=404)

    monkeypatch.setattr(metrics_page.requests, "get", _get, raising=False)

    metrics_page.render_metrics_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: None,
        metrics_func=lambda *_args: None,
        roc_curve_path="/tmp/roc.png",
        feature_imp_path="/tmp/feat.png",
        class_report_path="/tmp/report.png",
    )

    st.markdown.assert_called()
    st.info.assert_called()


def test_render_metrics_page_mlflow_with_images(monkeypatch) -> None:
    """Deve renderizar imagens quando artefatos estão disponíveis."""
    st = make_streamlit_mock()
    monkeypatch.setattr(metrics_page, "st", st)

    def _get(url: str, *_args: Any, **_kwargs: Any) -> DummyResponse:
        if url.endswith("/model-metrics"):
            return DummyResponse(
                status_code=200,
                json_data={
                    "source": "mlflow",
                    "metrics": {"accuracy": 0.9, "roc_auc": 0.95},
                    "params": {"model_type": "RF"},
                    "run_id": "abc123",
                },
            )
        return DummyResponse(status_code=200)

    monkeypatch.setattr(metrics_page.requests, "get", _get, raising=False)

    df = FakeDataFrame()
    monkeypatch.setattr(metrics_page.px, "pie", lambda *_args, **_kwargs: FakeFigure())
    monkeypatch.setattr(metrics_page.px, "histogram", lambda *_args, **_kwargs: FakeFigure())

    metrics_page.render_metrics_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: df,
        metrics_func=lambda *_args: None,
        roc_curve_path="/tmp/roc.png",
        feature_imp_path="/tmp/feat.png",
        class_report_path="/tmp/report.png",
    )

    assert st.image.call_count >= 2


def test_render_metrics_page_mlflow_source_local(monkeypatch) -> None:
    """Deve exibir aviso quando MLflow não tem run vinculada."""
    st = make_streamlit_mock()
    monkeypatch.setattr(metrics_page, "st", st)

    def _get(url: str, *_args: Any, **_kwargs: Any) -> DummyResponse:
        return DummyResponse(status_code=200, json_data={"source": "local"})

    monkeypatch.setattr(metrics_page.requests, "get", _get, raising=False)

    metrics_page.render_metrics_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: None,
        metrics_func=lambda *_args: None,
        roc_curve_path="/tmp/roc.png",
        feature_imp_path="/tmp/feat.png",
        class_report_path="/tmp/report.png",
    )

    st.warning.assert_called()


def test_render_metrics_page_mlflow_artifact_exception(monkeypatch) -> None:
    """Deve usar artefatos locais quando requisição falha."""
    st = make_streamlit_mock()
    monkeypatch.setattr(metrics_page, "st", st)

    def _get(url: str, *_args: Any, **_kwargs: Any) -> DummyResponse:
        if url.endswith("/model-metrics"):
            return DummyResponse(
                status_code=200,
                json_data={"source": "mlflow", "metrics": {"accuracy": 0.9}},
            )
        raise requests.exceptions.RequestException("fail")

    monkeypatch.setattr(metrics_page.requests, "get", _get, raising=False)
    monkeypatch.setattr(metrics_page.os.path, "exists", lambda _path: True)

    metrics_page.render_metrics_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: None,
        metrics_func=lambda *_args: None,
        roc_curve_path="/tmp/roc.png",
        feature_imp_path="/tmp/feat.png",
        class_report_path="/tmp/report.png",
    )

    st.image.assert_called()


def test_render_metrics_page_local_no_metrics(monkeypatch) -> None:
    """Deve avisar quando métricas locais não estão disponíveis."""
    st = make_streamlit_mock()
    monkeypatch.setattr(metrics_page, "st", st)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise requests.exceptions.RequestException

    monkeypatch.setattr(metrics_page.requests, "get", _raise, raising=False)

    metrics_page.render_metrics_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: None,
        metrics_func=lambda *_args: None,
        roc_curve_path="/tmp/roc.png",
        feature_imp_path="/tmp/feat.png",
        class_report_path="/tmp/report.png",
    )

    st.warning.assert_called()


def test_render_metrics_page_local(monkeypatch) -> None:
    """Deve renderizar métricas locais quando API falha."""
    st = make_streamlit_mock()
    monkeypatch.setattr(metrics_page, "st", st)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise requests.exceptions.RequestException

    monkeypatch.setattr(metrics_page.requests, "get", _raise, raising=False)

    metrics = {
        "accuracy": 0.9,
        "roc_auc": 0.95,
        "report": {"weighted avg": {"precision": 0.9, "recall": 0.9}},
    }

    monkeypatch.setattr(metrics_page.px, "pie", lambda *_args, **_kwargs: FakeFigure())
    monkeypatch.setattr(metrics_page.px, "histogram", lambda *_args, **_kwargs: FakeFigure())
    monkeypatch.setattr(metrics_page.os.path, "exists", lambda _path: True)

    df = pd.DataFrame({"TARGET": [0, 1], "IDADE": [10, 12]})

    metrics_page.render_metrics_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: df,
        metrics_func=lambda *_args: metrics,
        roc_curve_path="/tmp/roc.png",
        feature_imp_path="/tmp/feat.png",
        class_report_path="/tmp/report.png",
    )

    st.info.assert_called()
    st.markdown.assert_called()


def test_render_prediction_page_model_none(monkeypatch) -> None:
    """Deve interromper se modelo não carregado."""
    st = make_streamlit_mock()
    st.stop.side_effect = RuntimeError("stop")
    monkeypatch.setattr(prediction_page, "st", st)

    with pytest.raises(RuntimeError):
        prediction_page.render_prediction_page(None, lambda _data: (0, 0.0))


def test_render_prediction_page_success(monkeypatch) -> None:
    """Deve renderizar predição com sucesso."""
    st = make_streamlit_mock()
    monkeypatch.setattr(prediction_page, "st", st)

    st.number_input.side_effect = [
        12,
        2022,
        5.0,
        5.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    ]
    st.selectbox.side_effect = [
        "Feminino",
        "1A",
        "Fase 1 (3° e 4° ano)",
        "Cursando",
    ]

    prediction_page.render_prediction_page(
        model=MagicMock(),
        predict_func=lambda _data: (1, 0.7),
    )

    st.warning.assert_called()


def test_render_prediction_page_error(monkeypatch) -> None:
    """Deve exibir erro quando API falha."""
    st = make_streamlit_mock()
    monkeypatch.setattr(prediction_page, "st", st)

    st.number_input.side_effect = [
        12,
        2022,
        5.0,
        5.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    ]
    st.selectbox.side_effect = [
        "Feminino",
        "1A",
        "Fase 1 (3° e 4° ano)",
        "Cursando",
    ]

    def _raise(_data: Dict[str, Any]) -> tuple[int, float]:
        raise ConnectionError("fail")

    prediction_page.render_prediction_page(model=MagicMock(), predict_func=_raise)

    st.error.assert_called()


def test_render_retrain_page_with_retrain(monkeypatch) -> None:
    """Deve iniciar retreinamento e armazenar métricas do candidato."""
    st = make_streamlit_mock()
    st.button.side_effect = [True, False, False]
    st.number_input.side_effect = [100, 2]
    st.selectbox.side_effect = ["Sem limite", "Todas"]
    st.slider.return_value = 20
    st.text_input.side_effect = ["lucas_admin", "test-api-key"]
    monkeypatch.setattr(retrain_page, "st", st)

    response = DummyResponse(
        status_code=200,
        json_data={"metrics": {"accuracy": 0.9, "roc_auc": 0.95}},
    )
    monkeypatch.setattr(
        retrain_page.requests,
        "post",
        lambda *_args, **_kwargs: response,
        raising=False,
    )

    retrain_page.render_retrain_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: None,
        metrics_func=lambda *_args: {
            "accuracy": 0.8,
            "roc_auc": 0.9,
            "report": {"weighted avg": {"f1-score": 0.8, "precision": 0.8, "recall": 0.8}},
        },
        load_model_func=MagicMock(),
    )

    assert st.session_state.get("candidate_ready") is True
    assert "candidate_metrics" in st.session_state


def test_render_retrain_page_with_error(monkeypatch) -> None:
    """Deve exibir erro quando API retorna falha."""
    st = make_streamlit_mock()
    st.button.return_value = True
    st.number_input.side_effect = [100, 2]
    st.selectbox.side_effect = ["Sem limite", "Todas"]
    st.slider.return_value = 20
    st.text_input.side_effect = ["lucas_admin", "test-api-key"]
    monkeypatch.setattr(retrain_page, "st", st)

    response = DummyResponse(status_code=500, json_data={"detail": "error"})
    monkeypatch.setattr(retrain_page.requests, "post", lambda *_a, **_k: response, raising=False)

    retrain_page.render_retrain_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: pd.DataFrame({"TARGET": [0, 1]}),
        metrics_func=lambda *_args: None,
        load_model_func=MagicMock(),
    )

    st.error.assert_called()


def test_render_retrain_page_promote_and_discard(monkeypatch) -> None:
    """Deve promover e descartar candidato com sucesso."""
    st = make_streamlit_mock()
    st.session_state["candidate_ready"] = True
    st.session_state["candidate_metrics"] = {"accuracy": 0.9, "roc_auc": 0.95}
    st.button.side_effect = [False, True, True]
    st.number_input.side_effect = [100, 2]
    st.selectbox.side_effect = ["Sem limite", "Todas"]
    st.slider.return_value = 20
    st.text_input.side_effect = ["lucas_admin", "test-api-key"]
    monkeypatch.setattr(retrain_page, "st", st)

    response = DummyResponse(status_code=200, json_data={})
    monkeypatch.setattr(retrain_page.requests, "post", lambda *_a, **_k: response, raising=False)

    load_model_func = MagicMock()
    load_model_func.clear = MagicMock()
    load_dataset_func = MagicMock()
    load_dataset_func.clear = MagicMock()

    retrain_page.render_retrain_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=load_dataset_func,
        metrics_func=lambda *_args: None,
        load_model_func=load_model_func,
    )

    load_model_func.clear.assert_called()
    load_dataset_func.clear.assert_called()


def test_render_retrain_page_promote_error(monkeypatch) -> None:
    """Deve exibir erro quando promoção falha."""
    st = make_streamlit_mock()
    st.session_state["candidate_ready"] = True
    st.session_state["candidate_metrics"] = {"accuracy": 0.9, "roc_auc": 0.95}
    st.button.side_effect = [False, True, False]
    st.number_input.side_effect = [100, 2]
    st.selectbox.side_effect = ["Sem limite", "Todas"]
    st.slider.return_value = 20
    st.text_input.side_effect = ["lucas_admin", "test-api-key"]
    monkeypatch.setattr(retrain_page, "st", st)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise requests.exceptions.RequestException("fail")

    monkeypatch.setattr(retrain_page.requests, "post", _raise, raising=False)

    retrain_page.render_retrain_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: pd.DataFrame({"TARGET": [0, 1]}),
        metrics_func=lambda *_args: None,
        load_model_func=MagicMock(),
    )

    st.error.assert_called()


def test_render_retrain_page_discard_error(monkeypatch) -> None:
    """Deve exibir erro quando descarte falha."""
    st = make_streamlit_mock()
    st.session_state["candidate_ready"] = True
    st.session_state["candidate_metrics"] = {"accuracy": 0.9, "roc_auc": 0.95}
    st.button.side_effect = [False, False, True]
    st.number_input.side_effect = [100, 2]
    st.selectbox.side_effect = ["Sem limite", "Todas"]
    st.slider.return_value = 20
    st.text_input.side_effect = ["lucas_admin", "test-api-key"]
    monkeypatch.setattr(retrain_page, "st", st)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise requests.exceptions.RequestException("fail")

    monkeypatch.setattr(retrain_page.requests, "post", _raise, raising=False)

    retrain_page.render_retrain_page(
        model=MagicMock(),
        api_url="http://api",
        load_dataset_func=lambda: pd.DataFrame({"TARGET": [0, 1]}),
        metrics_func=lambda *_args: None,
        load_model_func=MagicMock(),
    )

    st.error.assert_called()


def test_render_retrain_page_model_none(monkeypatch) -> None:
    """Deve exibir aviso quando não há modelo carregado."""
    st = make_streamlit_mock()
    st.button.return_value = False
    st.number_input.side_effect = [100, 2]
    st.selectbox.side_effect = ["Sem limite", "Todas"]
    st.slider.return_value = 20
    st.text_input.side_effect = ["lucas_admin", "test-api-key"]
    monkeypatch.setattr(retrain_page, "st", st)

    retrain_page.render_retrain_page(
        model=None,
        api_url="http://api",
        load_dataset_func=lambda: None,
        metrics_func=lambda *_args: None,
        load_model_func=MagicMock(),
    )

    st.warning.assert_called()
