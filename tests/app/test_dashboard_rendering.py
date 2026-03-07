"""
Testes de renderização e utilidades do dashboard.
"""

from __future__ import annotations

import json
import runpy
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import pandas as pd
except ImportError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = MagicMock
    pandas_stub.Series = MagicMock
    sys.modules["pandas"] = pandas_stub
    import pandas as pd

try:
    from requests import exceptions as requests_exceptions
except ImportError:
    requests_stub = types.ModuleType("requests")

    class RequestError(Exception):
        """Exceção base stub."""

    class RequestBaseError(RequestError):
        """Exceção genérica stub."""

    class HTTPError(RequestError):
        """Exceção HTTP stub."""

        def __init__(self, response: object | None = None) -> None:
            super().__init__("HTTP error")
            self.response = response

    requests_stub.exceptions = types.SimpleNamespace(
        RequestException=RequestBaseError,
        HTTPError=HTTPError,
    )
    sys.modules["requests"] = requests_stub
    from requests import exceptions as requests_exceptions


@dataclass
class DummyResponse:
    """Resposta HTTP simulada para testes."""

    status_code: int
    payload: Dict[str, Any] | None = None
    content: bytes = b""

    def json(self) -> Dict[str, Any]:
        """Retorna o payload JSON simulado."""
        return self.payload or {}

    @property
    def text(self) -> str:
        """Texto serializado do payload."""
        return json.dumps(self.payload or {})

    def raise_for_status(self) -> None:
        """Levanta erro para status >= 400."""
        if self.status_code >= 400:
            raise requests_exceptions.HTTPError(response=self)


def _load_dashboard_data_module():
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_resource = lambda func: func
    streamlit_stub.cache_data = lambda func: func
    streamlit_stub.error = MagicMock()
    sys.modules["streamlit"] = streamlit_stub

    # Limpar módulo em cache se existir
    sys.modules.pop("app.dashboard.data", None)
    sys.modules.pop("app.dashboard", None)

    import app.dashboard.data as dashboard_data

    return dashboard_data


def _context_manager_mock() -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__.return_value = ctx
    ctx.__exit__.return_value = None
    return ctx


def _make_streamlit_mock() -> MagicMock:
    st = MagicMock()
    ctx = _context_manager_mock()

    def _columns(n: int | List[Any]) -> List[MagicMock]:
        count = n if isinstance(n, int) else len(n)
        return [_context_manager_mock() for _ in range(count)]

    def _tabs(labels: List[str]) -> List[MagicMock]:
        return [_context_manager_mock() for _ in labels]

    st.sidebar = ctx
    st.spinner.return_value = ctx
    st.expander.return_value = ctx
    st.columns.side_effect = _columns
    st.tabs.side_effect = _tabs
    st.button.return_value = False
    st.radio.return_value = "🔮 Predição"

    def _number_input(*_args: Any, **_kwargs: Any) -> Any:
        return _kwargs.get("value", 0)

    def _selectbox(*_args: Any, **_kwargs: Any) -> Any:
        return _args[1][0] if len(_args) > 1 and _args[1] else None

    def _slider(*_args: Any, **_kwargs: Any) -> Any:
        return _kwargs.get("value", 0.2)

    st.number_input.side_effect = _number_input
    st.selectbox.side_effect = _selectbox
    st.slider.side_effect = _slider
    st.progress.return_value = MagicMock(progress=MagicMock(), empty=MagicMock())
    st.session_state = {}
    return st


@pytest.mark.unit
class TestDashboardConfigAndStyles:
    """Testes de config e estilos."""

    def test_configure_page_calls_streamlit(self) -> None:
        """Testa configure_page executando set_page_config."""
        from app.dashboard import config

        st_mock = _make_streamlit_mock()
        with patch.object(config, "st", st_mock):
            config.configure_page()

        st_mock.set_page_config.assert_called()

    def test_apply_custom_css_calls_markdown(self) -> None:
        """Testa aplicação de CSS customizado."""
        from app.dashboard import styles

        st_mock = _make_streamlit_mock()
        with patch.object(styles, "st", st_mock):
            styles.apply_custom_css()

        st_mock.markdown.assert_called()


@pytest.mark.unit
class TestDashboardPages:
    """Testes de renderização das páginas."""

    def test_render_about_page(self) -> None:
        """Testa renderização da página Sobre."""
        from app.dashboard.pages import about

        st_mock = _make_streamlit_mock()
        with patch.object(about, "st", st_mock):
            about.render_about_page()

        st_mock.markdown.assert_called()

    def test_render_prediction_page_high_risk(self) -> None:
        """Testa renderização da página de predição com risco."""
        from app.dashboard.pages import prediction

        st_mock = _make_streamlit_mock()

        def _number_input(*_args: Any, **_kwargs: Any) -> Any:
            return _kwargs.get("value", 0)

        st_mock.number_input.side_effect = _number_input
        predict_func = MagicMock(return_value=(1, 0.7, []))

        with patch.object(prediction, "st", st_mock), patch.object(prediction, "go", MagicMock()):
            prediction.render_prediction_page(predict_func=predict_func, api_healthy=True)

        predict_func.assert_called()

    def test_render_prediction_page_low_risk(self) -> None:
        """Testa renderização da página de predição sem risco."""
        from app.dashboard.pages import prediction

        st_mock = _make_streamlit_mock()
        predict_func = MagicMock(return_value=(0, 0.2, []))

        with patch.object(prediction, "st", st_mock), patch.object(prediction, "go", MagicMock()):
            prediction.render_prediction_page(predict_func=predict_func, api_healthy=True)

        predict_func.assert_called()

    def test_render_drift_page_with_report(self, tmp_path: Path) -> None:
        """Testa renderização de drift quando relatório existe."""
        from app.dashboard.pages import drift

        data_path = tmp_path / "data.csv"
        data_path.write_text("col1,col2\n1,2\n", encoding="utf-8")

        report_path = tmp_path / "report.html"
        report_path.write_text("<html>ok</html>", encoding="utf-8")

        st_mock = _make_streamlit_mock()
        st_mock.button.return_value = False

        with patch.object(drift, "st", st_mock):
            drift.render_drift_page(str(data_path), str(report_path))

        st_mock.markdown.assert_called()

    def test_render_drift_page_missing_report(self) -> None:
        """Testa renderização de drift quando relatório não existe."""
        from app.dashboard.pages import drift

        st_mock = _make_streamlit_mock()
        st_mock.button.return_value = False

        with (
            patch.object(drift, "st", st_mock),
            patch.object(drift.os.path, "exists", return_value=False),
        ):
            drift.render_drift_page("data.csv", "missing.html")

        st_mock.warning.assert_called()

    def test_render_metrics_page_mlflow(self) -> None:
        """Testa renderização da página de métricas com dados do MLflow."""
        from app.dashboard.pages import metrics

        st_mock = _make_streamlit_mock()
        mlflow_payload = {
            "source": "mlflow",
            "run_id": "abc123",
            "metrics": {
                "accuracy": 0.9,
                "roc_auc": 0.95,
                "f1_score": 0.9,
                "precision": 0.9,
                "recall": 0.9,
            },
            "params": {"model_type": "RF", "n_estimators": 100, "max_depth": 10},
        }

        def _requests_get(url: str, timeout: int = 10, **kwargs) -> DummyResponse:
            if "model-metrics" in url:
                return DummyResponse(200, mlflow_payload)
            return DummyResponse(200, {"ok": True}, content=b"img")

        requests_mock = MagicMock()
        requests_mock.get.side_effect = _requests_get
        requests_mock.exceptions.RequestException = requests_exceptions.RequestException
        requests_mock.exceptions.HTTPError = requests_exceptions.HTTPError

        with (
            patch.object(metrics, "st", st_mock),
            patch.object(metrics, "px", MagicMock()),
            patch.object(metrics, "requests", requests_mock),
        ):
            metrics.render_metrics_page(
                MagicMock(),
                "http://api",
                lambda: None,
                lambda *_: None,
                "roc.png",
                "imp.png",
                "report.png",
            )

        st_mock.markdown.assert_called()

    def test_render_metrics_page_local_fallback(self) -> None:
        """Testa renderização da página de métrtricas local."""
        from app.dashboard.pages import metrics

        st_mock = _make_streamlit_mock()
        df = pd.DataFrame({"IDADE": [10, 11], "TARGET": [0, 1]})
        metrics_dict = {
            "accuracy": 0.8,
            "roc_auc": 0.7,
            "report": {"weighted avg": {"precision": 0.8, "recall": 0.75}},
        }

        with (
            patch.object(metrics, "st", st_mock),
            patch.object(metrics, "px", MagicMock()),
            patch.object(metrics, "requests") as requests_mock,
            patch.object(metrics.os.path, "exists", return_value=False),
        ):
            requests_mock.exceptions = requests_exceptions
            requests_mock.get.side_effect = requests_exceptions.RequestException("fail")
            metrics.render_metrics_page(
                MagicMock(),
                "http://api",
                lambda: df,
                lambda *_: metrics_dict,
                "roc.png",
                "imp.png",
                "report.png",
            )

        st_mock.info.assert_called()

    def test_render_retrain_page_success(self) -> None:
        """Testa fluxo de retreinamento com sucesso."""
        from app.dashboard.pages import retrain

        st_mock = _make_streamlit_mock()
        st_mock.button.side_effect = [True, False, False, False]
        st_mock.session_state = {"candidate_ready": False}  # Initialize with default value

        response = DummyResponse(200, {"metrics": {"accuracy": 0.9}})

        with (
            patch.object(retrain, "st", st_mock),
            patch.object(retrain, "requests") as requests_mock,
        ):
            requests_mock.exceptions = requests_exceptions
            requests_mock.post.return_value = response
            load_dataset_func = MagicMock(return_value=pd.DataFrame({"A": [1]}))
            load_dataset_func.clear = MagicMock()
            retrain.render_retrain_page(
                MagicMock(named_steps={"classifier": MagicMock()}),
                "http://api",
                load_dataset_func,
                lambda *_: {
                    "accuracy": 0.8,
                    "roc_auc": 0.9,
                    "report": {
                        "weighted avg": {
                            "precision": 0.8,
                            "recall": 0.8,
                            "f1-score": 0.8,
                        }
                    },
                },
                MagicMock(clear=MagicMock()),
            )

        # Verificar que a função foi chamada (session_state pode ter sido atualizado)
        assert st_mock.markdown.called or st_mock.success.called

    def test_render_retrain_page_promote_discard(self) -> None:
        """Testa promoções e descartes no retreinamento."""
        from app.dashboard.pages import retrain

        st_mock = _make_streamlit_mock()
        st_mock.button.side_effect = [False, True, True, False]
        st_mock.text_input.return_value = "test-api-key"  # pragma: allowlist secret
        st_mock.session_state = {
            "candidate_ready": True,
            "candidate_metrics": {"accuracy": 0.9},
        }

        response = DummyResponse(200, {"status": "ok"})

        with (
            patch.object(retrain, "st", st_mock),
            patch.object(retrain, "requests") as requests_mock,
        ):
            requests_mock.exceptions = requests_exceptions
            requests_mock.post.return_value = response
            load_dataset_func = MagicMock(return_value=pd.DataFrame({"A": [1]}))
            load_dataset_func.clear = MagicMock()
            retrain.render_retrain_page(
                MagicMock(named_steps={"classifier": MagicMock()}),
                "http://api",
                load_dataset_func,
                lambda *_: {
                    "accuracy": 0.8,
                    "roc_auc": 0.9,
                    "report": {
                        "weighted avg": {
                            "precision": 0.8,
                            "recall": 0.8,
                            "f1-score": 0.8,
                        }
                    },
                },
                MagicMock(clear=MagicMock()),
            )

        assert requests_mock.post.called

    def test_render_sidebar_with_model(self) -> None:
        """Testa sidebar quando modelo está carregado."""
        streamlit_stub = types.ModuleType("streamlit")
        streamlit_stub.cache_resource = lambda func: func
        streamlit_stub.cache_data = lambda func: func
        streamlit_stub.error = MagicMock()
        sys.modules["streamlit"] = streamlit_stub

        # Clear modules to force re-import with mocked streamlit
        sys.modules.pop("app.dashboard.sidebar", None)
        sys.modules.pop("app.dashboard", None)

        from app.dashboard import sidebar

        st_mock = _make_streamlit_mock()
        st_mock.button.return_value = False
        st_mock.radio.return_value = "📊 Métricas do Modelo"

        model = MagicMock()
        model.feature_names_in_ = ["A", "B"]
        model.named_steps = {"classifier": MagicMock()}

        with patch.object(sidebar, "st", st_mock):
            page = sidebar.render_sidebar(
                model, MagicMock(clear=MagicMock()), MagicMock(clear=MagicMock())
            )

        assert page == "📊 Métricas do Modelo"

    def test_render_sidebar_reload(self) -> None:
        """Testa botão de recarregar na sidebar."""
        streamlit_stub = types.ModuleType("streamlit")
        streamlit_stub.cache_resource = lambda func: func
        streamlit_stub.cache_data = lambda func: func
        streamlit_stub.error = MagicMock()
        sys.modules["streamlit"] = streamlit_stub

        # Clear modules to force re-import with mocked streamlit
        sys.modules.pop("app.dashboard.sidebar", None)
        sys.modules.pop("app.dashboard", None)

        from app.dashboard import sidebar

        st_mock = _make_streamlit_mock()
        st_mock.button.return_value = True

        load_model_func = MagicMock(clear=MagicMock())
        load_dataset_func = MagicMock(clear=MagicMock())

        with patch.object(sidebar, "st", st_mock):
            sidebar.render_sidebar(None, load_model_func, load_dataset_func)

        load_model_func.clear.assert_called()
        load_dataset_func.clear.assert_called()

    def test_dashboard_script_runs(self) -> None:
        """Testa execução do script do dashboard sem falhas."""
        dashboard_path = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"

        # Create robust mocks
        streamlit_stub = types.ModuleType("streamlit")
        streamlit_stub.session_state = {}
        streamlit_stub.cache_resource = lambda *_args, **_kwargs: lambda func: func
        streamlit_stub.cache_data = lambda *_args, **_kwargs: lambda func: func
        streamlit_stub.error = MagicMock()

        # Clear all dashboard modules to force fresh import
        sys.modules.pop("app.dashboard.config", None)
        sys.modules.pop("app.dashboard.styles", None)
        sys.modules.pop("app.dashboard.data", None)
        sys.modules.pop("app.dashboard.sidebar", None)
        sys.modules.pop("app.dashboard", None)

        with (
            patch.dict(sys.modules, {"streamlit": streamlit_stub}),
            patch("app.dashboard.config.configure_page"),
            patch("app.dashboard.styles.apply_custom_css"),
            patch("app.dashboard.data.load_model", return_value=MagicMock()),
            patch("app.dashboard.sidebar.render_sidebar", return_value="🔮 Predição"),
            patch("app.dashboard.pages.prediction.render_prediction_page"),
        ):
            runpy.run_path(str(dashboard_path), run_name="__main__")
