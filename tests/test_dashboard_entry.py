from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def _install_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _remove_module(name: str) -> None:
    sys.modules.pop(name, None)


def _build_stub_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _import_dashboard(page_name: str) -> tuple[types.ModuleType, dict[str, MagicMock]]:
    _remove_module("dashboard_entry")

    st = types.ModuleType("streamlit")
    st.session_state = {}
    _install_module("streamlit", st)

    calls: dict[str, MagicMock] = {
        "about": MagicMock(),
        "drift": MagicMock(),
        "metrics": MagicMock(),
        "prediction": MagicMock(),
        "retrain": MagicMock(),
    }

    _install_module(
        "app.dashboard.config",
        _build_stub_module(
            "app.dashboard.config",
            API_URL="http://api",
            CLASS_REPORT_PATH="/tmp/report.png",
            DRIFT_REPORT_PATH="/tmp/drift.html",
            FEATURE_IMP_PATH="/tmp/feat.png",
            ROC_CURVE_PATH="/tmp/roc.png",
            configure_page=MagicMock(),
        ),
    )

    _install_module(
        "app.dashboard.data",
        _build_stub_module(
            "app.dashboard.data",
            get_model_cache_buster=MagicMock(return_value=-1),
            get_model_metrics=MagicMock(),
            load_dataset=MagicMock(),
            load_model=MagicMock(return_value=MagicMock()),
            predict_via_api=MagicMock(),
        ),
    )

    _install_module(
        "app.dashboard.sidebar",
        _build_stub_module(
            "app.dashboard.sidebar",
            render_sidebar=MagicMock(return_value=page_name),
        ),
    )

    _install_module(
        "app.dashboard.styles",
        _build_stub_module(
            "app.dashboard.styles",
            apply_custom_css=MagicMock(),
        ),
    )

    _install_module(
        "app.dashboard.pages.about",
        _build_stub_module("app.dashboard.pages.about", render_about_page=calls["about"]),
    )
    _install_module(
        "app.dashboard.pages.drift",
        _build_stub_module("app.dashboard.pages.drift", render_drift_page=calls["drift"]),
    )
    _install_module(
        "app.dashboard.pages.metrics",
        _build_stub_module("app.dashboard.pages.metrics", render_metrics_page=calls["metrics"]),
    )
    _install_module(
        "app.dashboard.pages.prediction",
        _build_stub_module(
            "app.dashboard.pages.prediction", render_prediction_page=calls["prediction"]
        ),
    )
    _install_module(
        "app.dashboard.pages.retrain",
        _build_stub_module("app.dashboard.pages.retrain", render_retrain_page=calls["retrain"]),
    )

    dashboard_path = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
    loader = importlib.machinery.SourceFileLoader("dashboard_entry", str(dashboard_path))
    spec = importlib.util.spec_from_loader("dashboard_entry", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("Não foi possível carregar app/dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_dashboard_routes_to_prediction() -> None:
    _module, calls = _import_dashboard("🔮 Predição")

    calls["prediction"].assert_called_once()


def test_dashboard_routes_to_metrics() -> None:
    _module, calls = _import_dashboard("📊 Métricas do Modelo")

    calls["metrics"].assert_called_once()


def test_dashboard_routes_to_drift() -> None:
    _module, calls = _import_dashboard("🔄 Monitoramento de Drift")

    calls["drift"].assert_called_once()


def test_dashboard_routes_to_retrain() -> None:
    _module, calls = _import_dashboard("⚙️ Retreinamento")

    calls["retrain"].assert_called_once()


def test_dashboard_routes_to_about() -> None:
    _module, calls = _import_dashboard("ℹ️ Sobre o Projeto")

    calls["about"].assert_called_once()
