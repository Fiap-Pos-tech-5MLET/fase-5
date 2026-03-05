from __future__ import annotations

import sys
import types

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

import app.dashboard.styles as dashboard_styles
from tests.utils_streamlit import make_streamlit_mock


def test_apply_custom_css_calls_markdown(monkeypatch) -> None:
    """Deve aplicar CSS usando st.markdown."""
    st = make_streamlit_mock()
    monkeypatch.setattr(dashboard_styles, "st", st)

    dashboard_styles.apply_custom_css()

    st.markdown.assert_called_once()
    args, kwargs = st.markdown.call_args
    assert "<style>" in args[0]
    assert kwargs.get("unsafe_allow_html") is True
