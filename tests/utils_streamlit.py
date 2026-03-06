from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List
from unittest.mock import MagicMock


class DummyContext:
    """Context manager simples para simular componentes do Streamlit."""

    def __enter__(self) -> DummyContext:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False


class DummyProgress:
    """Simula barra de progresso do Streamlit."""

    def progress(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def empty(self) -> None:
        return None


def make_streamlit_mock() -> MagicMock:
    """
    Cria um mock configurado para o Streamlit.

    Returns:
        MagicMock: Mock configurado para uso em testes.
    """
    st = MagicMock()
    st.session_state = {}
    st.sidebar = DummyContext()

    def _columns(count: int | list[int] | tuple[int, ...]) -> List[DummyContext]:
        size = len(count) if isinstance(count, list | tuple) else int(count)
        return [DummyContext() for _ in range(size)]

    def _tabs(labels: Iterable[str]) -> List[DummyContext]:
        return [DummyContext() for _ in labels]

    st.columns.side_effect = _columns
    st.tabs.side_effect = _tabs
    st.expander.side_effect = lambda *_args, **_kwargs: DummyContext()
    st.spinner.side_effect = lambda *_args, **_kwargs: DummyContext()
    st.progress.side_effect = lambda *_args, **_kwargs: DummyProgress()
    st.components = SimpleNamespace(v1=SimpleNamespace(html=MagicMock()))

    return st
