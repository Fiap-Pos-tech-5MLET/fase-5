"""
Sidebar do dashboard.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st


def render_sidebar(
    model: Any,
    load_model_func: Callable[[], Any],
    load_dataset_func: Callable[[], Any],
) -> str:
    """
    Renderiza a barra lateral e retorna a página selecionada.

    Args:
        model (Any): Modelo carregado.
        load_model_func (Callable[[], Any]): Função de carregamento do modelo.
        load_dataset_func (Callable[[], Any]): Função de carregamento do dataset.

    Returns:
        str: Página selecionada.
    """
    with st.sidebar:
        st.markdown("## 🎓 Passos Mágicos")
        st.markdown("##### Predição de Risco Escolar")
        st.markdown("---")

        page = st.radio(
            "Navegação",
            [
                "🔮 Predição",
                "📊 Métricas do Modelo",
                "🔄 Monitoramento de Drift",
                "⚙️ Retreinamento",
                "ℹ️ Sobre o Projeto",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if model is not None:
            st.success("✅ Modelo carregado")
            try:
                n_features = len(model.feature_names_in_)
                model_type = type(model.named_steps["classifier"]).__name__
                st.caption(f"**Tipo:** {model_type}")
                st.caption(f"**Features:** {n_features}")
            except (AttributeError, KeyError, TypeError):
                pass
        else:
            st.error("❌ Modelo não encontrado")

        st.markdown("---")

        if st.button(
            "🔄 Recarregar Modelo",
            use_container_width=True,
            help="Limpa o cache e recarrega o modelo do disco. Use após retreinar.",
        ):
            load_model_func.clear()
            load_dataset_func.clear()
            st.success("Cache limpo! Recarregando...")
            st.rerun()

        st.caption("Última atualização: " + st.session_state.get("last_refresh", ""))

    return page
