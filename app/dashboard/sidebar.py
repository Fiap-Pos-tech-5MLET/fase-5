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
    api_healthy: bool = False,
) -> str:
    """
    Renderiza a barra lateral e retorna a página selecionada.

    Args:
        model (Any): Modelo carregado (None para usar health check via API).
        load_model_func (Callable[[], Any]): Função de carregamento do modelo.
        load_dataset_func (Callable[[], Any]): Função de carregamento do dataset.
        api_healthy (bool): Status de saúde da API. Default: False.

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

        # Mostra status da API (novo) ou do modelo (legado)
        if api_healthy:
            st.success("✅ API disponível")
            st.caption("Status: Modelo carregado na API")
        elif model is not None:
            st.success("✅ Modelo carregado")
            try:
                n_features = len(model.feature_names_in_)
                model_type = type(model.named_steps["classifier"]).__name__
                st.caption(f"**Tipo:** {model_type}")
                st.caption(f"**Features:** {n_features}")
            except (AttributeError, KeyError, TypeError):
                pass
        else:
            st.error("❌ API/Modelo não disponível")
            st.caption("Verifique a conexão e se a API está em execução.")

        st.markdown("---")

        if st.button(
            "🔄 Recarregar Modelo",
            use_container_width=True,
            help="Limpa o cache e recarrega o modelo. Use após retreinar.",
        ):
            load_model_func.clear()
            load_dataset_func.clear()
            st.rerun()

        st.caption("Última atualização: " + st.session_state.get("last_refresh", ""))

    return page
