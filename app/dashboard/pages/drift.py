"""
Página de monitoramento de drift.
"""

from __future__ import annotations

import os

import streamlit as st

from scripts.monitoring import generate_drift_report


def render_drift_page(data_path: str, drift_report_path: str, api_healthy: bool = False) -> None:
    """
    Renderiza a página de monitoramento de data drift.

    Args:
        data_path (str): Caminho do arquivo de dados.
        drift_report_path (str): Caminho do relatório de drift.
        api_healthy (bool): Status de saúde da API (não usado nesta página). Default: False.

    Returns:
        None
    """
    st.markdown("# 🔄 Monitoramento de Data Drift")
    st.markdown(
        "Análise da estabilidade das distribuições de dados ao longo do tempo usando **Evidently**."
    )

    st.markdown('<div class="section-header">Relatório de Drift</div>', unsafe_allow_html=True)

    if os.path.exists(drift_report_path):
        with open(drift_report_path, encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=800, scrolling=True)

        st.markdown("")
        st.markdown(
            '<div class="section-header">Estratégia de Monitoramento</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            <div class="info-box">
                <strong>🔍 Quando monitorar?</strong><br>
                • Semestralmente (alinhado ao ciclo escolar)<br>
                • Sempre que novos dados PEDE forem recebidos<br>
                • Quando há suspeita de mudança no perfil dos alunos
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="info-box">
                <strong>🚨 Quando retreinar?</strong><br>
                • Drift significativo em > 30% das features<br>
                • Queda observada na acurácia do modelo<br>
                • Mudança na distribuição da variável target
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        if st.button("🔄 Regenerar Relatório de Drift", type="primary"):
            with st.spinner("Gerando relatório de drift..."):
                try:
                    generate_drift_report(data_path, drift_report_path)
                    st.success(
                        "✅ Relatório regenerado com sucesso! "
                        "Recarregue a página para ver o novo relatório."
                    )
                    st.rerun()
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Erro ao gerar relatório: {exc}")
                    st.exception(exc)

    else:
        st.warning("⚠️ Relatório de drift não encontrado.")
        st.info("Execute `python scripts/monitoring.py` ou clique no botão abaixo para gerar.")

        if st.button("📊 Gerar Relatório de Drift", type="primary"):
            with st.spinner("Gerando relatório de drift..."):
                try:
                    generate_drift_report(data_path, drift_report_path)
                    st.success("✅ Relatório gerado com sucesso!")
                    st.rerun()
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Erro ao gerar relatório: {exc}")
                    st.exception(exc)
