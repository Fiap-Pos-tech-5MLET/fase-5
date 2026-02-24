"""
Página Sobre o Projeto.
"""

from __future__ import annotations

import streamlit as st


def render_about_page() -> None:
    """
    Renderiza a página sobre o projeto.

    Returns:
        None
    """
    st.markdown("# ℹ️ Sobre o Projeto")

    st.markdown(
        """
    ### Datathon — Machine Learning Engineering
    **PósTech FIAP — Fase 5**

    Sistema de predição de risco de defasagem escolar para alunos da
    **Associação Passos Mágicos** (Embu-Guaçu/SP).
    """
    )

    st.markdown(
        '<div class="section-header">Arquitetura do Sistema</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.6rem;">🔧</div>
            <div class="card-title">Pipeline ML</div>
            <p class="card-desc">Limpeza → Feature Engineering → Treinamento → Avaliação</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.6rem;">🚀</div>
            <div class="card-title">API FastAPI</div>
            <p class="card-desc">/predict, /drift, /model-info, /retrain</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.6rem;">📊</div>
            <div class="card-title">Monitoramento</div>
            <p class="card-desc">MLflow + Evidently + Logging Estruturado</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Modelo</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            """
        | Componente | Tecnologia |
        |-----------|-----------|
        | **Algoritmo** | Random Forest Classifier |
        | **Preprocessing** | StandardScaler + OneHotEncoder |
        | **Feature Selection** | SelectKBest (f_classif) |
        | **Serialização** | joblib |
        | **Experimentos** | MLflow |
        """
        )

    with m2:
        st.markdown(
            """
        | Métrica | Valor |
        |---------|-------|
        | **Accuracy** | 93.5% |
        | **F1-Score** | 93.5% |
        | **Precision** | 93.6% |
        | **Recall** | 93.5% |
        | **ROC-AUC** | 0.9897 |
        """
        )

    st.markdown('<div class="section-header">Stack Tecnológica</div>', unsafe_allow_html=True)

    st.markdown(
        """
    | Camada | Tecnologias |
    |--------|-----------|
    | **ML** | Python 3.12+, scikit-learn, pandas, numpy |
    | **API** | FastAPI, Uvicorn |
    | **Dashboard** | Streamlit, Plotly |
    | **Container** | Docker |
    | **Testes** | pytest (92% cobertura) |
    | **Monitoring** | Evidently, MLflow, Logging Python |
    """
    )

    st.markdown(
        '<div class="section-header">Estratégia de Retreinamento</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    1. 📥 **Receber** novos dados PEDE do período escolar
    2. 🔍 **Verificar** drift via Evidently (aba Monitoramento de Drift)
    3. 🔄 **Retreinar** se drift > 30% das features
    4. 📊 **Comparar** métricas novo vs. produção
    5. ✅ **Promover** se performance igual ou superior
    6. 🔙 **Rollback** via versionamento MLflow se necessário
    """
    )

    st.markdown('<div class="section-header">Cenários de Produção</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🆕 Aluno Novo", "📉 Degradação", "❓ Dados Incompletos"])

    with tab1:
        st.markdown(
            """
        **Aluno sem histórico (primeiro ano na associação):**
        - Campos INDE_22 e INDE_23 preenchidos com 0
        - Flag `HAS_HISTORY_23 = 0` sinaliza ausência de dados prévios
        - Predição baseada em dados demográficos e escolares
        - **Recomendação:** Acompanhamento presencial reforçado
        """
        )

    with tab2:
        st.markdown(
            """
        **Mudança no perfil dos alunos ao longo do tempo:**
        - Monitoramento contínuo via Data Drift (Evidently)
        - **Threshold:** drift em > 30% das features dispara alerta
        - **Ação:** Retreinar modelo com dados atualizados
        - **Rollback:** Versão anterior mantida via MLflow
        """
        )

    with tab3:
        st.markdown(
            """
        **Input da API com campos faltantes ou inválidos:**
        - Todos os campos são opcionais
        - Numéricos faltantes → preenchidos com 0
        - Categóricos faltantes → preenchidos com "UNKNOWN"
        - OneHotEncoder trata categorias desconhecidas automaticamente
        - Modelo retorna predição mesmo com dados parciais
        """
        )

    st.markdown('<div class="section-header">Desenvolvedores</div>', unsafe_allow_html=True)

    st.markdown(
        "\n".join(
            [
                "**Turma 5MLET — FIAP Pós Tech**",
                "",
                "| Nome | RM | GitHub |",
                "|------|----|--------|",
                "| Antônio Teixeira Santana Neto | RM364480 | [@antonioteixeirasn]("
                "https://github.com/antonioteixeirasn) |",
                "| Erik Douglas Alves Gomes | RM364379 | [@Erik-DAG]("
                "https://github.com/Erik-DAG) |",
                "| Gabriela Moreno Rocha dos Santos | RM364538 | [@gabrielaMSantos]("
                "https://github.com/gabrielaMSantos) |",
                "| Leonardo Fernandes Soares | RM364648 | [@leferso](https://github.com/leferso) |",
                "| Lucas Felipe de Jesus Machado | RM364306 | [@lfjmachado](https://github.com/"
                "lfjmachado) |",
            ]
        )
    )
