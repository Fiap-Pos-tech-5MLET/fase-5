"""
Página de predição.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import plotly.graph_objects as go
import streamlit as st

PredictFunc = Callable[[Dict[str, Any]], Tuple[int, float]]


def render_prediction_page(predict_func: PredictFunc, api_healthy: bool = False) -> None:
    """
    Renderiza a página de predição de risco.

    Args:
        predict_func (PredictFunc): Função que chama a API para predição.
        api_healthy (bool): Status de saúde da API. Default: False.

    Returns:
        None
    """
    st.markdown("# 🔮 Predição de Risco de Defasagem")
    st.markdown("Preencha as 13 variáveis do modelo para obter a predição de risco.")

    if not api_healthy:
        st.error("⚠️ API não disponível. Verifique a conexão com o servidor de predição.")
        st.stop()

    st.markdown('<div class="section-header">Dados do Aluno</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        nivel_de_defasagem = st.number_input(
            "Nível de Defasagem", min_value=-6.0, max_value=6.0, value=0.0, step=0.1
        )
        idade = st.number_input("Idade", min_value=6, max_value=30, value=12, step=1)
        genero_label = st.selectbox("Gênero", ["Feminino", "Masculino"])
        genero = 1 if genero_label == "Masculino" else 0
        ano_de_ingresso = st.number_input(
            "Ano de Ingresso", min_value=2010, max_value=2035, value=2022, step=1
        )
        qtde_aval_realizadas = st.number_input(
            "Qtde. Avaliações Realizadas", min_value=0, max_value=20, value=4, step=1
        )

    with col2:
        veterano = st.selectbox("Veterano", ["Não", "Sim"])
        em_fase = st.selectbox("Em Fase", ["Não", "Sim"])
        iaa = st.number_input(
            "IAA (Índice de Autoavaliação)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )
        ieg = st.number_input(
            "IEG (Índice de Engajamento)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )
        ips = st.number_input(
            "IPS (Índice Psicossocial)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )

    with col3:
        ida = st.number_input(
            "IDA (Índice de Aprendizagem)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )
        ipv = st.number_input(
            "IPV (Índice de Ponto de Virada)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )
        ian = st.number_input(
            "IAN (Índice de Adequação Nivelar)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.1f",
        )

    student_data = {
        "nivel_de_defasagem": float(nivel_de_defasagem),
        "idade": float(idade),
        "genero": float(genero),
        "ano_de_ingresso": float(ano_de_ingresso),
        "veterano": 1.0 if veterano == "Sim" else 0.0,
        "em_fase": 1.0 if em_fase == "Sim" else 0.0,
        "qtde_aval_realizadas": float(qtde_aval_realizadas),
        "iaa": float(iaa),
        "ieg": float(ieg),
        "ips": float(ips),
        "ida": float(ida),
        "ipv": float(ipv),
        "ian": float(ian),
    }

    with st.expander("Ver formato JSON enviado para /predict", expanded=False):
        st.json({"data": student_data})

    with st.spinner("Processando predição via API..."):
        try:
            prediction, probability = predict_func(student_data)

            st.markdown("")

            col_result, col_gauge = st.columns([1, 1])

            with col_result:
                if prediction == 1:
                    st.markdown(
                        f"""
                    <div class="risk-high">
                        <div class="risk-title">⚠️ EM RISCO</div>
                        <div class="risk-prob" style="color: #EF4444;">{probability:.1%}</div>
                        <p class="risk-desc">Probabilidade de defasagem escolar</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="risk-low">
                        <div class="risk-title">✅ SEM RISCO</div>
                        <div class="risk-prob" style="color: #22C55E;">{1 - probability:.1%}</div>
                        <p class="risk-desc">Probabilidade de estar adequado</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with col_gauge:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={"suffix": "%", "font": {"size": 40}},
                        title={"text": "Nível de Risco", "font": {"size": 16}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": "#7C3AED"},
                            "bgcolor": "#1A1D29",
                            "steps": [
                                {"range": [0, 30], "color": "#132A1F"},
                                {"range": [30, 70], "color": "#2A2510"},
                                {"range": [70, 100], "color": "#2A1215"},
                            ],
                            "threshold": {
                                "line": {"color": "#E2E8F0", "width": 3},
                                "thickness": 0.8,
                                "value": 50,
                            },
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font={"color": "#E2E8F0"},
                    height=300,
                    margin=dict(t=60, b=30, l=30, r=30),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("")
            if prediction == 1:
                st.warning(
                    "**Recomendação:** Este aluno apresenta risco elevado de defasagem. "
                    "Sugerimos acompanhamento pedagógico reforçado e avaliação presencial."
                )
            else:
                st.info(
                    "**Resultado:** O aluno apresenta baixo risco de defasagem. "
                    "Manter o acompanhamento regular."
                )

        except (ConnectionError, RuntimeError) as exc:
            st.error(f"Erro na predição: {exc}")
