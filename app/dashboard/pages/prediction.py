"""
Página de predição.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import plotly.graph_objects as go
import streamlit as st

PredictFunc = Callable[[Dict[str, Any]], Tuple[int, float]]


def render_prediction_page(model: Any, predict_func: PredictFunc) -> None:
    """
    Renderiza a página de predição de risco.

    Args:
        model (Any): Modelo carregado.
        predict_func (PredictFunc): Função que chama a API para predição.

    Returns:
        None
    """
    st.markdown("# 🔮 Predição de Risco de Defasagem")
    st.markdown("Preencha os dados do aluno para obter a predição de risco de defasagem escolar.")

    if model is None:
        st.error("⚠️ Modelo não carregado. Execute `python scripts/train.py` primeiro.")
        st.stop()

    st.markdown('<div class="section-header">Dados do Aluno</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        idade = st.number_input("Idade", min_value=6, max_value=30, value=12, step=1)
        genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
        fase = st.selectbox(
            "Fase",
            [
                "ALFA",
                "1A",
                "1B",
                "2A",
                "2B",
                "3A",
                "3B",
                "4A",
                "4B",
                "5A",
                "5B",
                "6A",
                "7A",
                "8A",
                "8B",
                "9",
            ],
        )

    with col2:
        ano_ingresso = st.number_input(
            "Ano de Ingresso", min_value=2016, max_value=2024, value=2022, step=1
        )
        fase_ideal = st.selectbox(
            "Fase Ideal",
            [
                "ALFA (1° e 2° ano)",
                "Fase 1 (3° e 4° ano)",
                "Fase 2 (5° e 6° ano)",
                "Fase 3 (7° e 8° ano)",
                "Fase 4 (9° ano)",
                "Fase 5 (1° EM)",
                "Fase 6 (2° EM)",
                "Fase 7 (3° EM)",
                "Fase 8 (Universitários)",
            ],
        )
        ativo = st.selectbox("Status", ["Cursando", "Inativo"])

    with col3:
        inde_22 = st.number_input(
            "INDE 2022",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            format="%.1f",
        )
        inde_23 = st.number_input(
            "INDE 2023",
            min_value=0.0,
            max_value=10.0,
            value=5.5,
            step=0.1,
            format="%.1f",
        )

    st.markdown(
        '<div class="section-header">Performance Histórica (Opcional)</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📋 Pedras e Conceitos (clique para expandir)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pedra_20 = st.number_input(
                "Pedra 2020", min_value=0.0, max_value=10.0, value=0.0, step=0.1
            )
            pedra_21 = st.number_input(
                "Pedra 2021", min_value=0.0, max_value=10.0, value=0.0, step=0.1
            )
        with c2:
            pedra_22 = st.number_input(
                "Pedra 2022", min_value=0.0, max_value=10.0, value=0.0, step=0.1
            )
            pedra_23 = st.number_input(
                "Pedra 2023", min_value=0.0, max_value=10.0, value=0.0, step=0.1
            )
        with c3:
            cg = st.number_input(
                "CG (Conceito Geral)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
            )
            cf = st.number_input(
                "CF (Conceito Final)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
            )
        with c4:
            ct = st.number_input(
                "CT (Conceito Total)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
            )
            n_av = st.number_input("Nº Avaliações", min_value=0, max_value=6, value=0, step=1)

    student_data = {
        "IDADE": idade,
        "GENERO": genero,
        "FASE": fase,
        "ANO_INGRESSO": ano_ingresso,
        "FASE_IDEAL": fase_ideal,
        "ATIVO/_INATIVO": ativo,
        "INDE_22": inde_22,
        "INDE_23": inde_23,
        "PEDRA_20": pedra_20,
        "PEDRA_21": pedra_21,
        "PEDRA_22": pedra_22,
        "PEDRA_23": pedra_23,
        "CG": cg,
        "CF": cf,
        "CT": ct,
        "Nº_AV": float(n_av),
    }

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
