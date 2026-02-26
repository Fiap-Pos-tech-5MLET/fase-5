"""
Página de retreinamento.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Optional

import pandas as pd
import requests
import streamlit as st

MetricsFunc = Callable[[Any, Optional[pd.DataFrame]], Optional[Dict[str, Any]]]
DatasetFunc = Callable[[], Optional[pd.DataFrame]]
CacheClearFunc = Callable[[], Any]


def render_retrain_page(
    model: Any,
    api_url: str,
    load_dataset_func: DatasetFunc,
    metrics_func: MetricsFunc,
    load_model_func: CacheClearFunc,
) -> None:
    """
    Renderiza a página de retreinamento e comparação de modelos.

    Args:
        model (Any): Modelo carregado.
        api_url (str): URL da API.
        load_dataset_func (DatasetFunc): Função de carregamento do dataset.
        metrics_func (MetricsFunc): Função de cálculo de métricas.
        load_model_func (CacheClearFunc): Função de cache do modelo para limpar.

    Returns:
        None
    """
    st.markdown("# ⚙️ Retreinamento do Modelo")
    st.markdown("Execute o pipeline de treinamento completo e compare as métricas antes e depois.")

    st.markdown('<div class="section-header">Modelo Atual</div>', unsafe_allow_html=True)

    if model is not None:
        col_info1, col_info2, col_info3 = st.columns(3)

        df_current = load_dataset_func()
        current_metrics = metrics_func(model, df_current) if df_current is not None else None

        with col_info1:
            try:
                model_type = type(model.named_steps["classifier"]).__name__
            except (AttributeError, KeyError, TypeError):
                model_type = "Desconhecido"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.4rem;">{model_type}</div>
                <div class="metric-label">Tipo do Modelo</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_info2:
            acc = f"{current_metrics['accuracy']:.1%}" if current_metrics else "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{acc}</div>
                <div class="metric-label">Accuracy Atual</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_info3:
            roc = f"{current_metrics['roc_auc']:.4f}" if current_metrics else "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{roc}</div>
                <div class="metric-label">ROC-AUC Atual</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ Nenhum modelo carregado. O treinamento criará um do zero.")
        current_metrics = None

    st.markdown("")
    st.markdown('<div class="section-header">Executar Retreinamento</div>', unsafe_allow_html=True)

    st.markdown(
        """
    O pipeline executa automaticamente:
    1. **Carga** dos dados brutos (PEDE 2024)
    2. **Limpeza** e tratamento de valores faltantes
    3. **Feature Engineering** (criação de variáveis derivadas)
    4. **Treinamento** com Random Forest + SelectKBest
    5. **Avaliação** de métricas (Accuracy, ROC-AUC, F1, Precision, Recall)
    6. **Salvamento** do modelo e registro no MLflow
    """
    )

    st.markdown('<div class="section-header">Hiperparâmetros</div>', unsafe_allow_html=True)

    hp1, hp2, hp3 = st.columns(3)

    with hp1:
        n_estimators = st.number_input(
            "Nº de Árvores (n_estimators)",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help=(
                "Número de árvores na Random Forest. Mais árvores = maior precisão, mas mais lento."
            ),
        )
        max_depth_option = st.selectbox(
            "Profundidade Máxima (max_depth)",
            ["Sem limite", "5", "10", "15", "20", "30", "50"],
            index=0,
            help="Limitar a profundidade pode reduzir overfitting.",
        )
        max_depth = None if max_depth_option == "Sem limite" else int(max_depth_option)

    with hp2:
        min_samples_split = st.number_input(
            "Min. Amostras para Split",
            min_value=2,
            max_value=50,
            value=2,
            step=1,
            help="Mínimo de amostras necessário para dividir um nó interno.",
        )
        k_option = st.selectbox(
            "Features Selecionadas (SelectKBest)",
            ["Todas", "5", "10", "15", "20", "25"],
            index=0,
            help="Selecionar apenas as K features mais relevantes.",
        )
        k_value = "all" if k_option == "Todas" else int(k_option)

    with hp3:
        test_size = st.slider(
            "Proporção de Teste (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentual dos dados reservados para teste.",
        )
        test_size_float = test_size / 100.0

        st.markdown("")
        depth_label = "ilimitada" if max_depth is None else max_depth
        summary_html = (
            '<div class="info-box">'
            "<strong>Resumo da configuração:</strong><br>"
            f"• {n_estimators} árvores, profundidade {depth_label}<br>"
            f"• Min. split: {min_samples_split}, Features: {k_option}<br>"
            f"• Teste: {test_size}% dos dados"
            "</div>"
        )
        st.markdown(summary_html, unsafe_allow_html=True)

    requested_by = st.text_input(
        "Solicitado por (nome ou e-mail)",
        value="",
        max_chars=120,
        help="Obrigatório para auditoria de governança do retreinamento.",
    ).strip()

    st.markdown("")

    col_btn, col_warn = st.columns([1, 2])
    with col_btn:
        retrain_btn = st.button(
            "🚀 Iniciar Retreinamento", use_container_width=True, type="primary"
        )
    with col_warn:
        st.caption(
            "⏱️ O treinamento leva aproximadamente 10-30 segundos. "
            "O modelo candidato será salvo separadamente para avaliação."
        )

    if retrain_btn:
        st.markdown("---")

        if len(requested_by) < 3:
            st.error("❌ Informe ao menos 3 caracteres no campo 'Solicitado por'.")
            return

        progress_bar = st.progress(0, text="Iniciando pipeline...")

        try:
            progress_bar.progress(20, text="📡 Enviando requisição para a API...")

            retrain_payload = {
                "requested_by": requested_by,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "k": k_value if k_value != "all" else None,
                "test_size": test_size_float,
            }

            progress_bar.progress(
                40,
                text=f"🤖 Treinando modelo candidato via API ({n_estimators} árvores)...",
            )

            response = requests.post(
                f"{api_url}/retrain",
                json=retrain_payload,
                timeout=120,
            )

            progress_bar.progress(80, text="📊 Processando resultados...")

            if response.status_code != 200:
                error_detail = ""
                try:
                    error_detail = response.json().get("detail", response.text)
                except ValueError:
                    error_detail = response.text
                raise RuntimeError(f"API retornou erro {response.status_code}: {error_detail}")

            result = response.json()
            metrics_dict = result.get("metrics", {})

            progress_bar.progress(100, text="✅ Modelo candidato treinado!")

            st.session_state["candidate_metrics"] = metrics_dict
            st.session_state["candidate_ready"] = True

        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            st.error(
                f"❌ Não foi possível conectar à API em `{api_url}`. "
                "Certifique-se de que a API está rodando (`uvicorn app.main:app`)."
            )
        except RuntimeError as exc:
            progress_bar.empty()
            st.error(f"❌ Erro durante o treinamento: {exc}")
            st.code(traceback.format_exc(), language="text")

    if st.session_state.get("candidate_ready", False):
        metrics_dict = st.session_state.get("candidate_metrics", {})

        st.markdown("")
        st.markdown(
            '<div class="section-header">Champion vs Challenger</div>',
            unsafe_allow_html=True,
        )

        st.warning(
            "⚠️ Um modelo **candidato** foi treinado mas **ainda não está em produção**. "
            "Compare as métricas abaixo e decida se deseja promovê-lo."
        )

        new_acc = metrics_dict.get("accuracy", 0)
        new_roc = metrics_dict.get("roc_auc", 0)
        new_f1 = metrics_dict.get("f1_score", 0)
        new_prec = metrics_dict.get("precision", 0)
        new_rec = metrics_dict.get("recall", 0)

        if current_metrics:
            old_acc = current_metrics["accuracy"]
            old_roc = current_metrics["roc_auc"]
            old_f1 = current_metrics["report"]["weighted avg"]["f1-score"]
            old_prec = current_metrics["report"]["weighted avg"]["precision"]
            old_rec = current_metrics["report"]["weighted avg"]["recall"]

            comparison_data = {
                "Métrica": ["Accuracy", "ROC-AUC", "F1-Score", "Precision", "Recall"],
                "🏆 Champion (Atual)": [
                    f"{old_acc:.4f}",
                    f"{old_roc:.4f}",
                    f"{old_f1:.4f}",
                    f"{old_prec:.4f}",
                    f"{old_rec:.4f}",
                ],
                "🥊 Challenger (Novo)": [
                    f"{new_acc:.4f}",
                    f"{new_roc:.4f}",
                    f"{new_f1:.4f}",
                    f"{new_prec:.4f}",
                    f"{new_rec:.4f}",
                ],
                "Variação": [
                    f"{(new_acc - old_acc):+.4f}",
                    f"{(new_roc - old_roc):+.4f}",
                    f"{(new_f1 - old_f1):+.4f}",
                    f"{(new_prec - old_prec):+.4f}",
                    f"{(new_rec - old_rec):+.4f}",
                ],
            }
        else:
            comparison_data = {
                "Métrica": ["Accuracy", "ROC-AUC", "F1-Score", "Precision", "Recall"],
                "🏆 Champion (Atual)": ["—", "—", "—", "—", "—"],
                "🥊 Challenger (Novo)": [
                    f"{new_acc:.4f}",
                    f"{new_roc:.4f}",
                    f"{new_f1:.4f}",
                    f"{new_prec:.4f}",
                    f"{new_rec:.4f}",
                ],
                "Variação": ["—", "—", "—", "—", "—"],
            }

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("")
        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.6rem;">{new_acc:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.6rem;">{new_roc:.4f}</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.6rem;">{new_f1:.1%}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.6rem;">{new_prec:.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.6rem;">{new_rec:.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown('<div class="section-header">Decisão</div>', unsafe_allow_html=True)

        col_promote, col_discard = st.columns(2)

        with col_promote:
            if st.button(
                "✅ Promover Modelo Candidato",
                use_container_width=True,
                type="primary",
                help="Substitui o modelo atual pelo candidato e recarrega a API.",
            ):
                with st.spinner("Promovendo modelo..."):
                    try:
                        resp = requests.post(f"{api_url}/promote", timeout=30)
                        resp.raise_for_status()

                        load_model_func.clear()
                        load_dataset_func.clear()
                        st.session_state["candidate_ready"] = False
                        st.session_state.pop("candidate_metrics", None)

                        st.success("🏆 Modelo candidato **promovido** para produção com sucesso!")
                        st.info(
                            "💡 O modelo em produção foi atualizado. As demais páginas do dashboard "
                            "já refletem o novo modelo. A decisão foi registrada nos logs."
                        )
                        st.rerun()
                    except requests.exceptions.RequestException as exc:
                        st.error(f"❌ Erro ao promover: {exc}")

        with col_discard:
            if st.button(
                "❌ Descartar Candidato",
                use_container_width=True,
                help="Remove o modelo candidato e mantém o modelo atual em produção.",
            ):
                with st.spinner("Descartando modelo candidato..."):
                    try:
                        resp = requests.post(f"{api_url}/discard", timeout=30)
                        resp.raise_for_status()

                        st.session_state["candidate_ready"] = False
                        st.session_state.pop("candidate_metrics", None)

                        st.info(
                            "🗑️ Modelo candidato **descartado**. "
                            "O modelo atual em produção foi mantido."
                        )
                        st.rerun()
                    except requests.exceptions.RequestException as exc:
                        st.error(f"❌ Erro ao descartar: {exc}")
