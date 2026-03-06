"""
Página de métricas do modelo.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app.dashboard.config import DASHBOARD_REQUESTED_BY

MetricsFunc = Callable[[Any, Optional[pd.DataFrame]], Optional[Dict[str, Any]]]
DatasetFunc = Callable[[], Optional[pd.DataFrame]]


def render_metrics_page(
    model: Any,
    api_url: str,
    load_dataset_func: DatasetFunc,
    metrics_func: MetricsFunc,
    roc_curve_path: str,
    feature_imp_path: str,
    class_report_path: str,
) -> None:
    """
    Renderiza a página de métricas do modelo.

    Args:
        model (Any): Modelo carregado.
        api_url (str): URL da API.
        load_dataset_func (DatasetFunc): Função para carregar dataset.
        metrics_func (MetricsFunc): Função para calcular métricas.
        roc_curve_path (str): Caminho do artefato ROC local.
        feature_imp_path (str): Caminho do artefato de importância local.
        class_report_path (str): Caminho do report local.

    Returns:
        None
    """
    st.markdown("# 📊 Métricas do Modelo")
    st.markdown("Desempenho do modelo em produção — métricas e artefatos do MLflow.")

    mlflow_metrics: Optional[Dict[str, Any]] = None
    mlflow_source: Optional[str] = None

    try:
        resp = requests.get(
            f"{api_url}/model-metrics",
            timeout=10,
            headers={"x-requested-by": DASHBOARD_REQUESTED_BY},
        )
        if resp.status_code == 200:
            mlflow_data = resp.json()
            mlflow_source = mlflow_data.get("source")
            if mlflow_source == "mlflow" and mlflow_data.get("metrics"):
                mlflow_metrics = mlflow_data
    except requests.exceptions.RequestException:
        mlflow_metrics = None

    if mlflow_metrics:
        metrics = mlflow_metrics["metrics"]
        params = mlflow_metrics.get("params", {})
        run_id = mlflow_metrics.get("run_id", "")

        st.markdown(
            '<div class="section-header">Informações da Run (MLflow)</div>',
            unsafe_allow_html=True,
        )

        info_cols = st.columns(4)
        with info_cols[0]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 0.9rem;">{run_id[:8]}...</div>
                <div class="metric-label">Run ID</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with info_cols[1]:
            model_type = params.get("model_type", "N/A")
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 0.9rem;">{model_type}</div>
                <div class="metric-label">Tipo do Modelo</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with info_cols[2]:
            n_est = params.get("n_estimators", "N/A")
            max_d = params.get("max_depth", "Ilimitada")
            hyperparams_label = f"{n_est} árvores / depth={max_d}"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 0.9rem;">{hyperparams_label}</div>
                <div class="metric-label">Hiperparâmetros</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with info_cols[3]:
            n_samples = params.get("n_samples", "N/A")
            n_feat = params.get("n_features", "N/A")
            train_data_label = f"{n_samples} amostras / {n_feat} features"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 0.9rem;">{train_data_label}</div>
                <div class="metric-label">Dados de Treino</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.caption(f"📋 Fonte: MLflow | Run ID: `{run_id}`")

        st.markdown("")

        st.markdown(
            '<div class="section-header">Indicadores de Performance</div>',
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get("accuracy", 0):.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get("roc_auc", 0):.4f}</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get("f1_score", 0):.1%}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get("precision", 0):.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k5:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get("recall", 0):.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-header">Curva ROC</div>', unsafe_allow_html=True)
            try:
                img_resp = requests.get(
                    f"{api_url}/model-artifact/roc_curve.png",
                    timeout=10,
                    headers={"x-requested-by": DASHBOARD_REQUESTED_BY},
                )
                if img_resp.status_code == 200:
                    st.image(img_resp.content, use_container_width=True)
                else:
                    st.info("Artefato de curva ROC não disponível.")
            except requests.exceptions.RequestException:
                if os.path.exists(roc_curve_path):
                    st.image(roc_curve_path)
                else:
                    st.info("Artefato de curva ROC não disponível.")

        with col_right:
            st.markdown(
                '<div class="section-header">Classification Report</div>',
                unsafe_allow_html=True,
            )
            try:
                img_resp = requests.get(
                    f"{api_url}/model-artifact/classification_report.png",
                    timeout=10,
                    headers={"x-requested-by": DASHBOARD_REQUESTED_BY},
                )
                if img_resp.status_code == 200:
                    st.image(img_resp.content, use_container_width=True)
                else:
                    st.info("Artefato de classification report não disponível.")
            except requests.exceptions.RequestException:
                report_local = os.path.join(
                    os.path.dirname(class_report_path),
                    "classification_report.png",
                )
                if os.path.exists(report_local):
                    st.image(report_local)
                else:
                    st.info("Artefato de classification report não disponível.")

        st.markdown(
            '<div class="section-header">Importância das Features</div>',
            unsafe_allow_html=True,
        )
        try:
            img_resp = requests.get(
                f"{api_url}/model-artifact/feature_importance.png",
                timeout=10,
                headers={"x-requested-by": DASHBOARD_REQUESTED_BY},
            )
            if img_resp.status_code == 200:
                st.image(img_resp.content, use_container_width=True)
            else:
                st.info("Artefato de feature importance não disponível.")
        except requests.exceptions.RequestException:
            if os.path.exists(feature_imp_path):
                st.image(feature_imp_path)
            else:
                st.info("Artefato de feature importance não disponível.")

        df = load_dataset_func()
        if df is not None:
            st.markdown(
                '<div class="section-header">Distribuição do Dataset</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                target_counts = df["TARGET"].value_counts().reset_index()
                target_counts.columns = ["Target", "Count"]
                target_counts["Label"] = target_counts["Target"].map(
                    {0: "Sem Risco", 1: "Em Risco"}
                )
                fig = px.pie(
                    target_counts,
                    values="Count",
                    names="Label",
                    color_discrete_sequence=["#22C55E", "#EF4444"],
                    title="Distribuição da Variável Target",
                )
                fig.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font={"color": "#E2E8F0"},
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if "IDADE" in df.columns:
                    fig = px.histogram(
                        df,
                        x="IDADE",
                        color="TARGET",
                        barmode="overlay",
                        color_discrete_map={0: "#22C55E", 1: "#EF4444"},
                        labels={"TARGET": "Risco", "IDADE": "Idade"},
                        title="Distribuição de Idade por Risco",
                    )
                    fig.update_layout(
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        font={"color": "#E2E8F0"},
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif mlflow_source == "local":
        st.warning(
            "⚠️ Nenhuma run do MLflow vinculada ao modelo em produção. "
            "Execute um retreinamento e promova o modelo para vincular as métricas do MLflow."
        )
        st.info(
            "💡 Dica: Vá para a aba **⚙️ Retreinamento**, treine um modelo e clique em **Promover**."
        )

    else:
        st.info("📡 Não foi possível conectar à API. Exibindo métricas calculadas localmente.")

        df = load_dataset_func()
        metrics = metrics_func(model, df) if model is not None else None

        if metrics:
            st.markdown(
                '<div class="section-header">Indicadores de Performance (Local)</div>',
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics["accuracy"]:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics["roc_auc"]:.4f}</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with k3:
                weighted_avg = metrics["report"]["weighted avg"]
                precision_value = weighted_avg["precision"]
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{precision_value:.1%}</div>
                    <div class="metric-label">Precision</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with k4:
                recall_value = weighted_avg["recall"]
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{recall_value:.1%}</div>
                    <div class="metric-label">Recall</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            if os.path.exists(roc_curve_path):
                st.image(roc_curve_path)
            if os.path.exists(feature_imp_path):
                st.image(feature_imp_path)
        else:
            st.warning("⚠️ Modelo ou dados não disponíveis para calcular métricas.")
