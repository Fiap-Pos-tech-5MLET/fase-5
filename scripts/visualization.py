"""
Funções de visualização para o projeto Passos Mágicos.

Este módulo contém funções auxiliares para criação de gráficos e visualizações
comuns em análises exploratórias de dados educacionais.

Funções principais:
- Gráficos de contagem com porcentagens
- Análise de correlação visual
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_exact_counter(
    df: pd.DataFrame,
    coluna: str,
    titulo: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    rotation: int = 45,
) -> plt.Figure:
    """
    Cria gráfico de barras com contagens e porcentagens exatas para uma coluna categórica.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        coluna (str): Nome da coluna a visualizar.
        titulo (Optional[str]): Título do gráfico. Se None, usa o nome da coluna.
        figsize (Tuple[int, int]): Tamanho da figura (largura, altura).
        rotation (int): Ângulo de rotação dos rótulos do eixo x.

    Returns:
        plt.Figure: Objeto Figure do matplotlib.

    Examples:
        >>> df = pd.DataFrame({"FASE": ["Fase 1", "Fase 2", "Fase 1", "Fase 3"]})
        >>> fig = plot_exact_counter(df, "FASE", titulo="Distribuição por Fase")
        >>> plt.close(fig)  # Fecha para não exibir em testes
    """
    if titulo is None:
        titulo = f"Distribuição de {coluna}"

    # Contagens e porcentagens
    contagens = df[coluna].value_counts().reset_index()
    contagens.columns = [coluna, "count"]
    total = contagens["count"].sum()
    contagens["percentage"] = (contagens["count"] / total) * 100

    # Criação do gráfico
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        contagens[coluna].astype(str),
        contagens["count"],
        color=sns.color_palette("viridis", len(contagens)),
        edgecolor="black",
        linewidth=1.2,
    )

    # Anotações com valor e porcentagem
    for bar, count, pct in zip(bars, contagens["count"], contagens["percentage"], strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(coluna, fontsize=12, fontweight="bold")
    ax.set_ylabel("Contagem", fontsize=12, fontweight="bold")
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def analyse_corr(
    df: pd.DataFrame,
    colunas: Optional[List[str]] = None,
    threshold: float = 0.3,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Matriz de Correlação",
) -> plt.Figure:
    """
    Cria heatmap de correlação e identifica correlações fortes.

    Args:
        df (pd.DataFrame): DataFrame com features numéricas.
        colunas (Optional[List[str]]): Lista de colunas a incluir. Se None, usa todas numéricas.
        threshold (float): Threshold absoluto para destacar correlações fortes (padrão: 0.3).
        figsize (Tuple[int, int]): Tamanho da figura.
        title (str): Título do gráfico.

    Returns:
        plt.Figure: Objeto Figure do matplotlib.

    Examples:
        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3, 4, 5],
        ...     "B": [2, 4, 6, 8, 10],
        ...     "C": [5, 4, 3, 2, 1]
        ... })
        >>> fig = analyse_corr(df, threshold=0.5)
        >>> plt.close(fig)  # Fecha para não exibir em testes
    """
    # Seleciona colunas
    df_corr = df.select_dtypes(include=[np.number]) if colunas is None else df[colunas]

    # Calcula correlação
    corr_matrix = df_corr.corr()

    # Cria figura
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    # Identifica correlações fortes (acima do threshold)
    print(f"\n{'='*60}")
    print(f"CORRELAÇÕES FORTES (|r| > {threshold}):")
    print(f"{'='*60}")

    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                strong_corrs.append((var1, var2, corr_value))

    # Ordena por valor absoluto decrescente
    strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    if strong_corrs:
        for var1, var2, corr_val in strong_corrs:
            direction = "positiva" if corr_val > 0 else "negativa"
            print(f"{var1:20} <-> {var2:20} : {corr_val:6.3f} ({direction})")
    else:
        print(f"Nenhuma correlação acima de |{threshold}| encontrada.")

    print(f"{'='*60}\n")

    return fig


def plot_distribuicao_target(
    y: pd.Series,
    labels: Optional[List[str]] = None,
    titulo: str = "Distribuição da Variável Target",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Cria gráfico de pizza para visualizar distribuição da variável target.

    Args:
        y (pd.Series): Série contendo a variável target.
        labels (Optional[List[str]]): Rótulos customizados. Se None, usa valores únicos.
        titulo (str): Título do gráfico.
        figsize (Tuple[int, int]): Tamanho da figura.

    Returns:
        plt.Figure: Objeto Figure do matplotlib.

    Examples:
        >>> y = pd.Series([0, 0, 0, 1, 1])
        >>> fig = plot_distribuicao_target(y, labels=["Não Evadiu", "Evadiu"])
        >>> plt.close(fig)
    """
    contagens = y.value_counts().sort_index()

    if labels is None:
        labels = [str(val) for val in contagens.index]

    fig, ax = plt.subplots(figsize=figsize)

    _wedges, _texts, autotexts = ax.pie(
        contagens.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(contagens)),
        textprops={"fontsize": 12, "fontweight": "bold"},
    )

    # Adiciona contagem absoluta
    for autotext, count in zip(autotexts, contagens.values):
        autotext.set_text(f"{autotext.get_text()}\n(n={count})")

    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    titulo: str = "Feature Importance",
) -> plt.Figure:
    """
    Cria gráfico de barras horizontais com as features mais importantes.

    Args:
        importances (np.ndarray): Array de importâncias das features.
        feature_names (List[str]): Lista com nomes das features.
        top_n (int): Número de features a exibir (padrão: 15).
        figsize (Tuple[int, int]): Tamanho da figura.
        titulo (str): Título do gráfico.

    Returns:
        plt.Figure: Objeto Figure do matplotlib.

    Examples:
        >>> importances = np.array([0.5, 0.3, 0.2])
        >>> names = ["Feature A", "Feature B", "Feature C"]
        >>> fig = plot_feature_importance(importances, names, top_n=3)
        >>> plt.close(fig)
    """
    # Cria DataFrame e ordena
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    # Cria gráfico
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("viridis", len(df_imp))

    bars = ax.barh(df_imp["feature"], df_imp["importance"], color=colors, edgecolor="black")

    # Anotações
    for bar, imp in zip(bars, df_imp["importance"], strict=False):
        ax.text(
            imp,
            bar.get_y() + bar.get_height() / 2.0,
            f" {imp:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Importância", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig
