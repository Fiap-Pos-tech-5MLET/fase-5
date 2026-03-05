"""
Testes unitários para funções de visualização.

Cobertura:
- plot_exact_counter: gráfico de contagens com porcentagens
- analyse_corr: análise de correlação visual
- plot_distribuicao_target: distribuição da variável target
- plot_feature_importance: importância de features

Autor: Equipe Passos Mágicos
Data: 2026-03-05
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend para testes
matplotlib.use("Agg")

from scripts.visualization import (
    analyse_corr,
    plot_distribuicao_target,
    plot_exact_counter,
    plot_feature_importance,
)


@pytest.mark.unit
class TestPlotExactCounter:
    """Testes para função plot_exact_counter."""

    def test_plot_basico(self):
        """Testa criação de plot básico de contagens."""
        # Arrange
        df = pd.DataFrame({"FASE": ["Fase 7", "Fase 8", "Fase 7", "Fase 9", "Fase 8"]})

        # Act
        fig = plot_exact_counter(df, "FASE")

        # Assert
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Cleanup
        plt.close(fig)

    def test_titulo_customizado(self):
        """Testa uso de título customizado."""
        # Arrange
        df = pd.DataFrame({"TURMA": ["A", "B", "A", "C"]})

        # Act
        fig = plot_exact_counter(df, "TURMA", titulo="Distribuição por Turma")

        # Assert
        ax = fig.axes[0]
        assert "Distribuição por Turma" in ax.get_title()

        plt.close(fig)

    def test_figsize_customizado(self):
        """Testa configuração de figsize."""
        # Arrange
        df = pd.DataFrame({"CAT": ["X", "Y", "Z"]})

        # Act
        fig = plot_exact_counter(df, "CAT", figsize=(8, 4))

        # Assert
        # Verifica dimensões (aproximadas devido a tight_layout)
        width, height = fig.get_size_inches()
        assert width == pytest.approx(8, abs=0.5)
        assert height == pytest.approx(4, abs=0.5)

        plt.close(fig)

    def test_rotation_labels(self):
        """Testa rotação de labels do eixo X."""
        # Arrange
        df = pd.DataFrame(
            {"CATEGORIA": ["Categoria Muito Longa A"] * 10 + ["Categoria B"] * 5}
        )

        # Act
        fig = plot_exact_counter(df, "CATEGORIA", rotation=90)

        # Assert
        ax = fig.axes[0]
        # Verifica que os ticks do X têm rotação
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 90

        plt.close(fig)

    def test_categorias_desbalanceadas(self):
        """Testa plot com categorias muito desbalanceadas."""
        # Arrange: 90% categoria A, 10% categoria B
        df = pd.DataFrame({"STATUS": ["A"] * 90 + ["B"] * 10})

        # Act
        fig = plot_exact_counter(df, "STATUS")

        # Assert: deve executar sem erros
        assert fig is not None

        plt.close(fig)

    def test_muitas_categorias(self):
        """Testa plot com muitas categorias únicas."""
        # Arrange: 20 categorias diferentes
        categorias = [f"Cat_{i}" for i in range(20)]
        df = pd.DataFrame({"TIPO": categorias * 3})

        # Act
        fig = plot_exact_counter(df, "TIPO", figsize=(14, 8))

        # Assert
        ax = fig.axes[0]
        # Deve ter 20 barras
        assert len(ax.patches) == 20

        plt.close(fig)


@pytest.mark.unit
class TestAnalyseCorr:
    """Testes para função analyse_corr."""

    def test_correlacao_basica(self, capsys):
        """Testa análise de correlação básica."""
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10], "C": [5, 4, 3, 2, 1]}
        )

        # Act
        fig = analyse_corr(df, threshold=0.5)

        # Assert
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Verifica output capturado
        captured = capsys.readouterr()
        assert "CORRELAÇÕES FORTES" in captured.out

        plt.close(fig)

    def test_threshold_alto(self, capsys):
        """Testa threshold alto que não encontra correlações."""
        # Arrange
        df = pd.DataFrame(
            {"X": np.random.randn(50), "Y": np.random.randn(50), "Z": np.random.randn(50)}
        )

        # Act
        fig = analyse_corr(df, threshold=0.99)

        # Assert
        captured = capsys.readouterr()
        # Com threshold tão alto e dados aleatórios, não deve encontrar correlações
        assert "Nenhuma correlação" in captured.out or "0.99" in captured.out

        plt.close(fig)

    def test_colunas_especificas(self):
        """Testa seleção de colunas específicas."""
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],
                "D": ["x", "y", "z"],  # Categórica
            }
        )

        # Act: selecionar apenas A, B, C (numéricas)
        fig = analyse_corr(df, colunas=["A", "B", "C"])

        # Assert
        ax = fig.axes[0]
        # Deve ter 3x3 células no heatmap
        assert len(ax.collections) > 0  # Tem elementos desenhados

        plt.close(fig)

    def test_figsize_customizado(self):
        """Testa configuração de figsize."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Act
        fig = analyse_corr(df, figsize=(8, 6))

        # Assert
        width, height = fig.get_size_inches()
        assert width == pytest.approx(8, abs=0.5)
        assert height == pytest.approx(6, abs=0.5)

        plt.close(fig)

    def test_titulo_customizado(self):
        """Testa título customizado."""
        # Arrange
        df = pd.DataFrame({"X": [1, 2, 3], "Y": [3, 2, 1]})

        # Act
        fig = analyse_corr(df, title="Minha Matriz de Correlação")

        # Assert
        ax = fig.axes[0]
        assert "Minha Matriz de Correlação" in ax.get_title()

        plt.close(fig)

    def test_dataframe_vazio(self):
        """Testa comportamento com DataFrame vazio."""
        # Arrange
        df_vazio = pd.DataFrame()

        # Act & Assert: deve lançar erro ou retornar None
        with pytest.raises((ValueError, KeyError, Exception)):
            _ = analyse_corr(df_vazio)


@pytest.mark.unit
class TestPlotDistribuicaoTarget:
    """Testes para função plot_distribuicao_target."""

    def test_plot_binario(self):
        """Testa plot de target binário."""
        # Arrange
        y = pd.Series([0, 0, 0, 1, 1])

        # Act
        fig = plot_distribuicao_target(y, labels=["Não Evadiu", "Evadiu"])

        # Assert
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_labels_default(self):
        """Testa uso de labels padrão (valores únicos da série)."""
        # Arrange
        y = pd.Series([0, 1, 0, 1, 1])

        # Act
        fig = plot_distribuicao_target(y)

        # Assert
        ax = fig.axes[0]
        # Deve ter 2 slices no pie chart (0 e 1)
        assert len(ax.patches) == 2

        plt.close(fig)

    def test_titulo_customizado(self):
        """Testa título customizado."""
        # Arrange
        y = pd.Series([0, 1])

        # Act
        fig = plot_distribuicao_target(y, titulo="Distribuição de Evasão")

        # Assert
        ax = fig.axes[0]
        assert "Distribuição de Evasão" in ax.get_title()

        plt.close(fig)

    def test_figsize_customizado(self):
        """Testa configuração de figsize."""
        # Arrange
        y = pd.Series([0, 0, 1, 1])

        # Act
        fig = plot_distribuicao_target(y, figsize=(6, 4))

        # Assert
        width, height = fig.get_size_inches()
        assert width == pytest.approx(6, abs=0.5)
        assert height == pytest.approx(4, abs=0.5)

        plt.close(fig)

    def test_multiclasse(self):
        """Testa plot com mais de 2 classes."""
        # Arrange
        y = pd.Series([0, 1, 2, 0, 1, 2, 2])

        # Act
        fig = plot_distribuicao_target(y, labels=["Classe A", "Classe B", "Classe C"])

        # Assert
        ax = fig.axes[0]
        # Deve ter 3 slices
        assert len(ax.patches) == 3

        plt.close(fig)

    def test_desbalanceado(self):
        """Testa plot com classes muito desbalanceadas."""
        # Arrange: 95% classe 0, 5% classe 1
        y = pd.Series([0] * 95 + [1] * 5)

        # Act
        fig = plot_distribuicao_target(y)

        # Assert: deve executar sem erros
        assert fig is not None

        plt.close(fig)


@pytest.mark.unit
class TestPlotFeatureImportance:
    """Testes para função plot_feature_importance."""

    def test_plot_basico(self):
        """Testa criação de plot básico de feature importance."""
        # Arrange
        importances = np.array([0.5, 0.3, 0.15, 0.05])
        names = ["Feature A", "Feature B", "Feature C", "Feature D"]

        # Act
        fig = plot_feature_importance(importances, names, top_n=4)

        # Assert
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_top_n_limitado(self):
        """Testa limitação do número de features exibidas."""
        # Arrange: 20 features mas pedir top 10
        importances = np.random.rand(20)
        names = [f"Feat_{i}" for i in range(20)]

        # Act
        fig = plot_feature_importance(importances, names, top_n=10)

        # Assert
        ax = fig.axes[0]
        # Deve ter exatamente 10 barras
        assert len(ax.patches) == 10

        plt.close(fig)

    def test_ordenacao_decrescente(self):
        """Testa se features são ordenadas por importância decrescente."""
        # Arrange
        importances = np.array([0.1, 0.9, 0.3, 0.7])
        names = ["Low", "High", "Medium-Low", "Medium-High"]

        # Act
        fig = plot_feature_importance(importances, names, top_n=4)

        # Assert
        ax = fig.axes[0]
        # A primeira barra (topo) deve ser a de maior importância
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        # "High" (0.9) deve estar no topo
        assert labels[0] == "High"

        plt.close(fig)

    def test_figsize_customizado(self):
        """Testa configuração de figsize."""
        # Arrange
        importances = np.array([0.5, 0.3, 0.2])
        names = ["A", "B", "C"]

        # Act
        fig = plot_feature_importance(importances, names, figsize=(8, 6))

        # Assert
        width, height = fig.get_size_inches()
        assert width == pytest.approx(8, abs=0.5)
        assert height == pytest.approx(6, abs=0.5)

        plt.close(fig)

    def test_titulo_customizado(self):
        """Testa título customizado."""
        # Arrange
        importances = np.array([0.6, 0.4])
        names = ["Importante", "Menos Importante"]

        # Act
        fig = plot_feature_importance(
            importances, names, titulo="Importância das Features - RandomForest"
        )

        # Assert
        ax = fig.axes[0]
        assert "RandomForest" in ax.get_title()

        plt.close(fig)

    def test_importancias_zero(self):
        """Testa features com importância zero."""
        # Arrange
        importances = np.array([0.8, 0.2, 0.0, 0.0])
        names = ["Important", "Mediano", "Zero_A", "Zero_B"]

        # Act
        fig = plot_feature_importance(importances, names, top_n=4)

        # Assert: deve executar sem erros
        assert fig is not None

        plt.close(fig)


@pytest.mark.unit
class TestEdgeCasesVisualization:
    """Testes de edge cases gerais para módulo de visualização."""

    def test_coluna_inexistente(self):
        """Testa plot_exact_counter com coluna inexistente."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})

        # Act & Assert
        with pytest.raises(KeyError):
            _ = plot_exact_counter(df, "coluna_inexistente")

    def test_serie_vazia_target(self):
        """Testa plot_distribuicao_target com série vazia."""
        # Arrange
        y_vazio = pd.Series([], dtype=int)

        # Act & Assert
        with pytest.raises((ValueError, IndexError, Exception)):
            _ = plot_distribuicao_target(y_vazio)

    def test_importances_tamanho_diferente(self):
        """Testa feature importance com arrays de tamanhos diferentes."""
        # Arrange: incompatível
        importances = np.array([0.5, 0.3, 0.2])
        names = ["A", "B"]  # Falta 1 nome

        # Act & Assert: deve lançar erro
        with pytest.raises((ValueError, IndexError, Exception)):
            _ = plot_feature_importance(importances, names)

    def test_importances_negativas(self):
        """Testa feature importance com valores negativos (não deve ocorrer, mas testa robustez)."""
        # Arrange
        importances = np.array([-0.1, 0.5, 0.6])
        names = ["Neg", "PosA", "PosB"]

        # Act: deve executar sem erros (matplotlib aceita valores negativos em barh)
        fig = plot_feature_importance(importances, names)

        # Assert
        assert fig is not None

        plt.close(fig)
