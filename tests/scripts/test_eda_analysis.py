"""
Testes unitários para funções de análise estatística e EDA.

Cobertura:
- verificar_normalidade: testes estatísticos de normalidade
- aplicar_transformacao_log: transformações logarítmicas
- perform_cross_validation: validação cruzada estratificada

Autor: Equipe Passos Mágicos
Data: 2026-03-05
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from scripts.eda_analysis import (
    aplicar_transformacao_log,
    perform_cross_validation,
    verificar_normalidade,
)


@pytest.mark.unit
class TestVerificarNormalidade:
    """Testes para função verificar_normalidade."""

    def test_distribuicao_normal(self):
        """Testa detecção de distribuição normal."""
        # Arrange: distribuição normal padrão
        np.random.seed(42)
        dados_normais = pd.Series(np.random.normal(0, 1, 1000))

        # Act
        resultado = verificar_normalidade(dados_normais, alpha=0.05, verbose=False)

        # Assert
        assert isinstance(resultado, dict)
        assert "normal" in resultado
        assert "shapiro_stat" in resultado
        assert "shapiro_p" in resultado
        assert "dagostino_stat" in resultado
        assert "dagostino_p" in resultado

        # Distribuição normal deve ser aceita
        assert resultado["normal"] is True
        assert resultado["shapiro_p"] > 0.05
        assert resultado["dagostino_p"] > 0.05

    def test_distribuicao_nao_normal(self):
        """Testa detecção de distribuição não normal (exponencial)."""
        # Arrange: distribuição exponencial (altamente skewed)
        np.random.seed(42)
        dados_exponenciais = pd.Series(np.random.exponential(1, 1000))

        # Act
        resultado = verificar_normalidade(dados_exponenciais, alpha=0.05, verbose=False)

        # Assert
        assert resultado["normal"] is False
        # Pelo menos um dos p-values deve ser < 0.05
        assert resultado["shapiro_p"] < 0.05 or resultado["dagostino_p"] < 0.05

    def test_com_valores_nan(self):
        """Testa handling de valores NaN."""
        # Arrange
        dados_com_nan = pd.Series([1, 2, 3, np.nan, 4, 5, np.nan, 6, 7, 8] * 100)

        # Act
        resultado = verificar_normalidade(dados_com_nan, verbose=False)

        # Assert: deve executar sem erros, tratando NaN
        assert isinstance(resultado, dict)
        assert "normal" in resultado

    def test_verbose_output(self, capsys):
        """Testa output do modo verbose."""
        # Arrange
        dados = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)

        # Act
        _ = verificar_normalidade(dados, verbose=True)
        captured = capsys.readouterr()

        # Assert: deve imprimir informações
        assert "Shapiro-Wilk" in captured.out
        assert "D'Agostino" in captured.out
        assert "p-value" in captured.out

    def test_alpha_customizado(self):
        """Testa nível de significância customizado."""
        # Arrange
        dados = pd.Series(np.random.normal(0, 1, 500))

        # Act
        resultado_005 = verificar_normalidade(dados, alpha=0.05, verbose=False)
        resultado_001 = verificar_normalidade(dados, alpha=0.01, verbose=False)

        # Assert: diferentes alphas podem dar resultados diferentes
        assert isinstance(resultado_005["normal"], bool)
        assert isinstance(resultado_001["normal"], bool)


@pytest.mark.unit
class TestAplicarTransformacaoLog:
    """Testes para função aplicar_transformacao_log."""

    def test_transformacao_basica(self):
        """Testa transformação logarítmica básica."""
        # Arrange
        df = pd.DataFrame({"A": [0, 1, 10, 100], "B": [5, 10, 15, 20]})

        # Act
        df_transformado = aplicar_transformacao_log(df, ["A", "B"])

        # Assert
        assert "A_log" in df_transformado.columns
        assert "B_log" in df_transformado.columns

        # Verifica cálculo: log1p(0) = 0
        assert df_transformado.loc[0, "A_log"] == pytest.approx(0.0)

        # Verifica cálculo: log1p(1) ≈ 0.693
        assert df_transformado.loc[1, "A_log"] == pytest.approx(0.693, abs=0.01)

    def test_preserva_dataframe_original(self):
        """Testa se o DataFrame original não é modificado (imutabilidade)."""
        # Arrange
        df_original = pd.DataFrame({"X": [1, 2, 3, 4, 5]})
        df_copia = df_original.copy()

        # Act
        _ = aplicar_transformacao_log(df_original, ["X"])

        # Assert: original deve permanecer inalterado
        pd.testing.assert_frame_equal(df_original, df_copia)
        assert "X_log" not in df_original.columns

    def test_coluna_inexistente(self):
        """Testa comportamento com coluna inexistente."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})

        # Act: tenta transformar coluna que não existe
        df_resultado = aplicar_transformacao_log(df, ["B"])

        # Assert: não deve criar colunas para features inexistentes
        assert "B_log" not in df_resultado.columns
        # Deve ter as mesmas colunas do original
        assert list(df_resultado.columns) == list(df.columns)

    def test_valores_zero(self):
        """Testa handling de valores zero."""
        # Arrange
        df = pd.DataFrame({"valores": [0, 0, 0, 1, 2]})

        # Act
        df_resultado = aplicar_transformacao_log(df, ["valores"])

        # Assert: log1p(0) = log(1) = 0
        assert df_resultado.loc[0, "valores_log"] == 0.0
        assert df_resultado.loc[1, "valores_log"] == 0.0

    def test_valores_negativos_nao_aceitos(self):
        """Testa que valores negativos geram warning/erro matemático."""
        # Arrange
        df = pd.DataFrame({"neg": [-5, -10, -15]})

        # Act & Assert: log1p de negativos deve gerar NaN
        df_resultado = aplicar_transformacao_log(df, ["neg"])

        # log1p(-5) = log(-4) = NaN
        assert df_resultado["neg_log"].isna().all()

    def test_multiplas_colunas(self):
        """Testa transformação de múltiplas colunas simultaneamente."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30], "C": [100, 200, 300]})

        # Act
        df_resultado = aplicar_transformacao_log(df, ["A", "B", "C"])

        # Assert
        assert "A_log" in df_resultado.columns
        assert "B_log" in df_resultado.columns
        assert "C_log" in df_resultado.columns

        # Verifica que não criou outras colunas indesejadas
        assert len(df_resultado.columns) == 6  # 3 originais + 3 _log


@pytest.mark.unit
class TestPerformCrossValidation:
    """Testes para função perform_cross_validation."""

    @pytest.fixture
    def dados_classificacao_balanceada(self):
        """Dataset sintético para classificação binária balanceada."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
            "feat3": np.random.randn(n_samples),
        })
        # Target: 50% classe 0, 50% classe 1
        y = pd.Series([0] * 100 + [1] * 100, name="target")

        return X, y

    @pytest.fixture
    def dados_classificacao_desbalanceada(self):
        """Dataset sintético desbalanceado (20% classe minoritária)."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
        })
        y = pd.Series([0] * 160 + [1] * 40, name="target")

        return X, y

    def test_validacao_cruzada_basica(self, dados_classificacao_balanceada):
        """Testa execução básica de validação cruzada."""
        # Arrange
        X, y = dados_classificacao_balanceada
        model = LogisticRegression(random_state=42, max_iter=1000)

        # Act
        resultado = perform_cross_validation(
            model=model, X=X, y=y, n_folds=3, verbose=False
        )

        # Assert
        assert isinstance(resultado, dict)
        assert "accuracy" in resultado
        assert "precision" in resultado
        assert "recall" in resultado
        assert "f1" in resultado

        # Deve ter 3 valores (3 folds)
        assert len(resultado["accuracy"]) == 3
        assert len(resultado["f1"]) == 3

        # Métricas devem estar em [0, 1]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            for valor in resultado[metric]:
                assert 0.0 <= valor <= 1.0

    def test_random_forest(self, dados_classificacao_desbalanceada):
        """Testa validação cruzada com RandomForest."""
        # Arrange
        X, y = dados_classificacao_desbalanceada
        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=1
        )

        # Act
        resultado = perform_cross_validation(
            model=model, X=X, y=y, n_folds=5, verbose=False
        )

        # Assert
        assert len(resultado["accuracy"]) == 5

        # RandomForest deve ter performance razoável (>60% accuracy)
        mean_accuracy = np.mean(resultado["accuracy"])
        assert mean_accuracy > 0.6

    def test_reproducibilidade(self, dados_classificacao_balanceada):
        """Testa se random_state garante reproducibilidade."""
        # Arrange
        X, y = dados_classificacao_balanceada
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        model2 = LogisticRegression(random_state=42, max_iter=1000)

        # Act
        resultado1 = perform_cross_validation(
            model=model1, X=X, y=y, n_folds=5, random_state=42, verbose=False
        )
        resultado2 = perform_cross_validation(
            model=model2, X=X, y=y, n_folds=5, random_state=42, verbose=False
        )

        # Assert: resultados devem ser idênticos
        np.testing.assert_array_almost_equal(
            resultado1["accuracy"], resultado2["accuracy"], decimal=10
        )
        np.testing.assert_array_almost_equal(
            resultado1["f1"], resultado2["f1"], decimal=10
        )

    def test_n_folds_customizado(self, dados_classificacao_balanceada):
        """Testa diferentes valores de n_folds."""
        # Arrange
        X, y = dados_classificacao_balanceada
        model = LogisticRegression(random_state=42, max_iter=1000)

        # Act
        resultado_3folds = perform_cross_validation(
            model=model, X=X, y=y, n_folds=3, verbose=False
        )
        resultado_10folds = perform_cross_validation(
            model=model, X=X, y=y, n_folds=10, verbose=False
        )

        # Assert
        assert len(resultado_3folds["accuracy"]) == 3
        assert len(resultado_10folds["accuracy"]) == 10

    def test_verbose_output(self, dados_classificacao_balanceada, capsys):
        """Testa modo verbose."""
        # Arrange
        X, y = dados_classificacao_balanceada
        model = LogisticRegression(random_state=42, max_iter=1000)

        # Act
        _ = perform_cross_validation(model=model, X=X, y=y, n_folds=2, verbose=True)
        captured = capsys.readouterr()

        # Assert: deve imprimir progresso
        assert "Fold" in captured.out or "fold" in captured.out.lower()


@pytest.mark.unit
class TestEdgeCasesEda:
    """Testes de edge cases gerais para módulo EDA."""

    def test_series_vazia(self):
        """Testa comportamento com série vazia."""
        # Arrange
        series_vazia = pd.Series([], dtype=float)

        # Act & Assert: deve lançar erro ou lidar gracefully
        with pytest.raises(ValueError):
            verificar_normalidade(series_vazia, verbose=False)

    def test_dataframe_vazio(self):
        """Testa transformação log em DataFrame vazio."""
        # Arrange
        df_vazio = pd.DataFrame()

        # Act
        resultado = aplicar_transformacao_log(df_vazio, ["coluna_inexistente"])

        # Assert: deve retornar DataFrame vazio
        assert resultado.empty

    def test_serie_com_valores_unicos(self):
        """Testa distribuição com apenas um valor único."""
        # Arrange: todos os valores iguais
        dados_constantes = pd.Series([5.0] * 100)

        # Act & Assert: distribuição constante pode lançar erro em testes estatísticos
        with pytest.raises((ValueError, RuntimeError, Exception)):
            verificar_normalidade(dados_constantes, verbose=False)
