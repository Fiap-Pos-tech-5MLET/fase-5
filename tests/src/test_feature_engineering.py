"""
Test suite para o módulo feature_engineering.

Organização:
- TestFeatureCreation: Testes para criação de features
- TestFeatureSelection: Testes para seleção de features
- TestFeatureValidation: Testes para validação de features
- TestBuildFeatureMatrix: Testes para build_feature_matrix_for_model
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_feature_matrix_for_model,
    create_features,
    select_features,
)


# ==================== FIXTURES ====================
@pytest.fixture
def sample_dataframe_with_inde():
    """DataFrame com colunas INDE para criar features."""
    return pd.DataFrame(
        {
            "INDE_2024": [10.0, 8.0, 12.0, 9.5, 11.0],
            "INDE_23": [8.0, 0.0, 10.0, 7.0, 10.5],
            "IDADE": [15, 16, 15, 17, 16],
            "FASE": ["1", "2", "2", "3", "1"],
        }
    )


@pytest.fixture
def sample_dataframe_with_target():
    """DataFrame completo com target para select_features."""
    return pd.DataFrame(
        {
            "INDE_23": [8.0, 6.0, 9.0, 7.5, 8.5],
            "INDE_2024": [10.0, 7.0, 11.0, 8.5, 10.0],
            "DEFASAGEM": [0, -1, 1, -0.5, 0.5],
            "TARGET": [0, 1, 0, 1, 0],
            "IDADE": [15, 16, 15, 17, 16],
            "FASE": ["1", "2", "2", "3", "1"],
            "RA": ["A001", "A002", "A003", "A004", "A005"],
        }
    )


@pytest.mark.unit
@pytest.mark.feature_engineering
class TestFeatureCreation:
    """Testes para a função create_features."""

    def test_create_features_type_error(self):
        """Testa erro com tipo inválido."""
        with pytest.raises(TypeError, match="Esperado pd.DataFrame"):
            create_features("not a dataframe")

    def test_create_features_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        df_empty = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame não pode estar vazio"):
            create_features(df_empty)

    def test_create_features_returns_dataframe(self, sample_dataframe_with_inde):
        """Testa que retorna DataFrame."""
        result = create_features(sample_dataframe_with_inde)
        assert isinstance(result, pd.DataFrame)

    def test_create_features_preserves_original_columns(self, sample_dataframe_with_inde):
        """Testa que colunas originais são preservadas."""
        result = create_features(sample_dataframe_with_inde)

        for col in sample_dataframe_with_inde.columns:
            assert col in result.columns

    def test_create_features_creates_growth_feature(self, sample_dataframe_with_inde):
        """Testa criação da feature INDE_GROWTH."""
        result = create_features(sample_dataframe_with_inde)

        assert "INDE_GROWTH" in result.columns
        # INDE_GROWTH = INDE_2024 - INDE_23
        expected = sample_dataframe_with_inde["INDE_2024"] - sample_dataframe_with_inde["INDE_23"]
        np.testing.assert_array_almost_equal(result["INDE_GROWTH"].values, expected.values)

    def test_create_features_creates_history_flag(self, sample_dataframe_with_inde):
        """Testa criação da feature HAS_HISTORY_23."""
        result = create_features(sample_dataframe_with_inde)

        assert "HAS_HISTORY_23" in result.columns
        # HAS_HISTORY_23 = 1 se INDE_23 > 0, 0 caso contrário
        expected = (sample_dataframe_with_inde["INDE_23"] > 0).astype(int)
        np.testing.assert_array_equal(result["HAS_HISTORY_23"].values, expected.values)

    def test_create_features_history_flag_correct_values(self, sample_dataframe_with_inde):
        """Testa que HAS_HISTORY_23 tem valores corretos."""
        result = create_features(sample_dataframe_with_inde)

        # Linha 1: INDE_23 = 8.0 > 0 -> HAS_HISTORY_23 = 1
        assert result.loc[0, "HAS_HISTORY_23"] == 1

        # Linha 2: INDE_23 = 0.0 = 0 -> HAS_HISTORY_23 = 0
        assert result.loc[1, "HAS_HISTORY_23"] == 0

        # Linha 3: INDE_23 = 10.0 > 0 -> HAS_HISTORY_23 = 1
        assert result.loc[2, "HAS_HISTORY_23"] == 1

    def test_create_features_no_nan_in_new_features(self, sample_dataframe_with_inde):
        """Testa que não há valores NaN nas novas features."""
        result = create_features(sample_dataframe_with_inde)

        assert not result["INDE_GROWTH"].isna().any()
        assert not result["HAS_HISTORY_23"].isna().any()

    def test_create_features_single_row(self):
        """Testa com DataFrame de uma linha."""
        df = pd.DataFrame(
            {
                "INDE_2024": [10.0],
                "INDE_23": [8.0],
            }
        )

        result = create_features(df)

        assert len(result) == 1
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "INDE_GROWTH"] == 2.0

    def test_create_features_large_dataframe(self):
        """Testa com DataFrame grande."""
        df = pd.DataFrame(
            {
                "INDE_2024": np.random.uniform(5, 15, 1000),
                "INDE_23": np.random.uniform(0, 12, 1000),
            }
        )

        result = create_features(df)

        assert len(result) == 1000
        assert "INDE_GROWTH" in result.columns
        assert "HAS_HISTORY_23" in result.columns

    def test_create_features_without_inde_columns(self):
        """Testa com DataFrame sem colunas INDE."""
        df = pd.DataFrame(
            {
                "IDADE": [15, 16, 17],
                "FASE": ["1", "2", "3"],
            }
        )

        result = create_features(df)

        # Deve retornar DataFrame sem adicionar features
        assert len(result.columns) == len(df.columns)


@pytest.mark.unit
@pytest.mark.feature_engineering
class TestFeatureSelection:
    """Testes para a função select_features."""

    def test_select_features_type_error(self):
        """Testa erro com tipo inválido."""
        with pytest.raises(TypeError):
            select_features("not a dataframe")

    def test_select_features_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        with pytest.raises(ValueError):
            select_features(pd.DataFrame())

    def test_select_features_missing_target(self):
        """Testa erro quando target não existe."""
        df = pd.DataFrame(
            {
                "INDE_23": [8.0, 6.0],
                "IDADE": [15, 16],
            }
        )

        with pytest.raises(ValueError, match="Target column not found"):
            select_features(df)

    def test_select_features_returns_tuple(self, sample_dataframe_with_target):
        """Testa que retorna tupla (X, y)."""
        result = select_features(sample_dataframe_with_target)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_select_features_returns_correct_types(self, sample_dataframe_with_target):
        """Testa que X é DataFrame e y é Series."""
        X, y = select_features(sample_dataframe_with_target)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_select_features_correct_sample_count(self, sample_dataframe_with_target):
        """Testa que X e y têm mesmo número de amostras."""
        X, y = select_features(sample_dataframe_with_target)

        assert len(X) == len(y)
        assert len(X) == len(sample_dataframe_with_target)

    def test_select_features_removes_target(self, sample_dataframe_with_target):
        """Testa que TARGET é removido de X."""
        X, _y = select_features(sample_dataframe_with_target)

        assert "TARGET" not in X.columns

    def test_select_features_extracts_correct_target(self, sample_dataframe_with_target):
        """Testa que y tem valores corretos do TARGET."""
        _X, y = select_features(sample_dataframe_with_target)

        np.testing.assert_array_equal(y.values, sample_dataframe_with_target["TARGET"].values)

    def test_select_features_removes_metadata(self, sample_dataframe_with_target):
        """Testa remoção de colunas de metadados."""
        X, _y = select_features(sample_dataframe_with_target)

        assert "RA" not in X.columns
        assert "TARGET" not in X.columns
        assert "DEFASAGEM" not in X.columns

    def test_select_features_with_remove_leakage_false(self, sample_dataframe_with_target):
        """Testa select_features com remove_leakage=False."""
        X, _y = select_features(sample_dataframe_with_target, remove_leakage=False)

        # Com remove_leakage=False, INDE_23 deve estar presente
        assert "INDE_23" in X.columns

    def test_select_features_with_remove_leakage_true(self, sample_dataframe_with_target):
        """Testa select_features com remove_leakage=True."""
        X, _y = select_features(sample_dataframe_with_target, remove_leakage=True)

        # Com remove_leakage=True, INDE_2024 não deve estar presente
        assert "INDE_2024" not in X.columns
        assert "DEFASAGEM" not in X.columns

    def test_select_features_preserves_numeric_dtypes(self, sample_dataframe_with_target):
        """Testa que tipos de dados são preservados."""
        X, _y = select_features(sample_dataframe_with_target)

        # INDE_23 e IDADE devem ser numéricos
        numeric_cols = X.select_dtypes(include=["number"]).columns
        assert len(numeric_cols) > 0

    def test_select_features_custom_target_column(self):
        """Testa select_features com coluna target customizada."""
        df = pd.DataFrame(
            {
                "INDE_23": [8.0, 6.0, 9.0],
                "IDADE": [15, 16, 15],
                "CUSTOM_TARGET": [0, 1, 0],
            }
        )

        X, y = select_features(df, target_col="CUSTOM_TARGET")

        assert "CUSTOM_TARGET" not in X.columns
        np.testing.assert_array_equal(y.values, [0, 1, 0])


@pytest.mark.unit
class TestFeatureValidation:
    """Testes para validação de features criadas."""

    def test_features_are_numeric(self):
        """Testa que features criadas são numéricas."""
        df = pd.DataFrame(
            {
                "IDADE": [15, 16, 17],
                "INDE_23": [0.5, 0.8, 0.3],
            }
        )
        result = create_features(df)
        numeric_cols = result.select_dtypes(include="number").columns
        assert len(numeric_cols) > 0

    def test_features_reasonable_ranges(self):
        """Testa que features têm valores em ranges razoáveis."""
        df = pd.DataFrame(
            {
                "IDADE": [15, 16, 17],
                "INDE_23": [0.5, 0.8, 0.3],
            }
        )
        result = create_features(df)
        # Features não devem ter valores infinitos
        assert (
            not result.select_dtypes(include="number")
            .isin([float("inf"), float("-inf")])
            .any()
            .any()
        )


@pytest.mark.unit
@pytest.mark.feature_engineering
class TestBuildFeatureMatrix:
    """Testes para build_feature_matrix_for_model."""

    def test_build_feature_matrix_normalizes_columns(self):
        """Testa normalização de nomes de colunas."""
        df = pd.DataFrame(
            {
                "IDADE": [15, 16],
                "GENERO": ["M", "F"],
                "ANO_DE_INGRESSO": [2020, 2021],
            }
        )

        result = build_feature_matrix_for_model(df)

        # Verifica nomes normalizados
        assert "idade" in result.columns
        assert "genero" in result.columns
        assert "ano_de_ingresso" in result.columns

    def test_build_feature_matrix_creates_veterano(self):
        """Testa criação automática da feature veterano."""
        df = pd.DataFrame(
            {
                "ano_de_ingresso": [2020, 2023, 2024, 2025],
            }
        )

        result = build_feature_matrix_for_model(df)

        assert "veterano" in result.columns
        # Veterano = 1 se ano_de_ingresso < 2024
        assert result.loc[0, "veterano"] == 1  # 2020
        assert result.loc[1, "veterano"] == 1  # 2023
        assert result.loc[2, "veterano"] == 0  # 2024
        assert result.loc[3, "veterano"] == 0  # 2025

    def test_build_feature_matrix_converts_genero(self):
        """Testa conversão de gênero para numérico."""
        df = pd.DataFrame(
            {
                "genero": ["M", "F", "MASCULINO", "FEMININO", "HOMEM", "MULHER"],
            }
        )

        result = build_feature_matrix_for_model(df)

        assert result.loc[0, "genero"] == 1  # M
        assert result.loc[1, "genero"] == 0  # F
        assert result.loc[2, "genero"] == 1  # MASCULINO
        assert result.loc[3, "genero"] == 0  # FEMININO
        assert result.loc[4, "genero"] == 1  # HOMEM
        assert result.loc[5, "genero"] == 0  # MULHER

    def test_build_feature_matrix_handles_aliases(self):
        """Testa mapeamento de aliases de colunas."""
        df = pd.DataFrame(
            {
                "IDADE": [15],
                "GÊNERO": ["M"],
                "ANO_INGRESSO": [2020],
            }
        )

        result = build_feature_matrix_for_model(df)

        assert "idade" in result.columns
        assert "genero" in result.columns
        assert "ano_de_ingresso" in result.columns

    def test_build_feature_matrix_fills_missing_with_zero(self):
        """Testa preenchimento de valores faltantes com zero."""
        df = pd.DataFrame(
            {
                "idade": [15, None, 17],
                "genero": ["M", "F", None],
            }
        )

        result = build_feature_matrix_for_model(df)

        # Valores None devem ser preenchidos com 0
        assert not result["idade"].isna().any()
        assert not result["genero"].isna().any()

    def test_build_feature_matrix_returns_only_model_features(self):
        """Testa que retorna apenas features esperadas pelo modelo."""
        df = pd.DataFrame(
            {
                "idade": [15, 16],
                "genero": ["M", "F"],
                "ano_de_ingresso": [2020, 2021],
                "qtde_aval_realizadas": [5, 3],
                "extra_column": ["A", "B"],  # Não deve aparecer no resultado
            }
        )

        result = build_feature_matrix_for_model(df)

        # Deve conter model features
        assert "idade" in result.columns
        assert "genero" in result.columns
        assert "ano_de_ingresso" in result.columns
        assert "veterano" in result.columns
        assert "qtde_aval_realizadas" in result.columns

        # Não deve conter colunas extras
        assert "extra_column" not in result.columns

    def test_build_feature_matrix_handles_empty_dataframe(self):
        """Testa comportamento com DataFrame vazio."""
        df = pd.DataFrame()

        result = build_feature_matrix_for_model(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_build_feature_matrix_converts_all_to_numeric(self):
        """Testa conversão de todas as features para numérico."""
        df = pd.DataFrame(
            {
                "idade": ["15", "16"],  # String
                "genero": ["M", "F"],
                "qtde_aval_realizadas": ["5", "3"],  # String
            }
        )

        result = build_feature_matrix_for_model(df)

        # Todas as colunas devem ser numéricas
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
