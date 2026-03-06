"""
Test suite para o módulo data_cleaning.

Organização:
- TestDataLoading: Testes para carregamento de dados
- TestDataCleaning: Testes para limpeza de dados
- TestMissingValues: Testes para tratamento de valores faltantes
- TestTargetCreation: Testes para criação da variável alvo
"""

import os
import tempfile
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    clean_data,
    create_target,
    handle_missing_values,
    load_data,
    validate_data_quality,
)


# ==================== FIXTURES ====================
@pytest.fixture
def sample_raw_dataframe():
    """Cria DataFrame bruto de amostra."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Nome Anonimizado": ["Aluno1", "Aluno2", "Aluno3", "Aluno4", "Aluno5"],
            "INDE 2024": [50.0, 60.5, np.nan, 75.0, 80.0],
            "IEG 2024": [0.5, 0.6, 0.7, np.nan, 0.8],
            "IDADE": [15, 16, 15, 17, 16],
            "FASE": ["1", "2", "2", "3", "1"],
        }
    )


@pytest.fixture
def sample_clean_dataframe():
    """Cria DataFrame limpo de amostra."""
    return pd.DataFrame(
        {
            "INDE_2024": [50.0, 60.5, 70.0, 75.0, 80.0],
            "IEG_2024": [0.5, 0.6, 0.7, 0.65, 0.8],
            "IDADE": [15, 16, 15, 17, 16],
            "FASE": ["1", "2", "2", "3", "1"],
        }
    )


@pytest.mark.unit
@pytest.mark.data_loading
class TestDataLoading:
    """Testes para a função load_data."""

    def test_load_data_file_not_found(self):
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            load_data("arquivo_inexistente_12345.xlsx")

    def test_load_data_invalid_extension(self):
        """Testa erro com extensão de arquivo inválida."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Formato de arquivo inválido"):
                load_data(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch("pandas.read_excel")
    @patch("os.path.exists", return_value=True)
    def test_load_data_excel_file(self, mock_exists, mock_read_excel):
        """Testa carregamento de arquivo Excel válido."""
        mock_df = pd.DataFrame({"COL1": [1, 2], "COL2": [3, 4]})
        mock_read_excel.return_value = mock_df

        result = load_data("dummy_path.xlsx")

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 2
        assert "COL1" in result.columns

    @patch("pandas.read_excel")
    @patch("os.path.exists", return_value=True)
    def test_load_data_with_custom_sheet(self, mock_exists, mock_read_excel):
        """Testa carregamento com sheet customizado."""
        mock_df = pd.DataFrame({"COL1": [1, 2]})
        mock_read_excel.return_value = mock_df

        result = load_data("dummy_path.xlsx", sheet_name="CUSTOM")

        assert isinstance(result, pd.DataFrame)
        mock_read_excel.assert_called_once_with("dummy_path.xlsx", sheet_name="CUSTOM")

    @patch("pandas.read_excel")
    @patch("os.path.exists", return_value=True)
    def test_load_data_sheet_not_found(self, mock_exists, mock_read_excel):
        """Testa erro quando sheet não existe."""
        mock_read_excel.side_effect = ValueError("Sheet não encontrado")

        with pytest.raises(ValueError):
            load_data("dummy_path.xlsx", sheet_name="INEXISTENTE")

    @patch("pandas.read_excel")
    @patch("os.path.exists", return_value=True)
    def test_load_data_empty_file(self, mock_exists, mock_read_excel):
        """Testa carregamento de arquivo vazio."""
        mock_df = pd.DataFrame()
        mock_read_excel.return_value = mock_df

        result = load_data("dummy_path.xlsx")

        assert isinstance(result, pd.DataFrame)


@pytest.mark.unit
@pytest.mark.data_cleaning
class TestDataCleaning:
    """Testes para a função clean_data."""

    def test_clean_data_type_error(self):
        """Testa erro com tipo inválido."""
        with pytest.raises(TypeError, match="Esperado pd.DataFrame"):
            clean_data("not a dataframe")

    def test_clean_data_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        df_empty = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame não pode estar vazio"):
            clean_data(df_empty)

    def test_clean_data_standardizes_column_names(self, sample_raw_dataframe):
        """Testa que nomes de colunas são padronizados."""
        result = clean_data(sample_raw_dataframe)

        # Verificar que todas colunas estão em uppercase
        assert all(c.isupper() or c.isdigit() or c == "_" for c in "".join(result.columns))

        # Verificar que espaços foram removidos
        assert all(" " not in col for col in result.columns)

    def test_clean_data_removes_identifier_columns(self, sample_raw_dataframe):
        """Testa remoção de colunas identificadoras."""
        result = clean_data(sample_raw_dataframe)

        # 'Nome Anonimizado' deve ser removido
        assert "NOME_ANONIMIZADO" not in result.columns
        assert "Nome Anonimizado" not in result.columns

    def test_clean_data_converts_numeric_columns(self, sample_raw_dataframe):
        """Testa conversão de colunas numéricas."""
        result = clean_data(sample_raw_dataframe)

        # INDE_2024 deve ser numérica
        assert result["INDE_2024"].dtype in [np.float64, np.float32, float]

    def test_clean_data_returns_dataframe(self, sample_raw_dataframe):
        """Testa que retorna DataFrame."""
        result = clean_data(sample_raw_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_clean_data_preserves_data_count(self, sample_raw_dataframe):
        """Testa que não remove linhas."""
        original_count = len(sample_raw_dataframe)
        result = clean_data(sample_raw_dataframe)

        assert len(result) == original_count

    def test_clean_data_with_simple_dataframe(self):
        """Testa limpeza com DataFrame simples."""
        df = pd.DataFrame(
            {
                "Nome": ["A", "B"],
                "INDE 2024": [50.0, 60.0],
            }
        )

        result = clean_data(df)
        assert isinstance(result, pd.DataFrame)
        assert "INDE_2024" in result.columns


@pytest.mark.unit
@pytest.mark.data_cleaning
class TestMissingValues:
    """Testes para a função handle_missing_values."""

    def test_handle_missing_values_type_error(self):
        """Testa erro com tipo inválido."""
        with pytest.raises(TypeError):
            handle_missing_values("not a dataframe")

    def test_handle_missing_values_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        df_empty = pd.DataFrame()
        with pytest.raises(ValueError):
            handle_missing_values(df_empty)

    def test_handle_missing_values_numeric_imputation(self):
        """Testa imputação de valores numéricos."""
        df = pd.DataFrame(
            {
                "A": [1.0, np.nan, 3.0, 4.0, 5.0],
                "B": [4.0, 5.0, np.nan, 6.0, 7.0],
            }
        )

        result = handle_missing_values(df)

        assert result["A"].isna().sum() == 0
        assert result["B"].isna().sum() == 0

    def test_handle_missing_values_categorical_imputation(self):
        """Testa imputação de valores categóricos."""
        df = pd.DataFrame(
            {
                "A": ["X", None, "Y", "X", "Z"],
                "B": ["foo", "bar", None, "baz", "foo"],
            }
        )

        result = handle_missing_values(df)

        assert result.isna().sum().sum() == 0

    def test_handle_missing_values_mixed_types(self):
        """Testa imputação com tipos mistos."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, np.nan, 3.0, 4.0, 5.0],
                "categorical": ["A", None, "C", "A", "B"],
            }
        )

        result = handle_missing_values(df)

        assert isinstance(result, pd.DataFrame)
        assert result.isna().sum().sum() == 0

    def test_handle_missing_values_no_missing(self):
        """Testa com DataFrame sem valores faltantes."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": [4, 5, 6, 7],
            }
        )

        result = handle_missing_values(df)

        assert len(result) == len(df)
        assert result.isna().sum().sum() == 0

    def test_handle_missing_values_all_missing(self):
        """Testa com coluna totalmente faltante."""
        df = pd.DataFrame(
            {
                "A": [np.nan, np.nan, np.nan],
                "B": [1, 2, 3],
            }
        )

        result = handle_missing_values(df)

        # Deve imputar os valores faltantes
        assert result.isna().sum().sum() == 0

    def test_handle_missing_values_preserves_shape(self):
        """Testa que shape do DataFrame é preservado."""
        df = pd.DataFrame(
            {
                "A": [1.0, np.nan, 3.0],
                "B": ["X", None, "Z"],
                "C": [10.0, 20.0, np.nan],
            }
        )

        original_shape = df.shape
        result = handle_missing_values(df)

        assert result.shape == original_shape


@pytest.mark.unit
@pytest.mark.data_cleaning
class TestTargetCreation:
    """Testes para a função create_target."""

    def test_create_target_type_error(self):
        """Testa erro com tipo inválido."""
        with pytest.raises(TypeError):
            create_target("not a dataframe")

    def test_create_target_missing_column(self):
        """Testa erro quando coluna target não existe."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        with pytest.raises(ValueError, match="não encontrada"):
            create_target(df, target_column="DEFASAGEM")

    def test_create_target_creates_binary_column(self):
        """Testa criação de coluna TARGET binária."""
        df = pd.DataFrame(
            {
                "DEFASAGEM": [-1, 0, 1, -2, 2],
            }
        )

        result = create_target(df)

        assert "TARGET" in result.columns
        assert set(result["TARGET"]) <= {0, 1}

    def test_create_target_correct_mapping(self):
        """Testa que mapeamento é correto (< 0 -> 1, >= 0 -> 0)."""
        df = pd.DataFrame(
            {
                "DEFASAGEM": [-1.0, 0.0, 1.0, -2.0, 2.5],
            }
        )

        result = create_target(df)

        # -1 < 0 -> TARGET = 1
        # 0 >= 0 -> TARGET = 0
        # 1 >= 0 -> TARGET = 0
        # -2 < 0 -> TARGET = 1
        # 2.5 >= 0 -> TARGET = 0
        expected = [1, 0, 0, 1, 0]

        np.testing.assert_array_equal(result["TARGET"].values, expected)


@pytest.mark.unit
@pytest.mark.data_cleaning
class TestDataQualityValidation:
    """Testes para validação de qualidade com Great Expectations."""

    def test_validate_data_quality_success(self, monkeypatch: pytest.MonkeyPatch):
        """Validação deve passar com dados válidos."""
        validator = MagicMock()
        validator.expect_column_values_to_be_between.return_value = {"success": True}
        validator.expect_column_values_to_not_be_null.return_value = {"success": True}

        ge_stub = types.SimpleNamespace(from_pandas=lambda _df: validator)
        monkeypatch.setitem(os.sys.modules, "great_expectations", ge_stub)

        df = pd.DataFrame({"INDE_2024": [10.0, 20.0], "DEFASAGEM": [0.0, 1.0]})

        validate_data_quality(df)

    def test_validate_data_quality_negative_inde(self, monkeypatch: pytest.MonkeyPatch):
        """Deve falhar quando INDE_2024 contém valores negativos."""
        validator = MagicMock()
        validator.expect_column_values_to_be_between.return_value = {"success": False}
        validator.expect_column_values_to_not_be_null.return_value = {"success": True}

        ge_stub = types.SimpleNamespace(from_pandas=lambda _df: validator)
        monkeypatch.setitem(os.sys.modules, "great_expectations", ge_stub)

        df = pd.DataFrame({"INDE_2024": [-1.0, 20.0], "DEFASAGEM": [0.0, 1.0]})

        with pytest.raises(ValueError, match="INDE_2024"):
            validate_data_quality(df)

    def test_validate_data_quality_defasagem_null_ratio(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Deve falhar quando taxa de nulos em DEFASAGEM excede limite."""
        validator = MagicMock()
        validator.expect_column_values_to_be_between.return_value = {"success": True}
        validator.expect_column_values_to_not_be_null.return_value = {"success": False}

        ge_stub = types.SimpleNamespace(from_pandas=lambda _df: validator)
        monkeypatch.setitem(os.sys.modules, "great_expectations", ge_stub)

        df = pd.DataFrame({"INDE_2024": [10.0, 20.0], "DEFASAGEM": [None, 1.0]})

        with pytest.raises(ValueError, match="DEFASAGEM"):
            validate_data_quality(df)

    def test_create_target_removes_null_rows(self):
        """Testa remoção de linhas com DEFASAGEM nula."""
        df = pd.DataFrame(
            {
                "DEFASAGEM": [-1, np.nan, 0, 1, np.nan],
                "OTHER": ["a", "b", "c", "d", "e"],
            }
        )

        result = create_target(df)

        # Deve ter removido 2 linhas (as com NaN)
        assert len(result) == 3

    def test_create_target_returns_dataframe(self):
        """Testa que retorna DataFrame."""
        df = pd.DataFrame(
            {
                "DEFASAGEM": [-1, 0, 1, -2],
            }
        )

        result = create_target(df)

        assert isinstance(result, pd.DataFrame)

    def test_create_target_with_custom_column(self):
        """Testa com nome de coluna customizado."""
        df = pd.DataFrame(
            {
                "CUSTOM_RISK": [-1, 0, 1, -0.5],
            }
        )

        result = create_target(df, target_column="CUSTOM_RISK")

        assert "TARGET" in result.columns

    def test_create_target_all_nan_error(self):
        """Testa erro quando todas linhas são NaN."""
        df = pd.DataFrame(
            {
                "DEFASAGEM": [np.nan, np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="Nenhuma linha válida"):
            create_target(df)

    def test_create_target_class_distribution(self):
        """Testa distribuição de classes no TARGET."""
        df = pd.DataFrame(
            {
                "DEFASAGEM": [-1, -2, -3, 0, 1, 2],  # 3 em risco, 3 sem risco
            }
        )

        result = create_target(df)

        # Deve ter 3 de cada classe
        assert result["TARGET"].value_counts()[0] == 3
        assert result["TARGET"].value_counts()[1] == 3
