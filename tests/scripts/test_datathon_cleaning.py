"""
Testes unitários para scripts.datathon_cleaning

Module: tests.scripts.test_datathon_cleaning
Author: GitHub Copilot
Date: 2025-03-05

Tests:
    - test_filter_columns_basic
    - test_filter_columns_multiple_filters
    - test_filter_columns_no_match
    - test_filter_columns_type_error
    - test_cleaning_dataset_basic
    - test_cleaning_dataset_with_nome_column
    - test_cleaning_dataset_type_error
    - test_create_annual_datasets
    - test_create_annual_datasets_returns_dict
    - test_analyze_student_continuity
    - test_analyze_student_continuity_with_duplicates
    - test_analyze_student_continuity_missing_nome_column
"""

import numpy as np
import pandas as pd
import pytest

from scripts.datathon_cleaning import (
    analyze_student_continuity,
    cleaning_dataset,
    create_annual_datasets,
    filter_columns,
)


class TestFilterColumns:
    """Testes para function filter_columns()"""

    def test_filter_columns_basic(self):
        """Testa remoção básica de colunas com padrão"""
        df = pd.DataFrame(
            {
                "NOME": ["Alice", "Bob"],
                "valor_2020": [10, 20],
                "valor_2021": [30, 40],
                "valor_2022": [50, 60],
            }
        )

        result = filter_columns(df, ["2021", "2022"])

        assert "NOME" in result.columns
        assert "valor_2020" in result.columns
        assert "valor_2021" not in result.columns
        assert "valor_2022" not in result.columns
        assert result.shape[1] == 2

    def test_filter_columns_multiple_filters(self):
        """Testa remoção com múltiplos filtros"""
        df = pd.DataFrame(
            {"col_a": [1, 2], "col_b_2021": [3, 4], "col_c_2022": [5, 6], "col_d": [7, 8]}
        )

        result = filter_columns(df, ["_2021", "_2022"])

        assert result.shape[1] == 2
        assert "col_a" in result.columns
        assert "col_d" in result.columns

    def test_filter_columns_no_match(self):
        """Testa quando nenhuma coluna contém o filtro"""
        df = pd.DataFrame({"NOME": ["Alice"], "valor": [10]})

        result = filter_columns(df, ["2021"])

        assert result.shape[1] == df.shape[1]
        assert list(result.columns) == list(df.columns)

    def test_filter_columns_type_error_df(self):
        """Testa TypeError quando df não é DataFrame"""
        with pytest.raises(TypeError):
            filter_columns([1, 2, 3], ["2021"])

    def test_filter_columns_type_error_filters(self):
        """Testa TypeError quando filters não é lista"""
        df = pd.DataFrame({"col": [1, 2]})

        with pytest.raises(TypeError):
            filter_columns(df, "2021")

    def test_filter_columns_returns_copy(self):
        """Verifica que retorna cópia, não referência"""
        df = pd.DataFrame({"NOME": ["Alice"], "valor_2021": [10]})

        result = filter_columns(df, ["2021"])
        result["NOME"] = "Modified"

        assert df["NOME"].iloc[0] == "Alice"


class TestCleaningDataset:
    """Testes para function cleaning_dataset()"""

    def test_cleaning_dataset_basic(self):
        """Testa remoção de linhas com todos NaN"""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6], "C": ["x", np.nan, "z"]})

        result = cleaning_dataset(df)

        # Linha do meio tem todos NaN (exceto None na string)
        assert result.shape[0] == 2

    def test_cleaning_dataset_with_nome_column(self):
        """Testa que coluna NOME é ignorada no check de NaN"""
        df = pd.DataFrame(
            {
                "NOME": ["Alice", "Bob", "Charlie"],
                "valor": [10, np.nan, np.nan],
                "score": [20, np.nan, np.nan],
            }
        )

        result = cleaning_dataset(df)

        # Apenas Alice tem valores em todas as colunas (exceto NOME)
        assert result.shape[0] == 1
        assert "Alice" in result["NOME"].values

    def test_cleaning_dataset_completely_empty_row(self):
        """Testa remoção de linha completamente vazia"""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})

        result = cleaning_dataset(df)

        assert result.shape[0] == 2
        assert not pd.isna(result["A"].iloc[1])  # Valor válido após limpeza

    def test_cleaning_dataset_type_error(self):
        """Testa TypeError quando input não é DataFrame"""
        with pytest.raises(TypeError):
            cleaning_dataset([1, 2, 3])

    def test_cleaning_dataset_returns_copy(self):
        """Verifica que retorna cópia"""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = cleaning_dataset(df)

        result["A"] = 999
        assert df["A"].iloc[0] == 1


class TestCreateAnnualDatasets:
    """Testes para function create_annual_datasets()"""

    @pytest.fixture
    def temporal_df(self):
        """Fixture com dados temporais de múltiplos anos"""
        return pd.DataFrame(
            {
                "NOME": ["Alice", "Bob", "Charlie"],
                "valor_2020": [10, 20, 30],
                "valor_2021": [11, 21, np.nan],
                "valor_2022": [12, np.nan, 32],
            }
        )

    def test_create_annual_datasets_returns_dict(self, temporal_df):
        """Testa que retorna dicionário comKeys 2020, 2021, 2022"""
        result = create_annual_datasets(temporal_df)

        assert isinstance(result, dict)
        assert set(result.keys()) == {2020, 2021, 2022}

    def test_create_annual_datasets_2020(self, temporal_df):
        """Testa dataset de 2020 (remove 2021, 2022)"""
        datasets = create_annual_datasets(temporal_df)
        df_2020 = datasets[2020]

        assert "valor_2020" in df_2020.columns
        assert "valor_2021" not in df_2020.columns
        assert "valor_2022" not in df_2020.columns

    def test_create_annual_datasets_2021_column_rename(self, temporal_df):
        """Testa que 2021 remove sufixo _2021"""
        datasets = create_annual_datasets(temporal_df)
        df_2021 = datasets[2021]

        assert "valor" in df_2021.columns
        assert "valor_2021" not in df_2021.columns

    def test_create_annual_datasets_type_error(self):
        """Testa TypeError com input inválido"""
        with pytest.raises(TypeError):
            create_annual_datasets([1, 2, 3])

    def test_create_annual_datasets_sizes(self, temporal_df):
        """Testa tamanho de cada dataset"""
        datasets = create_annual_datasets(temporal_df)

        # Todos devem ter pelo menos 1 linha (Alice está em todos)
        assert datasets[2020].shape[0] >= 1
        assert datasets[2021].shape[0] >= 1
        assert datasets[2022].shape[0] >= 1


class TestAnalyzeStudentContinuity:
    """Testes para function analyze_student_continuity()"""

    @pytest.fixture
    def continuity_dfs(self):
        """Fixture com 3 datasets para dois anos"""
        df_2020 = pd.DataFrame({"NOME": ["Alice", "Bob", "Charlie"], "valor": [10, 20, 30]})
        df_2021 = pd.DataFrame(
            {
                "NOME": ["Alice", "Bob", "David"],  # Charlie saiu, David entrou
                "valor": [11, 21, 31],
            }
        )
        df_2022 = pd.DataFrame(
            {
                "NOME": ["Alice", "David", "Eve"],  # Bob saiu, Eve entrou
                "valor": [12, 32, 33],
            }
        )
        return df_2020, df_2021, df_2022

    def test_analyze_student_continuity_basic(self, continuity_dfs):
        """Testa análise básica de continuidade"""
        df_2020, df_2021, df_2022 = continuity_dfs
        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        # Verificações básicas
        assert result["alunos_2020"] == 3
        assert result["alunos_2021"] == 3
        assert result["alunos_2022"] == 3

    def test_analyze_student_continuity_2020_2021(self, continuity_dfs):
        """Testa continuidade 2020→2021"""
        df_2020, df_2021, df_2022 = continuity_dfs
        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        # Alice e Bob continuaram (Charlie não)
        assert result["continuidade_2020_2021"] == 2
        assert result["taxa_2020_2021"] == pytest.approx(2 / 3 * 100, rel=1)

    def test_analyze_student_continuity_2021_2022(self, continuity_dfs):
        """Testa continuidade 2021→2022"""
        df_2020, df_2021, df_2022 = continuity_dfs
        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        # Alice e David continuaram (Bob não)
        assert result["continuidade_2021_2022"] == 2
        assert result["taxa_2021_2022"] == pytest.approx(2 / 3 * 100, rel=1)

    def test_analyze_student_continuity_novos(self, continuity_dfs):
        """Testa detecção de alunos novos"""
        df_2020, df_2021, df_2022 = continuity_dfs
        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        # David e Eve são novos em 2022 (David já existia em 2021)
        # Novos = alunos em 2022 que não estavam em 2021
        assert result["novos_2022"] == 1  # Eve

    def test_analyze_student_continuity_sets(self, continuity_dfs):
        """Testa que retorna Sets de alunos"""
        df_2020, df_2021, df_2022 = continuity_dfs
        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        assert isinstance(result["alunos_2020_set"], set)
        assert isinstance(result["alunos_2021_set"], set)
        assert isinstance(result["alunos_2022_set"], set)
        assert "Alice" in result["alunos_2020_set"]

    def test_analyze_student_continuity_type_error(self):
        """Testa TypeError com input inválido"""
        df = pd.DataFrame({"NOME": ["Alice"]})

        with pytest.raises(TypeError):
            analyze_student_continuity([1, 2], df, df)

    def test_analyze_student_continuity_missing_nome(self):
        """Testa KeyError quando coluna NOME falta"""
        df1 = pd.DataFrame({"outros": [1]})
        df2 = pd.DataFrame({"NOME": ["Alice"]})
        df3 = pd.DataFrame({"NOME": ["Bob"]})

        with pytest.raises(KeyError):
            analyze_student_continuity(df1, df2, df3)

    def test_analyze_student_continuity_with_whitespace(self):
        """Testa tratamento de whitespace em nomes"""
        df_2020 = pd.DataFrame({"NOME": ["Alice  ", "  Bob"]})
        df_2021 = pd.DataFrame({"NOME": ["Alice", "Bob  "]})
        df_2022 = pd.DataFrame({"NOME": ["Alice", "Bob"]})

        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        # Com strip(), deve reconhecer como mesmos nomes
        assert result["continuidade_2020_2021"] >= 2

    def test_analyze_student_continuity_with_duplicates(self):
        """Testa que remove duplicatas antes de contar"""
        df_2020 = pd.DataFrame({"NOME": ["Alice", "Alice", "Bob"]})
        df_2021 = pd.DataFrame({"NOME": ["Alice", "Bob", "Bob"]})
        df_2022 = pd.DataFrame({"NOME": ["Alice", "Bob"]})

        result = analyze_student_continuity(df_2020, df_2021, df_2022)

        # Com unique(), deve contar Alice e Bob como 1 cada
        assert result["alunos_2020"] == 2
        assert result["alunos_2021"] == 2
        assert result["alunos_2022"] == 2

    def test_analyze_student_continuity_empty_dataframe(self):
        """Testa com dataframe vazio"""
        df_empty = pd.DataFrame({"NOME": []})
        df_normal = pd.DataFrame({"NOME": ["Alice"]})

        result = analyze_student_continuity(df_empty, df_normal, df_normal)

        assert result["alunos_2020"] == 0
        assert result["taxa_2020_2021"] == 0
