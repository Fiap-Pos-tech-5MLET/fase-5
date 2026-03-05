"""
Testes unitários para o módulo scripts.data_processing.

Testa funções de processamento e limpeza de dados, incluindo:
- Padronização de colunas por ano
- Análise diagnóstica de qualidade
- Cálculo de idade
- Operações de conjuntos
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from scripts.data_processing import (
    padronizar_colunas_ano,
    adicionar_colunas_vazias,
    analise_nulos,
    calcular_idade_2023,
    obter_elementos_comuns,
    renomear_colunas_ano,
    filtrar_colunas_relevantes,
    consolidar_dataframes,
)


class TestPadronizarColunasAno:
    """Testes para padronizar_colunas_ano."""

    def test_padronizacao_basica(self):
        """Testa padronização básica de colunas com sufixo de ano."""
        df = pd.DataFrame({"Nome Aluno": ["Ana"], "Idade": [12]})
        resultado = padronizar_colunas_ano(df, 2022)

        assert "NOME_ALUNO_22" in resultado.columns
        assert "IDADE_22" in resultado.columns
        assert len(resultado.columns) == 2

    def test_ignora_colunas(self):
        """Testa que colunas ignoradas são apenas convertidas para maiúsculas."""
        df = pd.DataFrame({"ID": [1], "Nome": ["Ana"]})
        resultado = padronizar_colunas_ano(df, 2022, ignorar_cols=["ID"])

        assert "ID" in resultado.columns
        assert "NOME_22" in resultado.columns

    def test_preserva_original(self):
        """Verifica que DataFrame original não é alterado."""
        df_original = pd.DataFrame({"col": [1]})
        colunas_antes = list(df_original.columns)

        padronizar_colunas_ano(df_original, 2022)

        assert list(df_original.columns) == colunas_antes

    def test_sufixo_ja_existente(self):
        """Testa que não adiciona sufixo duplicado."""
        df = pd.DataFrame({"NOME_22": [1]})
        resultado = padronizar_colunas_ano(df, 2022)

        assert list(resultado.columns) == ["NOME_22"]

    def test_ano_completo_e_curto(self):
        """Testa que reconhece sufixos com ano completo e abreviado."""
        df = pd.DataFrame({"COL_2022": [1], "COL2_22": [2]})
        resultado = padronizar_colunas_ano(df, 2022)

        assert "COL_2022" in resultado.columns
        assert "COL2_22" in resultado.columns


class TestAdicionarColunasVazias:
    """Testes para adicionar_colunas_vazias."""

    def test_adiciona_colunas_novas(self):
        """Testa adição de colunas inexistentes."""
        df = pd.DataFrame({"A": [1, 2]})
        resultado = adicionar_colunas_vazias(df, ["B", "C"])

        assert "B" in resultado.columns
        assert "C" in resultado.columns
        assert resultado["B"].isna().all()

    def test_nao_sobrescreve_existentes(self):
        """Testa que colunas existentes não são sobrescritas."""
        df = pd.DataFrame({"A": [1, 2]})
        resultado = adicionar_colunas_vazias(df, ["A", "B"])

        assert list(resultado["A"]) == [1, 2]
        assert resultado["B"].isna().all()

    def test_preserva_original(self):
        """Verifica que DataFrame original não é alterado."""
        df_original = pd.DataFrame({"A": [1]})
        adicionar_colunas_vazias(df_original, ["B"])

        assert "B" not in df_original.columns


class TestAnaliseNulos:
    """Testes para analise_nulos."""

    def test_identifica_nulos(self):
        """Testa identificação correta de valores nulos."""
        df = pd.DataFrame({"A": [1, None, 3], "B": [1, 2, 3]})
        resultado = analise_nulos(df)

        assert len(resultado) == 1
        assert resultado.iloc[0]["coluna"] == "A"
        assert resultado.iloc[0]["qtd_nulos"] == 1
        assert abs(resultado.iloc[0]["perc_nulos"] - 33.33) < 0.01

    def test_sem_nulos(self):
        """Testa DataFrame sem nulos."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        resultado = analise_nulos(df)

        assert len(resultado) == 0

    def test_ordenacao_por_percentual(self):
        """Testa que resultados são ordenados por percentual decrescente."""
        df = pd.DataFrame({"A": [1, None, None], "B": [None, 2, 3]})
        resultado = analise_nulos(df)

        assert resultado.iloc[0]["coluna"] == "A"  # 66.67% nulos
        assert resultado.iloc[1]["coluna"] == "B"  # 33.33% nulos


class TestCalcularIdade2023:
    """Testes para calcular_idade_2023."""

    def test_calculo_correto(self):
        """Testa cálculo correto de idade."""
        idade = calcular_idade_2023("05/15/2010")
        assert idade == 13

    def test_aniversario_apos_referencia(self):
        """Testa idade quando aniversário é após data de referência."""
        idade = calcular_idade_2023("06/01/2010")
        assert idade == 12  # Ainda não fez aniversário em 01/01/2023

    def test_data_invalida(self):
        """Testa tratamento de data inválida."""
        idade = calcular_idade_2023("invalid_date")
        assert idade is None

    def test_none_input(self):
        """Testa tratamento de None."""
        idade = calcular_idade_2023(None)
        assert idade is None


class TestObterElementosComuns:
    """Testes para obter_elementos_comuns."""

    def test_intersecao_basica(self):
        """Testa interseção básica entre DataFrames."""
        df1 = pd.DataFrame({"ID": [1, 2, 3]})
        df2 = pd.DataFrame({"ID": [2, 3, 4]})
        resultado = obter_elementos_comuns(df1, df2, "ID")

        assert sorted(resultado) == [2, 3]

    def test_sem_intersecao(self):
        """Testa DataFrames sem elementos comuns."""
        df1 = pd.DataFrame({"ID": [1, 2]})
        df2 = pd.DataFrame({"ID": [3, 4]})
        resultado = obter_elementos_comuns(df1, df2, "ID")

        assert resultado == []

    def test_intersecao_completa(self):
        """Testa DataFrames com total overlap."""
        df1 = pd.DataFrame({"ID": [1, 2, 3]})
        df2 = pd.DataFrame({"ID": [1, 2, 3]})
        resultado = obter_elementos_comuns(df1, df2, "ID")

        assert sorted(resultado) == [1, 2, 3]


class TestRenomearColunasAno:
    """Testes para renomear_colunas_ano."""

    def test_renomeacao_basica(self):
        """Testa renomeação de colunas."""
        df = pd.DataFrame({"col_antiga": [1, 2]})
        mapeamento = {"col_antiga": "COL_NOVA_22"}
        resultado = renomear_colunas_ano(df, mapeamento)

        assert "COL_NOVA_22" in resultado.columns
        assert "col_antiga" not in resultado.columns

    def test_preserva_original(self):
        """Testa que DataFrame original não é alterado."""
        df_original = pd.DataFrame({"A": [1]})
        renomear_colunas_ano(df_original, {"A": "B"})

        assert "A" in df_original.columns


class TestFiltrarColunasRelevantes:
    """Testes para filtrar_colunas_relevantes."""

    def test_filtragem_basica(self):
        """Testa filtro mantendo apenas colunas especificadas."""
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        resultado = filtrar_colunas_relevantes(df, ["A", "C"])

        assert list(resultado.columns) == ["A", "C"]

    def test_coluna_inexistente(self):
        """Testa que colunas inexistentes são ignoradas."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        resultado = filtrar_colunas_relevantes(df, ["A", "X"])

        assert list(resultado.columns) == ["A"]


class TestConsolidarDataframes:
    """Testes para consolidar_dataframes."""

    def test_consolidacao_basica(self):
        """Testa consolidação básica de 2 DataFrames."""
        df1 = pd.DataFrame({"NOME": ["Ana"], "NOTA_22": [8.5]})
        df2 = pd.DataFrame({"NOME": ["Ana"], "NOTA_23": [9.0]})
        resultado = consolidar_dataframes([df1, df2], "NOME")

        assert "NOME" in resultado.columns
        assert "NOTA_22" in resultado.columns
        assert "NOTA_23" in resultado.columns

    def test_lista_vazia(self):
        """Testa que retorna DataFrame vazio para lista vazia."""
        resultado = consolidar_dataframes([])
        assert len(resultado) == 0

    def test_merge_outer(self):
        """Testa que usa outer join (mantém todos os registros)."""
        df1 = pd.DataFrame({"NOME": ["Ana"], "NOTA": [8.5]})
        df2 = pd.DataFrame({"NOME": ["Bruno"], "NOTA": [7.0]})
        resultado = consolidar_dataframes([df1, df2], "NOME")

        assert len(resultado) == 2
        assert "Ana" in resultado["NOME"].values
        assert "Bruno" in resultado["NOME"].values
