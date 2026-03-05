"""
Testes unitários para o módulo scripts.notebook_feature_engineering.

Testa funções de feature engineering específicas do domínio educacional,
incluindo normalização de fases e criação de flags de condição.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.notebook_feature_engineering import (
    aplicar_transformacoes_fase_turma,
    criar_coluna_em_fase,
    criar_coluna_veterano,
    obter_nova_fase,
    obter_nova_fase_24,
    obter_nova_turma,
    obter_nova_turma_24,
)


class TestObterNovaTurma:
    """Testes para obter_nova_turma."""

    def test_turma_valida(self):
        """Testa extração de turma válida."""
        assert obter_nova_turma("8A") == "A"
        assert obter_nova_turma("1B") == "B"
        assert obter_nova_turma("3A") == "A"

    def test_turma_especial(self):
        """Testa casos especiais sem turma."""
        assert obter_nova_turma("ALFA") == "NÃO SE APLICA"
        assert obter_nova_turma("9") == "NÃO SE APLICA"

    def test_nan_input(self):
        """Testa tratamento de NaN."""
        assert obter_nova_turma(np.nan) == "NÃO SE APLICA"

    def test_case_insensitive(self):
        """Testa que conversão é case-insensitive."""
        assert obter_nova_turma("8a") == "A"
        assert obter_nova_turma("alfa") == "NÃO SE APLICA"


class TestObterNovaFase:
    """Testes para obter_nova_fase (anos 2022/2023)."""

    def test_fase_alfa(self):
        """Testa normalização de ALFA."""
        assert obter_nova_fase("ALFA") == "Fase 0 (1° e 2° ano)"

    def test_fases_com_turma(self):
        """Testa normalização de fases com turmas A/B."""
        assert obter_nova_fase("1A") == "Fase 1 (3° e 4° ano)"
        assert obter_nova_fase("1B") == "Fase 1 (3° e 4° ano)"
        assert obter_nova_fase("8A") == "Fase 8 (Universitários)"
        assert obter_nova_fase("8B") == "Fase 8 (Universitários)"

    def test_fase_9_descartada(self):
        """Testa que '9' retorna NÃO SE APLICA em 2022/2023."""
        assert obter_nova_fase("9") == "NÃO SE APLICA"

    def test_nan_invalido(self):
        """Testa tratamento de valores inválidos."""
        assert obter_nova_fase(np.nan) == "NÃO SE APLICA"
        assert obter_nova_fase("INVALIDO") == "NÃO SE APLICA"

    def test_case_insensitive(self):
        """Testa normalização case-insensitive."""
        assert obter_nova_fase("alfa") == "Fase 0 (1° e 2° ano)"
        assert obter_nova_fase("8a") == "Fase 8 (Universitários)"


class TestObterNovaFase24:
    """Testes para obter_nova_fase_24 (ano 2024)."""

    def test_fase_9_valida(self):
        """Testa que '9' é uma fase válida em 2024."""
        assert obter_nova_fase_24("9") == "Fase 4 (9° ano)"

    def test_outras_fases_iguais(self):
        """Testa que outras fases têm mesmo mapeamento de 2022/2023."""
        assert obter_nova_fase_24("ALFA") == "Fase 0 (1° e 2° ano)"
        assert obter_nova_fase_24("8A") == "Fase 8 (Universitários)"

    def test_nan_input(self):
        """Testa tratamento de NaN."""
        assert obter_nova_fase_24(np.nan) == "NÃO SE APLICA"


class TestObterNovaTurma24:
    """Testes para obter_nova_turma_24."""

    def test_comportamento_identico(self):
        """Testa que comportamento é idêntico a obter_nova_turma."""
        assert obter_nova_turma_24("8A") == obter_nova_turma("8A")
        assert obter_nova_turma_24("9") == obter_nova_turma("9")
        assert obter_nova_turma_24("ALFA") == obter_nova_turma("ALFA")


class TestCriarColunaVeterano:
    """Testes para criar_coluna_veterano."""

    def test_veterano_antes_corte(self):
        """Testa que alunos antes do corte são veteranos."""
        df = pd.DataFrame({"ANO_INGRESSO": [2020, 2021, 2022, 2023]})
        resultado = criar_coluna_veterano(df, "ANO_INGRESSO", ano_corte=2022)

        assert list(resultado["VETERANO"]) == [1, 1, 0, 0]

    def test_ano_corte_customizado(self):
        """Testa corte customizado."""
        df = pd.DataFrame({"ANO_INGRESSO": [2019, 2020, 2021]})
        resultado = criar_coluna_veterano(df, "ANO_INGRESSO", ano_corte=2020)

        assert list(resultado["VETERANO"]) == [1, 0, 0]

    def test_preserva_original(self):
        """Testa que DataFrame original não é alterado."""
        df_original = pd.DataFrame({"ANO_INGRESSO": [2020]})
        criar_coluna_veterano(df_original, "ANO_INGRESSO")

        assert "VETERANO" not in df_original.columns


class TestCriarColunaEmFase:
    """Testes para criar_coluna_em_fase."""

    def test_aluno_em_fase(self):
        """Testa alunos na fase ideal."""
        df = pd.DataFrame({
            "FASE_ATUAL": ["Fase 2", "Fase 1"],
            "FASE_IDEAL": ["Fase 2", "Fase 2"]
        })
        resultado = criar_coluna_em_fase(df, "FASE_ATUAL", "FASE_IDEAL")

        assert list(resultado["EM_FASE"]) == [1, 0]

    def test_case_sensitive(self):
        """Testa que comparação é case-sensitive."""
        df = pd.DataFrame({
            "FASE_ATUAL": ["fase 2", "Fase 2"],
            "FASE_IDEAL": ["Fase 2", "Fase 2"]
        })
        resultado = criar_coluna_em_fase(df, "FASE_ATUAL", "FASE_IDEAL")

        assert list(resultado["EM_FASE"]) == [0, 1]

    def test_preserva_original(self):
        """Testa que DataFrame original não é alterado."""
        df_original = pd.DataFrame({
            "FASE_ATUAL": ["Fase 2"],
            "FASE_IDEAL": ["Fase 2"]
        })
        criar_coluna_em_fase(df_original, "FASE_ATUAL", "FASE_IDEAL")

        assert "EM_FASE" not in df_original.columns


class TestAplicarTransformacoesFaseTurma:
    """Testes para aplicar_transformacoes_fase_turma."""

    def test_transformacao_2023(self):
        """Testa transformações para ano 2023."""
        df = pd.DataFrame({"FASE_ORIGINAL": ["8A", "ALFA", "9"]})
        resultado = aplicar_transformacoes_fase_turma(df, "FASE_ORIGINAL", ano=2023)

        assert "FASE_PADRONIZADA" in resultado.columns
        assert "TURMA" in resultado.columns
        assert resultado.loc[0, "FASE_PADRONIZADA"] == "Fase 8 (Universitários)"
        assert resultado.loc[2, "FASE_PADRONIZADA"] == "NÃO SE APLICA"

    def test_transformacao_2024(self):
        """Testa transformações para ano 2024."""
        df = pd.DataFrame({"FASE_ORIGINAL": ["9"]})
        resultado = aplicar_transformacoes_fase_turma(df, "FASE_ORIGINAL", ano=2024)

        assert resultado.loc[0, "FASE_PADRONIZADA"] == "Fase 4 (9° ano)"

    def test_sem_turma(self):
        """Testa opção de não criar coluna de turma."""
        df = pd.DataFrame({"FASE_ORIGINAL": ["8A"]})
        resultado = aplicar_transformacoes_fase_turma(
            df, "FASE_ORIGINAL", criar_turma=False
        )

        assert "FASE_PADRONIZADA" in resultado.columns
        assert "TURMA" not in resultado.columns

    def test_preserva_original(self):
        """Testa que DataFrame original não é alterado."""
        df_original = pd.DataFrame({"FASE": ["8A"]})
        aplicar_transformacoes_fase_turma(df_original, "FASE")

        assert "FASE_PADRONIZADA" not in df_original.columns


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_strings_com_espacos(self):
        """Testa tratamento de strings com espaços."""
        assert obter_nova_fase(" 8A ") == "Fase 8 (Universitários)"
        assert obter_nova_turma(" 8A ") == "A"

    def test_valores_numericos(self):
        """Testa entrada numérica."""
        df = pd.DataFrame({"FASE": [8, 1]})
        # Deve converter para string internamente
        df["FASE_NORM"] = df["FASE"].apply(lambda x: obter_nova_fase(str(x)))
        assert "NÃO SE APLICA" in df["FASE_NORM"].values

    def test_dataframe_vazio(self):
        """Testa transformações em DataFrame vazio."""
        df_vazio = pd.DataFrame({"FASE": []})
        resultado = aplicar_transformacoes_fase_turma(df_vazio, "FASE")

        assert len(resultado) == 0
        assert "FASE_PADRONIZADA" in resultado.columns
