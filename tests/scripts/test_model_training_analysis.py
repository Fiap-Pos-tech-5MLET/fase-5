"""Testes para utilitários de modelagem do notebook de treinamento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.model_training_analysis import (
    criar_modelo_lgbm,
    executar_validacao_cruzada_lgbm,
    preparar_dados_modelagem,
    resumir_resultados_cv,
    treinar_e_avaliar_lgbm,
)


@pytest.fixture(name="dataset_sintetico")
def fixture_dataset_sintetico() -> pd.DataFrame:
    """Gera dataframe sintético com features numéricas e categóricas."""
    rng = np.random.default_rng(42)
    n_amostras = 180
    df = pd.DataFrame(
        {
            "idade": rng.integers(8, 18, size=n_amostras),
            "iaa": rng.normal(6.5, 1.2, size=n_amostras),
            "ieg": rng.normal(6.0, 1.1, size=n_amostras),
            "ips": rng.normal(5.8, 1.0, size=n_amostras),
            "genero": rng.choice(["M", "F"], size=n_amostras),
            "evadiu": np.array([0] * 126 + [1] * 54),
        }
    )
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


@pytest.mark.unit
class TestPrepararDadosModelagem:
    """Valida preparação de dados para treino/teste."""

    def test_preparar_dados_usa_apenas_numericas(self, dataset_sintetico: pd.DataFrame) -> None:
        """Ignora colunas categóricas quando feature_columns não é informado."""
        X_train, X_test, y_train, y_test, features = preparar_dados_modelagem(dataset_sintetico)

        assert all(col in ["idade", "iaa", "ieg", "ips"] for col in features)
        assert "genero" not in features
        assert len(X_train) + len(X_test) == len(dataset_sintetico)
        assert len(y_train) + len(y_test) == len(dataset_sintetico)

    def test_preparar_dados_com_features_explicitas(self, dataset_sintetico: pd.DataFrame) -> None:
        """Seleciona apenas features válidas explicitamente informadas."""
        features_alvo = ["idade", "iaa", "inexistente"]
        _X_train, _X_test, _y_train, _y_test, features = preparar_dados_modelagem(
            dataset_sintetico,
            feature_columns=features_alvo,
        )

        assert features == ["idade", "iaa"]


@pytest.mark.unit
class TestValidacaoCruzadaLgbm:
    """Valida execução e formato das métricas de cross-validation."""

    def test_execucao_retorna_metricas_esperadas(self, dataset_sintetico: pd.DataFrame) -> None:
        """Retorna métricas de treino/validação para cada fold."""
        X_train, _X_test, y_train, _y_test, _features = preparar_dados_modelagem(dataset_sintetico)
        modelo = criar_modelo_lgbm({"n_estimators": 20, "learning_rate": 0.05})

        resultados = executar_validacao_cruzada_lgbm(
            X_train=X_train,
            y_train=y_train,
            modelo=modelo,
            n_splits=3,
            random_state=42,
        )

        for dataset in ["train", "validation"]:
            assert dataset in resultados
            assert "accuracy" in resultados[dataset]
            assert "log_loss" in resultados[dataset]
            assert len(resultados[dataset]["accuracy"]) == 3
            assert len(resultados[dataset]["pr_auc"]) == 3


@pytest.mark.unit
class TestResumoEAvaliacaoFinal:
    """Valida resumo de CV e avaliação final."""

    def test_resumo_cv_gera_dataframe(self, dataset_sintetico: pd.DataFrame) -> None:
        """Converte resultados de folds para tabela com médias e desvios."""
        X_train, _X_test, y_train, _y_test, _features = preparar_dados_modelagem(dataset_sintetico)
        modelo = criar_modelo_lgbm({"n_estimators": 20})

        resultados = executar_validacao_cruzada_lgbm(X_train, y_train, modelo, n_splits=3)
        resumo = resumir_resultados_cv(resultados)

        assert not resumo.empty
        assert {"metrica", "treino_media", "validacao_media"}.issubset(set(resumo.columns))

    def test_treino_final_retorna_metricas_teste(self, dataset_sintetico: pd.DataFrame) -> None:
        """Treina modelo final e retorna métricas em faixa válida."""
        X_train, X_test, y_train, y_test, _features = preparar_dados_modelagem(dataset_sintetico)
        modelo = criar_modelo_lgbm({"n_estimators": 30})

        resultados = treinar_e_avaliar_lgbm(X_train, X_test, y_train, y_test, modelo)

        metricas = resultados["test_metrics"]
        assert {
            "accuracy",
            "f1_score",
            "recall",
            "precision",
            "log_loss",
            "roc_auc",
            "pr_auc",
        }.issubset(metricas.keys())
        assert 0.0 <= metricas["accuracy"] <= 1.0
        assert 0.0 <= metricas["f1_score"] <= 1.0
        assert metricas["log_loss"] >= 0.0
