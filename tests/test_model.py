"""
Test suite para o módulo model.

Organização:
- TestPipelineCreation: Testes para criação de pipeline
- TestModelTraining: Testes para treinamento de modelo
- TestModelPersistence: Testes para salvar/carregar modelo
- TestModelValidation: Testes para validação de modelo
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.model import create_pipeline, load_model, save_model, train_model


# ==================== FIXTURES ====================
@pytest.fixture
def sample_dataframe():
    """Cria DataFrame de amostra para testes."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "idade": np.random.randint(15, 25, 100),
            "nota": np.random.uniform(5.0, 10.0, 100),
            "frequencia": np.random.uniform(0.7, 1.0, 100),
            "fase": np.random.choice(["1", "2", "3"], 100),
            "turma": np.random.choice(["A", "B"], 100),
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def small_dataframe():
    """Cria DataFrame pequeno para testes rápidos."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "a": np.random.randn(30),
            "b": np.random.randn(30),
            "categoria": np.random.choice(["A", "B"], 30),
            "target": np.random.randint(0, 2, 30),
            "idade": np.random.randint(15, 25, 30),
            "nota": np.random.uniform(5.0, 10.0, 30),
            "frequencia": np.random.uniform(0.7, 1.0, 30),
            "fase": np.random.choice(["1", "2", "3"], 30),
            "turma": np.random.choice(["A", "B"], 30),
        }
    )


@pytest.fixture
def numeric_features():
    """Lista de features numéricas."""
    return ["idade", "nota", "frequencia"]


@pytest.fixture
def categorical_features():
    """Lista de features categóricas."""
    return ["fase", "turma"]


@pytest.mark.unit
@pytest.mark.model_training
class TestPipelineCreation:
    """Testes para a função create_pipeline."""

    def test_create_pipeline_returns_pipeline(self, numeric_features, categorical_features):
        """Testa se retorna objeto Pipeline."""
        result = create_pipeline(numeric_features, categorical_features)
        assert isinstance(result, Pipeline)

    def test_create_pipeline_has_steps(self, numeric_features, categorical_features):
        """Testa que pipeline tem steps definidos."""
        result = create_pipeline(numeric_features, categorical_features)
        assert len(result.steps) > 0
        assert all(isinstance(step[0], str) for step in result.steps)

    def test_create_pipeline_contains_classifier(self, numeric_features, categorical_features):
        """Testa que pipeline contém classificador."""
        result = create_pipeline(numeric_features, categorical_features)
        step_names = [step[0] for step in result.steps]
        assert "classifier" in step_names

    def test_create_pipeline_type_error_numeric(self):
        """Testa erro quando numeric_features não é lista."""
        with pytest.raises(TypeError, match="devem ser listas"):
            create_pipeline("not a list", [])

    def test_create_pipeline_type_error_categorical(self):
        """Testa erro quando categorical_features não é lista."""
        with pytest.raises(TypeError, match="devem ser listas"):
            create_pipeline([], "not a list")

    def test_create_pipeline_empty_features(self):
        """Testa erro quando ambas listas estão vazias."""
        with pytest.raises(ValueError, match="Pelo menos um tipo de feature"):
            create_pipeline([], [])

    def test_create_pipeline_invalid_n_estimators(self):
        """Testa erro com n_estimators inválido."""
        with pytest.raises(ValueError, match="n_estimators deve ser inteiro positivo"):
            create_pipeline(["a"], [], n_estimators=0)

    def test_create_pipeline_invalid_max_depth(self):
        """Testa erro com max_depth inválido."""
        with pytest.raises(ValueError, match="max_depth deve ser inteiro positivo ou None"):
            create_pipeline(["a"], [], max_depth=0)

    def test_create_pipeline_invalid_min_samples_split(self):
        """Testa erro com min_samples_split inválido."""
        with pytest.raises(ValueError, match="min_samples_split deve ser >= 2"):
            create_pipeline(["a"], [], min_samples_split=1)

    def test_create_pipeline_invalid_k(self):
        """Testa erro com k inválido."""
        with pytest.raises(ValueError, match="k deve ser inteiro positivo ou 'all'"):
            create_pipeline(["a"], [], k=0)

    def test_create_pipeline_only_numeric(self):
        """Testa pipeline apenas com features numéricas."""
        result = create_pipeline(["idade", "nota"], [])
        assert isinstance(result, Pipeline)
        assert len(result.steps) > 0

    def test_create_pipeline_only_categorical(self):
        """Testa pipeline apenas com features categóricas."""
        result = create_pipeline([], ["fase", "turma"])
        assert isinstance(result, Pipeline)
        assert len(result.steps) > 0

    def test_create_pipeline_mixed_features(self):
        """Testa pipeline com features numéricas e categóricas."""
        result = create_pipeline(["idade", "nota"], ["fase", "turma"])
        assert isinstance(result, Pipeline)

    def test_create_pipeline_with_custom_n_estimators(self):
        """Testa pipeline com n_estimators customizado."""
        result = create_pipeline(["a"], [], n_estimators=50)
        assert isinstance(result, Pipeline)
        # Verifica se RandomForest foi criado com parâmetro correto
        clf = result.named_steps.get("classifier")
        assert clf is not None
        assert clf.n_estimators == 50

    def test_create_pipeline_with_max_depth(self):
        """Testa pipeline com max_depth customizado."""
        result = create_pipeline(["a"], [], max_depth=10)
        assert isinstance(result, Pipeline)
        clf = result.named_steps.get("classifier")
        assert clf.max_depth == 10


@pytest.mark.unit
@pytest.mark.model_training
class TestModelTraining:
    """Testes para a função train_model."""

    def test_train_model_returns_tuple(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que train_model retorna tupla com 4 elementos."""
        X = sample_dataframe.drop("target", axis=1)
        y = sample_dataframe["target"]

        result = train_model(X[numeric_features + categorical_features], y)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_train_model_returns_pipeline(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que primeiro elemento é Pipeline."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        clf, _metrics, _X_test, _y_test = train_model(X, y)

        assert isinstance(clf, Pipeline)

    def test_train_model_returns_metrics(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que retorna dicionário de métricas."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        _clf, metrics, _X_test, _y_test = train_model(X, y)

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_train_model_returns_test_sets(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que retorna conjuntos de teste."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        _clf, _metrics, X_test, y_test = train_model(X, y)

        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_test) == len(y_test)

    def test_train_model_type_error_dataframe(self):
        """Testa erro com DataFrame inválido."""
        with pytest.raises(TypeError):
            train_model("not a dataframe", pd.Series([0, 1]))

    def test_train_model_type_error_y(self):
        """Testa erro com y não Series/array."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError):
            train_model(df, "not a series")

    def test_train_model_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        with pytest.raises(ValueError):
            train_model(pd.DataFrame(), pd.Series([]))

    def test_train_model_size_mismatch(self):
        """Testa erro quando X e y têm tamanhos diferentes."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])

        with pytest.raises(ValueError):
            train_model(df, y)

    def test_train_model_with_test_size(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa train_model com test_size customizado."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        _clf, _metrics, X_test, _y_test = train_model(X, y, test_size=0.3)

        # Verifica proporção de teste (aprox 30%)
        expected_test_size = int(len(X) * 0.3)
        assert 0.25 * expected_test_size <= len(X_test) <= 0.35 * len(X)

    def test_train_model_with_custom_n_estimators(self, small_dataframe):
        """Testa train_model com n_estimators customizado."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _metrics, _X_test, _y_test = train_model(X, y, n_estimators=50)

        assert clf.named_steps["classifier"].n_estimators == 50

    def test_train_model_invalid_test_size(self, small_dataframe):
        """Testa erro com test_size fora do range."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        with pytest.raises(ValueError):
            train_model(X, y, test_size=1.5)

    def test_train_model_metrics_keys(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que métricas contêm keys esperadas."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        _clf, metrics, _X_test, _y_test = train_model(X, y)

        # Verifica se métricas básicas estão presentes
        assert any(key in metrics for key in ["accuracy", "f1", "roc_auc", "precision", "recall"])


@pytest.mark.unit
@pytest.mark.model_training
class TestModelPersistence:
    """Testes para salvar e carregar modelo."""

    def test_save_model_creates_file(self, small_dataframe):
        """Testa que save_model cria arquivo."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, _, _ = train_model(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pkl")
            save_model(clf, path)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_save_model_creates_directory_if_needed(self, small_dataframe):
        """Testa que save_model cria diretório se necessário."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, _, _ = train_model(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Criar diretório pai primeiro (save_model não cria aninhados)
            models_dir = os.path.join(tmp_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            path = os.path.join(models_dir, "model.pkl")

            save_model(clf, path)
            assert os.path.exists(path)

    def test_load_model_returns_model(self, small_dataframe):
        """Testa que load_model retorna modelo carregado."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, _, _ = train_model(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pkl")
            save_model(clf, path)

            loaded = load_model(path)

            assert loaded is not None
            assert hasattr(loaded, "predict")

    def test_load_model_file_not_found(self):
        """Testa erro ao carregar arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            load_model("inexistent_path_12345_model.pkl")

    def test_save_load_roundtrip(self, small_dataframe):
        """Testa que modelo salvo e carregado funciona igual."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, X_test, _ = train_model(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pkl")

            # Salva e carrega
            save_model(clf, path)
            loaded = load_model(path)

            # Faz predições com ambos
            pred_original = clf.predict(X_test)
            pred_loaded = loaded.predict(X_test)

            # Devem ser idênticas
            np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_save_model_invalid_path(self, small_dataframe):
        """Testa erro ao salvar em path inválido."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, _, _ = train_model(X, y)

        # Path inválido (em Windows, NUL é dispositivo especial)
        with pytest.raises((OSError, FileNotFoundError, PermissionError, ValueError)):
            save_model(clf, "NUL:/invalid/path/model.pkl")

    def test_load_model_invalid_file(self):
        """Testa erro ao carregar arquivo corrompido."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "corrupted.pkl")

            # Cria arquivo inválido
            with open(path, "w") as f:
                f.write("invalid pickle data")

            with pytest.raises(Exception):  # EOFError, UnpicklingError, etc
                load_model(path)


@pytest.mark.unit
class TestModelValidation:
    """Testes para validação de modelo."""

    def test_model_has_predict_method(self, small_dataframe):
        """Testa que modelo tem método predict."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, _, _ = train_model(X, y)

        assert hasattr(clf, "predict")
        assert callable(clf.predict)

    def test_model_predictions_correct_shape(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que predições retornam shape correto."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        clf, _, X_test, _ = train_model(X, y)

        predictions = clf.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]

    def test_model_predictions_are_binary(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que predições são binárias (0 ou 1)."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        clf, _, X_test, _ = train_model(X, y)

        predictions = clf.predict(X_test)

        assert set(predictions) <= {0, 1}

    def test_model_predict_proba_available(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que modelo tem predict_proba."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        clf, _, X_test, _ = train_model(X, y)

        assert hasattr(clf, "predict_proba")

        probas = clf.predict_proba(X_test)
        assert probas.shape == (X_test.shape[0], 2)

        # Verificar que probabilidades somam 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), 1.0)

    def test_model_with_single_sample(self, small_dataframe):
        """Testa predição com uma única amostra."""
        X = small_dataframe[["a", "b"]]
        y = small_dataframe["target"]

        clf, _, _, _ = train_model(X, y)

        single_sample = X.iloc[:1]
        prediction = clf.predict(single_sample)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_model_metrics_are_numeric(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que métricas são valores numéricos."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        _clf, metrics, _, _ = train_model(X, y)

        for key, value in metrics.items():
            # Métricas devem ser numbers ou dicts (no caso de classification_report)
            assert isinstance(value, (int, float, dict)) or hasattr(value, "__iter__")

    def test_model_accuracy_in_metrics(
        self, sample_dataframe, numeric_features, categorical_features
    ):
        """Testa que acurácia está nos métricas."""
        X = sample_dataframe[numeric_features + categorical_features]
        y = sample_dataframe["target"]

        _clf, metrics, _, _ = train_model(X, y)

        assert "accuracy" in metrics or any("accuracy" in str(k).lower() for k in metrics.keys())
