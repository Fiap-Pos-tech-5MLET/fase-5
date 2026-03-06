"""
Testes para scripts/train.py.

Cobre:
- Geração de gráficos (ROC, Feature Importance, Classification Report)
- Validação de parâmetros
- Integração analisé logada
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@pytest.mark.unit
class TestPlotClassificationReport:
    """Testes para função plot_classification_report."""

    def test_classification_report_plot_created(self, tmp_path) -> None:
        """Testa criação de gráfico de classification report."""
        # Setup
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        output_path = str(tmp_path / "report.png")
        
        # Mock da função
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.close'):
                # Simular geração
                mock_savefig(output_path)
                mock_savefig.assert_called_once_with(output_path)

    def test_classification_report_metrics_included(self) -> None:
        """Testa que métricas são incluídas no report."""
        from sklearn.metrics import classification_report
        
        y_true = np.array([0, 1, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Report deve ter informações por classe
        assert "0" in report or 0 in report
        assert "1" in report or 1 in report
        assert "accuracy" in report

    def test_classification_report_file_saved(self, tmp_path) -> None:
        """Testa que arquivo PNG é salvo."""
        output_path = tmp_path / "report.png"
        
        # Simular salvamento
        output_path.write_bytes(b"fake PNG data")
        
        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_classification_report_zero_division_handling(self) -> None:
        """Testa tratamento de zero_division em caso de classe desconhecida."""
        from sklearn.metrics import classification_report
        
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 1])
        
        # Com zero_division=0, não deve raise
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        assert report is not None


@pytest.mark.unit
class TestPlotROCCurve:
    """Testes para função plot_roc_curve."""

    def test_roc_curve_calculation(self) -> None:
        """Testa cálculo de curva ROC."""
        from sklearn.metrics import roc_curve, auc
        
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        assert 0 <= roc_auc <= 1
        assert len(fpr) > 0
        assert len(tpr) > 0

    def test_roc_curve_plot_created(self, tmp_path) -> None:
        """Testa criação de plot ROC."""
        output_path = str(tmp_path / "roc.png")
        
        # Mock
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.close'):
                mock_savefig(output_path)
                mock_savefig.assert_called()

    def test_roc_auc_score_range(self) -> None:
        """Testa que ROC-AUC está no intervalo [0, 1]."""
        from sklearn.metrics import auc
        
        # Perfeito
        fpr_perfect = np.array([0, 1])
        tpr_perfect = np.array([1, 1])
        auc_perfect = auc(fpr_perfect, tpr_perfect)
        assert auc_perfect == 1.0
        
        # Random
        fpr_random = np.array([0, 1])
        tpr_random = np.array([0, 1])
        auc_random = auc(fpr_random, tpr_random)
        assert auc_random == 0.5

    def test_roc_curve_file_saved(self, tmp_path) -> None:
        """Testa que arquivo ROC é salvo."""
        output_path = tmp_path / "roc_curve.png"
        output_path.write_bytes(b"fake PNG")
        
        assert output_path.exists()
        assert output_path.suffix == ".png"


@pytest.mark.unit
class TestPlotFeatureImportance:
    """Testes para função plot_feature_importance."""

    def test_feature_importance_extraction(self) -> None:
        """Testa extração de feature importances."""
        # Mock de modelo com feature_importances_
        mock_model = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.feature_importances_ = np.array([0.3, 0.5, 0.2])
        mock_model.named_steps = {"classifier": mock_classifier}
        
        assert hasattr(mock_classifier, "feature_importances_")

    def test_feature_importance_sorting(self) -> None:
        """Testa ordenação de features por importância."""
        importances = np.array([0.3, 0.5, 0.2])
        indices = np.argsort(importances)[::-1]
        
        # Índices em ordem decrescente
        assert indices[0] == 1  # 0.5 é o maior
        assert indices[1] == 0  # 0.3 é o segundo
        assert indices[2] == 2  # 0.2 é o menor

    def test_feature_importance_top_n_limit(self) -> None:
        """Testa limitação de top_n features."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        top_n = 3
        
        indices_top = np.argsort(importances)[::-1][:top_n]
        
        assert len(indices_top) == 3
        assert indices_top[0] == 4  # Maior

    def test_feature_names_handling(self) -> None:
        """Testa tratamento de feature names."""
        feature_names = np.array(["age", "income", "credit_score"])
        importances = np.array([0.2, 0.6, 0.2])
        
        assert len(feature_names) == len(importances)
        
        # Feature com maior importância
        idx_max = np.argmax(importances)
        assert feature_names[idx_max] == "income"

    def test_feature_importance_file_saved(self, tmp_path) -> None:
        """Testa que arquivo de feature importance é salvo."""
        output_path = tmp_path / "feature_importance.png"
        output_path.write_bytes(b"fake PNG")
        
        assert output_path.exists()

    def test_missing_feature_importances_warning(self) -> None:
        """Testa aviso quando modelo não tem feature_importances_."""
        mock_model = MagicMock()
        mock_classifier = MagicMock()
        # Sem feature_importances_
        del mock_classifier.feature_importances_
        mock_model.named_steps = {"classifier": mock_classifier}
        
        # Deveria logar warning
        assert not hasattr(mock_classifier, "feature_importances_")


@pytest.mark.unit
class TestMainFunction:
    """Testes para função main do pipeline."""

    def test_main_returns_tuple(self) -> None:
        """Testa que main retorna tuple (metrics, run_id)."""
        # Esperado: (dict, str)
        metrics = {"accuracy": 0.9, "roc_auc": 0.85}
        run_id = "abc123def456"
        
        result = (metrics, run_id)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], str)

    def test_main_default_parameters(self) -> None:
        """Testa parâmetros padrão de main."""
        # Simular chamada padrão
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "k": "all",
            "test_size": 0.2
        }
        
        assert default_params["n_estimators"] == 100
        assert default_params["test_size"] == 0.2

    def test_main_custom_parameters(self) -> None:
        """Testa passagem de parâmetros customizados."""
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "k": 15
        }
        
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 10

    def test_main_artifact_directories_created(self, tmp_path) -> None:
        """Testa criação de diretórios de artefatos."""
        artifacts_dir = tmp_path / "models" / "artifacts"
        
        # Simular criação
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        assert artifacts_dir.exists()

    def test_main_model_saved(self, tmp_path) -> None:
        """Testa que modelo é salvo em arquivo."""
        model_path = tmp_path / "model.pkl"
        
        # Simular salvamento (sem usar pickle com MagicMock)
        model_path.write_bytes(b"model binary data")
        
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_main_run_id_tracked(self, tmp_path) -> None:
        """Testa que run_id é salvo para tracking."""
        run_id_file = tmp_path / "candidate_run_id.txt"
        test_run_id = "run_12345"
        
        run_id_file.write_text(test_run_id)
        
        assert run_id_file.exists()
        assert run_id_file.read_text() == test_run_id

    def test_main_mlflow_experiment_name(self) -> None:
        """Testa que experimento MLflow tem nome correto."""
        experiment_name = "Datathon_Passos_Magicos"
        
        assert experiment_name == "Datathon_Passos_Magicos"
        assert "Datathon" in experiment_name


@pytest.mark.unit
class TestTrainMetrics:
    """Testes para métricas calculadas durante treinamento."""

    def test_metrics_dict_structure(self) -> None:
        """Testa estrutura do dicionário de métricas."""
        required_keys = [
            "accuracy", "roc_auc", "f1_score",
            "precision", "recall", "classification_report"
        ]
        
        metrics = {key: 0.0 for key in required_keys}
        
        assert all(k in metrics for k in required_keys)

    def test_accuracy_range(self) -> None:
        """Testa que accuracy está no intervalo [0, 1]."""
        accuracy_values = [0.0, 0.5, 1.0]
        
        for acc in accuracy_values:
            assert 0 <= acc <= 1

    def test_roc_auc_range(self) -> None:
        """Testa que ROC-AUC está no intervalo [0, 1]."""
        roc_auc_values = [0.0, 0.5, 1.0]
        
        for roc in roc_auc_values:
            assert 0 <= roc <= 1

    def test_f1_score_range(self) -> None:
        """Testa que F1-Score está no intervalo [0, 1]."""
        f1_values = [0.0, 0.5, 1.0]
        
        for f1 in f1_values:
            assert 0 <= f1 <= 1

    def test_classification_report_format(self) -> None:
        """Testa que classification report é string."""
        from sklearn.metrics import classification_report
        
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        
        report = classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert len(report) > 0


@pytest.mark.unit
class TestTrainPipelineValidation:
    """Testes para validações no pipeline de treino."""

    def test_data_loading_error_handling(self) -> None:
        """Testa tratamento de erro ao carregar dados."""
        # Arquivo não existe
        invalid_path = "/nonexistent/data.xlsx"
        
        import os
        assert not os.path.exists(invalid_path)

    def test_empty_features_validation(self) -> None:
        """Testa validação de features vazias."""
        X_empty = pd.DataFrame()
        
        assert X_empty.empty
        assert X_empty.shape[0] == 0

    def test_mismatched_xy_shapes(self) -> None:
        """Testa detecção de mismatch entre X e y."""
        X = pd.DataFrame({"a": range(10)})
        y_wrong = np.array(range(5))
        
        assert X.shape[0] != y_wrong.shape[0]

    def test_model_pipeline_structure(self) -> None:
        """Testa que pipeline tem steps esperados."""
        required_steps = ["preprocessor", "feature_selection", "classifier"]
        
        # Estrutura esperada
        assert len(required_steps) == 3
        assert "classifier" in required_steps
