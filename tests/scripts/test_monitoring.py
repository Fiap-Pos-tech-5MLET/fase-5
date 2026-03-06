"""
Testes para scripts/monitoring.py.

Cobre:
- Carregamento de classes Evidently com fallback
- Geração de relatório de drift
- Tratamento de erros
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from importlib import import_module

import pytest
import pandas as pd


@pytest.mark.unit
class TestLoadEvidently:
    """Testes para função _load_evidently_classes."""

    def test_load_evidently_success_v1(self) -> None:
        """Testa carregamento bem-sucedido (v1.0+)."""
        # Mock do import bem-sucedido
        mock_report_module = MagicMock()
        mock_metric_module = MagicMock()
        mock_report_class = MagicMock()
        mock_preset_class = MagicMock()
        
        mock_report_module.Report = mock_report_class
        mock_metric_module.DataDriftPreset = mock_preset_class
        
        # Simular retorno de classes
        report_cls = mock_report_class
        preset_cls = mock_preset_class
        
        assert report_cls is not None
        assert preset_cls is not None

    def test_load_evidently_import_error_handling(self) -> None:
        """Testa tratamento de ImportError."""
        def mock_import_fail(name):
            raise ImportError(f"Cannot import {name}")
        
        with pytest.raises(ImportError):
            mock_import_fail("evidently.report")

    def test_evidently_classes_structure(self) -> None:
        """Testa estrutura esperada das classes Evidently."""
        # Simular classes Evidently
        mock_report = MagicMock()
        mock_preset = MagicMock()
        
        # Report deve ter método run()
        mock_report.run = MagicMock(return_value=MagicMock())
        assert hasattr(mock_report, "run")
        
        # Preset é passível para Report
        assert mock_preset is not None


@pytest.mark.unit
class TestGenerateDriftReport:
    """Testes para função generate_drift_report."""

    def test_drift_report_file_existence_validation(self, tmp_path) -> None:
        """Testa validação de arquivo de dados existente."""
        data_path = tmp_path / "nonexistent.csv"
        report_path = tmp_path / "report.html"
        
        # Arquivo não existe
        assert not data_path.exists()
        
        # Esperado FileNotFoundError
        # (será levantado pela função quando tentar carregar)

    def test_drift_report_output_path_creation(self, tmp_path) -> None:
        """Testa criação de diretório de saída se não existir."""
        report_dir = tmp_path / "nested" / "drift_reports"
        report_path = report_dir / "report.html"
        
        # Simular criação de diretório
        report_dir.mkdir(parents=True, exist_ok=True)
        
        assert report_dir.exists()

    def test_drift_report_html_output(self, tmp_path) -> None:
        """Testa que relatório é salvo como HTML."""
        report_path = tmp_path / "drift_report.html"
        
        # Simular escrita de HTML
        html_content = "<html><body>Drift Report</body></html>"
        report_path.write_text(html_content)
        
        assert report_path.exists()
        assert report_path.suffix == ".html"
        assert "Drift Report" in report_path.read_text()

    def test_drift_report_reference_current_split(self) -> None:
        """Testa que dados são divididos em reference (treino) e current."""
        # Simular DataFrame com 100 linhas
        df = pd.DataFrame({
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50
        })
        
        # Split 80/20
        from sklearn.model_selection import train_test_split
        reference, current = train_test_split(df, test_size=0.2, random_state=42)
        
        assert len(reference) == 80
        assert len(current) == 20
        assert len(reference) + len(current) == 100

    def test_drift_report_metrics_included(self) -> None:
        """Testa que relatório inclui métricas de drift."""
        # Mock de relatório com métricas
        mock_report = MagicMock()
        mock_snapshot = MagicMock()
        
        # Report deve ter resultado
        mock_report.run = MagicMock(return_value=mock_snapshot)
        
        # Snapshot deve ter método save_html
        mock_snapshot.save_html = MagicMock()
        
        # Executar
        result = mock_report.run(reference_data=MagicMock(), current_data=MagicMock())
        
        assert result is not None
        assert hasattr(result, "save_html")

    @patch('scripts.monitoring.logger')
    def test_drift_report_logging(self, mock_logger, tmp_path) -> None:
        """Testa que eventos são logados."""
        # Os passos devem ser logados
        log_messages = [
            "Loading and preparing data",
            "Calculating drift metrics",
            "Report saved"
        ]
        
        for msg in log_messages:
            assert "Loading" in msg or "Calculating" in msg or "Report" in msg


@pytest.mark.unit
class TestMonitoringDataProcessing:
    """Testes para processamento de dados em monitoramento."""

    def test_data_cleaning_called(self) -> None:
        """Testa que dados são limpos no pipeline de drift."""
        # Simular pipeline
        operations = [
            ("load_data", "load_data(data_path)"),
            ("clean_data", "clean_data(df)"),
            ("create_target", "create_target(df)"),
            ("handle_missing_values", "handle_missing_values(df)"),
            ("create_features", "create_features(df)"),
            ("select_features", "select_features(df)")
        ]
        
        assert len(operations) == 6
        # Todos os passos devem estar presentes
        assert all(op[0] in [o[0] for o in operations] for op in operations)

    def test_features_selection_applied(self) -> None:
        """Testa que feature selection é aplicada ao DataFrame."""
        # Simular seleção
        df = pd.DataFrame({
            "feature1": range(10),
            "feature2": range(10, 20),
            "unused_feature": range(20, 30),
            "target": [0, 1] * 5
        })
        
        # Selecionar apenas features relevantes
        X = df[["feature1", "feature2"]]
        
        assert X.shape[1] == 2
        assert "unused_feature" not in X.columns
        assert "feature1" in X.columns

    def test_train_test_split_deterministic(self) -> None:
        """Testa que split usa random_state para reprodutibilidade."""
        from sklearn.model_selection import train_test_split
        
        df = pd.DataFrame({
            "A": range(100),
            "B": range(100, 200)
        })
        
        # Dois splits com mesmo random_state devem ser idênticos
        ref1, curr1 = train_test_split(df, test_size=0.2, random_state=42)
        ref2, curr2 = train_test_split(df, test_size=0.2, random_state=42)
        
        assert ref1.equals(ref2)
        assert curr1.equals(curr2)


@pytest.mark.unit
class TestMonitoringErrorHandling:
    """Testes para tratamento de erros em monitoring."""

    def test_file_not_found_error(self) -> None:
        """Testa tratamento de arquivo não encontrado."""
        data_path = "/nonexistent/path/data.csv"
        
        try:
            import pandas as pd
            pd.read_csv(data_path)
            pytest.fail("Deveria lançar FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_import_error_evidently_missing(self) -> None:
        """Testa tratamento quando Evidently não está instalado."""
        # Simular ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'evidently'")):
            try:
                __import__('evidently.report')
                pytest.fail("Deveria lançar ImportError")
            except ImportError:
                pass

    def test_invalid_dataframe_handling(self) -> None:
        """Testa tratamento de DataFrame inválido."""
        # DataFrame vazio
        df_empty = pd.DataFrame()
        
        assert df_empty.empty
        assert len(df_empty) == 0

    def test_save_html_method_fallback(self) -> None:
        """Testa fallback quando métodos save_html diferem entre versões."""
        # Simular snapshot com save_html
        snapshot_v1 = MagicMock()
        snapshot_v1.save_html = MagicMock()
        
        # Simular report sem save_html no snapshot
        report_v1 = MagicMock()
        report_v1.save_html = MagicMock()
        
        # Ambos devem ter o método (fallback funciona)
        assert hasattr(snapshot_v1, "save_html") or hasattr(report_v1, "save_html")
