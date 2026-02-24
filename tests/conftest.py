"""
Pytest configuration and fixtures for the entire test suite.
"""

import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock mlflow before any app imports to avoid hang during collection
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["scripts.train"] = MagicMock()

# Create robust streamlit mock
_streamlit_mock = MagicMock()
_streamlit_mock.components = MagicMock()
_streamlit_mock.components.v1 = MagicMock()
_streamlit_mock.cache_resource = lambda func: func
_streamlit_mock.cache_data = lambda func: func
_streamlit_mock.error = MagicMock()
_streamlit_mock.set_page_config = MagicMock()
_streamlit_mock.sidebar = MagicMock()
_streamlit_mock.markdown = MagicMock()
sys.modules["streamlit"] = _streamlit_mock
sys.modules["streamlit.components"] = _streamlit_mock.components
sys.modules["streamlit.components.v1"] = _streamlit_mock.components.v1

sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["scripts.monitoring"] = MagicMock()

# Add project root and directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "app"))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Testes unitários para funções/classes individuais")
    config.addinivalue_line("markers", "integration: Testes de integração")
    config.addinivalue_line("markers", "slow: Testes que demoram tempo")
    config.addinivalue_line("markers", "data_loading: Testes de carregamento de dados")
    config.addinivalue_line("markers", "data_cleaning: Testes de limpeza de dados")
    config.addinivalue_line("markers", "feature_engineering: Testes de feature engineering")
    config.addinivalue_line("markers", "model_training: Testes de treinamento de modelo")
    config.addinivalue_line("markers", "api: Testes de endpoints da API")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment."""
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(autouse=True)
def reset_random_for_each_test():
    """Reset random seeds for each test for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    yield
