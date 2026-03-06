"""
Configurações centralizadas da aplicação.

Define paths, parâmetros de modelo e configurações de ambiente
para toda a aplicação do Datathon Passos Mágicos.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "app" / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "app" / "artifacts"

# Data paths
RAW_DATA_PATH: str = str(BASE_DIR / "app" / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed" / "processed_data.csv")

# Model paths
MODEL_PATH: str = str(MODEL_DIR / "model.pkl")
CHAMPION_RUN_ID_PATH: str = str(MODEL_DIR / "champion_run_id.txt")

# Artifacts
DRIFT_REPORT_PATH: str = str(ARTIFACTS_DIR / "data_drift_report.html")

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
}

# Training parameters
TRAIN_PARAMS = {"test_size": 0.2, "random_state": 42}

# API configuration
API_TITLE: str = "Datathon Passos Mágicos API"
API_VERSION: str = "1.0.0"


# MLflow configuration
def get_mlflow_tracking_uri() -> str:
    """
    Retorna URI de tracking do MLflow.

    Returns:
        URI do MLflow (variável de ambiente ou padrão local).
    """
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


# Criar diretórios se não existirem
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
