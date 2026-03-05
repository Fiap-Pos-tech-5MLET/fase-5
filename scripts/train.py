"""
Script de treinamento do pipeline de Machine Learning.

Este script executa o pipeline completo de treinamento incluindo:
- Carregamento e limpeza de dados
- Feature engineering
- Treinamento do modelo RandomForest
- Avaliação de métricas
- Logging no MLflow
- Geração de artefatos visuais
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    classification_report,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import mlflow
import mlflow.sklearn

from src.data_cleaning import (
    clean_data,
    create_target,
    handle_missing_values,
    load_data,
    validate_data_quality,
)
from src.feature_engineering import create_features, select_features
from src.feature_store import persist_dataset_version, register_feature_view
from src.model import save_model, train_model


def plot_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """
    Gera e salva heatmap do classification report.

    Args:
        y_true: Valores reais do target.
        y_pred: Valores preditos pelo modelo.
        output_path: Caminho para salvar a imagem.
    """
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.debug(f"Classification report salvo em: {output_path}")


def plot_roc_curve(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, output_path: str) -> None:
    """
    Gera e salva curva ROC.

    Args:
        estimator: Modelo treinado com método predict_proba.
        X: Features de teste.
        y: Target de teste.
        output_path: Caminho para salvar a imagem.
    """
    y_prob = estimator.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.debug(f"ROC curve salva em: {output_path}")


def plot_feature_importance(
    model: Pipeline, feature_names: Optional[np.ndarray], output_path: str, top_n: int = 20
) -> None:
    """
    Gera e salva gráfico de importância de features.

    Args:
        model: Pipeline treinado contendo RandomForest.
        feature_names: Nomes das features (opcional).
        output_path: Caminho para salvar a imagem.
        top_n: Número de features mais importantes a exibir.
    """
    # Get feature importances from the classifier step
    if not hasattr(model.named_steps["classifier"], "feature_importances_"):
        logger.warning("Modelo não possui feature_importances_")
        return

    importances = model.named_steps["classifier"].feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot top N
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances (Top {top_n})")
    n_show = min(top_n, len(importances))
    plt.bar(range(n_show), importances[indices][:n_show], align="center")

    # Use feature names if available
    if feature_names is not None and len(feature_names) == len(importances):
        plt.xticks(range(n_show), [feature_names[i] for i in indices[:n_show]], rotation=90)
    else:
        if feature_names is not None:
            logger.warning(
                f"Feature names mismatch: {len(feature_names)} names vs {len(importances)} features"
            )
        plt.xticks(range(n_show), [str(i) for i in indices[:n_show]])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.debug(f"Feature importance salva em: {output_path}")


def main(
    data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    k: Union[int, str] = "all",
    test_size: float = 0.2,
) -> Tuple[Dict[str, Any], str]:
    """Executa o pipeline de treinamento com rastreabilidade completa.

    Args:
        data_path (Optional[str]): Caminho para os dados brutos.
        model_path (Optional[str]): Caminho de saída do modelo treinado.
        artifacts_dir (Optional[str]): Diretório para artefatos visuais.
        n_estimators (int): Número de árvores no RandomForest.
        max_depth (Optional[int]): Profundidade máxima das árvores.
        min_samples_split (int): Mínimo de amostras para split.
        k (Union[int, str]): Número de features para SelectKBest.
        test_size (float): Proporção para conjunto de teste.

    Returns:
        Tuple[Dict[str, Any], str]: Métricas de avaliação e run_id do MLflow.
    """
    DATA_PATH = data_path or "app/data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    MODEL_PATH = model_path or "app/models/model.pkl"
    ARTIFACTS_DIR = artifacts_dir or "app/models/artifacts"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # MLflow Setup
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    mlflow.set_experiment("Datathon_Passos_Magicos")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        logger.info("Loading data...")
        try:
            df = load_data(DATA_PATH)
        except FileNotFoundError:
            logger.error("Data file not found at %s", DATA_PATH)
            return {}, ""

        logger.info("Cleaning data...")
        df = clean_data(df)

        logger.info("Validating data quality with Great Expectations...")
        validate_data_quality(df)

        logger.info("Creating target...")
        df = create_target(df)

        logger.info("Handling missing values...")
        df = handle_missing_values(df)

        logger.info("Engineering features...")
        df = create_features(df)

        logger.info("Selecting features...")
        X, y = select_features(df)

        logger.info("Persisting feature store metadata and dataset versions...")
        feature_registry_path = register_feature_view(
            feature_df=X,
            target_name="TARGET",
        )
        features_snapshot_path, dataset_version = persist_dataset_version(
            df=X,
            dataset_name="train_features",
        )

        # Log Data Params
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("dataset_version", dataset_version)

        logger.info(f"Training on {X.shape[0]} samples and {X.shape[1]} features...")
        model, metrics, X_test, y_test = train_model(
            X,
            y,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            k=k,
            test_size=test_size,
        )

        # Log Model Params (Random Forest)
        try:
            rf = model.named_steps["classifier"]
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", rf.n_estimators)
            mlflow.log_param("max_depth", rf.max_depth)
            mlflow.log_param("min_samples_split", rf.min_samples_split)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("k_best", k)
        except (AttributeError, KeyError, TypeError) as exc:
            logger.warning("Não foi possível registrar alguns hiperparâmetros: %s", exc)

        logger.info("--- Model Metrics ---")
        logger.info(f"\n{metrics['classification_report']}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

        # Log Metrics
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])

        # Visual Artifacts
        logger.info("Generating visual artifacts...")
        roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
        report_path = os.path.join(ARTIFACTS_DIR, "classification_report.png")
        feat_imp_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")

        plot_roc_curve(model, X_test, y_test, roc_path)
        plot_classification_report(y_test, model.predict(X_test), report_path)

        # Try to infer feature names from the pipeline
        feature_names = None
        try:
            # Access preprocessor
            preprocessor = model.named_steps["preprocessor"]
            # Access feature selection (SelectKBest) mask
            selector = model.named_steps["feature_selection"]

            # Get names from preprocessor (Num + Cat OneHot)
            if hasattr(preprocessor, "get_feature_names_out"):
                all_features = preprocessor.get_feature_names_out()
            else:
                # Fallback if get_feature_names_out fails on ColumnTransformer (older sklearn versions)
                all_features = np.array(
                    [f"Feat_{i}" for i in range(model.named_steps["classifier"].n_features_in_)]
                )

            # Apply selection mask if SelectKBest was used
            if hasattr(selector, "get_support"):
                selected_mask = selector.get_support()
                feature_names = all_features[selected_mask]
            else:
                feature_names = all_features

        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
            logger.warning(f"Error extracting feature names: {e}")
            feature_names = None

        plot_feature_importance(model, feature_names, feat_imp_path)

        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(feat_imp_path)
        mlflow.log_artifact(feature_registry_path)
        mlflow.log_artifact(features_snapshot_path)

        # Log Model Artifact
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Saving model local backup to {MODEL_PATH}...")
        save_model(model, MODEL_PATH)

        # Save run_id alongside the model for tracking
        run_id_path = os.path.join(os.path.dirname(MODEL_PATH), "candidate_run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(run_id)
        logger.info(f"Run ID {run_id} saved to {run_id_path}")

        logger.info("Done! MLflow run logged with images.")

        return metrics, run_id


if __name__ == "__main__":
    main()
