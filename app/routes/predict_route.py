"""
Rota de predição de risco de defasagem escolar.

Endpoint principal da API para inferência do modelo.
"""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models.schemas import PredictionResponse, StudentData
from app.utils.model_loader import get_current_model
from src.data_cleaning import clean_data, handle_missing_values
from src.feature_engineering import create_features

logger = logging.getLogger("predict_route")

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    """
    Realiza predição de risco de defasagem escolar.

    **Input:** JSON com `{"data": {...}}` contendo campos do aluno.
    Todos os campos são opcionais — campos faltantes são preenchidos com defaults seguros.

    **Output:** `risk_prediction` (0 ou 1) e `risk_probability` (0.0 a 1.0).

    Args:
        student: Dados do aluno (schema StudentData).

    Returns:
        PredictionResponse com predição e probabilidade.

    Raises:
        HTTPException: 503 se modelo não estiver carregado, 400 se erro de processamento.
    """
    model = get_current_model()

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert dict to DataFrame
        df = pd.DataFrame([student.data])

        # Preprocessing pipeline
        df = clean_data(df)
        df = handle_missing_values(df)
        df = create_features(df)

        # Drop leakage columns
        leakage_cols = [
            "INDE_2024",
            "PEDRA_2024",
            "IAA",
            "IEG",
            "IPS",
            "IPP",
            "IDA",
            "MAT",
            "POR",
            "ING",
            "IPV",
            "IAN",
            "DESTAQUE_IEG",
            "DESTAQUE_IDA",
            "DESTAQUE_IPV",
            "ATINGIU_PV",
            "INDICADO",
            "REC_AV1",
            "REC_AV2",
            "REC_PSICOLOGIA",
            "DEFASAGEM",
            "RA",
            "NOME_ANONIMIZADO",
            "PEDRA_2024",
        ]
        feature_matrix = df.drop(
            columns=[c for c in leakage_cols if c in df.columns], errors="ignore"
        )

        # Feature Alignment
        if hasattr(model, "feature_names_in_"):
            try:
                preprocessor = model.named_steps["preprocessor"]
                numeric_cols_model = []
                categorical_cols_model = []

                for name, _, cols in preprocessor.transformers_:
                    if name == "num":
                        numeric_cols_model = cols
                    elif name == "cat":
                        categorical_cols_model = cols

                for col in numeric_cols_model:
                    if col not in feature_matrix.columns:
                        feature_matrix[col] = 0.0

                for col in categorical_cols_model:
                    if col not in feature_matrix.columns:
                        feature_matrix[col] = "UNKNOWN"
            except ValueError:
                pass

            expected_cols = model.feature_names_in_
            missing_cols = set(expected_cols) - set(feature_matrix.columns)
            for col_name in missing_cols:
                feature_matrix[col_name] = 0

            feature_matrix = feature_matrix[expected_cols]

        prediction = model.predict(feature_matrix)
        proba = model.predict_proba(feature_matrix)[:, 1]

        logger.info("Prediction: risk=%d, probability=%.4f", int(prediction[0]), float(proba[0]))

        return {
            "risk_prediction": int(prediction[0]),
            "risk_probability": float(proba[0]),
        }
    except ValueError as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
