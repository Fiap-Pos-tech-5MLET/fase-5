"""
Rota de predicao de risco de defasagem escolar.

Este endpoint entrega inferencia do modelo champion em producao e resolve o
problema central do projeto: antecipar risco de defasagem com dados reais,
frequentemente incompletos. O pipeline aplica limpeza, preenchimento seguro
de faltantes, engenharia de features e alinhamento de colunas com o modelo
treinado para reduzir erros de schema e garantir consistencia.
"""

import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.models.schemas import FeatureContribution, PredictionResponse, StudentData
from app.utils.model_loader import get_current_model, load_model
from app.utils.security import validate_requested_by
from app.utils.structured_logging import log_with_request
from app.utils.xai import explain_prediction
from src.data_cleaning import clean_data, handle_missing_values
from src.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    build_feature_matrix_for_model,
    create_features,
)

logger = logging.getLogger("predict_route")

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    student: Annotated[
        StudentData,
        Body(
            examples=[
                {
                    "data": {
                        "nivel_de_defasagem": 0.0,
                        "idade": 12.0,
                        "genero": 0.0,
                        "ano_de_ingresso": 2022.0,
                        "veterano": 0.0,
                        "em_fase": 1.0,
                        "qtde_aval_realizadas": 4.0,
                        "iaa": 6.0,
                        "ieg": 6.0,
                        "ips": 6.0,
                        "ida": 6.0,
                        "ipv": 6.0,
                        "ian": 6.0,
                    }
                }
            ]
        ),
    ],
    request: Request,
    requested_by: Annotated[str, Depends(validate_requested_by)],
) -> PredictionResponse:
    """
    Realiza predicao de risco de defasagem escolar.

    Valor de negocio: permite triagem preventiva de alunos com maior risco,
    apoiando decisoes de acompanhamento pedagogo antes do fechamento do ciclo.

    **Input:** JSON com `{"data": {...}}` contendo campos do aluno.
    Todos os campos sao opcionais — faltantes sao preenchidos com defaults seguros.

    **Output:** `risk_prediction` (0 ou 1) e `risk_probability` (0.0 a 1.0).

    Fluxo tecnico:
    1) limpeza e tratamento de faltantes
    2) engenharia de features
    3) remocao de colunas de leakage
    4) alinhamento das features esperadas pelo modelo

    Args:
        student (StudentData): Wrapper com dados do aluno no campo `data`.

    Returns:
        PredictionResponse: Predicao binaria e probabilidade de risco.

    Raises:
        HTTPException: 503 se o modelo nao estiver carregado.
        HTTPException: 400 se houver erro no preprocessamento.
    """
    model = get_current_model()
    if model is None:
        model, _ = load_model()
    if not model:
        log_with_request(
            logger=logger,
            level=logging.WARNING,
            event="prediction_unavailable",
            request=request,
            requested_by=requested_by,
            status="model_not_loaded",
        )
        raise HTTPException(status_code=503, detail="Model not loaded")

    def _predict_sync() -> PredictionResponse:
        # Convert dict to DataFrame
        df = pd.DataFrame([student.data])

        # Preprocessing pipeline
        df = clean_data(df)
        df = handle_missing_values(df)
        df = create_features(df)

        # Caminho principal do modelo atual: 13 variáveis canônicas
        feature_matrix = build_feature_matrix_for_model(df)

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
            except (ValueError, AttributeError, KeyError, TypeError):
                pass

            expected_cols = model.feature_names_in_
            missing_cols = set(expected_cols) - set(feature_matrix.columns)
            for col_name in missing_cols:
                feature_matrix[col_name] = 0

            feature_matrix = feature_matrix[expected_cols]

        # Compatibilidade para modelos sem feature_names_in_
        if not hasattr(model, "feature_names_in_"):
            available_cols = [col for col in MODEL_FEATURE_COLUMNS if col in feature_matrix.columns]
            feature_matrix = feature_matrix[available_cols]

        prediction = model.predict(feature_matrix)
        proba = model.predict_proba(feature_matrix)[:, 1]
        top_features, explanation_method = explain_prediction(model, feature_matrix)
        feature_contributions = [FeatureContribution(**feature) for feature in top_features]

        log_with_request(
            logger=logger,
            level=logging.INFO,
            event="prediction_inference",
            request=request,
            requested_by=requested_by,
            status="success",
            risk_prediction=int(prediction[0]),
            risk_probability=float(proba[0]),
            explanation_method=explanation_method,
        )

        return PredictionResponse(
            risk_prediction=int(prediction[0]),
            risk_probability=float(proba[0]),
            explanation_method=explanation_method,
            top_features=feature_contributions,
        )

    try:
        return await run_in_threadpool(_predict_sync)
    except ValueError as exc:
        log_with_request(
            logger=logger,
            level=logging.ERROR,
            event="prediction_inference",
            request=request,
            requested_by=requested_by,
            status="failed",
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
