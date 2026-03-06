"""
Rotas de treinamento e gestao de modelos.

Implementa o ciclo champion/challenger para MLOps:
- /retrain cria um modelo candidato (challenger) sem afetar producao
- /promote substitui o champion pelo challenger aprovado
- /discard remove o challenger rejeitado
- /model-metrics e /model-artifact trazem rastreabilidade via MLflow
"""

import logging
import os
import shutil
from typing import Annotated, Any, Dict

import mlflow
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

try:
    from mlflow.exceptions import MlflowException
except (ImportError, ModuleNotFoundError, AttributeError):

    class MlflowException(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Fallback para ambientes sem pacote mlflow.exceptions disponível."""


from app.models.schemas import (
    DiscardResponse,
    ModelMetricsResponse,
    PromoteResponse,
    RetrainRequest,
)
from app.utils.model_loader import get_model_paths, reload_model
from app.utils.security import validate_api_key
from app.utils.structured_logging import log_with_request
from scripts.train import main as run_training

logger = logging.getLogger("train_route")

router = APIRouter()


def _get_latest_dataset_path() -> str:
    """
    Obtém o caminho do arquivo de dados "atual" em app/data/raw.

    Retorna o nome do arquivo mais recente ou "unknown" se não houver arquivo.
    Usado para rastreabilidade: qual versão do dataset foi usada para treinar o modelo.
    """
    from pathlib import Path

    raw_data_dir = Path("app/data/raw")
    if not raw_data_dir.exists():
        return "unknown"

    # Procura arquivos .csv ou .xlsx
    files = list(raw_data_dir.glob("*.csv")) + list(raw_data_dir.glob("*.xlsx"))
    if not files:
        return "unknown"

    # Retorna os nomes dos arquivos (não versionados), exemplo: dados.csv
    current_files = [f for f in files if "_" not in f.stem or f.stem.split("_")[0].isalpha()]
    if current_files:
        return str(current_files[0].name)

    return str(max(files, key=lambda f: f.stat().st_mtime).name)


@router.post("/retrain")
async def retrain(
    params: Annotated[RetrainRequest, Body(...)],
    request: Request,
    _authenticated: Annotated[None, Depends(validate_api_key)],
) -> Dict[str, Any]:
    """
    Treina um modelo candidato (challenger) SEM sobrescrever o modelo em produção (champion).

    O modelo candidato é salvo em models/model_candidate.pkl.
    Use POST /promote para promovê-lo ou POST /discard para descartá-lo.

    Quando usar: apos identificar drift, queda de metricas ou chegada de
    novos dados do ciclo escolar. O retorno inclui metricas e run_id do MLflow
    para comparacao objetiva com o champion atual.

    Args:
        params (RetrainRequest): Hiperparametros de treinamento.

    Returns:
        Dict[str, Any]: Status do treino, metricas e run_id do MLflow.

    Raises:
        HTTPException: 500 se o treinamento falhar.
    """
    try:
        _, _, candidate_path = get_model_paths()

        # Convert k=None to 'all' for SelectKBest
        k_value = params.k if params.k is not None else "all"

        log_with_request(
            logger=logger,
            level=logging.INFO,
            event="model_retrain",
            request=request,
            requested_by=params.requested_by,
            status="started",
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            k=k_value,
            test_size=params.test_size,
        )

        result = await run_in_threadpool(
            run_training,
            model_path=candidate_path,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            k=k_value,
            test_size=params.test_size,
        )

        # Unpack metrics and run_id
        if isinstance(result, tuple):
            metrics, run_id = result
        else:
            metrics, run_id = result, None

        log_with_request(
            logger=logger,
            level=logging.INFO,
            event="model_retrain",
            request=request,
            requested_by=params.requested_by,
            status="success",
            run_id=run_id,
            dataset_path=_get_latest_dataset_path(),
        )

        return {
            "status": "success",
            "message": (
                "Modelo candidato treinado. Use /promote para "
                "promovê-lo ou /discard para descartá-lo."
            ),
            "promoted": False,
            "requested_by": params.requested_by,
            "candidate_path": candidate_path,
            "run_id": run_id,
            "metrics": metrics,
            "hyperparameters": {
                "n_estimators": params.n_estimators,
                "max_depth": params.max_depth,
                "min_samples_split": params.min_samples_split,
                "k": k_value,
                "test_size": params.test_size,
            },
        }
    except ImportError as exc:
        log_with_request(
            logger=logger,
            level=logging.ERROR,
            event="model_retrain",
            request=request,
            requested_by=params.requested_by,
            status="failed",
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Import failed: {exc!s}",
        ) from exc
    except ValueError as exc:
        log_with_request(
            logger=logger,
            level=logging.ERROR,
            event="model_retrain",
            request=request,
            requested_by=params.requested_by,
            status="failed",
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {exc!s}",
        ) from exc


@router.post("/promote", response_model=PromoteResponse)
async def promote(
    request: Request,
    _authenticated: Annotated[None, Depends(validate_api_key)],
) -> PromoteResponse:
    """
    Promove o modelo candidato (challenger) para produção (champion).

    Copia model_candidate.pkl → model.pkl e recarrega o modelo na API.
    Atualiza o run_id do champion para rastreabilidade no MLflow.

    Quando usar: somente apos avaliacao de metricas, regressao e alinhamento
    com os objetivos do projeto.

    Returns:
        PromoteResponse: Status da promocao e timestamp de carregamento.

    Raises:
        HTTPException: 404 se candidato nao existir.
        HTTPException: 500 se a promocao falhar.
    """
    models_dir, model_path, candidate_path = get_model_paths()
    candidate_run_id_path = os.path.join(models_dir, "candidate_run_id.txt")
    champion_run_id_path = os.path.join(models_dir, "champion_run_id.txt")

    if not os.path.exists(candidate_path):
        raise HTTPException(
            status_code=404,
            detail="Nenhum modelo candidato encontrado. Execute /retrain primeiro.",
        )

    try:
        # Copy candidate → champion
        shutil.copy2(candidate_path, model_path)
        logger.info("Candidate model promoted: %s → %s", candidate_path, model_path)

        # Copy candidate run_id → champion run_id
        promoted_run_id = None
        if os.path.exists(candidate_run_id_path):
            shutil.copy2(candidate_run_id_path, champion_run_id_path)
            with open(champion_run_id_path, encoding="utf-8") as f:
                promoted_run_id = f.read().strip()
            logger.info("Champion run_id updated to %s", promoted_run_id)

        # Reload champion
        _, loaded_at = reload_model()

        # Log rastreabilidade: modelo promovido com qual dataset
        log_with_request(
            logger=logger,
            level=logging.INFO,
            event="model_promoted",
            request=request,
            run_id=promoted_run_id,
            champion_path=str(model_path),
            dataset_path=_get_latest_dataset_path(),
        )

        # Remove candidate files
        os.remove(candidate_path)
        if os.path.exists(candidate_run_id_path):
            os.remove(candidate_run_id_path)
        logger.info("Candidate files removed after promotion.")

        return PromoteResponse(
            status="promoted",
            message="Modelo candidato promovido para produção com sucesso!",
            loaded_at=loaded_at or "N/A",
            champion_run_id=promoted_run_id,
        )
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"File error: {exc!s}",
        ) from exc
    except OSError as exc:
        logger.error("Promotion failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Promotion failed: {exc!s}",
        ) from exc


@router.post("/discard", response_model=DiscardResponse)
async def discard(
    _authenticated: Annotated[None, Depends(validate_api_key)],
) -> DiscardResponse:
    """
    Descarta o modelo candidato (challenger) e mantém o modelo atual (champion).

    Quando usar: se o challenger tiver metricas piores, sinais de overfitting
    ou degradacao em dados recentes.

    Returns:
        DiscardResponse: Status do descarte.

    Raises:
        HTTPException: 404 se candidato nao existir.
        HTTPException: 500 se o descarte falhar.
    """
    models_dir, _, candidate_path = get_model_paths()
    candidate_run_id_path = os.path.join(models_dir, "candidate_run_id.txt")

    if not os.path.exists(candidate_path):
        raise HTTPException(status_code=404, detail="Nenhum modelo candidato para descartar.")

    try:
        os.remove(candidate_path)
        if os.path.exists(candidate_run_id_path):
            os.remove(candidate_run_id_path)
        logger.info("Candidate model and run_id discarded. Champion model unchanged.")

        return DiscardResponse(
            status="discarded",
            message="Modelo candidato descartado. O modelo atual em produção foi mantido.",
        )
    except OSError as exc:
        logger.error("Discard failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Discard failed: {exc!s}",
        ) from exc


@router.get("/model-metrics", response_model=ModelMetricsResponse)
async def model_metrics() -> ModelMetricsResponse:
    """
    Retorna métricas, parâmetros e artefatos do modelo champion em produção.

    Busca da run do MLflow correspondente ao champion (via champion_run_id.txt).
    Fallback: retorna informações básicas se MLflow não estiver disponível.

    Valor: permite auditoria e comparacao entre versoes, suportando decisao
    de promover ou descartar candidatos.

    Returns:
        ModelMetricsResponse: Metricas e artefatos do champion.
    """
    models_dir, _, _ = get_model_paths()
    champion_run_id_path = os.path.join(models_dir, "champion_run_id.txt")

    champion_run_id = None
    if os.path.exists(champion_run_id_path):
        with open(champion_run_id_path, encoding="utf-8") as f:
            champion_run_id = f.read().strip()

    if not champion_run_id:
        logger.info("No champion_run_id.txt found. Returning basic model info.")
        return ModelMetricsResponse(
            source="local",
            message=(
                "Nenhuma run do MLflow vinculada ao modelo em produção. "
                "Execute /retrain e /promote."
            ),
            run_id=None,
            metrics=None,
            params=None,
            artifacts=[],
        )

    try:
        run = mlflow.get_run(champion_run_id)

        metrics = dict(run.data.metrics)
        params = dict(run.data.params)

        # List available artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts_list = client.list_artifacts(champion_run_id)
        artifact_names = [a.path for a in artifacts_list if a.path.endswith(".png")]

        return ModelMetricsResponse(
            source="mlflow",
            run_id=champion_run_id,
            run_name=run.info.run_name,
            start_time=run.info.start_time,
            end_time=run.info.end_time,
            status=run.info.status,
            metrics=metrics,
            params=params,
            artifacts=artifact_names,
        )
    except ImportError as exc:
        logger.warning("MLflow not available: %s", exc)
        return ModelMetricsResponse(
            source="error",
            run_id=champion_run_id,
            message=f"MLflow não disponível: {exc!s}",
            metrics=None,
            params=None,
            artifacts=[],
        )
    except (ValueError, MlflowException) as exc:
        logger.warning(
            "Failed to fetch from MLflow (run_id=%s): %s",
            champion_run_id,
            exc,
        )
        return ModelMetricsResponse(
            source="local",
            run_id=champion_run_id,
            message=(
                "Run do MLflow não encontrada para o champion atual. "
                "Retreinhe e promova novamente para atualizar o run_id. "
                f"Detalhe: {exc!s}"
            ),
            metrics=None,
            params=None,
            artifacts=[],
        )


@router.get("/model-artifact/{artifact_name}")
async def model_artifact(artifact_name: str) -> FileResponse:
    """
    Serve um artefato (imagem) da run do champion no MLflow.

    Fallback: serve do diretório local models/artifacts/.

    Valor: disponibiliza graficos de ROC, feature importance e reports
    que sustentam a explicacao do modelo.

    Args:
        artifact_name (str): Nome do arquivo de artefato.

    Returns:
        FileResponse: Imagem PNG do artefato.

    Raises:
        HTTPException: 404 se artefato nao for encontrado.
    """
    models_dir, _, _ = get_model_paths()
    champion_run_id_path = os.path.join(models_dir, "champion_run_id.txt")

    # Try MLflow first
    champion_run_id = None
    if os.path.exists(champion_run_id_path):
        with open(champion_run_id_path, encoding="utf-8") as f:
            champion_run_id = f.read().strip()

    if champion_run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            local_path = client.download_artifacts(champion_run_id, artifact_name)

            if os.path.exists(local_path):
                logger.info(
                    "Serving artifact '%s' from MLflow run %s",
                    artifact_name,
                    champion_run_id,
                )
                return FileResponse(local_path, media_type="image/png")
        except ImportError as exc:
            logger.warning("MLflow import error: %s. Falling back to local.", exc)
        except (ValueError, MlflowException) as exc:
            logger.warning(
                "Failed to fetch artifact from MLflow: %s. Falling back to local.",
                exc,
            )

    # Fallback to local artifacts
    local_fallback = os.path.join(models_dir, "artifacts", artifact_name)
    if os.path.exists(local_fallback):
        logger.info("Serving artifact '%s' from local fallback", artifact_name)
        return FileResponse(local_fallback, media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Artefato '{artifact_name}' não encontrado.")
