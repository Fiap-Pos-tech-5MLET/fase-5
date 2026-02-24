"""
API FastAPI para predição de risco de defasagem escolar.

Aplicação principal que registra rotas e middlewares.
API desenvolvida para o Datathon Passos Mágicos.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import uvicorn
from fastapi import FastAPI, Request

from app.routes.audit_route import router as audit_router
from app.routes.predict_route import router as predict_router
from app.routes.train_route import router as train_router
from app.utils.model_loader import load_model

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("datathon_api")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Gerencia eventos de ciclo de vida da aplicação."""
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info("MLflow tracking URI set to: %s", mlflow_tracking_uri)

    model, loaded_at = load_model()
    if model:
        logger.info("Model loaded at startup: %s", loaded_at)
    else:
        logger.warning("No model loaded at startup. Train a model first.")

    yield


# Create FastAPI app
app = FastAPI(
    title="Datathon Passos Mágicos — API de Predição",
    description="""
API para predição de risco de defasagem escolar dos alunos da Associação Passos Mágicos.

## Endpoints

### Predição
- **POST /predict** — Predição de risco para um aluno

### Treinamento
- **POST /retrain** — Treina modelo candidato
- **POST /promote** — Promove candidato para produção
- **POST /discard** — Descarta modelo candidato
- **GET /model-metrics** — Métricas do modelo champion
- **GET /model-artifact/{name}** — Download de artefatos

### Auditoria
- **GET /model-info** — Metadados e estratégia de retreinamento
- **GET /drift** — Relatório de Data Drift (Evidently)
- **GET /** — Health check
""",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api" if os.getenv("ENVIRONMENT") == "production" else "",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para log de todas as requisições com timing."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# Register routers
app.include_router(audit_router, tags=["Auditoria"])
app.include_router(predict_router, tags=["Predição"])
app.include_router(train_router, tags=["Treinamento"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
