"""
API FastAPI para predição de risco de defasagem escolar.

Aplicação principal que registra rotas e middlewares.
API desenvolvida para o Datathon Passos Mágicos.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import mlflow
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.routes.audit_route import router as audit_router
from app.routes.predict_route import router as predict_router
from app.routes.train_route import router as train_router
from app.utils.keep_alive import start_keep_alive
from app.utils.model_loader import load_model
from app.utils.structured_logging import log_with_request

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("datathon_api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Gerencia eventos de ciclo de vida da aplicação."""
    # Iniciar keep-alive para Render Free Tier
    try:
        start_keep_alive()
    except (OSError, RuntimeError) as e:
        logger.warning("Failed to start keep-alive: %s", e)

    # Configurar MLflow
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info("MLflow tracking URI set to: %s", mlflow_tracking_uri)

    # Carregar modelo
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
    root_path="/api",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para log de todas as requisições com timing."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    log_with_request(
        logger=logger,
        level=logging.INFO,
        event="http_request",
        request=request,
        requested_by=request.headers.get("x-requested-by", "unknown"),
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


@app.get("/health")
async def health_check():
    """
    Health check endpoint para keep-alive (Render Free Tier).

    Responde de forma leve sem fazer operações pesadas.

    Returns:
        dict: Status da aplicação com timestamp.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "datathon-api",
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "development"),
        },
    )


@app.get("/", response_class=HTMLResponse)
def serve_homepage() -> HTMLResponse:
    """
    Serve a página inicial HTML com links para API Docs e Dashboard.

    Returns:
        HTMLResponse: Página HTML de boas-vindas.
    """
    index_path = Path(__file__).parent.parent / "index.html"
    try:
        with open(index_path, encoding="utf-8") as f:
            html_content = f.read()
        # Injeta a URL do Dashboard se disponível
        dashboard_url = os.getenv("DASHBOARD_URL", "")
        if dashboard_url:
            html_content = html_content.replace("window.DASHBOARD_URL", f'"{dashboard_url}"')
        return HTMLResponse(content=html_content)
    except (FileNotFoundError, OSError) as e:
        logger.warning("Could not load index.html: %s", e)
        fallback_html = (
            "<h1>Datathon API</h1>"
            "<p>Acesse <a href='/docs'>/docs</a> para a documentação da API.</p>"
        )
        return HTMLResponse(content=fallback_html, status_code=200)


# Register routers
app.include_router(audit_router, tags=["Auditoria"])
app.include_router(predict_router, tags=["Predição"])
app.include_router(train_router, tags=["Treinamento"])


if __name__ == "__main__":
    # Binding em 0.0.0.0 é intencional para ambiente de desenvolvimento/container
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
