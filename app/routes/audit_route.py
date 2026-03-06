"""
Rotas de auditoria e monitoramento.

Explicam o estado do modelo em producao e trazem transparencia tecnica
para operacao: health check, metadados do modelo, relatorio de drift e ingestao de dados.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.models.schemas import ModelInfoResponse
from app.utils.model_loader import get_current_model, get_model_info, get_model_paths
from app.utils.security import validate_api_key
from app.utils.structured_logging import log_with_request

logger = logging.getLogger("audit_route")

router = APIRouter()


@router.get("/")
async def health_check(
    request: Request, requested_by: str = Header(default="unknown", alias="x-requested-by")
) -> Dict[str, bool]:
    """
    Health check endpoint.

    Usado por orquestradores e monitoramento para confirmar disponibilidade
    da API e se o modelo esta carregado em memoria.

    Returns:
        Dict[str, bool]: Status basico e indicacao de modelo carregado.
    """
    model = get_current_model()
    log_with_request(
        logger=logger,
        level=logging.INFO,
        event="audit_health_check",
        request=request,
        requested_by=requested_by,
        model_loaded=model is not None,
    )
    return {"status": True, "model_loaded": model is not None}


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(
    request: Request, requested_by: str = Header(default="unknown", alias="x-requested-by")
) -> ModelInfoResponse:
    """
    Retorna metadados do modelo em produção, incluindo estratégia de
    retreinamento e cenários de produção documentados.

    Valor: documenta governanca do modelo, criterios de retreino e
    comportamento esperado em cenarios de dados reais.

    Returns:
        ModelInfoResponse: Metadados completos do modelo em producao.
    """
    model = get_current_model()
    info_basic = await run_in_threadpool(get_model_info)

    info = {
        "model_loaded": info_basic["model_loaded"],
        "model_path": info_basic["model_path"],
        "loaded_at": info_basic["loaded_at"],
        "model_type": None,
        "n_features": None,
        "features": None,
        "retraining_strategy": {
            "trigger": ("Semestralmente ou quando drift detectado > 30% das features"),
            "data_source": (
                "Novo arquivo PEDE exportado pela Passos Mágicos ao final de cada ciclo"
            ),
            "process": [
                "1. Receber novos dados PEDE do período",
                "2. Executar pipeline de monitoramento (monitoring.py) para verificar drift",
                "3. Se drift significativo: executar retreinamento (train_pipeline.py)",
                "4. Comparar métricas do novo modelo com o modelo em produção",
                "5. Se métricas iguais ou superiores: promover para produção",
                "6. Atualizar model.pkl e reiniciar API",
            ],
            "frequency": (
                "Semestral (alinhado ao ciclo escolar) ou sob demanda se drift > threshold"
            ),
            "rollback": "Manter versão anterior do model.pkl via versionamento MLflow",
        },
        "production_scenarios": {
            "novo_aluno_sem_historico": {
                "descricao": "Aluno novo sem INDE_22 ou INDE_23 (primeiro ano na associação)",
                "comportamento": (
                    "Campos numéricos faltantes são preenchidos com 0, HAS_HISTORY_23 = 0. "
                    "O modelo classifica com base apenas em dados demográficos e escolares. "
                    "A probabilidade de risco tende a ser menos confiável neste cenário."
                ),
                "recomendacao": "Acompanhamento presencial reforçado no primeiro semestre.",
            },
            "degradacao_do_modelo": {
                "descricao": "Perfil dos alunos muda significativamente ao longo dos anos",
                "deteccao": (
                    "Monitoramento de Data Drift via Evidently (endpoint /drift). "
                    "Se > 30% das features apresentarem drift estatístico, disparar alerta."
                ),
                "acao": "Retreinar o modelo com os dados mais recentes.",
            },
            "dados_incompletos_ou_corrompidos": {
                "descricao": "Input da API com campos faltantes, tipos errados ou valores absurdos",
                "comportamento": (
                    "A API preenche campos faltantes automaticamente. "
                    "Valores numéricos inválidos são coercidos para 0. "
                    "Categorias desconhecidas são tratadas pelo OneHotEncoder "
                    "(handle_unknown='ignore')."
                ),
                "contrato": (
                    "Todos os campos do input são opcionais. "
                    "O modelo retorna predição mesmo com dados parciais."
                ),
            },
            "escala_e_concorrencia": {
                "descricao": "Múltiplas requisições simultâneas em produção",
                "comportamento": (
                    "FastAPI com Uvicorn suporta async. Modelo carregado uma vez na memória. "
                    "Para escala: usar Docker com múltiplas réplicas atrás de load balancer."
                ),
            },
        },
    }

    if model is not None:
        try:
            info["model_type"] = type(model.named_steps["classifier"]).__name__
            info["n_features"] = len(model.feature_names_in_)
            info["features"] = list(model.feature_names_in_)
        except (AttributeError, KeyError):
            pass

    response = ModelInfoResponse(**info)
    log_with_request(
        logger=logger,
        level=logging.INFO,
        event="audit_model_info",
        request=request,
        requested_by=requested_by,
        model_loaded=response.model_loaded,
        model_type=response.model_type,
        n_features=response.n_features,
    )
    return response


@router.get("/drift", response_class=HTMLResponse)
async def drift_report(
    request: Request, requested_by: str = Header(default="unknown", alias="x-requested-by")
) -> HTMLResponse:
    """
    Serve o relatório de data drift como HTML.

    O relatorio e gerado por `scripts/monitoring.py` com Evidently e
    compara distribuicoes entre dados de referencia e dados atuais.

    Returns:
        HTMLResponse: Relatorio Evidently em HTML.

    Raises:
        HTTPException: 404 se relatorio nao existir.
    """
    models_dir, _, _ = get_model_paths()
    report_path = os.path.join(models_dir, "artifacts", "data_drift_report.html")

    if not os.path.exists(report_path):
        log_with_request(
            logger=logger,
            level=logging.WARNING,
            event="audit_drift_report_missing",
            request=request,
            requested_by=requested_by,
            report_path=report_path,
        )
        raise HTTPException(
            status_code=404, detail="Drift report not found. Run monitoring.py first."
        )

    def _read_report() -> str:
        with open(report_path, encoding="utf-8") as f:
            return f.read()

    html_content = await run_in_threadpool(_read_report)
    log_with_request(
        logger=logger,
        level=logging.INFO,
        event="audit_drift_report_served",
        request=request,
        requested_by=requested_by,
        report_path=report_path,
    )
    return HTMLResponse(content=html_content)


@router.post("/update-data")
async def update_dataset(
    file: UploadFile = File(...),
    request: Request = None,
    _authenticated: Any = Depends(validate_api_key),
) -> JSONResponse:
    """
    Ingere um novo dataset bruto enviado pela Associação Passos Mágicos.

    Substitui o arquivo antigo para ser usado no próximo retreinamento com versionamento
    automático. Valida formato, estrutura e gera logs de auditoria.

    Args:
        file (UploadFile): Arquivo CSV ou XLSX com dados novos.
        request (Request): Objeto de requisição FastAPI.
        _authenticated (Any): Validação de API Key via dependency.

    Returns:
        JSONResponse: Confirmação de sucesso com informações do arquivo salvo.

    Raises:
        HTTPException: 400 formato inválido, 401 API Key inválida, 422 validação falhou.
    """
    # 1. Validar formato do arquivo
    if not file.filename or not file.filename.endswith((".csv", ".xlsx")):
        log_with_request(
            logger=logger,
            level=logging.WARNING,
            event="data_ingestion_invalid_format",
            request=request,
            filename=file.filename,
            allowed_formats=["csv", "xlsx"],
        )
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie arquivo .csv ou .xlsx",
        )

    # 2. Preparar diretório de dados com versionamento
    raw_data_dir = Path("app/data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 3. Gerar nome versionado com timestamp
    file_stem = Path(file.filename).stem
    file_ext = Path(file.filename).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"{file_stem}_{timestamp}{file_ext}"
    versioned_path = raw_data_dir / versioned_filename

    # 4. Salvar arquivo versionado
    try:
        with open(versioned_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 5. Se foi sucesso, copiar para o arquivo "atual" (sobrescrever)
        current_path = raw_data_dir / file.filename
        shutil.copy(versioned_path, current_path)

        log_with_request(
            logger=logger,
            level=logging.INFO,
            event="data_ingestion_success",
            request=request,
            filename=file.filename,
            versioned_filename=versioned_filename,
            file_size_bytes=len(content),
            archive_path=str(versioned_path),
            current_path=str(current_path),
        )

        return JSONResponse(
            status_code=201,
            content={
                "status": "sucesso",
                "mensagem": f"Arquivo {file.filename} atualizado com sucesso.",
                "arquivo_versionado": versioned_filename,
                "timestamp": timestamp,
                "tamanho_bytes": len(content),
                "proximo_passo": "Acesse GET /drift para monitorar degradação e decida se retreino é necessário.",
                "auditoria": {
                    "acao": "ingestao_dados",
                    "timestamp": timestamp,
                    "arquivo": file.filename,
                },
            },
        )

    except OSError as e:
        log_with_request(
            logger=logger,
            level=logging.ERROR,
            event="data_ingestion_file_error",
            request=request,
            filename=file.filename,
            error=str(e),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Erro ao salvar arquivo: {str(e)}",
        )
