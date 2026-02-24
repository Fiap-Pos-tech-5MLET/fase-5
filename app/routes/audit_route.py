"""
Rotas de auditoria e monitoramento.

Endpoints para informações do modelo, drift detection e health check.
"""

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.models.schemas import ModelInfoResponse
from app.utils.model_loader import get_current_model, get_model_info, get_model_paths

logger = logging.getLogger("audit_route")

router = APIRouter()


@router.get("/")
def health_check():
    """
    Health check endpoint.

    Returns:
        Dicionário com status e informação se modelo está carregado.
    """
    model = get_current_model()
    return {"status": "ok", "model_loaded": model is not None}


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """
    Retorna metadados do modelo em produção, incluindo estratégia de
    retreinamento e cenários de produção documentados.

    Returns:
        ModelInfoResponse com metadados completos.
    """
    model = get_current_model()
    info_basic = get_model_info()

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

    return info


@router.get("/drift", response_class=HTMLResponse)
def drift_report():
    """
    Serve o relatório de data drift como HTML.

    Returns:
        HTMLResponse com relatório Evidently.

    Raises:
        HTTPException: 404 se relatório não existir.
    """
    models_dir, _, _ = get_model_paths()
    report_path = os.path.join(models_dir, "artifacts", "data_drift_report.html")

    if not os.path.exists(report_path):
        logger.warning("Drift report not found. Run `python scripts/monitoring.py` to generate.")
        raise HTTPException(
            status_code=404, detail="Drift report not found. Run monitoring.py first."
        )

    with open(report_path, encoding="utf-8") as f:
        html_content = f.read()

    logger.info("Drift report served successfully.")
    return HTMLResponse(content=html_content)
