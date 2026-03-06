"""Helpers para logs estruturados em JSON.

Facilita instrumentação consistente de eventos da API para futura ingestão em
stacks de observabilidade como Datadog e Elasticsearch.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timezone
from typing import Any

from fastapi import Request


def log_json(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emite log estruturado em formato JSON.

    Args:
        logger (logging.Logger): Logger de destino.
        level (int): Nível do log (`logging.INFO`, `logging.ERROR`, etc.).
        event (str): Nome canônico do evento.
        **fields (Any): Campos adicionais do contexto.
    """
    payload: dict[str, Any] = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **fields,
    }
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))


def get_requester_context(request: Request, requested_by: str | None = None) -> dict[str, Any]:
    """Monta contexto padrão do solicitante para auditoria.

    Args:
        request (Request): Objeto da requisição FastAPI.
        requested_by (str | None): Identificador informado do solicitante.

    Returns:
        dict[str, Any]: Contexto padronizado para logs estruturados.
    """
    forwarded_for = request.headers.get("x-forwarded-for", "")
    real_ip = request.headers.get("x-real-ip", "")
    request_id = request.headers.get("x-request-id", "")
    user_agent = request.headers.get("user-agent", "unknown")

    client_ip = "unknown"
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    elif real_ip:
        client_ip = real_ip.strip()
    elif request.client is not None and request.client.host:
        client_ip = request.client.host

    return {
        "requested_by": requested_by or "unknown",
        "client_ip": client_ip,
        "forwarded_for": forwarded_for or None,
        "request_id": request_id or None,
        "user_agent": user_agent,
        "request_method": request.method,
        "request_path": request.url.path,
    }


def log_with_request(
    logger: logging.Logger,
    level: int,
    event: str,
    request: Request,
    requested_by: str | None = None,
    **fields: Any,
) -> None:
    """Registra log estruturado com contexto padronizado da requisição.

    Args:
        logger (logging.Logger): Logger de destino.
        level (int): Nível do log.
        event (str): Nome canônico do evento.
        request (Request): Requisição FastAPI.
        requested_by (str | None): Usuário/ator informado para governança.
        **fields (Any): Campos adicionais do evento.
    """
    context = get_requester_context(request=request, requested_by=requested_by)
    log_json(logger=logger, level=level, event=event, **context, **fields)
