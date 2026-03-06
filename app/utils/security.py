"""Utilitários de segurança para rotas sensíveis da API."""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Header, HTTPException, status


def validate_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-KEY")] = None,
) -> None:
    """Valida a API key enviada no cabeçalho HTTP.

    Args:
        x_api_key (str | None): Valor enviado no header `X-API-KEY`.

    Raises:
        HTTPException: 401 quando a chave está ausente ou inválida.
    """
    configured_api_key = os.getenv("API_KEY", "")

    if not configured_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: API key não configurada.",
        )

    if not x_api_key or not hmac.compare_digest(x_api_key, configured_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: X-API-KEY inválida.",
        )


def validate_requested_by(
    x_requested_by: Annotated[str | None, Header(alias="x-requested-by")] = None,
) -> str:
    """Valida o header de autoria para auditoria das requisições.

    Args:
        x_requested_by (str | None): Valor enviado no header `x-requested-by`.

    Returns:
        str: Valor normalizado (sem espaços nas bordas).

    Raises:
        HTTPException: 422 quando ausente, vazio, somente espaços ou `unknown`.
    """
    if x_requested_by is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Validation error: x-requested-by é obrigatório.",
        )

    normalized = x_requested_by.strip()
    if not normalized or normalized.lower() == "unknown":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Validation error: x-requested-by não pode ser vazio, apenas espaços ou 'unknown'."
            ),
        )

    return normalized
