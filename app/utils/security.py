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
