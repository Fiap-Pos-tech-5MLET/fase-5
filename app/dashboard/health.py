"""
Health check para API — Substitui carregamento local de modelo.

Verifica disponibilidade da API e se o modelo está carregado,
sem necessidade de acessar o filesystem.
"""

from typing import Optional

import requests
import streamlit as st

from app.dashboard.config import DASHBOARD_REQUESTED_BY


def check_api_health(api_url: str, timeout: int = 5) -> bool:
    """
    Verifica se a API está disponível e com modelo carregado.

    Substitui o carregamento local de modelo por um health check
    via HTTP, reduzindo overhead de I/O e memória no dashboard.

    Args:
        api_url (str): URL base da API (ex: 'http://127.0.0.1:8000/api')
        timeout (int): Timeout em segundos. Default: 5.

    Returns:
        bool: True se API está saudável e modelo carregado, False caso contrário.

    Example:
        >>> api_available = check_api_health('http://127.0.0.1:8000/api')
        >>> if api_available:
        ...     st.success("✅ API disponível")
        ... else:
        ...     st.error("❌ API indisponível")
    """
    try:
        # Usa cache para evitar múltiplas requisições na mesma sessão
        @st.cache_data(ttl=60)  # Cache de 60 segundos
        def _health_check() -> bool:
            response = requests.get(
                f"{api_url}/model-info",
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "x-requested-by": DASHBOARD_REQUESTED_BY,
                },
            )
            return response.status_code == 200

        return _health_check()

    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.RequestException:
        return False


def get_api_status(api_url: str, timeout: int = 5) -> Optional[dict]:
    """
    Obtém informações de status da API (versão do modelo, etc).

    Args:
        api_url (str): URL base da API
        timeout (int): Timeout em segundos. Default: 5.

    Returns:
        Optional[dict]: Dados do modelo ou None se indisponível.

    Example:
        >>> status = get_api_status('http://127.0.0.1:8000/api')
        >>> if status:
        ...     st.write(f"Modelo: {status.get('model_id')}")
    """
    try:
        response = requests.get(
            f"{api_url}/model-info",
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "x-requested-by": DASHBOARD_REQUESTED_BY,
            },
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None
