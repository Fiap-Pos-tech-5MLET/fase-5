"""Keep-alive para evitar sleep no Render Free Tier.

Este módulo implementa um sistema automático de keep-alive que ping a
aplicação a cada 10 minutos para evitar que o Render Free Tier durma
o container após 15 minutos de inatividade.
"""

import os
import threading
import time
from typing import Optional

import requests

logger_keepalive = __import__("logging").getLogger("keep_alive")


def ping_self() -> None:
    """
    Pinga a aplicação a cada 10 minutos para evitar sleep.

    Executa em thread daemon para não bloquear a aplicação.
    """
    app_url = os.getenv("RENDER_APP_URL", "http://localhost:8080")
    interval_seconds = int(os.getenv("KEEP_ALIVE_INTERVAL", "600"))  # 10 min padrão

    while True:
        try:
            time.sleep(interval_seconds)
            response = requests.get(
                f"{app_url}/health",
                timeout=5,
                allow_redirects=False,
            )
            if response.status_code == 200:
                logger_keepalive.info("✅ Keep-alive ping enviado com sucesso")
            else:
                logger_keepalive.warning(
                    "⚠️ Keep-alive recebeu status %d",
                    response.status_code,
                )
        except requests.exceptions.RequestException as e:
            logger_keepalive.error("❌ Keep-alive falhou: %s", str(e))
        except Exception as e:
            logger_keepalive.error("❌ Erro inesperado no keep-alive: %s", str(e))


def start_keep_alive() -> Optional[threading.Thread]:
    """
    Inicia thread de keep-alive se estiver em produção.

    Returns:
        Optional[threading.Thread]: Thread de keep-alive ou None se não aplicável.
    """
    if os.getenv("ENVIRONMENT") == "production":
        thread = threading.Thread(target=ping_self, daemon=True)
        thread.start()
        logger_keepalive.info(
            "🔄 Keep-alive ativado para Render Free Tier (interval: %s seg)",
            os.getenv("KEEP_ALIVE_INTERVAL", "600"),
        )
        return thread
    return None
