"""Testes para utilitários de log estruturado com contexto de requisição."""

import logging
from unittest.mock import MagicMock

from app.utils.structured_logging import get_requester_context


class _Client:
    """Cliente fake para simular request.client."""

    def __init__(self, host: str) -> None:
        self.host = host


def test_get_requester_context_uses_forwarded_for_first_ip() -> None:
    """Deve priorizar o primeiro IP de x-forwarded-for."""
    request = MagicMock()
    request.headers = {
        "x-forwarded-for": "203.0.113.10, 10.0.0.1",
        "x-real-ip": "198.51.100.7",
        "user-agent": "pytest-agent",
        "x-request-id": "req-123",
    }
    request.client = _Client("127.0.0.1")
    request.method = "POST"
    request.url.path = "/retrain"

    context = get_requester_context(request=request, requested_by="lucas_admin")

    assert context["client_ip"] == "203.0.113.10"
    assert context["requested_by"] == "lucas_admin"
    assert context["request_path"] == "/retrain"


def test_get_requester_context_falls_back_to_client_host() -> None:
    """Deve usar request.client.host quando headers de proxy não existirem."""
    request = MagicMock()
    request.headers = {}
    request.client = _Client("192.168.0.10")
    request.method = "GET"
    request.url.path = "/model-info"

    context = get_requester_context(request=request)

    assert context["client_ip"] == "192.168.0.10"
    assert context["requested_by"] == "unknown"
    assert context["request_method"] == "GET"
