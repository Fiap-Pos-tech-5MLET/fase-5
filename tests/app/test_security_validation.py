"""Testes de validações de segurança da API."""

from __future__ import annotations

from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.utils.security import validate_requested_by


@pytest.fixture
def security_client() -> TestClient:
    """Cria API mínima para testar dependências de validação."""
    app = FastAPI()

    @app.get("/validated")
    def validated_endpoint(
        requested_by: Annotated[str, Depends(validate_requested_by)],
    ) -> dict[str, str]:
        return {"requested_by": requested_by}

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "header_value",
    ["", "   ", "unknown", "UNKNOWN", " Unknown "],
)
def test_validate_requested_by_rejects_invalid_text_values(
    security_client: TestClient, header_value: str
) -> None:
    """Rejeita header vazio, só com espaços e valor padrão unknown."""
    response = security_client.get("/validated", headers={"x-requested-by": header_value})

    assert response.status_code == 422
    assert "x-requested-by" in response.json()["detail"]


def test_validate_requested_by_rejects_missing_header(security_client: TestClient) -> None:
    """Rejeita ausência do header x-requested-by."""
    response = security_client.get("/validated")

    assert response.status_code == 422
    assert "x-requested-by" in response.json()["detail"]


def test_validate_requested_by_accepts_and_normalizes_value(security_client: TestClient) -> None:
    """Aceita valor válido e remove espaços nas bordas."""
    response = security_client.get("/validated", headers={"x-requested-by": "  lucas_admin  "})

    assert response.status_code == 200
    assert response.json()["requested_by"] == "lucas_admin"
