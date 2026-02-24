"""
Testes para validar configuração do Dockerfile e deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.unit
class TestDeploymentConfig:
    """Testes de configuração de deployment."""

    def test_dockerfile_exists(self) -> None:
        """Verifica se Dockerfile existe."""
        dockerfile_path = Path("Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile não encontrado"

    def test_dockerfile_has_supervisor(self) -> None:
        """Verifica se Dockerfile instala Supervisor."""
        with open("Dockerfile", encoding="utf-8") as f:
            content = f.read()
        assert "supervisor" in content.lower(), "Supervisor não está instalado no Dockerfile"
        assert "nginx" in content.lower(), "Nginx não está instalado no Dockerfile"

    def test_dockerfile_exposes_port_80(self) -> None:
        """Verifica se Dockerfile expõe porta 80."""
        with open("Dockerfile", encoding="utf-8") as f:
            content = f.read()
        assert "EXPOSE 80" in content, "Porta 80 não está exposta no Dockerfile"

    def test_dockerfile_has_supervisord_config(self) -> None:
        """Verifica se Dockerfile contém configuração do Supervisor."""
        with open("Dockerfile", encoding="utf-8") as f:
            content = f.read()
        assert "supervisord.conf" in content, "Configuração do Supervisor não encontrada"
        assert "[program:nginx]" in content, "Programa nginx não configurado"
        assert "[program:api]" in content, "Programa api não configurado"
        assert "[program:dashboard]" in content, "Programa dashboard não configurado"
        assert "[program:mlflow]" in content, "Programa mlflow não configurado"

    def test_nginx_conf_uses_localhost(self) -> None:
        """Verifica se nginx.conf usa localhost para proxy."""
        nginx_conf = Path("nginx.conf")
        assert nginx_conf.exists(), "nginx.conf não encontrado"

        with open(nginx_conf, encoding="utf-8") as f:
            content = f.read()

        assert "localhost:8000" in content, "Nginx não está usando localhost para API"
        assert "localhost:8501" in content, "Nginx não está usando localhost para Dashboard"
        assert "localhost:5000" in content, "Nginx não está usando localhost para MLflow"

    def test_nginx_conf_has_all_routes(self) -> None:
        """Verifica se nginx.conf tem todas as rotas necessárias."""
        with open("nginx.conf", encoding="utf-8") as f:
            content = f.read()

        assert "location /api/" in content, "Rota /api/ não encontrada"
        assert "location /dashboard/" in content, "Rota /dashboard/ não encontrada"
        assert "location /mlflow/" in content, "Rota /mlflow/ não encontrada"
        assert "location = /" in content or "location /" in content, "Rota raiz não encontrada"

    def test_render_yaml_exists(self) -> None:
        """Verifica se render.yaml existe."""
        render_yaml = Path("render.yaml")
        assert render_yaml.exists(), "render.yaml não encontrado"

    def test_render_yaml_has_docker_config(self) -> None:
        """Verifica se render.yaml está configurado para Docker."""
        with open("render.yaml", encoding="utf-8") as f:
            content = f.read()

        assert "env: docker" in content, "render.yaml não está configurado para Docker"
        assert "type: web" in content, "render.yaml não tem tipo web"

    def test_deployment_md_exists(self) -> None:
        """Verifica se documentação de deployment existe."""
        deployment_doc = Path("DEPLOYMENT.md")
        assert deployment_doc.exists(), "DEPLOYMENT.md não encontrado"

    def test_streamlit_config_has_baseurlpath(self) -> None:
        """Verifica se Streamlit está configurado com baseUrlPath."""
        streamlit_config = Path(".streamlit/config.toml")
        assert streamlit_config.exists(), "config.toml do Streamlit não encontrado"

        with open(streamlit_config, encoding="utf-8") as f:
            content = f.read()

        assert 'baseUrlPath = "/dashboard"' in content, (
            "baseUrlPath não está configurado corretamente"
        )

    def test_app_main_has_root_path(self) -> None:
        """Verifica se FastAPI tem root_path configurado."""
        app_main = Path("app/main.py")
        assert app_main.exists(), "app/main.py não encontrado"

        with open(app_main, encoding="utf-8") as f:
            content = f.read()

        assert 'root_path="/api"' in content, "root_path não está configurado no FastAPI"
        assert "ENVIRONMENT" in content, "Verificação de ENVIRONMENT não encontrada"

    def test_docker_compose_exists_for_dev(self) -> None:
        """Verifica se docker-compose.yml existe para desenvolvimento."""
        docker_compose = Path("docker-compose.yml")
        assert docker_compose.exists(), "docker-compose.yml não encontrado"
