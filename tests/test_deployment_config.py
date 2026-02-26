"""
Testes para validar configuração do Dockerfile e deployment.
"""

from __future__ import annotations

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

    def test_dockerfile_exposes_port_8080(self) -> None:
        """Verifica se Dockerfile expõe porta 8080."""
        with open("Dockerfile", encoding="utf-8") as f:
            content = f.read()
        assert "EXPOSE 8080" in content, "Porta 8080 não está exposta no Dockerfile"

    def test_dockerfile_has_supervisord_config(self) -> None:
        """Verifica se Dockerfile contém configuração do Supervisor."""
        with open("Dockerfile", encoding="utf-8") as f:
            content = f.read()
        assert "supervisord.conf" in content, "Configuração do Supervisor não encontrada"
        assert "COPY supervisord.conf" in content, "Dockerfile não copia supervisord.conf"

    def test_nginx_conf_uses_localhost(self) -> None:
        """Verifica se nginx.conf usa localhost para proxy."""
        nginx_conf = Path("nginx.conf")
        assert nginx_conf.exists(), "nginx.conf não encontrado"

        with open(nginx_conf, encoding="utf-8") as f:
            content = f.read()

        assert "127.0.0.1:8000" in content, "Nginx não está usando loopback para API"
        assert "127.0.0.1:8501" in content, "Nginx não está usando loopback para Dashboard"

    def test_nginx_conf_has_all_routes(self) -> None:
        """Verifica se nginx.conf tem todas as rotas necessárias."""
        with open("nginx.conf", encoding="utf-8") as f:
            content = f.read()

        assert "location /api/" in content, "Rota /api/ não encontrada"
        assert "location /dashboard/" in content, "Rota /dashboard/ não encontrada"
        assert "location /health" in content, "Rota /health não encontrada"
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

    def test_docker_compose_exists_for_dev(self) -> None:
        """Verifica se docker-compose.yml existe para desenvolvimento."""
        docker_compose = Path("docker-compose.yml")
        assert docker_compose.exists(), "docker-compose.yml não encontrado"
