FROM python:3.13.3-slim

LABEL maintainer="Fiap Pos-tech 5MLET"
LABEL description="Datathon Passos Mágicos - Simplified single-stage image (Nginx + Supervisor + App)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libffi-dev \
    libpq-dev \
    libssl-dev \
       nginx \
       supervisor \
       curl \
       gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (simple single-stage install)
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .
RUN mkdir -p /app/app/models /app/app/artifacts /app/mlruns /app/data /var/log/supervisor

# Definir variável de ambiente para produção
ENV ENVIRONMENT=production

# Copiar arquivo de configuração do Nginx
COPY nginx.conf /etc/nginx/sites-available/default

# Copiar página de landing
COPY index.html /app/index.html

# Criar arquivo de configuração do supervisor para rodar múltiplos processos
RUN mkdir -p /var/log/supervisor
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
user=root

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/nginx.err.log
stdout_logfile=/var/log/supervisor/nginx.out.log

[program:api]
command=uvicorn app.main:app --host localhost --port 8000
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/api.err.log
stdout_logfile=/var/log/supervisor/api.out.log

[program:dashboard]
command=streamlit run app/dashboard.py --server.port 8501 --server.address localhost --server.baseUrlPath /dashboard --server.headless true
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/dashboard.err.log
stdout_logfile=/var/log/supervisor/dashboard.out.log

[program:mlflow]
command=mlflow server --host localhost --port 5000 --backend-store-uri file:///app/mlruns --default-artifact-root file:///app/app/models
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/mlflow.err.log
stdout_logfile=/var/log/supervisor/mlflow.out.log
EOF

# Expor porta 80 (Nginx)
EXPOSE 80

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Rodar supervisor que gerencia todos os 3 processos
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]