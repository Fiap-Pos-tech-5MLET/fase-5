FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    ENVIRONMENT=production

WORKDIR /app

# Instalação de dependências do sistema e compiladores para ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nginx \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Instalação das dependências Python
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Cópia do código e artefatos
COPY app /app/app
COPY src /app/src
COPY scripts /app/scripts
COPY index.html /app/index.html
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Criação de usuário não-root para executar os serviços
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/app/models /app/app/artifacts /app/mlruns /app/data /var/log/supervisor && \
    chown -R appuser:appuser /app /var/log/supervisor /var/log/nginx /var/lib/nginx && \
    chmod -R 755 /app

# Nginx precisa rodar como root para bind na porta 8080
# mas os workers podem rodar como appuser (configurado no nginx.conf)
USER root

EXPOSE 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]