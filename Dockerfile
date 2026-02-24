FROM python:3.11-slim AS builder

LABEL maintainer="Fiap Pos-tech 5MLET"
LABEL description="Datathon Passos Mágicos - Builder (wheels)"

ENV PIP_NO_CACHE_DIR=off

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy only requirements first to leverage build cache
COPY requirements.txt /wheels/requirements.txt

# Build wheels for all requirements (so final image doesn't need build deps)
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /wheels/requirements.txt

FROM python:3.11-slim

LABEL maintainer="Fiap Pos-tech 5MLET"
LABEL description="Datathon Passos Mágicos - Runtime (Nginx + Supervisor + App)"

# Install only runtime system dependencies (no build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nginx \
        supervisor \
        curl \
        gettext-base \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy wheels from builder and install without building
COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt

# Create necessary directories
RUN mkdir -p /app/app/models /app/app/artifacts /app/mlruns /app/data /var/log/supervisor

# Copy minimal application files (context is reduced by .dockerignore)
COPY . /app

# Place nginx template
COPY nginx.conf /etc/nginx/sites-available/default.template

# Place landing page
COPY index.html /app/index.html

# Supervisor config to run multiple services
RUN mkdir -p /var/log/supervisor && cat > /etc/supervisor/conf.d/supervisord.conf <<'EOF'
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
user=root

[program:nginx]
command=/usr/sbin/nginx -g 'daemon off;'
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

# Expose the port used by Nginx (Render will set PORT at runtime)
EXPOSE 80

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
