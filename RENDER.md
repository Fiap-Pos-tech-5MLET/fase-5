# Render deployment — multi-service setup

Este repositório foi preparado para deploy no Render com 4 serviços independentes:

- `datathon-api` (FastAPI)
- `datathon-dashboard` (Streamlit)
- `datathon-mlflow` (MLflow UI)
- `datathon-frontend` (Nginx proxy + landing page)

Mudanças principais
- Separação de dependências em `requirements-api.txt`, `requirements-dashboard.txt`, `requirements-mlflow.txt`.
- Dockerfiles dedicados: `Dockerfile.api`, `Dockerfile.dashboard`, `Dockerfile.mlflow`, `Dockerfile.nginx` (multi-stage builds para wheels).
- `render.yaml` atualizado com 4 serviços (cada serviço referencia seu `dockerfilePath`).
- CI ajustado para publicar imagens no GHCR (GitHub Container Registry) e usar cache.

Como funciona no Render
- Para cada serviço em `render.yaml`, Render irá construir a imagem a partir do Dockerfile correspondente ou usar uma imagem pública (`image:`) se preferir.
- No plano gratuito, cuide dos limites: executar MLflow + Streamlit + API num único container é instável. Recomendamos manter serviços separados e, se necessário, mover MLflow para storage externo.

Comandos úteis

- Testar local (requer Docker):
```bash
docker build -f Dockerfile.api -t fase5-api:local .
docker build -f Dockerfile.dashboard -t fase5-dashboard:local .
docker build -f Dockerfile.mlflow -t fase5-mlflow:local .
docker build -f Dockerfile.nginx -t fase5-frontend:local .

# Run minimal smoke tests (map ports)
docker run --rm -p 8000:8000 fase5-api:local
docker run --rm -p 8501:8501 fase5-dashboard:local
docker run --rm -p 5000:5000 fase5-mlflow:local
docker run --rm -p 8080:80 fase5-frontend:local
```

- Build + push via GitHub Actions: o workflow `.github/workflows/ci-cd-pipeline.yml` agora publica imagens no GHCR como `ghcr.io/<ORG>/fase-5:ci-${{ github.sha }}`.

Recomendações finais
- Faça push das alterações para rodar o workflow no GitHub (verificará lint, testes, build e publicará imagens no GHCR).
- Se desejar reduzir ainda mais o tamanho das imagens, eu posso revisar `requirements.txt`/remover dependências que não são necessárias para produção.
- Para deploy no Render com builds mais rápidos, recomendo usar imagens publicadas no GHCR e apontar `render.yaml` para `image:` em vez de construir na plataforma.
