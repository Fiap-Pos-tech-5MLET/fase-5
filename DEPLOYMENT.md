# Deploy (Render + Docker)

Guia objetivo para reproduzir o deploy de produção.

## 1) Arquitetura de Produção

- Runtime: `Docker` (container único)
- Process manager: `supervisord`
- Entry point: `nginx` na porta `8080`
- Serviços internos:
  - FastAPI: `127.0.0.1:8000`
  - Streamlit: `127.0.0.1:8501`

## 2) Rotas Públicas

- `/` → Landing page
- `/api/docs` → Swagger
- `/dashboard/` → Dashboard Streamlit
- `/health` → Health check

## 3) Pré-requisitos no Render

- Repositório conectado ao Render
- Arquivo `render.yaml` versionado
- Secret `RENDER_DEPLOY_HOOK_URL` configurado no GitHub (pipeline de `main`)

## 4) Variáveis de Ambiente

Defina variáveis não sensíveis no `render.yaml` e segredos no painel do Render.

Variáveis operacionais esperadas:
- `ENVIRONMENT=production`
- `API_URL=https://SEU_APP.onrender.com/api`
- `MODEL_PATH=app/models/model.pkl`
- `DATASET_PATH=app/data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`
- `ARTIFACTS_DIR=app/artifacts`
- `KEEP_ALIVE_INTERVAL=600`
- `MLFLOW_TRACKING_URI=file:./mlruns`

Segredos (não versionar):
- `API_KEY`

## 5) Deploy (Caminho Feliz)

1. Merge em `main` após aprovação no GitFlow.
2. Pipeline `main-pipeline.yml` executa smoke tests + build Docker.
3. Deploy hook do Render é acionado automaticamente.

## 6) Validação Pós-Deploy

```bash
curl -fsSL https://SEU_APP.onrender.com/health
curl -fsSL https://SEU_APP.onrender.com/api/docs > /dev/null
```

Valide também manualmente:
- `https://SEU_APP.onrender.com/`
- `https://SEU_APP.onrender.com/dashboard/`

## 7) Rollback

Rollback é **GitOps-only**:
- Reverter commit ou ajustar referência do champion (`app/models/champion_run_id.txt`) via PR.
- Nunca corrigir produção manualmente fora da esteira.

## 8) Troubleshooting Rápido

- Erro em `/dashboard/`: verificar `nginx.conf` + `supervisord.conf` + logs do processo `dashboard`.
- Erro em `/api/docs`: validar processo `api` e health check interno.
- Build falhou no CI: reproduzir com `docker build -f Dockerfile .`.

---

Referências: [README.md](README.md), [TESTING.md](TESTING.md), [CONTRIBUTING.md](CONTRIBUTING.md).
