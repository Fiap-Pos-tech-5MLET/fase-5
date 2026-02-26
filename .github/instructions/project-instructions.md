# Contexto do Projeto para Agentes de IA

Resumo operacional para agentes que atuam neste repositório.

## 1) Objetivo do Projeto

Predizer risco de defasagem escolar para apoiar intervenções educacionais da Associação Passos Mágicos, com rastreabilidade técnica e operação contínua.

## 2) Arquitetura Atual

- API: FastAPI (`app/main.py`)
- Dashboard: Streamlit (`app/dashboard.py`)
- Reverse proxy: Nginx (`nginx.conf`)
- Process manager: Supervisor (`supervisord.conf`)
- Modelo: scikit-learn com artefatos em `app/models/`
- MLOps: ciclo champion/challenger (`/retrain`, `/promote`, `/discard`)

## 3) Rotas-Chave

- `GET /health`
- `POST /predict`
- `GET /model-info`
- `GET /drift`
- `POST /retrain`
- `POST /promote`
- `POST /discard`
- `GET /model-metrics`

## 4) Qualidade e CI/CD

Workflows oficiais:

- `.github/workflows/feature-pipeline.yml`
- `.github/workflows/develop-pipeline.yml`
- `.github/workflows/main-pipeline.yml`
- `.github/workflows/issue-ops-rollback.yml`

Gates por branch:

- Feature/Bugfix: qualidade + testes rápidos + build Docker
- Develop: qualidade + testes completos + segurança + build Docker
- Main: smoke + build Docker + deploy hook Render

## 5) Regras de alteração de código

- Priorizar correção na causa raiz.
- Evitar mudanças fora do escopo solicitado.
- Manter tipagem, docstrings e consistência de estilo.
- Atualizar documentação afetada na mesma entrega.

## 6) Operação e incidentes

- Deploy é orientado por GitOps (merge + CI/CD).
- Rollback é via PR (reversão de commit ou ajuste do `champion_run_id.txt`).
- Não executar correções manuais diretas em produção.

## 7) Comandos úteis

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov=app --cov-report=term-missing
make quality
docker build -f Dockerfile .
```
