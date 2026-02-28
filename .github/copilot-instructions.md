# Instruções do Copilot — Projeto Fase 5 (Datathon Passos Mágicos)

## 1) Objetivo

Garantir que sugestões e revisões do Copilot preservem:
- estabilidade de deploy (Render + Docker + Nginx + Supervisor),
- segurança de rotas sensíveis,
- consistência entre código, workflows e documentação,
- qualidade técnica com foco pragmático para entrega acadêmica.

---

## 2) Stack real do projeto

- Backend: FastAPI (`app/main.py`) com `root_path="/api"`.
- Dashboard: Streamlit (`app/dashboard.py`) servido via Nginx em `/dashboard`.
- Runtime produção: container único com Nginx + FastAPI + Streamlit + Supervisord.
- MLOps: champion/challenger com MLflow (`/retrain`, `/promote`, `/discard`).
- CI/CD: GitHub Actions em fluxo GitFlow (`feature/*` → `develop` → `main`).

> Não introduzir padrões que contradigam a arquitetura atual (ex.: separar serviços sem solicitação explícita).

---

## 3) Regras obrigatórias para mudanças de código

### 3.1 API e rotas
- Preservar `root_path="/api"` no FastAPI.
- Não remover endpoints de governança de modelo (`/retrain`, `/promote`, `/discard`).
- Rotas sensíveis devem continuar protegidas por `X-API-KEY` (`validate_api_key`).

### 3.2 Dashboard ↔ API
- Toda chamada do dashboard para backend deve usar `API_URL`.
- Em produção, `API_URL` deve incluir `/api` (ex.: `http://127.0.0.1:8080/api`).
- Evitar fallback silencioso que mascare erro de configuração de ambiente.

### 3.3 Segurança
- Proibido hardcode de segredos.
- `API_KEY` deve vir de variável de ambiente.
- Evitar `except Exception:` e `bare except:`.

### 3.4 Qualidade e estilo
- Type hints e docstrings em português (Google style) para código novo/alterado.
- Nomes descritivos em snake_case/PascalCase conforme contexto.
- Não criar complexidade desnecessária para resolver problemas operacionais simples.

---

## 4) Regras de deploy e configuração

### 4.1 Fonte de verdade de configuração
- `render.yaml` é o contrato IaC do projeto.
- Em Render sem Blueprint (comum no plano free), tratar `render.yaml` como referência e replicar env vars manualmente em **Service → Environment**.

### 4.2 Variáveis operacionais esperadas
- `ENVIRONMENT`
- `API_URL`
- `MODEL_PATH`
- `DATASET_PATH`
- `ARTIFACTS_DIR`
- `KEEP_ALIVE_INTERVAL`
- `MLFLOW_TRACKING_URI`
- `API_KEY` (segredo)

### 4.3 Cuidados para evitar regressão em produção
- Não alterar paths internos Nginx/FastAPI/Streamlit sem validar `nginx.conf` + `supervisord.conf`.
- Manter coerência entre `render.yaml`, `.env.example`, `README.md` e `DEPLOYMENT.md`.

---

## 5) Regras de CI/CD (GitHub Actions)

- `feature-pipeline.yml`: qualidade rápida + testes rápidos + build Docker + PR automático.
- `develop-pipeline.yml`: qualidade completa, testes com coverage, segurança, build Docker.
- `main-pipeline.yml`: valida `render.yaml`, smoke tests, build Docker, deploy Render, smoke pós-deploy e rollback automático.
- `issue-ops-rollback.yml`: rollback via label `ops:rollback` em issue.

Ao editar workflows:
- evitar gatilhos duplicados,
- manter guardrails de segurança,
- não remover validações de deploy sem justificativa explícita.

---

## 6) Testes e validação local

### 6.1 Antes de concluir mudanças
- Rodar testes focados no escopo alterado.
- Se alterar deploy/configuração, validar `tests/test_deployment_config.py`.
- Se alterar dashboard/config API, validar testes de dashboard.

### 6.2 Protocolo mínimo de validação por tipo de mudança

Use esta matriz para validar se código, scripts, docs e testes ficaram coerentes:

- **Mudança em `app/routes/*`, `app/main.py` ou segurança**:
  - confirmar contratos de rota e `root_path="/api"`,
  - confirmar proteção `X-API-KEY` em `/retrain`, `/promote`, `/discard`,
  - executar testes de API relacionados.

- **Mudança em `app/dashboard/*`**:
  - confirmar uso de `API_URL` em chamadas HTTP,
  - validar que mensagens de erro não mascaram problema de configuração,
  - executar testes de dashboard (`tests/test_dashboard_*.py`).

- **Mudança em `scripts/*` ou pipeline de dados/treino**:
  - validar paths com env vars esperadas (`MODEL_PATH`, `DATASET_PATH`, `ARTIFACTS_DIR`),
  - evitar caminhos hardcoded sem fallback controlado,
  - executar testes focados de script/rota impactada.

- **Mudança em deploy/workflows/configuração** (`render.yaml`, `nginx.conf`, `supervisord.conf`, `.github/workflows/*`):
  - confirmar consistência com `README.md`, `DEPLOYMENT.md` e `.env.example`,
  - executar `tests/test_deployment_config.py`,
  - validar gatilhos/guardrails dos workflows sem duplicidade.

### 6.3 Cobertura
- O projeto usa `--cov-fail-under=85` em `pytest.ini`.
- Para testes focados de configuração local, pode usar `--no-cov` quando o objetivo for validar regressão funcional pontual.

---

## 7) Documentação e Mermaid

- Toda mudança relevante em deploy/env/workflow deve atualizar docs correspondentes:
  - `README.md`
  - `DEPLOYMENT.md`
  - `.env.example`
  - `render.yaml`
- Se houver diagrama Mermaid relacionado, atualizar texto + diagrama juntos.

---

## 8) Anti-padrões a evitar neste projeto

- Reintroduzir instruções genéricas de PyTorch/LSTM não usadas no código atual.
- Presumir que Render aplicará `render.yaml` automaticamente em serviço manual legado.
- Corrigir problema operacional com workaround no código sem antes explicitar causa raiz.
- Alterar contratos de API/dashboard sem atualizar testes e docs.

---

## 9) Checklist objetivo para PR

- [ ] Mudança resolve causa raiz sem quebrar fluxos existentes.
- [ ] Rotas sensíveis continuam com `X-API-KEY`.
- [ ] Deploy/configs coerentes (`render.yaml`, `.env.example`, docs).
- [ ] Testes focados do escopo alterado passaram.
- [ ] Mermaid (quando aplicável) atualizado e válido.
- [ ] Se houve alteração em código/scripts, existe teste novo/ajustado cobrindo o comportamento alterado.
- [ ] Se houve alteração de comportamento, README/DEPLOYMENT foram revisados e atualizados quando necessário.

---

## 10) Runbook complementar

Para diretrizes operacionais detalhadas (Render free, troubleshooting de 502/404, validação pós-deploy), consulte:

- `.github/copilot-operational-runbook.md`
