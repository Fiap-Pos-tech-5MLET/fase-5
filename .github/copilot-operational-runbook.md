# Runbook Operacional para Copilot — Fase 5

Este runbook complementa `.github/copilot-instructions.md` com práticas para evitar incidentes recorrentes em deploy e integração.

## 1) Render (plano free) — comportamento esperado

- Sem Blueprint ativo, `render.yaml` pode não ser aplicado automaticamente em serviço manual existente.
- Nesse cenário, env vars devem ser conferidas/replicadas em **Service → Environment**.
- Mudanças em `render.yaml` só passam a valer no runtime quando houver sync/aplicação do serviço correspondente.

## 2) Erros comuns e diagnóstico

### 2.1 `502 Bad Gateway` no início
Possíveis causas:
- cold start do plano free,
- Nginx aceitando tráfego antes de API/dashboard estarem prontos.

Diagnóstico:
- verificar logs de `nginx`, `api` e `dashboard` no startup,
- validar `/health`, `/api/docs` e `/dashboard/` após deploy.

### 2.2 `404` em `/promote` vindo do dashboard
Possíveis causas:
- `API_URL` incorreta (sem `/api` em produção),
- variável ausente no ambiente do serviço.

Diagnóstico:
- validar `API_URL` efetiva no runtime,
- garantir que chamadas partem de `API_URL + /promote`.

### 2.3 `401 Unauthorized` em retreinamento/promote/discard
Possível causa:
- `API_KEY` ausente ou divergente entre dashboard e API.

Diagnóstico:
- confirmar header `X-API-KEY`,
- validar `API_KEY` no ambiente da API.

## 3) Contrato mínimo de envs (produção)

- `ENVIRONMENT=production`
- `API_URL=http://127.0.0.1:8080/api`
- `MODEL_PATH=/app/app/models/model.pkl`
- `DATASET_PATH=/app/app/data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`
- `ARTIFACTS_DIR=/app/app/artifacts`
- `KEEP_ALIVE_INTERVAL=600`
- `MLFLOW_TRACKING_URI=file:./mlruns`
- `API_KEY` (secret)

## 4) Procedimento de validação pós-deploy

1. Health: `GET /health`
2. Docs: `GET /api/docs`
3. Dashboard: `GET /dashboard/`
4. Predição simples em `POST /api/predict`
5. Fluxo governado (quando aplicável): `POST /retrain` → `POST /promote` com `X-API-KEY`

## 5) Quando atualizar documentação

Atualize obrigatoriamente em mudanças operacionais:
- `README.md` (setup e visão geral),
- `DEPLOYMENT.md` (runbook de deploy),
- `.env.example` (baseline local),
- `render.yaml` (contrato IaC).

## 6) Princípio de decisão para agentes

Antes de aplicar workaround em código para erro de produção:
1. confirmar causa raiz operacional (env/deploy/runtime),
2. corrigir configuração na fonte,
3. só então ajustar código se o problema persistir por falha real de aplicação.

## 7) Fluxo de validação pré-PR (código + docs + testes)

Antes de concluir qualquer tarefa, validar nesta ordem:

1. **Escopo alterado**
	- listar arquivos modificados e classificar por tipo: API, dashboard, scripts, deploy, docs.

2. **Padrões de código/scripts**
	- confirmar type hints/docstrings em trechos novos,
	- remover comentários obsoletos e imports/variáveis não usados,
	- evitar hardcode de segredos e caminhos de produção.

3. **Reflexo em documentação**
	- se mudou comportamento, revisar e atualizar:
	  - `README.md` (uso/setup/fluxo),
	  - `DEPLOYMENT.md` (deploy/env/troubleshooting),
	  - `.env.example` (variáveis de referência),
	  - `render.yaml` (contrato IaC).

4. **Testes obrigatórios por categoria**
	- API/rotas: rodar testes de rota afetada.
	- Dashboard/config API: rodar `tests/test_dashboard_*.py`.
	- Deploy/config: rodar `tests/test_deployment_config.py`.
	- Quando não houver teste cobrindo a mudança, adicionar ou ajustar teste focado.

5. **Fechamento**
	- registrar o que foi validado e o resultado,
	- não aprovar mudança com comportamento novo sem teste e sem revisão de docs correspondentes.
