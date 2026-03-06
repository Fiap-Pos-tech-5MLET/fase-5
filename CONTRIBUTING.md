# Guia de Contribuição

> Guia completo para contribuição técnica no projeto, seguindo padrões GitFlow, qualidade de código e boas práticas de Machine Learning Engineering.

---

## 📋 Índice

- [1. Fluxo GitFlow](#1-fluxo-gitflow)
- [2. Requisitos Mínimos](#2-requisitos-mínimos)
- [3. Padrões de Código](#3-padrões-de-código)
- [4. Segurança](#4-segurança)
- [5. Mensagens de Commit](#5-mensagens-de-commit)
- [6. Gestão de Incidentes](#6-gestão-de-incidentes)
- [7. Checklist de PR](#7-checklist-de-pr)
- [8. Code Review](#8-code-review)
- [9. CI/CD Gates](#9-cicd-gates)

---

## 1. Fluxo GitFlow

### 🌊 Branches e Estratégia

```mermaid
flowchart LR
    Feature["feature/*<br/>bugfix/*"] -->|PR| Develop[develop]
    Develop -->|PR<br/>aprovado| Main[main]
    Main -->|auto| Deploy[🚀 Produção<br/>Render]
    
    classDef feat fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef dev fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef prod fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    
    class Feature feat;
    class Develop dev;
    class Main,Deploy prod;
```

### 📋 Regras de Branch

| Branch | Origem | Destino | Proteção | CI/CD |
|--------|--------|---------|----------|-------|
| `feature/*` | `develop` | `develop` | ❌ Nenhuma | `feature-pipeline.yml` |
| `bugfix/*` | `develop` | `develop` | ❌ Nenhuma | `feature-pipeline.yml` |
| `develop` | `main` | `main` | ✅ Requer aprovação | `develop-pipeline.yml` |
| `main` | - | - | ✅ Protegida | `main-pipeline.yml` |

### 🎯 Convenções de Nomenclatura

**Branches:**
```bash
feature/nome-descritivo        # Nova funcionalidade
bugfix/descricao-do-bug        # Correção de bug
hotfix/correcao-urgente        # Correção em produção (direto da main)
```

**Exemplos:**
- `feature/adicionar-endpoint-drift`
- `feature/lucas_machado` (nome do desenvolvedor para trabalho longo)
- `bugfix/corrigir-validacao-api-key`
- `hotfix/corrigir-502-dashboard`

---

## 2. Requisitos Mínimos

### ✅ Antes de Abrir PR

**Checklist obrigatório:**

```bash
# 1. Formatar código
make format

# 2. Validar lint
make lint

# 3. Validar tipos
make type-check

# 4. Validar segurança
make security

# 5. Executar testes
make test

# 6. Validar build Docker (quando aplicável)
docker build -f Dockerfile -t fase-5:local .
```

**Ou execute tudo de uma vez:**
```bash
make quality  # Executa: format → lint → type → security
make test     # Executa suite completa de testes
```

### 🔍 Validações por Categoria

**1. Qualidade de Código:**
```bash
ruff check .                    # Lint completo
ruff format --check .           # Validar formatação
mypy src/ app/ --strict         # Type checking rigoroso
```

**2. Segurança:**
```bash
bandit -r src/ app/ -ll         # Scan de vulnerabilidades
detect-secrets scan --baseline .secrets.baseline  # Secrets scan
```

**3. Testes:**
```bash
pytest tests/ -v --cov=src --cov=app --cov-report=term-missing
# Target: ≥ 85% cobertura
```

---

## 3. Padrões de Código

### 🐍 Python Standards

**Type Hints Obrigatórios:**
```python
# ✅ BOM
def calcular_media(notas: list[float]) -> float:
    \"\"\"Calcula média aritmética de uma lista de notas.
    
    Args:
        notas: Lista de notas numéricas.
        
    Returns:
        Média aritmética das notas.
        
    Raises:
        ValueError: Se lista estiver vazia.
    \"\"\"
    if not notas:
        raise ValueError("Lista de notas não pode estar vazia")
    return sum(notas) / len(notas)

# ❌ RUIM (sem tipos, sem docstring)
def calcular_media(notas):
    return sum(notas) / len(notas)
```

**Docstrings em Português (Google Style):**
```python
def processar_aluno(
    dados: dict[str, Any],
    ano: int = 2024
) -> pd.DataFrame:
    \"\"\"Processa dados brutos de um aluno para feature engineering.
    
    Args:
        dados: Dicionário com dados brutos do aluno.
        ano: Ano de referência para cálculos (padrão: 2024).
        
    Returns:
        DataFrame com features processadas.
        
    Raises:
        KeyError: Se campo obrigatório estiver ausente.
        ValueError: Se ano for inválido.
        
    Examples:
        >>> dados = {\"IDADE\": 12, \"NOTA\": 8.5}
        >>> df = processar_aluno(dados, ano=2024)
        >>> assert \"IDADE\" in df.columns
    \"\"\"
    ...
```

### 📏 Nomenclatura (PEP 8)

| Tipo | Padrão | Exemplo |
|------|--------|---------|
| **Variáveis e funções** | `snake_case` | `calcular_media`, `dados_processados` |
| **Classes** | `PascalCase` | `ModelLoader`, `StudentData` |
| **Constantes** | `UPPER_SNAKE_CASE` | `MAX_IDADE`, `API_TIMEOUT` |
| **Privado** | `_prefixo` | `_validar_interno` |
| **Módulos** | `snake_case` | `data_cleaning.py` |

### 🚫 Anti-Padrões Proibidos

**1. Segredos Hardcoded:**
```python
# ❌ NUNCA FAZER ISSO
API_KEY = "sk-1234567890abcdef"
DATABASE_URL = "postgresql://user:senha@host/db"

# ✅ CORRETO
import os
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise EnvironmentError("API_KEY não configurada")
```

**2. Print em Produção:**
```python
# ❌ EVITAR
print(f"Processando aluno: {aluno_id}")

# ✅ CORRETO
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processando aluno: {aluno_id}")
```

**3. Bare Except:**
```python
# ❌ EVITAR
try:
    processar_dados()
except:  # Captura tudo, incluindo KeyboardInterrupt
    pass

# ✅ CORRETO
try:
    processar_dados()
except ValueError as e:
    logger.error(f"Erro ao processar: {e}")
    raise
```

---

## 4. Segurança

### 🔒 Regras de Segurança

**Nunca commitar:**
- ❌ Credenciais, tokens, API keys
- ❌ Arquivos `.env` reais (apenas `.env.example`)
- ❌ Certificados, chaves privadas
- ❌ Dados pessoais de alunos
- ❌ Logs com informações sensíveis

**Validações Automatizadas:**
```bash
# Pre-commit hook (recomendado)
detect-secrets scan --baseline .secrets.baseline

# CI/CD validation
bandit -r src/ app/ -ll         # Security scan
```

### 🛡️ Boas Práticas

1. **Validação de Input:**
```python
from pydantic import BaseModel, validator

class StudentInput(BaseModel):
    idade: int
    nota: float
    
    @validator('idade')
    def validar_idade(cls, v):
        if not 5 <= v <= 18:
            raise ValueError('Idade deve estar entre 5 e 18 anos')
        return v
```

2. **Sanitização de Logs:**
```python
# ✅ Remove informações sensíveis
def safe_log(data: dict) -> dict:
    sensitive_keys = ['cpf', 'senha', 'api_key']
    return {k: '***' if k in sensitive_keys else v 
            for k, v in data.items()}
```

3. **Rate Limiting (Nginx):**
```nginx
# Configurar em nginx.conf
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
```

---

## 5. Mensagens de Commit

### 📝 Padrão Conventional Commits

Formato: `<tipo>(<escopo>): <descrição>`

**Tipos Principais:**

| Tipo | Quando Usar | Exemplo |
|------|-------------|---------|
| `feat` | Nova funcionalidade | `feat(api): adicionar endpoint /drift` |
| `fix` | Correção de bug | `fix(dashboard): corrigir layout em mobile` |
| `docs` | Documentação apenas | `docs(readme): atualizar guia de deploy` |
| `test` | Adicionar/modificar testes | `test(api): adicionar testes de /retrain` |
| `refactor` | Refatoração sem mudar comportamento | `refactor(model): simplificar pipeline` |
| `style` | Formatação, lint | `style: aplicar ruff format` |
| `perf` | Melhoria de performance | `perf(api): otimizar query de dados` |
| `chore` | Tarefas de manutenção | `chore(deps): atualizar requirements.txt` |
| `ci` | Mudanças em CI/CD | `ci: adicionar step de security scan` |

**Exemplos Completos:**
```bash
feat(api): implementar sistema champion/challenger

- Adiciona endpoints /retrain, /promote, /discard
- Integra com MLflow para rastreabilidade
- Adiciona validação de métricas antes de promoção

Closes #45
```

```bash
fix(dashboard): corrigir erro ao carregar modelo

Quando API_URL não estava configurado, dashboard falhava silenciosamente.
Agora exibe mensagem clara de erro.

Fixes #67
```

---

## 6. Gestão de Incidentes

### 🚨 Estratégia GitOps-Only

**Rollback é EXCLUSIVAMENTE via Git:**
- ✅ Reverter commit com `git revert`
- ✅ Ajustar `app/models/champion_run_id.txt` via PR
- ✅ Merge controlado com CI/CD
- ❌ NUNCA rollback manual direto no servidor

### 🔄 Procedimento de Rollback

**1. Via Issue Label `ops:rollback`:**
```bash
# No GitHub Issues:
1. Criar issue com título: "Rollback para <commit-sha>"
2. Adicionar label: ops:rollback
3. Workflow .github/workflows/issue-ops-rollback.yml executa automaticamente
```

**2. Via Git Revert (manual):**
```bash
# 1. Identificar commit problemático
git log --oneline -10

# 2. Criar branch de rollback
git checkout main
git pull origin main
git checkout -b hotfix/rollback-commit-abc123

# 3. Reverter commit
git revert abc123 --no-edit

# 4. Abrir PR para main
git push origin hotfix/rollback-commit-abc123
# No GitHub: criar PR hotfix → main
```

### 📋 Decisão Tree de Incidentes

```mermaid
graph TD
    A[🚨 Incidente detectado] --> B{Severidade?}
    B -->|Crítico<br/>Prod quebrada| C[Rollback imediato<br/>ops:rollback]
    B -->|Alto<br/>Funcionalidade afetada| D[Hotfix urgente<br/>30min]
    B -->|Médio<br/>Problema parcial| E[bugfix/ branch<br/>2h-4h]
    B -->|Baixo<br/>Cosmético| F[feature/ branch<br/>próximo sprint]
    
    C --> G[main-pipeline.yml<br/>valida e deploya]
    D --> H[PR direto para main<br/>aprovação rápida]
    E --> I[PR para develop<br/>teste completo]
    F --> I
    
    style A fill:#ffebee,stroke:#c62828
    style C fill:#fff3e0,stroke:#f57c00
    style G fill:#e8f5e9,stroke:#388e3c
```

---

## 7. Checklist de PR

### ✅ Validação Pré-Merge

Antes de aprovar PR, confirmar:

- [ ] **Código e Documentação:**
  - [ ] Mudança resolve causa raiz sem quebrar fluxos existentes
  - [ ] Type hints em todas as funções novas/alteradas
  - [ ] Docstrings em português (Google style)
  - [ ] README/DEPLOYMENT atualizados (quando aplicável)
  - [ ] Diagramas Mermaid atualizados (quando aplicável)

- [ ] **Segurança:**
  - [ ] Rotas sensíveis continuam com `X-API-KEY`
  - [ ] Sem credenciais hardcoded
  - [ ] Validação de input em endpoints novos
  - [ ] Logs não expõem dados sensíveis

- [ ] **Testes:**
  - [ ] Testes focados do escopo alterado passaram
  - [ ] Cobertura ≥ 85% mantida
  - [ ] Testes novos cobrem comportamento alterado

- [ ] **Configuração:**
  - [ ] Deploy/configs coerentes (`render.yaml`, `.env.example`, docs)
  - [ ] Env vars novas documentadas em `.env.example` + README
  - [ ] Paths configuráveis via env vars (não hardcoded)

- [ ] **CI/CD:**
  - [ ] Pipeline passou sem erros
  - [ ] Build Docker bem-sucedido (quando aplicável)
  - [ ] Smoke tests passaram (develop/main)

---

## 8. Code Review

### 👀 Critérios de Review

**Obrigatórios:**
1. **Funcionalidade:** Resolve o problema descrito?
2. **Testes:** Tem cobertura adequada (≥85%)?
3. **Segurança:** Sem vulnerabilidades óbvias?
4. **Documentação:** Código auto-explicativo + docstrings?
5. **Manutenibilidade:** Código simples e direto?

**Desejáveis:**
6. **Performance:** Algoritmo eficiente?
7. **Escalabilidade:** Suporta crescimento de dados?
8. **Observabilidade:** Logs estruturados adequados?

### 💡 Boas Práticas de Reviewer

**O que procurar:**
- 🔍 Validação de input em endpoints
- 🔍 Tratamento de exceções específico (não `except:`)
- 🔍 Uso de logging ao invés de prints
- 🔍 Comentários explicam "por quê", não "o quê"
- 🔍 Funções pequenas (<50 linhas)
- 🔍 Coesão e baixo acoplamento

**O que evitar no review:**
- ❌ Criticar estilo pessoal (use formatador automático)
- ❌ Exigir perfeição (priorize "bom o suficiente")
- ❌ Reescrever código completo (sugira melhorias incrementais)
  
**Comunicação efetiva:**
```markdown
# ✅ BOM (construtivo)
Considere adicionar validação de `idade` entre 5-18 anos nesta linha:
\`\`\`python
if not 5 <= dados['idade'] <= 18:
    raise ValueError("Idade inválida")
\`\`\`

# ❌ RUIM (vago)
Código ruim, reescreva isso.
```

---

## 9. CI/CD Gates

### 🚦 Quality Gates por Branch

| Branch | Pipeline | Gates | Tempo Esperado |
|--------|----------|-------|----------------|
| `feature/*` | `feature-pipeline.yml` | Lint, Type, Build Docker | ~3min |
| `develop` | `develop-pipeline.yml` | Acima + Testes (≥85%), Security, Smoke | ~5min |
| `main` | `main-pipeline.yml` | Acima + Deploy Render, Smoke Prod, Rollback Auto | ~8min |

### 📊 Métricas de Qualidade

**Thresholds obrigatórios:**

```python
# pytest.ini
[tool:pytest]
addopts = --cov-fail-under=85  # Cobertura mínima
timeout = 120                  # Timeout por teste
```

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
disallow_untyped_calls = true
```

**Métricas monitoradas:**

| Métrica | Target | Falha se |
|---------|--------|----------|
| **Cobertura de testes** | ≥85% | <85% |
| **Tempo de execução testes** | <2min | >5min (timeout) |
| **Erros de lint** | 0 | >0 |
| **Erros de type-check** | 0 | >0 |
| **Vulnerabilidades (bandit)** | 0 | Severidade ≥ Medium |
| **Secrets detectados** | 0 | >0 |

### 🎯 Validação de Deploy (main)

**Smoke Tests Pós-Deploy:**
```bash
# 1. Health check API
curl https://api-url/api/health

# 2. Verificar modelo carregado
curl https://api-url/api/model/info

# 3. Teste de predição básico
curl -X POST https://api-url/api/predict \
  -H "Content-Type: application/json" \
  -d '{"IDADE": 12, "NOTA_PORT": 7.5, ...}'

# 4. Dashboard acessível
curl https://api-url/dashboard/ | grep "Painel de Predições"
```

**Rollback Automático:**
```yaml
# .github/workflows/main-pipeline.yml
- name: Smoke Test Produção
  run: |
    pytest tests/smoke/ --url=$RENDER_URL
  continue-on-error: false  # Falha interrompe pipeline
  
- name: Rollback se smoke falhar
  if: failure()
  run: |
    echo "Smoke tests falharam. Iniciando rollback..."
    git revert HEAD --no-edit
    git push origin main
```

---

## 📚 Documentação de Referência

**Guias complementares:**
- [TESTING.md](TESTING.md) - Estratégia detalhada de testes
- [DEPLOYMENT.md](DEPLOYMENT.md) - Guia de deploy e troubleshooting
- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Pirâmide de testes e métricas
- [README.md](README.md) - Visão geral do projeto

**Runbooks operacionais:**
- `.github/copilot-instructions.md` - Regras para Pull Requests
- `.github/copilot-operational-runbook.md` - Troubleshooting de produção

---

## 🤝 Dúvidas e Suporte

**Canais de comunicação:**
- 💬 Issues do GitHub para bugs e melhorias
- 📧 Email do time: [contato projeto]
- 📚 Docs: consulte os guias na raiz do repositório

**Antes de abrir issue:**
1. Verifique se já existe issue similar
2. Inclua steps de reprodução
3. Adicione logs/screenshots quando aplicável
4. Use labels apropriados: `bug`, `feature`, `docs`, `ops:rollback`

---

**Última atualização:** 2024-12-20  
**Versão do guia:** 2.0
| `chore` | Tarefas de manutenção | `chore(deps): atualizar requirements.txt` |
| `ci` | Mudanças em CI/CD | `ci: adicionar step de security scan` |

**Exemplos Completos:**
```bash
feat(api): implementar sistema champion/challenger

- Adiciona endpoints /retrain, /promote, /discard
- Integra com MLflow para rastreabilidade
- Adiciona validação de métricas antes de promoção

Closes #45
```

```bash
fix(dashboard): corrigir erro ao carregar modelo

Quando API_URL não estava configurado, dashboard falhava silenciosamente.
Agora exibe mensagem clara de erro.

Fixes #67
```

---

## 6. Gestão de Incidentes

### 🚨 Estratégia GitOps-Only

**Rollback é EXCLUSIVAMENTE via Git:**
- ✅ Reverter commit com `git revert`
- ✅ Ajustar `app/models/champion_run_id.txt` via PR
- ✅ Merge controlado com CI/CD
- ❌ NUNCA rollback manual direto no servidor

- Rollback de produção é **exclusivamente via GitOps**.
- Estratégia padrão:
  1. Ajustar `app/models/champion_run_id.txt` para uma versão estável.
  2. Abrir PR com justificativa do incidente.
  3. Fazer merge e deixar CI/CD aplicar a mudança.
- É proibido rollback manual direto no servidor para evitar drift operacional.

## 6) Checklist de PR

- Objetivo da mudança descrito de forma clara.
- Evidências de validação local (comandos e resultado resumido).
- Impacto em API, modelo e infraestrutura identificado.
- Mudanças em documentação incluídas quando aplicável.

## 7) Workflows oficiais

- `.github/workflows/feature-pipeline.yml`
- `.github/workflows/develop-pipeline.yml`
- `.github/workflows/main-pipeline.yml`

---

Dúvidas: abra uma issue no repositório.
