# Guia Completo de Testes e Qualidade

> Runbook técnico abrangente para execução local, validação no CI/CD e garantia de qualidade do código e ML pipeline.

---

## 📋 Índice

- [1. Objetivo e Princípios](#1-objetivo-e-princípios)
- [2. Comandos Essenciais](#2-comandos-essenciais)
- [3. Estrutura de Testes](#3-estrutura-de-testes)
- [4. Cobertura e Métricas](#4-cobertura-e-métricas)
- [5. Markers e Filtros](#5-markers-e-filtros)
- [6. Fixtures e Configuração](#6-fixtures-e-configuração)
- [7. Boas Práticas](#7-boas-práticas)
- [8. Integração com CI/CD](#8-integração-com-cicd)
- [9. Troubleshooting](#9-troubleshooting)

---

## 1. Objetivo e Princípios

### 🎯 Objetivos

- **Regressão Funcional:** Garantir que mudanças não quebrem comportamento esperado da API e pipeline ML
- **Qualidade Estática:** Manter código limpo, tipado e seguro (lint, type checking, security scans)
- **Cobertura Mínima:** Manter cobertura de código em **85%+** conforme `pytest.ini`
- **Reprodutibilidade:** Testes executam rapidamente sem dependências externas pesadas (MLflow, Streamlit mockados)
- **Rastreabilidade:** Cada teste documentado e categorizado por tipo e escopo

### 📐 Princípios

1. **AAA Pattern:** Arrange → Act → Assert (estrutura clara)
2. **Independência:** Testes não compartilham estado mutável
3. **Determinismo:** Execuções repetidas produzem mesmo resultado
4. **Velocidade:** Suite completa < 2 min (mocks pesados, fixtures eficientes)
5. **Clareza:** Nomes descritivos indicam cenário testado

### 🎯 Metas Quantitativas

| Métrica | Target | Justificativa |
|---------|--------|---------------|
| **Cobertura Total** | ≥ 85% | Confiança para refatoração e evolução |
| **Tempo de Execução** | < 2 min | Feedback rápido para desenvolvedores |
| **Taxa de Falso Positivo** | < 5% | Testes flaky prejudicam confiança |
| **Lint/Type/Security** | 0 erros críticos | Código limpo e seguro |

---

## 2. Comandos Essenciais

### ⚡ Testes Básicos

```bash
# Todos os testes com output verbose
pytest tests/ -v

# Testes com cobertura completa
pytest tests/ --cov=src --cov=app --cov-report=term-missing -v

# Relatório HTML de cobertura (htmlcov/index.html)
pytest tests/ --cov=src --cov=app --cov-report=html

# Teste específico
pytest tests/app/test_predict_route.py -v

# Testes de dashboard
pytest tests/app/test_dashboard*.py -v

# Testes de health check da API
pytest tests/app/test_dashboard_health.py -v

# Testes com palavra-chave
pytest -k "test_predict" -v
```

**Output esperado (exemplo):**
```bash
$ pytest tests/ --cov=src --cov=app --cov-report=term

========================= test session starts ==========================
platform linux -- Python 3.11.5, pytest-7.4.3, pluggy-1.3.0
collected 600 items

tests/app/test_main.py ............                               [  2%]
tests/app/test_predict_route.py ................                  [  4%]
tests/src/test_data_cleaning.py .............................     [ 10%]
...
========================= 600 passed in 98.45s ==========================

----------- coverage: platform linux, python 3.11.5 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src/data_cleaning.py                120      8    93%
src/feature_engineering.py          150     12    92%
app/routes/predict_route.py          80      5    94%
-----------------------------------------------------
TOTAL                               830     77    91%
```

### 🎯 Testes com Markers

```bash
# Apenas testes unitários (rápidos)
pytest -m unit -v

# Apenas testes de integração (médios)
pytest -m integration -v

# Apenas testes de API
pytest -m api -v

# Apenas testes de dashboard
pytest -m dashboard -v

# Testes de pipeline de dados
pytest -m "data_loading or data_cleaning or feature_engineering" -v

# Testes rápidos (excluindo lentos)
pytest -m "not slow" -v

# Testes que requerem GPU (skip se não tiver)
pytest -m gpu -v
```

**Exemplo de uso de markers:**
```python
# tests/app/test_predict_route.py
import pytest

@pytest.mark.api
@pytest.mark.integration
def test_predict_endpoint_retorna_200(client):
    \"\"\"Testa que endpoint /predict retorna 200 OK.\"\"\"
    response = client.post("/api/predict", json=sample_data)
    assert response.status_code == 200

@pytest.mark.unit
def test_processar_idades_validas():
    \"\"\"Testa função de limpeza de idades (unitário).\"\"\"
    df = pd.DataFrame({"IDADE": [10, 15, 12]})
    result = limpar_idades(df)
    assert result["IDADE"].min() >= 5
```

### 🛠️ Qualidade de Código (Local)

```bash
# Formatar código automaticamente
make format
# Equivale a: ruff format .

# Validar lint
make lint
# Equivale a: ruff check .

# Validar tipos
make type-check
# Equivale a: mypy src/ app/ --strict

# Scan de segurança
make security
# Equivale a: bandit -r src/ app/ -ll && detect-secrets scan

# Executar todas as verificações de qualidade
make quality
# Equivale a: format + lint + type-check + security (sequencial)

# Executar tudo (qualidade + testes)
make all
# Equiv a: quality + test
```

**Output esperado (make quality):**
```bash
$ make quality
Running ruff format...
✅ 120 files formatted

Running ruff lint...
✅ All checks passed!

Running mypy type check...
Success: no issues found in 85 source files

Running bandit security scan...
[main]  INFO  Run metrics:
        Total issues (by severity):
                Low: 0
                Medium: 0
                High: 0
✅ No security issues found

Running detect-secrets...
✅ No secrets detected

🎉 All quality checks passed!
```

### 📊 Comandos Avançados

**1. Ver duração dos testes mais lentos:**
```bash
pytest tests/ --durations=10

# Output:
# slowest 10 durations
# ==========================================
# 5.23s call     tests/app/test_dashboard_pages.py::test_prediction_page_renders
# 3.45s call     tests/src/test_model.py::test_train_model_completo
# 2.78s call     tests/app/test_train_route.py::test_retrain_endpoint
# ...
```

**2. Executar testes em paralelo (pytest-xdist):**
```bash
# Requer: pip install pytest-xdist
pytest tests/ -n auto  # Usa todos os CPU cores disponíveis
pytest tests/ -n 4     # Usa 4 workers

# Reduz tempo de ~2min → ~40s em máquinas com 8 cores
```

**3. Executar apenas testes que falharam na última execução:**
```bash
pytest --lf  # --last-failed
pytest --ff  # --failed-first (tenta failed primeiro, depois passa para restantes)
```

**4. Modo verboso com output capturado:**
```bash
pytest -vv -s  # -vv = extra verbose, -s = mostra prints
```

**5. Gerar relatório XML (para CI/CD):**
```bash
pytest tests/ --junitxml=test-report.xml
```

**6. Executar com debugger (pdb):**
```bash
pytest tests/app/test_predict_route.py::test_especifico --pdb
# Abre debugger quando teste falha
```

---

## 3. Estrutura de Testes

### 📂 Estrutura Hierárquica de Testes (NOVO)

O projeto utiliza uma estrutura organizada de testes para facilitar navegação e manutenção:

```
tests/
├── conftest.py                    # Configuração global (fixtures, mocks)
├── utils_streamlit.py             # Utilitários para tests do dashboard
├── app/                           # Testes de app/ (FastAPI + Dashboard)
│   ├── 18 arquivos de teste
│   └── Cobertura: endpoints, dashboard, config, logging, XAI
├── src/                           # Testes de src/ (Pipeline ML)
│   ├── 4 arquivos de teste
│   └── Cobertura: data cleaning, feature engineering, model
├── scripts/                       # Testes de scripts/ (Utilitários) ⭐ NOVO
│   ├── test_data_processing.py       (28 testes)
│   ├── test_eda_analysis.py          (22 testes)
│   ├── test_notebook_feature_engineering.py (22 testes)
│   ├── test_visualization.py         (26 testes)
│   └── Cobertura total: 98+ testes
└── integration/                   # Testes de integração
    └── test_deployment_config.py  (validação de render.yaml)
```

**Execução por diretório:**

```bash
pytest tests/app/ -v              # Testes de API e Dashboard
pytest tests/src/ -v              # Testes de Pipeline ML
pytest tests/scripts/ -v          # Testes de Scripts Utilitários (NOVO)
pytest tests/integration/ -v      # Testes de Integração
pytest tests/ -v                  # Todos os testes
```

---

Os testes do diretório `tests/` cobrem os seguintes diretórios e módulos:

### Diretórios cobertos pela cobertura (conforme pytest.ini)

- **`src/`** — Pipeline de dados e ML
  - `data_cleaning.py` — limpeza e preparação de dados
  - `feature_engineering.py` — engenharia de features
  - `feature_store.py` — armazenamento de features
  - `model.py` — definição e treinamento de modelos

- **`app/`** — Aplicação completa (API + Dashboard)
  - `main.py` — aplicação FastAPI principal
  - `dashboard.py` — entry point do dashboard
  - `config.py` — configurações da aplicação
  - `routes/` — rotas da API (predict, train, audit)
  - `utils/` — utilitários (model_loader, security, xai, logging, keep_alive)
  - `dashboard/` — módulos do dashboard (config, data, sidebar, styles, pages/)

### Categorias de testes

#### API e Rotas
- `test_main.py` — aplicação FastAPI e health checks
- `test_predict_route.py` — endpoint de predição e XAI
- `test_train_route.py` — endpoints /retrain, /promote, /discard
- `test_audit_route.py` — endpoints de auditoria e métricas
- `test_schemas.py` — validação de schemas Pydantic

#### Pipeline de Dados e ML
- `test_data_cleaning.py` — carregamento, limpeza e valores faltantes
- `test_feature_engineering.py` — criação e transformação de features
- `test_feature_store.py` — armazenamento e recuperação de features
- `test_model.py` — treinamento, avaliação e artefatos

#### Dashboard Streamlit
- `test_dashboard.py` — testes gerais
- `test_dashboard_config.py` — configuração (API_URL, etc.)
- `test_dashboard_data.py` — funções de carregamento de dados
- `test_dashboard_health.py` — health check da API e status do modelo ⭐ **NOVO**
- `test_dashboard_pages.py` — páginas (prediction, metrics, drift, retrain, about)
- `test_dashboard_sidebar.py` — barra lateral
- `test_dashboard_styles.py` — estilos CSS customizados
- `test_dashboard_entry.py` — entry point e roteamento
- `test_dashboard_rendering.py` — renderização de componentes

#### Utilitários e Infraestrutura
- `test_model_loader.py` — carregamento e validação de modelos
- `test_xai_utils.py` — explicabilidade (SHAP/LIME)
- `test_structured_logging.py` — logging estruturado em JSON
- `test_keep_alive.py` — keep-alive para Render
- `test_app_config.py` — configurações da aplicação
- `test_deployment_config.py` — validação de render.yaml

#### Scripts Utilitários (tests/scripts/) ⭐ NOVO
- `test_data_processing.py` — ETL e consolidação de dataframes (28 testes)
  - Padronização de colunas por ano
  - Análise de valores nulos
  - Cálculo de idade a partir de data de nascimento
  - Consolidação de múltiplos dataframes
  
- `test_notebook_feature_engineering.py` — transformações de domínio (22 testes)
  - Criação de features: NOVA_TURMA, NOVA_FASE, VETERANO
  - Validação de regras de negócio
  - Edge cases: valores '9' como válidos em 2024
  
- `test_eda_analysis.py` — análise estatística (22 testes)
  - Testes de normalidade (Shapiro-Wilk, D'Agostino K²)
  - Transformações logarítmicas para redução de skewness
  - Validação cruzada estratificada com métricas completas
  
- `test_visualization.py` — funções de plotagem (26 testes)
  - Gráficos de contagem com porcentagens
  - Heatmaps de correlação
  - Distribuição de target
  - Feature importance

### Markers disponíveis

Os testes podem ser filtrados usando markers pytest:

- `unit` — testes unitários
- `integration` — testes de integração
- `api` — testes de endpoints
- `schemas` — testes de schemas
- `dashboard` — testes do dashboard
- `data_loading` — testes de carregamento
- `data_cleaning` — testes de limpeza
- `feature_engineering` — testes de features
- `model_training` — testes de treinamento
- `slow` — testes demorados
- `gpu` — testes que requerem GPU

## 4) Critérios para aprovação de PR

- Testes passando localmente ou no CI;
- Cobertura não inferior ao baseline do projeto;
- Sem erro crítico em `ruff`, `mypy`, `bandit` e `detect-secrets`;
- Build Docker válido quando aplicável.

## 5) CI/CD relacionado

Workflows ativos:

- `.github/workflows/feature-pipeline.yml`
- `.github/workflows/develop-pipeline.yml`
- `.github/workflows/main-pipeline.yml`

Cada workflow publica summary por etapa com logs de falha para troubleshooting rápido.

## 6) Boas práticas de escrita de testes

- Estrutura AAA (Arrange, Act, Assert);
- Testes determinísticos e independentes;
- Uso de fixtures para setup compartilhado;
- Nome de teste descritivo por cenário.

## 7) Fixtures e configuração

O arquivo `tests/conftest.py` contém configurações críticas para todos os testes:

### Mocks globais

Para garantir execução rápida e sem dependências externas pesadas, os seguintes módulos são mockados:

- **MLflow** (`mlflow`, `mlflow.sklearn`) — tracking de experimentos
- **Streamlit** (`streamlit`, `streamlit.components.v1`) — framework do dashboard
- **Plotly** (`plotly`, `plotly.express`, `plotly.graph_objects`) — visualizações
- **Scripts** (`scripts.train`, `scripts.monitoring`) — scripts de automação

### Configuração de paths

O conftest adiciona automaticamente os diretórios ao `sys.path`:
- Raiz do projeto
- `src/` — módulos de pipeline
- `app/` — módulos da aplicação

### Decorators de cache

Os decorators `@st.cache_resource` e `@st.cache_data` do Streamlit são substituídos por funções passthrough para evitar problemas em testes.

### Fixtures compartilhadas

Fixtures definidas no conftest ficam disponíveis para todos os testes automaticamente, sem necessidade de import explícito.

### Markers pytest

O conftest registra todos os markers disponíveis para uso nos testes, permitindo execução seletiva e categorização.

## 8) Estrutura de um teste típico

Exemplo de estrutura recomendada:

```python
import pytest
from app.routes.predict_route import predict_endpoint

@pytest.mark.api
@pytest.mark.unit
def test_predict_endpoint_success(monkeypatch):
    """Testa predição bem-sucedida com dados válidos."""
    # Arrange (preparação)
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8]
    monkeypatch.setattr("app.utils.model_loader.load_model", lambda: mock_model)
    
    input_data = {"IDADE": 16, "FASE": "Fase 1"}
    
    # Act (ação)
    result = predict_endpoint(input_data)
    
    # Assert (verificação)
    assert result["prediction"] == 0.8
    assert "explanation" in result
```

## 9) Relatórios de cobertura

Após executar testes com `--cov-report=html`, os relatórios ficam em:

- **`htmlcov/index.html`** — relatório principal navegável
- **`htmlcov/`** — arquivos HTML por módulo com linhas cobertas/não cobertas destacadas
- **`.coverage`** — arquivo de dados binário da cobertura (usado pelo pytest-cov)

Para visualizar:

```bash
# Windows
start htmlcov/index.html

# Linux/macOS
open htmlcov/index.html
```

---

Referência estratégica: [TESTING_STRATEGY.md](TESTING_STRATEGY.md).
