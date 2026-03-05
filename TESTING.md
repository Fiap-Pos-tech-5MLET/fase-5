# Testes e Qualidade

Runbook técnico para execução local e validação no CI.

## 1) Objetivo

- Garantir regressão funcional da API e pipeline ML.
- Garantir qualidade estática (lint, tipos, segurança).
- Manter cobertura mínima do projeto em **85%**.

## 2) Comandos essenciais

### Testes

```bash
# Todos os testes com output verbose
pytest tests/ -v

# Testes com cobertura completa
pytest tests/ --cov=src --cov=app --cov-report=term-missing -v

# Relatório HTML de cobertura (htmlcov/index.html)
pytest tests/ --cov=src --cov=app --cov-report=html

# Teste específico
pytest tests/test_predict_route.py -v

# Testes com palavra-chave
pytest -k "test_predict" -v
```

### Testes com markers

```bash
# Apenas testes unitários
pytest -m unit -v

# Apenas testes de integração
pytest -m integration -v

# Apenas testes de API
pytest -m api -v

# Apenas testes de dashboard
pytest -m dashboard -v

# Testes de pipeline de dados
pytest -m "data_loading or data_cleaning or feature_engineering" -v

# Testes rápidos (excluindo lentos)
pytest -m "not slow" -v
```

### Qualidade (local)

```bash
make format
make lint
make type-check
make security
make quality
```

## 3) Escopo coberto

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
