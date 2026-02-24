# Projeto Tech Challenge Fase 5

---

## 📌 Índice

- [📝 Sobre o Projeto](#-sobre-o-projeto)
- [🎯 Desafio](#-desafio)
- [🛠 Tecnologias e Ferramentas](#-tecnologias-e-ferramentas)
- [🧱 Arquitetura da Solução](#-arquitetura-da-solução)
- [🗂️ Estrutura de Diretórios](#-estrutura-de-diretórios)
- [🚀 Como Configurar e Executar o Projeto](#-como-configurar-e-executar-o-projeto)
- [✅ Testes e Validações](#-testes-e-validações)
- [🔄 CI/CD Pipeline](#-cicd-pipeline)
- [🤖 IA para Code Review](#-ia-para-code-review)
- [📖 Documentação da API](#-documentação-da-api)
- [📊 Monitoramento e MLflow](#-monitoramento-e-mlflow)
- [🎥 Vídeo Demonstrativo](#-vídeo-demonstrativo)
- [🤝 Desenvolvedores](#-desenvolvedores)
- [⚖️ Licença](#-licença)

---

## 📝 Sobre o Projeto

Este repositório contém a implementação do **Tech Challenge Fase 5 da Pós-Graduação em Machine Learning**, focado em **análise e predição do desenvolvimento educacional** de crianças e jovens atendidos pela **Associação Passos Mágicos**.

### 🌟 Associação Passos Mágicos

**Mudando a vida de crianças e jovens por meio da educação**

A Associação Passos Mágicos tem uma trajetória de **32 anos de atuação** e trabalha na transformação da vida de crianças e jovens de baixa renda, os levando a melhores oportunidades de vida. A transformação, idealizada por **Michelle Flues** e **Dimetri Ivanoff**, começou em **1992**, atuando dentro de orfanatos no município de **Embu-Guaçu**.

Em **2016**, depois de anos de atuação, eles decidem ampliar o programa para que mais jovens tivessem acesso a essa fórmula mágica para transformação que inclui:
- ✨ **Educação de qualidade**
- 🧠 **Auxílio psicológico/psicopedagógico**
- 🌍 **Ampliação de sua visão de mundo**
- 💪 **Protagonismo**

Passaram então a atuar como um projeto social e educacional, criando assim a **Associação Passos Mágicos**.

A associação busca instrumentalizar o uso da educação como ferramenta para a mudança das condições de vida das crianças e jovens em vulnerabilidade social.

### 🎯 Desafio

Com base no **dataset de pesquisa extensiva** do desenvolvimento educacional no período de **2022, 2023 e 2024**, este projeto apresenta um desafio de **engenharia de Machine Learning** para trazer um **impacto real** na vida dessas crianças.

### ✨ Funcionalidades Principais

- **Análise de Dados Educacionais**: Exploração e visualização dos dados de desenvolvimento dos alunos ao longo de 3 anos
- **Predição de Desempenho**: Utiliza Machine Learning para prever o desenvolvimento educacional futuro
- **Identificação de Padrões**: Detecta fatores que influenciam positiva ou negativamente o progresso dos alunos
- **API REST Completa**: Endpoints para predições, análises e visualizações
- **Dashboard Interativo**: Interface visual para exploração dos dados e insights
- **Pipeline de Treinamento**: Sistema automatizado de treinamento e validação de modelos
- **Monitoramento com MLflow**: Rastreamento completo de experimentos, parâmetros e métricas
- **Containerização**: Deploy simplificado via Docker e Docker Compose (dev) ou container único (produção)
- **Deploy no Render**: Container único gerenciado pelo Supervisor com todos os serviços (ver [DEPLOYMENT.md](DEPLOYMENT.md))
- **CI/CD Automatizado**: Pipeline completo de integração e entrega contínua com GitHub Actions
- **Cobertura de Testes**: ≥85% de cobertura de código com testes automatizados
- **IA para Code Review**: Revisão automática de código usando GitHub Copilot

---

## 🛠 Tecnologias e Ferramentas

### 🎯 Stack Principal

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🐍 **Python 3.11+** | Linguagem de Programação | Linguagem base para **ML, API e pipeline de dados** |
| ⚡ **FastAPI** | Framework Web | **API REST** de alta performance com documentação automática |
| 🎨 **Streamlit** | Framework de Dashboard | **Interface interativa** para visualização e análise de dados |
| 🌐 **Nginx** | Reverse Proxy | **Ponto de entrada único** (porta 80) para roteamento de serviços |

### 🤖 Machine Learning & Ciência de Dados

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 📦 **scikit-learn** | Biblioteca de ML | **RandomForestClassifier** para predições educacionais |
| 📊 **NumPy & Pandas** | Processamento de Dados | Manipulação e análise de **dados educacionais estruturados** |
| 📈 **Matplotlib & Seaborn** | Visualização | Gráficos e _dashboards_ de insights educacionais |
| 🔍 **MLflow** | Plataforma MLOps | Rastreamento de **experimentos, parâmetros e métricas** |

### 🧪 Qualidade & Testes

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🧪 **Pytest** | Framework de Testes | Testes automatizados com **≥85% de cobertura** |
| 🎯 **pytest-cov** | Coverage | Medição de cobertura de código (_threshold: 85%_) |
| ✨ **Ruff** | Linter & Formatter | **Formatação automática**, linting e organização de imports |
| 🔤 **MyPy** | Type Checker | Verificação de **type hints** para maior segurança de tipos |

### 🔒 Segurança

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🛡️ **Bandit** | Security Scanner | Detecção de **vulnerabilidades** no código Python |
| 🔑 **detect-secrets** | Secret Scanner | Prevenção de **commit de credenciais** e API keys |

### 🐳 Infraestrutura & DevOps

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🐳 **Docker** | Containerização | Ambiente **isolado e reprodutível** |
| 🐙 **Docker Compose** | Orquestração | **4 serviços**: nginx, api, dashboard, mlflow |
| 🔄 **GitHub Actions** | CI/CD | Pipeline **totalmente paralelizado** (5-8 min) |
| 🤖 **GitHub Copilot** | IA Code Review | Revisão automática seguindo **padrões de qualidade** |

---

## 🧱 Arquitetura da Solução

O sistema é construído sobre uma **arquitetura modular e escalável**:
- **Desenvolvimento**: Docker Compose com serviços separados
- **Produção (Render)**: Container único com Supervisor gerenciando Nginx, API, Streamlit e MLflow

### 🏗️ Arquitetura de Produção (Docker Compose)

```
                        ┌──────────────────────────────┐
                        │    🌐 NGINX (Porta 80)       │
                        │    Reverse Proxy Único       │
                        └──────────────┬───────────────┘
                                       │
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                                                              ┃
        ▼                        ▼                      ▼              ▼
┌───────────────┐      ┌────────────────┐    ┌──────────────┐  ┌─────────────┐
│  📊 Dashboard │      │   ⚡ FastAPI    │    │  🔍 MLflow   │  │ 📄 Landing  │
│  (Streamlit)  │      │     API        │    │  Tracking    │  │    Page     │
│   :8501       │      │    :8000       │    │   :5000      │  │ index.html  │
└───────┬───────┘      └────────┬───────┘    └──────────────┘  └─────────────┘
        │                       │                                     
        └───────────────────────┘                                     
                 │                                                    
                 ▼                                                    
        ┌─────────────────┐                                          
        │  🤖 ML Models   │                                          
        │ (RandomForest)  │                                          
        └─────────────────┘                                          
```

**Roteamento via Nginx:**
- `http://localhost/` → Landing page (index.html)
- `http://localhost/api/*` → FastAPI (porta interna 8000)
- `http://localhost/dashboard/*` → Streamlit (porta interna 8501)
- `http://localhost/mlflow/*` → MLflow (porta interna 5000)

### 🔄 Pipeline de Dados e ML

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Dataset       │────▶│  Data Pipeline   │────▶│  Data Cleaning  │
│   (CSV)         │     │  (data_loader)   │     │  (limpeza)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit     │◀────│  ML Models       │◀────│  Feature Eng.   │
│   Dashboard     │     │  (Scikit-Learn)  │     │  (features)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │
         └────────▶ FastAPI ◀────┘
                    API REST
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐     ┌──────────────────┐
│   Docker        │     │    MLflow        │
│   (Deploy)      │     │  (Monitoring)    │
└─────────────────┘     └──────────────────┘
```

### 🎯 Componentes Principais

1. **🌐 Nginx (Reverse Proxy)**: Ponto de entrada único na porta 80, roteia todo tráfego HTTP
2. **📊 Camada de Dados**: Carregamento, validação e limpeza dos dados educacionais (2022-2024)
3. **🛠️ Camada de Processamento**: Feature engineering e transformações para ML
4. **🤖 Camada de Modelo**: **RandomForestClassifier** (Scikit-Learn) para predição de desempenho
5. **⚡ Camada de Serviço**: **FastAPI** com endpoints REST para predições e análises
6. **🎨 Camada de Interface**: **Streamlit** dashboard interativo para visualização
7. **🔍 Camada de Monitoramento**: **MLflow** para rastreamento de experimentos, métricas e modelos
8. **🐳 Infraestrutura**: **Docker Compose** orquestrando 4 serviços containerizados

---

## 🗂️ Estrutura de Diretórios

O projeto está organizado da seguinte forma para facilitar a navegação e o entendimento:

```
fase-5/
│
├── .github/
│   ├── copilot-instructions.md      # Instruções para IA Code Review (padrões de qualidade)
│   └── workflows/
│       └── ci-cd-pipeline.yml       # Pipeline CI/CD automatizado em português
│
├── app/                             # Aplicação principal (API + Dashboard)
│   ├── __init__.py
│   ├── config.py                    # Configurações da aplicação
│   ├── main.py                      # Ponto de entrada da API FastAPI
│   ├── dashboard.py                 # Ponto de entrada do Dashboard Streamlit
│   │
│   ├── artifacts/                   # Artefatos de treinamento (não versionado)
│   │   ├── model.pkl                # Modelo treinado
│   │   └── scaler.pkl               # Scaler para normalização
│   │
│   ├── dashboard/                   # Módulos do Dashboard Streamlit
│   │   ├── __init__.py
│   │   ├── config.py                # Configurações do dashboard
│   │   ├── data.py                  # Funções de carregamento de dados
│   │   ├── sidebar.py               # Componente da sidebar
│   │   ├── styles.py                # Estilos CSS customizados
│   │   └── pages/                   # Páginas do dashboard
│   │       ├── __init__.py
│   │       ├── about.py             # Página Sobre
│   │       ├── drift.py             # Monitoramento de Drift
│   │       ├── metrics.py           # Métricas do Modelo
│   │       ├── prediction.py        # Predições Interativas
│   │       └── retrain.py           # Retreinamento do Modelo
│   │
│   ├── data/                        # Dados da aplicação
│   │   └── __init__.py
│   │
│   ├── models/                      # Modelos e Schemas
│   │   ├── __init__.py
│   │   ├── model.pkl                # Modelo champion em produção
│   │   ├── champion_run_id.txt      # ID do run do MLflow do modelo champion
│   │   └── schemas/                 # Schemas Pydantic (Request/Response)
│   │       ├── __init__.py
│   │       ├── discard_response.py
│   │       ├── model_info_response.py
│   │       ├── model_metrics_response.py
│   │       ├── prediction_response.py
│   │       ├── promote_response.py
│   │       ├── retrain_request.py
│   │       ├── student_data.py
│   │       └── student_input.py
│   │
│   ├── routes/                      # Rotas da API FastAPI
│   │   ├── __init__.py
│   │   ├── audit_route.py           # Rotas de auditoria e logs
│   │   ├── predict_route.py         # Rota de predição
│   │   └── train_route.py           # Rotas de treinamento e MLOps
│   │
│   └── utils/                       # Utilitários da aplicação
│       ├── __init__.py
│       └── model_loader.py          # Carregamento de modelos
│
├── src/                             # Código-fonte do pipeline ML
│   ├── __init__.py
│   ├── data_cleaning.py             # Limpeza e tratamento de dados
│   ├── feature_engineering.py       # Engenharia de features
│   └── model.py                     # Modelos ML (RandomForestClassifier)
│
├── scripts/                         # Scripts auxiliares
│   ├── __init__.py
│   ├── monitoring.py                # Scripts de monitoramento
│   └── train.py                     # Script de treinamento
│
├── tests/                           # Testes automatizados
│   ├── conftest.py                  # Configurações e fixtures do pytest
│   ├── test_*.py                    # Testes unitários e de integração
│   └── ...
│
├── notebooks/                       # Notebooks Jupyter
│   └── EDA_and_Training.ipynb       # Análise exploratória e treinamento
│
├── data/                            # Datasets (não versionado)
│   ├── raw/                         # Dados brutos
│   ├── interim/                     # Dados intermediários
│   ├── processed/                   # Dados processados
│   └── external/                    # Dados externos
│
├── docs/                            # Documentação adicional
│   ├── datathon/                    # Documentos do Datathon
│   └── relatorios/                  # Relatórios técnicos
│
├── dev/                             # Arquivos de desenvolvimento
│
├── .streamlit/
│   └── config.toml                  # Configuração do Streamlit
│
├── .github/
│   └── workflows/
│       └── ci-cd-pipeline.yml       # Pipeline CI/CD
│
├── docker-compose.yml               # Orquestração de 4 serviços (nginx, api, dashboard, mlflow)
├── Dockerfile                       # Definição da imagem Docker para API
├── nginx.conf                       # Configuração do Nginx (reverse proxy)
├── index.html                       # Landing page (porta 80)
│
├── pyproject.toml                   # Configuração do projeto Python (Ruff, etc)
├── pytest.ini                       # Configuração do pytest
├── Makefile                         # Comandos automatizados
│
├── requirements.txt                 # Dependências de produção
├── requirements-dev.txt             # Dependências de desenvolvimento
│
├── run_tests.py                     # Script para executar testes
├── get_coverage.py                  # Script para gerar relatório de cobertura
│
├── .pre-commit-config.yaml          # Configuração de pre-commit hooks
├── .dockerignore                    # Arquivos ignorados no build Docker
├── .gitignore                       # Arquivos ignorados no Git
├── .env.example                     # Template de variáveis de ambiente
│
├── ARCHITECTURE.md                  # Documentação da arquitetura
├── CONTRIBUTING.md                  # Guia de contribuição
├── DEPLOYMENT.md                    # Guia de deployment
├── IMPLEMENTATION_SUMMARY.md        # Resumo da implementação
├── TESTING.md                       # Documentação de testes
├── TESTING_DEPLOYMENT.md            # Testes de deployment
├── TESTING_STRATEGY.md              # Estratégia de testes
├── LICENSE                          # Licença MIT
└── README.md                        # Este arquivo
```
├── TESTING_STRATEGY.md              # Estratégia de testes
├── LICENSE                          # Licença MIT
└── README.md                        # Este arquivo
```

---

## 🚀 Como Configurar e Executar o Projeto

### Pré-requisitos
- **Python**: 3.11 ou superior
- **Docker & Docker Compose** (opcional para execução em contêiner)
- **Git**
- **Make** (opcional, para comandos automatizados)

---

### Opção A: Execução Local (Desenvolvimento)

#### 1. Clone e Instale Dependências

```bash
# Clone o repositório
git clone https://github.com/Fiap-Pos-tech-5MLET/fase-5.git
cd fase-5

# Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

#### 2. Configure Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:
```bash
# .env
PROJECT_NAME="Tech Challenge Fase 5 - Associação Passos Mágicos"
SECRET_KEY=sua_chave_secreta_aqui
ACCESS_TOKEN_EXPIRE_MINUTES=60
ALGORITHM=HS256
DATASET_PATH=data/raw/passos_magicos_2022_2024.csv
MODEL_PATH=app/models/model.pkl
```

> **Nota**: O arquivo `.env.example` contém um template com todas as variáveis disponíveis.

#### 3. Prepare os Dados

Coloque o dataset de desenvolvimento educacional na pasta `data/raw/`:
```bash
# Estrutura esperada
data/
└── raw/
    └── passos_magicos_2022_2024.csv
```

#### 4. Treine o Modelo Inicial

```bash
# Executar treinamento inicial
python scripts/train.py

# Ou usando Make (Nota: Makefile precisa ser atualizado)
make train
```

Isso criará os artefatos:
- `app/models/model.pkl` - Modelo treinado (champion em produção)
- `app/artifacts/` - Artefatos de experimentos (scaler, preprocessors, etc.)

#### 5. Execute a API

```bash
# Rodar FastAPI
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Ou usando Make
make run-api
```

**API disponível em:** http://localhost:8000
**Documentação:** http://localhost:8000/api/docs

#### 6. Execute o Dashboard Streamlit

Em outro terminal:

```bash
# Rodar Streamlit
streamlit run app/dashboard.py --server.port=8501 --server.address=127.0.0.1

# Ou usando Make (Nota: Makefile precisa ser atualizado para usar app/dashboard.py)
make run-streamlit
```

**Dashboard disponível em:** http://localhost:8501

---

### 🐳 Opção B: Execução com Docker Compose (Recomendado)

**Arquitetura de porta única** - Todos os serviços acessíveis via **porta 80**

```bash
# Construir e executar todos os serviços
docker-compose up --build

# Ou em background (modo daemon)
docker-compose up -d --build

# Ver logs em tempo real
docker-compose logs -f

# Parar todos os serviços
docker-compose down

# Parar e remover volumes
docker-compose down -v
```

#### 🌐 Serviços Disponíveis (Porta Única)

Todos os serviços são acessados via **http://localhost** (porta 80):

| Serviço | URL | Descrição |
|---------|-----|-----------|
| 🏠 **Landing Page** | http://localhost/ | Página inicial com links para todos os serviços |
| ⚡ **API REST** | http://localhost/api/docs | Documentação interativa FastAPI (Swagger) |
| 📊 **Dashboard** | http://localhost/dashboard/ | Interface Streamlit para visualização |
| 🔍 **MLflow** | http://localhost/mlflow/ | Tracking de experimentos e modelos |

#### 🔧 Comandos Docker Úteis

```bash
# Ver status dos containers
docker-compose ps

# Reiniciar um serviço específico
docker-compose restart api
docker-compose restart dashboard

# Ver logs de um serviço específico
docker-compose logs -f api
docker-compose logs -f dashboard

# Rebuild apenas um serviço
docker-compose up -d --build api

# Executar comando dentro do container
docker-compose exec api python --version
docker-compose exec api pytest tests/ -v
```

---

## ✅ Testes e Validações

O projeto possui uma **cobertura de testes ≥85%** com testes automatizados para todos os componentes principais.

### Executar Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Ou usando Make
make test

# Rodar com cobertura de código
pytest tests/ --cov=src --cov=app --cov-report=term-missing -v

# Ou usando Make
make coverage

# Gerar relatório HTML de cobertura
make coverage-html
# Abrir: htmlcov/index.html

# Rodar teste específico
pytest tests/test_model.py -v
pytest tests/test_data_cleaning.py -v
```

### Verificação de Qualidade

```bash
# Rodar todos os checks de qualidade
make quality

# Checks individuais
make format        # Ruff format (formatação automática)
make lint          # Ruff check (linting + imports)
make type-check    # MyPy (verificação de type hints)
make security      # Bandit + detect-secrets
```

Para mais detalhes sobre testes, consulte o arquivo [TESTING.md](TESTING.md).

---

## 🔄 CI/CD Pipeline

O projeto implementa um **pipeline completo de CI/CD** usando **GitHub Actions** em **português**, garantindo qualidade e confiabilidade do código.

### Pipeline Automatizado (Paralelizado)

**Estágio 1 - Qualidade de Código:**
1. 🔍 **Qualidade de Código**: Ruff (formatação, linting, imports) + MyPy (type hints)

**Estágio 2 - Build & Validação (Paralelo):**
2. 📦 **Build da API**: Instalação de dependências e validação de imports
3. 🧪 **Testes e Cobertura**: Suite de testes com ≥85% de cobertura
4. 🔒 **Análise de Segurança**: Bandit (vulnerabilidades) + detect-secrets (credenciais)
5. 🐳 **Build Docker**: Build e smoke test do container

**Estágio 3 - Treinamento (apenas em main):**
6. 🤖 **Treinar e Validar Modelo**: Validação do pipeline ML (Scikit-Learn) e dados

**Estágio 4 - Relatório:**
7. 📊 **Relatório Detalhado**: Summary com status, tempo, métricas e ações corretivas

### Características do Pipeline

- ⚡ **Paralelização Completa**: Estágios 2 executam em paralelo
- ⏱️ **Duração**: 5-8 minutos (otimizado)
- 🇧🇷 **Interface em Português**: Logs e relatórios em PT-BR
- 📊 **Relatórios Detalhados**: Status, métricas, erros e soluções
- 🔄 **Cache Inteligente**: pip dependencies e Docker layers

### Triggers do Pipeline

- **Push** para branch `main`
- **Pull Requests** para `main`

---

## 🤖 IA para Code Review

O projeto utiliza **GitHub Copilot** com instruções customizadas para realizar revisão automática de código, garantindo qualidade, segurança e boas práticas.

### Padrões de Qualidade Verificados

- ✅ **Type Hints**: Todos os parâmetros e retornos têm type hints
- ✅ **Docstrings**: Google Style em português para todas as funções
- ✅ **Convenções de Nomenclatura**: snake_case, PascalCase, UPPER_SNAKE_CASE
- ✅ **Comprimento de Linhas**: Máximo 100 caracteres
- ✅ **Tratamento de Erros**: Try/except com exceções específicas
- ✅ **Segurança**: Validação de entrada, sem secrets hardcoded
- ✅ **Performance**: Operações vetorizadas, gerenciamento de memória
- ✅ **Testes**: Cobertura mínima de 85%

As instruções de code review estão em [.github/copilot-instructions.md](.github/copilot-instructions.md).

---

## 📖 Documentação da API

A API REST expõe endpoints para análise, predição e monitoramento.

### Documentação Interativa

Acesse a documentação interativa do Swagger UI:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Principais

#### 1. Verificação de Saúde
```http
GET /health
```

#### 2. Análise de Dados
```http
GET /api/analysis
```

#### 3. Predição de Desempenho
```http
POST /api/predict
```

#### 4. Treinamento de Modelo
```http
POST /api/train
```

---

## 📊 Monitoramento e MLflow

O projeto utiliza **MLflow** para rastreamento de experimentos, parâmetros, métricas e artefatos.

### Acessar MLflow

**Com Docker Compose (Recomendado):**
- MLflow UI disponível em: http://localhost/mlflow/
- Serviço já configurado e iniciado automaticamente

**Execução Local (Desenvolvimento):**
```bash
# Iniciar servidor MLflow localmente
mlflow ui --port 5000
```
**MLflow UI disponível em:** http://localhost:5000

### Métricas Rastreadas

- Acurácia do modelo
- Precisão, Recall e F1-Score
- Matriz de confusão
- Curvas ROC e AUC
- Hiperparâmetros utilizados
- Tempo de treinamento

---

## 🎥 Vídeo Demonstrativo
Assista ao vídeo explicativo do projeto e seu funcionamento:
- 📹 **Link do vídeo**: [Em breve]
- 💎 **Link API Pública**: [API](https://datathon-machine-learning-engineering.onrender.com/)
- 📊 **Conteúdo**: Arquitetura, demonstração da API, pipeline de treinamento e resultados

### 📸 Screenshots da Aplicação

**Principais Interfaces:**
- 🏠 **Landing Page**: Página inicial unificando acesso via porta 80
- 📊 **Dashboard Streamlit**: Interface interativa para visualização, predições e retreinamento
- 📖 **API Documentation**: Swagger UI com documentação completa dos endpoints
- 🔍 **MLflow UI**: Tracking de experimentos, métricas e modelos

> Para screenshots atualizados, acesse a [documentação do projeto](docs/datathon/) ou visite a API em produção.

---

## 🤝 Desenvolvedores

Este projeto foi desenvolvido como parte do **Tech Challenge Fase 5** da **Pós-Graduação em Machine Learning Engineering da FIAP**.

**Equipe 5MLET**:

| Nome | RM | GitHub |
|------|-----|--------|
| Lucas Felipe de Jesus Machado | RM364306 | [@lfjmachado](https://github.com/lfjmachado) |
| Antônio Teixeira Santana Neto | RM364480 | [@antonioteixeirasn](https://github.com/antonioteixeirasn) |
| Gabriela Moreno Rocha dos Santos | RM364538 | [@gabrielaMSantos](https://github.com/gabrielaMSantos) |
| Erik Douglas Alves Gomes | RM364379 | [@Erik-DAG](https://github.com/Erik-DAG) |
| Leonardo Fernandes Soares | RM364648 | [@leferso](https://github.com/leferso) |

---

## ⚖️ Licença

Este projeto está licenciado sob a **Licença MIT** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 📚 Documentação Adicional


### Guias de Testes e Qualidade
- **[TESTING.md](TESTING.md)** - Guia completo de testes e cobertura
- **[TESTING_DEPLOYMENT.md](TESTING_DEPLOYMENT.md)** - Testes de configuração de deployment
- **[TESTING_STRATEGY.md](TESTING_STRATEGY.md)** - Estratégia de testes do projeto
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Resumo da implementação

### Guias de Desenvolvimento
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Instruções para IA Code Review
  - Padrões de qualidade de código
  - Convenções de nomenclatura
  - Checklist de revisão

## 🌟 Agradecimentos
- **FIAP** - Pela excelente estrutura do curso de Pós-Graduação em Machine Learning
- **Professores** - Pelo conhecimento compartilhado e orientação
- **Associação Passos Mágicos** por disponibilizar os dados e pela inspiração deste projeto que visa contribuir para a transformação da vida de crianças e jovens através da educação.

---