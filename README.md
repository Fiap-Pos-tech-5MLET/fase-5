# Projeto Tech Challenge Fase 5

> Esta solução consolida uma arquitetura MLOps completa: inferência assíncrona escalável via FastAPI, orquestração Cloud Native (pronta para K8s), esteira CI/CD rigorosa (GitFlow) e explicabilidade de IA (XAI) para garantir que cada predição do risco de defasagem do aluno é fundamentada e auditável para a Associação Passos Mágicos.

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
- [📚 Documentação Adicional](#-documentação-adicional)
- [🌟 Agradecimentos](#-agradecimentos)

---

## 📝 Sobre o Projeto

Este repositório contém a implementação do **Tech Challenge Fase 5 da Pós-Graduação em Machine Learning**, com foco em **análise e predição do desenvolvimento educacional** de crianças e jovens atendidos pela **Associação Passos Mágicos**.

### 🌟 Associação Passos Mágicos

**Mudando a vida de crianças e jovens por meio da educação.**

A Associação Passos Mágicos possui mais de três décadas de atuação e trabalha na transformação da vida de crianças e jovens em vulnerabilidade social, ampliando suas oportunidades por meio da educação. A iniciativa foi idealizada por **Michelle Flues** e **Dimetri Ivanoff**, com início em **1992**, em **Embu-Guaçu**.

Em **2016**, o programa foi ampliado para alcançar mais jovens, sustentado por quatro pilares:
- ✨ **Educação de qualidade**
- 🧠 **Auxílio psicológico/psicopedagógico**
- 🌍 **Ampliação da visão de mundo**
- 💪 **Protagonismo**

### ✨ Funcionalidades Principais

- **Análise de dados educacionais** com foco em evolução histórica dos alunos.
- **Predição de desempenho** com modelos de Machine Learning.
- **Identificação de padrões** que influenciam o progresso escolar.
- **API REST** para predição, auditoria, monitoramento e ciclo de retreinamento.
- **Dashboard interativo** em Streamlit para visualização e exploração dos dados.
- **Pipeline de treinamento** com estratégia champion/challenger.
- **Monitoramento com MLflow** para métricas, parâmetros e artefatos.
- **Containerização com Docker** para execução padronizada.
- **Deploy no Render** com container único, Nginx e Supervisor.
- **CI/CD com GitHub Actions** em fluxo GitFlow (feature → develop → main).
- **Cobertura de testes** com meta mínima de **85%**.
- **IA para code review** com instruções customizadas de qualidade.

---

## 🎯 Desafio

Com base no dataset de desenvolvimento educacional dos anos **2022, 2023 e 2024**, o projeto propõe um desafio de **Machine Learning Engineering** com impacto social real: antecipar risco de defasagem e apoiar decisões pedagógicas mais assertivas.

---

## 🛠 Tecnologias e Ferramentas

### 🎯 Stack Principal

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🐍 **Python 3.11+** | Linguagem | Base para ML, API e automações |
| ⚡ **FastAPI** | Framework Web | API REST com documentação automática |
| 🎨 **Streamlit** | Dashboard | Interface de análise e operação de modelo |
| 🌐 **Nginx** | Reverse Proxy | Roteamento unificado (`/`, `/api`, `/dashboard`) |

### 🤖 Machine Learning & Ciência de Dados

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 📦 **scikit-learn** | ML | Modelo preditivo principal |
| 📊 **NumPy & Pandas** | Dados | Processamento e transformação de dados |
| 📈 **Matplotlib & Seaborn** | Visualização | Gráficos e análises de apoio |
| 🔍 **MLflow** | MLOps | Tracking de experimentos e artefatos |

### 🧪 Qualidade & Testes

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🧪 **Pytest** | Testes | Testes unitários e de integração |
| 🎯 **pytest-cov** | Coverage | Medição e reporte de cobertura |
| ✨ **Ruff** | Lint/Format | Qualidade de código e formatação |
| 🔤 **MyPy** | Type Checking | Verificação de tipos estáticos |

### 🔒 Segurança

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🛡️ **Bandit** | Scanner | Análise de vulnerabilidades Python |
| 🔑 **detect-secrets** | Scanner | Prevenção de vazamento de segredos |

### 🐳 Infraestrutura & DevOps

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🐳 **Docker** | Containerização | Imagem única para execução em produção |
| 🐙 **Docker Compose** | Orquestração local | Execução local simplificada |
| 🔄 **GitHub Actions** | CI/CD | Pipelines por branch (feature/develop/main) |
| 🤖 **GitHub Copilot** | IA | Apoio em revisão de código e padronização |

---

## 🧱 Arquitetura da Solução

Arquitetura modular para cobrir o ciclo completo de dados, treino, inferência e operação:
- **Desenvolvimento local**: execução com Docker Compose (porta de entrada única em `http://localhost`).
- **Produção no Render**: container único com **Nginx + FastAPI + Streamlit** sob supervisão do **Supervisor**.

### 1. 🏗️ Arquitetura de Execução

```mermaid
flowchart TD
    %% Acesso Externo com ícone e círculo menor
    User@{ shape: circle, label: "👤 Usuários ONG"} -->|Acesso HTTPS| Nginx[Nginx Reverse Proxy :8080]
    
    %% Roteamento Nginx e Orquestração
    subgraph Docker [Produção: Container Único]
        Nginx -->|Raiz /| Landing[Landing Page Estática]
        Nginx -->|Rota /api| FastAPI[FastAPI Backend :8000]
        Nginx -->|Rota /dashboard| Streamlit[Streamlit UI :8501]
    end
    
    %% Comunicação Interna 
    Streamlit -->|Chamadas REST| FastAPI
    
    %% Motor MLOps e Persistência
    subgraph MLOps [MLOps Engine]
        FastAPI -->|Injeção de Modelo| ModelPKL[(app/models/model.pkl)]
        FastAPI -->|XAI Explainer| SHAP[SHAP / LIME]
    end
    
    %% Saídas e Monitoramento
    FastAPI -->|Auditoria| JSONLogs[(Logs JSON)]
    
    %% Estilização de Cores e Contrastes
    classDef proxy fill:#5c2d91,stroke:#333,stroke-width:2px,color:#fff;
    classDef api fill:#0078d4,stroke:#333,stroke-width:2px,color:#fff;
    classDef ui fill:#107c10,stroke:#333,stroke-width:2px,color:#fff;
    classDef ops fill:#d83b01,stroke:#333,stroke-width:2px,color:#fff;
    classDef db fill:#f3f2f1,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    
    class Nginx proxy;
    class FastAPI api;
    class Streamlit,Landing ui;
    class SHAP ops;
    class ModelPKL,JSONLogs db;
```

**Roteamento principal via Nginx:**
- `/` → landing page
- `/api/*` → FastAPI
- `/dashboard/*` → Streamlit
- `/health` → health check da API

### 🔄 Pipeline de Dados e ML

```mermaid
flowchart LR
    %% Fontes de Dados
    Data[(Dataset Excel)] --> Load[1. Carga: load_data]
    
    %% Etapas de Processamento (Conforme src/)
    subgraph Prep [Preparação e Treino]
        direction TB
        Load --> Clean[2. Limpeza: clean_data]
        Clean --> Target[3. Target: create_target]
        Target --> Missing[4. Imputação: handle_missing_values]
        Missing --> Feature[5. Engenharia: create_features]
        Feature --> Train[6. Treino: scripts/train.py]
    end

    %% Artefatos
    Train --> Champion[(app/models/model.pkl)]
    Train --> MLflow[(MLflow Artifacts)]

    %% Ciclo de Predição (Conforme predict_route.py)
    subgraph Predict [Fluxo de Inferência]
        direction TB
        Input[Input Aluno] --> Align[Alinhamento de Features]
        Align --> Champion
        Champion --> Result[Predição + XAI]
    end

    %% Estilização
    classDef step_first fill:#e3f2fd,stroke:#1976d2,stroke-width:1px,color:#000;
    classDef step_second fill:#f3f2f1,stroke:#616161,stroke-width:1px,color:#000;
    classDef dbs fill:#fff4dd,stroke:#d4a017,stroke-width:2px,color:#000;

    class Prep step_first;
    class Predict step_second;
    class Data,Champion,MLflow dbs;
```

### 🎯 Componentes Principais

1. **🌐 Nginx (Reverse Proxy)**: padroniza entrada única e reduz acoplamento entre cliente e serviços internos.
2. **📊 Camada de Dados**: garante limpeza e consistência antes de qualquer inferência.
3. **🛠️ Feature Engineering**: transforma sinais pedagógicos em variáveis úteis ao modelo.
4. **🤖 Camada de Modelo**: mantém ciclo de evolução champion/challenger com rastreabilidade.
5. **⚡ API FastAPI**: expõe endpoints operacionais e auditáveis para integração externa.
6. **🎨 Dashboard Streamlit**: acelera análise funcional por usuários não técnicos.
7. **🔍 Monitoramento com MLflow**: viabiliza comparabilidade entre versões de modelo.
8. **🐳 Infraestrutura Docker**: assegura reprodutibilidade entre desenvolvimento e produção.

---

## 🗂️ Estrutura de Diretórios

Estrutura atual (resumo dos diretórios e arquivos mais relevantes):

```text
fase-5/
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/
│       ├── feature-pipeline.yml
│       ├── develop-pipeline.yml
│       ├── main-pipeline.yml
│       └── issue-ops-rollback.yml
├── app/
│   ├── main.py
│   ├── dashboard.py
│   ├── routes/
│   ├── models/
│   ├── dashboard/
│   └── utils/
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── model.py
├── scripts/
│   ├── train.py
│   └── monitoring.py
├── tests/
├── notebooks/
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
├── supervisord.conf
├── render.yaml
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── Makefile
├── DEPLOYMENT.md
├── TESTING.md
├── TESTING_STRATEGY.md
└── README.md
```

---

## 🚀 Como Configurar e Executar o Projeto

### Pré-requisitos

- **Python 3.11+**
- **Docker** e **Docker Compose** (opcional)
- **Git**
- **Make** (opcional)

### ✅ Caminho Feliz (Recomendado): Docker Compose

```bash
docker compose up --build

# Em segundo plano
docker compose up -d --build

# Logs
docker compose logs -f

# Encerrar
docker compose down
```

Serviços via `http://localhost`:

| Serviço | URL | Descrição |
|---------|-----|-----------|
| 🏠 Landing Page | `http://localhost/` | Página inicial |
| ⚡ API Docs | `http://localhost/api/docs` | Swagger da API |
| 📊 Dashboard | `http://localhost/dashboard/` | Interface Streamlit |
| ❤️ Health Check | `http://localhost/health` | Saúde da aplicação |

### Opção B: Execução Local (Secundária)

#### 1. Clone e instale dependências

```bash
git clone https://github.com/Fiap-Pos-tech-5MLET/fase-5.git
cd fase-5

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Configure variáveis de ambiente

Crie o arquivo `.env` na raiz do projeto (use `.env.example` como base):

```bash
PROJECT_NAME="Tech Challenge Fase 5 - Associação Passos Mágicos"
ENVIRONMENT=development
API_URL=http://127.0.0.1:8000
MODEL_PATH=app/models/model.pkl
DATASET_PATH=app/data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx
ARTIFACTS_DIR=app/artifacts
MLFLOW_TRACKING_URI=file:./mlruns
API_KEY=troque_para_uma_chave_forte_em_producao
```

#### 3. Prepare os dados

Coloque o dataset em `app/data/raw/` com o nome esperado pelo pipeline.

#### 4. Treine o modelo inicial

```bash
python scripts/train.py
```

Artefatos principais gerados:
- `app/models/model.pkl`
- `app/artifacts/`

#### 5. Execute a API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API local: `http://127.0.0.1:8000`

#### 6. Execute o Dashboard Streamlit

Em outro terminal:

```bash
streamlit run app/dashboard.py --server.port=8501 --server.address=127.0.0.1
```

Dashboard local: `http://127.0.0.1:8501`

---

## ✅ Testes e Validações

O projeto utiliza testes automatizados e validações de qualidade com meta mínima de **85%** de cobertura.

### Executar testes

```bash
# Todos os testes
pytest tests/ -v

# Cobertura
pytest tests/ --cov=src --cov=app --cov-report=term-missing -v

# Relatório HTML
pytest tests/ --cov=src --cov=app --cov-report=html
```

### Verificação de qualidade

```bash
make format
make lint
make type-check
make security
make quality
```

Para mais detalhes, consulte [TESTING.md](TESTING.md).

---

## 2. 🔄 Esteira CI/CD (Fluxo GitFlow Horizontal)

O projeto adota pipelines GitHub Actions por branch, alinhados ao fluxo GitFlow.
```mermaid
flowchart LR
    classDef event fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#000;
    classDef feat fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef dev fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000;
    classDef prod fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;

    Start([Push feature/*]):::event --> P1_1
    subgraph P1 [1. Pipeline Feature]
        P1_1(Linting Ruff):::feat --> P1_2(Testes Pytest):::feat
        P1_1 --> P1_3(Build Docker):::feat
        P1_2 --> P1_4(Automação PR):::feat
        P1_3 --> P1_4
    end
    P1_4 --> APR1([Merge develop]):::event
    APR1 --> P2_1
    subgraph P2 [2. Pipeline Develop]
    P2_1(Qualidade: Ruff + MyPy):::dev --> P2_3(Testes + Coverage):::dev
    P2_1 --> P2_4(Segurança: Bandit + Secrets):::dev
        P2_3 --> P2_5(Build Docker):::dev
        P2_4 --> P2_5
        P2_5 --> P2_6(Automação PR Release):::dev
    end
    P2_6 --> APR2([Merge main]):::event
    APR2 --> P3_1
    subgraph P3 [3. Pipeline Main]
    P3_1(Validar Origem PR):::prod --> P3_2(Smoke Tests):::prod
        P3_2 --> P3_3(Build Docker):::prod
    P3_3 --> P3_4(Deploy Render):::prod
    P3_4 --> P3_5(Post-Deploy Smoke):::prod
    P3_5 --> P3_6(Auto Rollback em Falha):::prod
    end
  P3_6 --> F([API em Produção]):::event
```
### Workflows ativos

1. **Feature Pipeline** (`feature-pipeline.yml`)
  - Executa em `feature/*` e `bugfix/*`
  - Valida qualidade, testes rápidos e build Docker
  - Pode abrir PR automático para `develop`

2. **Develop Pipeline** (`develop-pipeline.yml`)
  - Executa em `push` na branch `develop`
  - Executa qualidade, testes completos, segurança e build Docker
  - Pode abrir PR de release para `main`

3. **Main Pipeline** (`main-pipeline.yml`)
  - Executa apenas em `push` na branch `main`
  - Faz smoke tests, build Docker e deploy no Render via deploy hook
  - Executa smoke pós-deploy e rollback automático em falhas

4. **IssueOps Rollback** (`issue-ops-rollback.yml`)
  - Workflow de rollback emergencial acionado por label `ops:rollback` em issue

### 3. 🚑 Gestão de Incidentes e Rollback (Resiliência)

```mermaid
flowchart LR
    classDef incident fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#c62828;
    classDef action fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef automation fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef final fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#000;

    Start((🚨 Incidente)):::incident --> Type{Tipo?}
    Type -- "Código" --> Code[Issue + Label: ops:rollback]:::action
    Code --> IssueOps[Workflow: IssueOps Rollback]:::automation
    IssueOps --> Revert[Git Revert na Main]:::automation

    Type -- "Modelo" --> Model[Editar champion_run_id.txt]:::action

    Revert --> MainPipeline
    Model --> MainPipeline[3. Main Pipeline: Deploy]:::automation
    MainPipeline --> Success([✅ Estabilizado]):::final
    Success --> Hotfix[Criar Branch: hotfix/*]:::action
    Hotfix --> PR[Merge Main e Develop]:::automation
```

### Características

- Logs e summaries detalhados por etapa
- Jobs paralelos para reduzir tempo de execução
- Gating por qualidade/segurança antes de avançar de estágio
- Governança de origem de PR entre branches

---

## 🤖 IA para Code Review

O projeto usa **GitHub Copilot** com instruções customizadas para padronização técnica.

### Padrões verificados

- ✅ Type hints em parâmetros e retornos
- ✅ Docstrings em estilo Google (português)
- ✅ Convenções de nomenclatura e organização de código
- ✅ Regras de segurança (inputs, segredos, erros)
- ✅ Boas práticas de testes e cobertura

As diretrizes estão em [.github/copilot-instructions.md](.github/copilot-instructions.md).

---

## 📖 Documentação da API

A API oferece endpoints para predição, auditoria e gestão do ciclo de modelos.

### Documentação interativa

- **Swagger UI (local direto na API):** `http://127.0.0.1:8000/docs`
- **Swagger UI (via Nginx):** `http://localhost/api/docs`
- **ReDoc (local direto na API):** `http://127.0.0.1:8000/redoc`

### Endpoints principais

```http
GET  /health
GET  /
POST /predict
GET  /model-info
GET  /drift
POST /retrain
POST /promote
POST /discard
GET  /model-metrics
GET  /model-artifact/{name}
```

### Autorização em rotas sensíveis

As rotas de escrita e retreinamento exigem o header `X-API-KEY` com o valor configurado na variável de ambiente `API_KEY`:

- `POST /retrain`
- `POST /promote`
- `POST /discard`

Sem chave válida, a API responde `401 Unauthorized`.

### Exemplo de ingestão via cURL

```bash
curl -X POST \
  'http://localhost/api/predict' \
  -H 'accept: application/json' \
  -H 'x-requested-by: banca_fiap' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": {
    "IDADE": 16,
    "FASE": "Fase 1 (3° e 4° ano)",
    "INDE_22": 6.5,
    "INDE_23": 7.1,
    "ANO_INGRESSO": 2022,
    "GÊNERO": "Feminino"
  }
}'
```

> Se estiver usando Docker com Nginx local, utilize `http://localhost/api/predict`.

### Estratégia Champion/Challenger (rastreabilidade)

- O endpoint `/retrain` gera um **candidato** e grava o `run_id` em `app/models/candidate_run_id.txt`.
- O endpoint `/promote` só promove para produção após validação, copiando o `run_id` para `app/models/champion_run_id.txt`.
- O endpoint `/model-metrics` usa `app/models/champion_run_id.txt` para consultar a run exata no MLflow.

Com isso, apenas modelos validados viram champion, e cada promoção fica auditável por `run_id`.

---

## 📊 Monitoramento e MLflow

O projeto utiliza **MLflow** para rastrear experimentos, parâmetros, métricas e artefatos de treinamento.

### Como acessar

**Execução local (manual):**

```bash
mlflow ui --port 5000
```

MLflow local: `http://127.0.0.1:5000`

> Em produção no Render, o foco principal é API + Dashboard. O uso de MLflow na nuvem depende do perfil de recursos e da configuração do ambiente.

### Racional de segurança e recursos em produção

No deploy de produção, o serviço de MLflow foi desativado para reduzir consumo de memória/CPU e evitar pressão de recursos no container principal (API + Dashboard + Nginx). O roteamento permanece centralizado no Nginx (`nginx.conf`) e a execução de processos é controlada pelo Supervisor (`supervisord.conf`), onde o bloco do MLflow fica desabilitado por padrão.

### Métricas rastreadas

- Acurácia, precisão, recall e F1-score
- Curvas e artefatos de avaliação
- Hiperparâmetros de treinamento
- Metadados de execução e versionamento de modelo

---

## 🎥 Vídeo Demonstrativo

Assista ao vídeo explicativo do projeto e de seu funcionamento:

- 📹 **Link do vídeo:** [Em breve]
- 💎 **API pública:** [datathon-machine-learning-engineering](https://datathon-machine-learning-engineering.onrender.com/)
- 📊 **Conteúdo:** arquitetura, API, pipeline de treinamento e resultados

### 📸 Screenshots da Aplicação

**Principais interfaces:**
- 🏠 Landing Page
- 📊 Dashboard Streamlit
- 📖 Swagger da API
- 🔍 Indicadores e artefatos de monitoramento

#### Landing Page

> Adicione aqui o screenshot atualizado da landing page quando disponível.

---

## 🤝 Desenvolvedores

Este projeto foi desenvolvido como parte do **Tech Challenge Fase 5** da **Pós-Graduação em Machine Learning Engineering da FIAP**.

**Equipe 5MLET:**

| Nome | RM | GitHub |
|------|-----|--------|
| Lucas Felipe de Jesus Machado | RM364306 | [@lfjmachado](https://github.com/lfjmachado) |
| Antônio Teixeira Santana Neto | RM364480 | [@antonioteixeirasn](https://github.com/antonioteixeirasn) |
| Gabriela Moreno Rocha dos Santos | RM364538 | [@gabrielaMSantos](https://github.com/gabrielaMSantos) |
| Erik Douglas Alves Gomes | RM364379 | [@Erik-DAG](https://github.com/Erik-DAG) |
| Leonardo Fernandes Soares | RM364648 | [@leferso](https://github.com/leferso) |

---

## ⚖️ Licença

Este projeto está licenciado sob a **Licença MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 📚 Documentação Adicional

### Guias de testes e qualidade

- **[TESTING.md](TESTING.md)** — Guia completo de testes e cobertura
- **[TESTING_STRATEGY.md](TESTING_STRATEGY.md)** — Estratégia de testes do projeto
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** — Resumo de implementação

### Guias de desenvolvimento e operação

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Guia de contribuição
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Guia de deploy
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** — Diretrizes de code review com IA

---

## 🌟 Agradecimentos

- **FIAP** — Pela estrutura e qualidade da pós-graduação.
- **Professores** — Pelo conhecimento compartilhado e orientação.
- **Associação Passos Mágicos** — Pelos dados e pela inspiração para um projeto com impacto social por meio da educação.

---