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
- **Containerização**: Deploy simplificado via Docker e Docker Compose
- **CI/CD Automatizado**: Pipeline completo de integração e entrega contínua com GitHub Actions
- **Cobertura de Testes**: >90% de cobertura de código com testes automatizados
- **IA para Code Review**: Revisão automática de código usando GitHub Copilot

---

## 🛠 Tecnologias e Ferramentas

| Ferramenta | Categoria | Utilização no Projeto |
|------------|-----------|----------------------|
| 🐍 Python 3.11+ | Linguagem de Programação | Linguagem principal para ML, API e pipeline de dados |
| 🔥 PyTorch | Framework de Deep Learning | Implementação de redes neurais LSTM |
| ⚡ FastAPI | Framework Web | API REST de alta performance |
| 📊 NumPy & Pandas | Bibliotecas de Dados | Manipulação e análise de dados educacionais |
| 📈 Matplotlib & Seaborn | Visualização | Gráficos e visualizações de dados |
| 🧪 Pytest | Framework de Testes | Testes automatizados com >90% de cobertura |
| 📦 scikit-learn | Biblioteca de ML | Pré-processamento, modelos e métricas |
| 🔍 MLflow | Plataforma MLOps | Rastreamento de experimentos e modelos |
| 🐳 Docker | Containerização | Ambiente isolado e reprodutível |
| 🔄 GitHub Actions | CI/CD | Pipeline automatizado de build, teste e deploy |
| 🤖 GitHub Copilot | IA Code Review | Revisão automática de código seguindo padrões |
| 🎨 Streamlit | Framework de Dashboard | Interface interativa para visualização de dados |

---

## 🧱 Arquitetura da Solução

O sistema é construído sobre uma arquitetura modular e escalável com suporte para deployment em produção via Docker e Nginx.

### Arquitetura em Desenvolvimento

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Dataset       │────▶│  Data Pipeline   │────▶│  Preprocessing  │
│   (CSV)         │     │  (data_loader)   │     │  (normalização) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit     │◀────│  ML Models       │◀────│  Feature Eng.   │
│   :8501         │     │  (Sklearn/PyTorch│     │  (análise)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │
         └────────▶ FastAPI ◀────┘
                    :8000
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐     ┌──────────────────┐
│   Docker        │     │    MLflow        │
│   (Deploy)      │     │  (Monitoring)    │
└─────────────────┘     └──────────────────┘
```

### Componentes Principais

1. **Camada de Dados**: Carregamento e validação dos dados educacionais (2022-2024)
2. **Camada de Processamento**: Limpeza, normalização e engenharia de features
3. **Camada de Modelo**: Modelos de Machine Learning para predição e classificação
4. **Camada de Serviço**: API REST com FastAPI expondo endpoints de análise e predição
5. **Camada de Interface**: Dashboard Streamlit para exploração visual dos dados
6. **Camada de Monitoramento**: MLflow para rastreamento de experimentos e métricas
7. **Infraestrutura**: Ambiente dockerizado para deploys reprodutíveis

---

## 🗂️ Estrutura de Diretórios

O projeto está organizado da seguinte forma para facilitar a navegação e o entendimento:

```
fase-5/
│
├── .github/
│   ├── copilot-instructions.md      # Instruções para IA Code Review
│   └── workflows/
│       └── ci-cd-pipeline.yml       # Pipeline de CI/CD automatizado
│
├── app/
│   ├── config.py                    # Configurações da aplicação
│   ├── main.py                      # Ponto de entrada da API FastAPI
│   ├── schemas.py                   # Schemas Pydantic (Request/Response)
│   ├── data/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── audit_route.py           # Rotas de auditoria
│   │   ├── predict_route.py         # Rota de predição
│   │   └── train_route.py           # Rota de treinamento
│   └── utils/
│
├── src/
│   ├── data_loader.py               # Carregamento de dados educacionais
│   ├── preprocessing.py             # Pré-processamento e limpeza
│   ├── feature_engineering.py       # Engenharia de features
│   ├── lstm_model.py                # Modelos de Machine Learning
│   ├── train.py                     # Pipeline de treinamento
│   ├── evaluate.py                  # Avaliação e métricas
│   ├── seed_manager.py              # Reprodutibilidade
│   └── utils.py                     # Funções auxiliares
│
├── tests/
│   ├── conftest.py                  # Configurações do pytest
│   ├── test_*.py                    # Testes unitários e de integração
│   └── ...
│
├── notebooks/                       # Notebooks Jupyter para análise exploratória
│
├── data/                            # Datasets (não versionado)
│   ├── raw/                         # Dados brutos
│   ├── processed/                   # Dados processados
│   └── .gitkeep
│
├── docs/                            # Documentação adicional
│   └── images/                      # Imagens para documentação
│
├── docker-compose.yml               # Orquestração de contêineres
├── Dockerfile                       # Definição da imagem Docker
├── nginx.conf                       # Configuração do Nginx
├── index.html                       # Landing page
├── streamlit_app.py                 # Dashboard interativo
├── .streamlit/
│   └── config.toml                  # Configuração do Streamlit
├── Makefile                         # Comandos automatizados
├── pytest.ini                       # Configuração do pytest
├── requirements.txt                 # Dependências de produção
├── requirements-dev.txt             # Dependências de desenvolvimento
├── run_tests.py                     # Script para executar testes
├── CONTRIBUTING.md                  # Guia de contribuição
├── TESTING.md                       # Documentação de testes
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
MODEL_PATH=app/artifacts/model.pkl
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
python -m src.train

# Ou usando Make
make train
```

Isso criará os artefatos em `app/artifacts/`:
- `model.pkl` - Modelo treinado
- `scaler.pkl` - Scaler para normalização

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
streamlit run streamlit_app.py --server.port=8501 --server.address=127.0.0.1

# Ou usando Make
make run-streamlit
```

**Dashboard disponível em:** http://localhost:8501

---

### Opção B: Execução com Docker

```bash
# Construir e executar
docker-compose up --build

# Ou em background
docker-compose up -d --build

# Parar
docker-compose down
```

**Serviços disponíveis:**
- API: http://localhost:8000
- Dashboard: Execute o Streamlit localmente

---

## ✅ Testes e Validações

O projeto possui uma cobertura de testes completa (>90%) com testes automatizados para todos os componentes principais.

### Executar Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Ou usando Make
make test

# Rodar com cobertura de código
pytest tests/ --cov=src --cov-report=term-missing -v

# Ou usando Make
make coverage

# Gerar relatório HTML de cobertura
make coverage-html
# Abrir: htmlcov/index.html

# Rodar teste específico
pytest tests/test_lstm_model.py -v
```

### Verificação de Qualidade

```bash
# Rodar todos os checks de qualidade
make quality

# Checks individuais
make lint          # Pylint + Flake8
make format        # Black + isort
make type-check    # MyPy
make security      # Bandit
```

Para mais detalhes sobre testes, consulte o arquivo [TESTING.md](TESTING.md).

---

## 🔄 CI/CD Pipeline

O projeto implementa um pipeline completo de CI/CD usando **GitHub Actions**, garantindo qualidade e confiabilidade do código.

### Pipeline Automatizado

1. **Code Quality Check**: Verifica formatação, linting e type hints
2. **Build**: Valida a construção da aplicação
3. **Unit Tests & Coverage**: Executa testes com validação de cobertura mínima (90%)
4. **Integration Tests**: Testa endpoints da API
5. **Model Training**: Treina modelo com dados de validação
6. **Security Scan**: Análise de segurança com Bandit

### Triggers do Pipeline

- **Push** para branches `main` ou `develop`
- **Pull Requests** para `main` ou `develop`

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
- ✅ **Testes**: Cobertura mínima de 90%

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

### Iniciar MLflow

```bash
# Iniciar servidor MLflow
mlflow ui --port 5000

# Ou usando Make
make mlflow
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

[Link para o vídeo demonstrativo será adicionado aqui]

---

## 🤝 Desenvolvedores

Este projeto foi desenvolvido como parte do **Tech Challenge Fase 5** da **Pós-Graduação em Machine Learning Engineering da FIAP**.

**Equipe 5MLET**:
- [Nome do Desenvolvedor 1]
- [Nome do Desenvolvedor 2]
- [Nome do Desenvolvedor 3]

---

## ⚖️ Licença

Este projeto está licenciado sob a **Licença MIT** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🌟 Agradecimentos

Agradecimento especial à **Associação Passos Mágicos** por disponibilizar os dados e pela inspiração deste projeto que visa contribuir para a transformação da vida de crianças e jovens através da educação.

Para mais informações sobre a Associação Passos Mágicos, visite: [Site oficial da Associação]

---

**Feito com ❤️ pela Equipe 5MLET - FIAP Pós-Tech Machine Learning Engineering**
