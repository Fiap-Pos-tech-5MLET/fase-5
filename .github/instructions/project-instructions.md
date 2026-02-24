# 🤖 Contexto do Projeto para Agentes de IA

> Este documento contém toda a informação necessária para que qualquer assistente de IA compreenda o projeto e consiga realizar ajustes com autonomia.

---

## 1. Visão Geral

**Projeto:** Datathon — Machine Learning Engineering (PósTech FIAP, Fase 5)  
**Domínio:** Educação Social — Associação Passos Mágicos  
**Objetivo:** Modelo preditivo de classificação binária que estima o **risco de defasagem escolar** de estudantes.  
**Repositório:** `pos-tech-fiap-techchallenge-fase5`

### Problema de Negócio
A Associação Passos Mágicos atende crianças e jovens de baixa renda em Embu-Guaçu/SP. Com dados educacionais de 2022–2024, o modelo identifica alunos com risco de insucesso para intervenção precoce.

### Variável Alvo
- **`DEFASAGEM < 0`** → TARGET = 1 (Em risco)
- **`DEFASAGEM >= 0`** → TARGET = 0 (Sem risco)
- Distribuição: 622 sem risco / 534 com risco (~balanceado)

---

## 2. Stack Tecnológica

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.12+ |
| ML | scikit-learn (RandomForestClassifier), pandas, numpy |
| API | FastAPI + Uvicorn |
| Serialização | joblib (`models/model.pkl`) |
| Testes | pytest + pytest-cov (cobertura: **92%**) |
| Container | Docker (`python:3.12-slim`) |
| Experimentos | MLflow |
| Drift | Evidently (DataDriftPreset → HTML report) |
| Logging | Python `logging` (formato estruturado) |

---

## 3. Estrutura de Diretórios

```
pos-tech-fiap-techchallenge-fase5/
├── data/
│   └── raw/
│       └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx   # Dataset principal (sheets: PEDE2022, PEDE2023, PEDE2024)
├── docs/                                              # PDFs de referência + dicionário de dados
├── models/
│   ├── model.pkl                                      # Modelo champion em produção (joblib)
│   ├── model_candidate.pkl                            # Modelo challenger temporário (pós-retrain)
│   ├── champion_run_id.txt                            # MLflow run_id do champion atual
│   ├── candidate_run_id.txt                           # MLflow run_id do candidato (temporário)
│   └── artifacts/                                     # ROC curve, classification report, feature importance, drift report
├── notebooks/
│   └── EDA_and_Training.ipynb                         # Notebook exploratório
├── src/                                               # Código-fonte principal
│   ├── app.py                                         # FastAPI (endpoints + middleware de logging)
│   ├── dashboard.py                                   # Dashboard Streamlit (predição, métricas MLflow, retreinamento, drift)
│   ├── data_cleaning.py                               # Carregamento, limpeza, target, missing values
│   ├── feature_engineering.py                         # Criação e seleção de features
│   ├── model.py                                       # Pipeline sklearn + treinamento + métricas
│   ├── monitoring.py                                  # Geração de drift report (Evidently)
│   └── train_pipeline.py                              # Orquestração completa do treinamento
├── tests/                                             # Testes unitários e integração
│   ├── conftest.py                                    # Fixtures compartilhadas
│   ├── test_api.py
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   ├── test_monitoring.py
│   └── test_train_pipeline.py
├── .streamlit/config.toml                             # Configuração de tema do Streamlit
├── Dockerfile
├── requirements.txt
├── README.md
└── AGENTS.md                                          # Este arquivo
```

---

## 4. Arquitetura do Pipeline ML

### Fluxo de Treinamento (`train_pipeline.py`)

```
Excel (PEDE2024) → load_data() → clean_data() → create_target() → handle_missing_values()
                                                        ↓
                                                  create_features() → select_features() → train_model()
                                                                                             ↓
                                                                            model.pkl + métricas + artefatos visuais
```

### Detalhamento dos Módulos

#### `src/data_cleaning.py`
| Função | Descrição |
|--------|-----------|
| `load_data(file_path)` | Carrega Excel, sheet `PEDE2024`. Retorna DataFrame. |
| `clean_data(df)` | Padroniza nomes de colunas (UPPER, underscore), remove `NOME_ANONIMIZADO`, coerce numeric em colunas INDE/PEDRA/NOTA. |
| `create_target(df)` | Cria coluna `TARGET`: 1 se `DEFASAGEM < 0`, 0 caso contrário. Remove linhas com DEFASAGEM nulo. |
| `handle_missing_values(df)` | Preenche numéricos com 0 e categóricos com `"UNKNOWN"`. Força tipo string nos categóricos. |

#### `src/feature_engineering.py`
| Função | Descrição |
|--------|-----------|
| `create_features(df)` | Cria `INDE_GROWTH` (INDE_2024 - INDE_23) e flag `HAS_HISTORY_23`. |
| `select_features(df)` | Retorna `(X, y)`. Remove colunas de leakage (INDE_2024, IAA, IEG, IPS, IPP, IDA, MAT, POR, ING, IPV, IAN, DESTAQUE_*, etc.), RA, NOME_ANONIMIZADO, DEFASAGEM, TARGET, INDE_GROWTH. |

> **⚠️ LEAKAGE — Regra Crítica:** Qualquer coluna que represente resultado de 2024 **NÃO pode** ser usada como feature. O modelo deve prever risco com base em dados históricos (2022/2023) e cadastrais. A lista de colunas de leakage está em `select_features()` e duplicada em `app.py`.

#### `src/model.py`
| Função | Descrição |
|--------|-----------|
| `create_pipeline(num, cat)` | Cria sklearn Pipeline: `ColumnTransformer(StandardScaler, OneHotEncoder)` → `SelectKBest(f_classif, k='all')` → `RandomForestClassifier(n_estimators=100)`. |
| `train_model(X, y)` | Split 80/20, treina, retorna `(model, metrics_dict, X_test, y_test)`. Métricas: classification_report, roc_auc, accuracy, f1_score, precision, recall. |
| `save_model(model, path)` | Salva com `joblib.dump`. |

#### `src/app.py` — API FastAPI
| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Health check: `{"status": "ok", "model_loaded": true/false}` |
| `/predict` | POST | Recebe `{"data": {...}}`, processa pelo pipeline, retorna `{"risk_prediction": 0|1, "risk_probability": float}`. Faz alinhamento automático de features com o modelo. |
| `/drift` | GET | Serve o relatório HTML de drift do Evidently. Retorna 404 se não existir. |
| `/model-info` | GET | Retorna metadados do modelo (tipo, features, estratégia de retreinamento, cenários de produção). |
| `/retrain` | POST | Treina modelo **candidato** (challenger) sem sobrescrever champion. Aceita hiperparâmetros: `n_estimators`, `max_depth`, `min_samples_split`, `k`, `test_size`. Salva `model_candidate.pkl` + `candidate_run_id.txt`. |
| `/promote` | POST | Promove candidato → champion: copia `model_candidate.pkl` → `model.pkl`, `candidate_run_id.txt` → `champion_run_id.txt`, recarrega modelo. |
| `/discard` | POST | Descarta modelo candidato (remove `model_candidate.pkl` e `candidate_run_id.txt`), mantém champion. |
| `/model-metrics` | GET | Retorna métricas, params e artefatos do champion via MLflow (busca por `champion_run_id.txt`). Fallback para info local. |
| `/model-artifact/{name}` | GET | Serve imagem de artefato (ex: `roc_curve.png`) do MLflow. Fallback para `models/artifacts/`. |
| `/docs` | GET | Swagger UI automática (FastAPI) com schemas tipados. |

**Champion/Challenger Flow:**
```
POST /retrain → Treina candidato (model_candidate.pkl) + salva candidate_run_id.txt
       ↓
✅ POST /promote (copia → model.pkl + champion_run_id.txt, recarrega)
❌ POST /discard (remove candidato, mantém champion)
```

**Input da API `/predict`:** Enviar dicionário com quaisquer campos. A API preenche campos faltantes com defaults seguros (0 para numéricos, "UNKNOWN" para categóricos). Exemplo:
```json
{"data": {"INDE_23": 6.5, "FASE": "1A", "TURMA": "A", "IDADE": 12}}
```

#### `src/monitoring.py`
- Carrega o dataset, processa pelo pipeline, faz split train/test como proxy de reference/current.
- Gera relatório HTML via `evidently.Report(DataDriftPreset())`.
- Salva em `models/artifacts/data_drift_report.html`.

#### `src/train_pipeline.py`
- Função `main(data_path=None, model_path=None, artifacts_dir=None)` aceita parâmetros opcionais (defaults apontam para paths padrão).
- Integra com **MLflow**: loga parâmetros, métricas e artefatos visuais.
- **Retorna `(metrics, run_id)`** — o `run_id` é o identificador da run MLflow, salvo em `candidate_run_id.txt`.
- Gera: `roc_curve.png`, `classification_report.png`, `feature_importance.png`.
- Usa `logging` estruturado (não `print()`).

#### `src/dashboard.py` — Dashboard Streamlit
| Página | Descrição |
|--------|-----------|
| 🔮 Predição | Formulário completo para inputar dados de aluno + gauge de probabilidade. Comunica com API via `POST /predict`. |
| 📊 Métricas do Modelo | KPIs vindos do MLflow (accuracy, roc_auc, f1, precision, recall), imagens de artefatos servidas via `GET /model-artifact/{name}`, info da run (run_id, model_type, hiperparâmetros). Fallback para cálculo local. |
| 🔄 Monitoramento de Drift | Relatório Evidently embarcado via iframe. |
| ⚙️ Retreinamento | Controle de hiperparâmetros (n_estimators, max_depth, min_samples_split, k, test_size), botão de retreinar, tabela comparativa **Champion vs Challenger**, botões de **Promover** ou **Descartar**. |
| ℹ️ Sobre o Projeto | Arquitetura, estratégia, cenários de produção. |

---

## 5. Métricas do Modelo Atual

| Métrica | Valor |
|---------|-------|
| Accuracy | 93.5% |
| F1-Score (weighted) | 93.5% |
| Precision (weighted) | 93.6% |
| Recall (weighted) | 93.5% |
| ROC-AUC | 0.9897 |

**Confiabilidade:** Prevenção rigorosa de leakage, distribuição balanceada, métricas equilibradas entre classes.

---

## 6. Dataset

**Arquivo:** `data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`

### Sheets disponíveis
| Sheet | Registros | Colunas |
|-------|-----------|---------|
| PEDE2022 | 860 | 42 |
| PEDE2023 | 1014 | 48 |
| PEDE2024 | 1156 | 50 |

### Colunas Principais (PEDE2024, pré-limpeza)
- **Identificação:** RA, Nome Anonimizado, Fase, Turma
- **Demográficas:** Data de Nasc, Idade, Gênero, Ano ingresso, Instituição de ensino
- **Histórico INDE:** INDE 22, INDE 23, INDE 2024 (⚠️ leakage)
- **Histórico Pedra:** Pedra 20, 21, 22, 23, 2024 (⚠️ leakage)
- **Índices 2024 (⚠️ leakage):** IAA, IEG, IPS, IPP, IDA, Mat, Por, Ing, IPV, IAN
- **Avaliações:** Avaliadores 1-6, Rec Av1/Av2, Rec Psicologia
- **Target source:** Defasagem
- **Outras:** Indicado, Atingiu PV, Destaque IEG/IDA/IPV, Fase Ideal, Escola, Ativo/Inativo

### Após limpeza (clean_data)
- Colunas ficam UPPERCASED com underscores: `INDE_23`, `PEDRA_2024`, `INSTITUIÇÃO_DE_ENSINO`, etc.
- `NOME_ANONIMIZADO` é removido.

### Após select_features
- ~28 features restantes (históricas + cadastrais), sem nenhuma coluna de 2024.

---

## 7. Testes

### Estrutura
```
tests/
├── conftest.py               # 5 fixtures: sample_raw_dataframe, sample_cleaned_dataframe,
│                              # sample_dataframe_with_target, sample_full_pipeline_dataframe, sample_X_y
├── test_data_cleaning.py     # 20 testes (load_data, clean_data, create_target, handle_missing_values)
├── test_feature_engineering.py # 16 testes (create_features, select_features, leakage prevention)
├── test_model.py             # 15 testes (create_pipeline, train_model, save_model)
├── test_api.py               # 17 testes (health, drift, predict, create_target)
├── test_monitoring.py        # 2 testes (drift report generation)
└── test_train_pipeline.py    # 8 testes (plot functions, main() integration, run_id tracking, missing data)
```

**Total: 86 testes**

### Executar
```bash
# Rodar testes
.venv/bin/python -m pytest tests/ -v

# Com cobertura
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing

# Cobertura atual: 92% (meta: ≥80%)
```

### Cobertura por módulo
| Módulo | Cobertura |
|--------|-----------|
| data_cleaning.py | 100% |
| feature_engineering.py | 100% |
| model.py | 100% |
| app.py | 87% |
| monitoring.py | 85% |
| train_pipeline.py | 92% |

---

## 8. Docker & Docker Compose

### Dockerfile (imagem base compartilhada)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY .streamlit/ .streamlit/
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (3 serviços)
| Serviço | Container | Porta | Command |
|---------|-----------|-------|---------|
| `api` | datathon-api | 8000 | `uvicorn src.app:app --host 0.0.0.0 --port 8000` |
| `dashboard` | datathon-dashboard | 8501 | `streamlit run src/dashboard.py --server.port 8501` |
| `mlflow` | datathon-mlflow | 5000 | `mlflow ui --host 0.0.0.0 --port 5000` |

**Volumes:**
- `models_data` — compartilhado entre `api` e `dashboard` (modelos e artifacts)
- `mlflow_data` — compartilhado entre `api` e `mlflow` (runs do MLflow)
- `./data` — bind mount read-only (dataset)

**Networking:**
- Dashboard conecta à API via hostname Docker interno: `API_URL=http://api:8000`
- API usa healthcheck: `api` deve estar healthy antes do `dashboard` iniciar

```bash
# Subir tudo
docker compose up -d

# Ver logs
docker compose logs -f

# Parar tudo
docker compose down

# Rebuild após alterações no código
docker compose build && docker compose up -d
```

---

## 9. Comandos Úteis

```bash
# Ambiente virtual
source .venv/bin/activate

# Treinar modelo
python src/train_pipeline.py

# Iniciar API
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

# Gerar drift report
python src/monitoring.py

# Testes com cobertura
pytest tests/ --cov=src --cov-report=term-missing

# MLflow UI
mlflow ui   # http://127.0.0.1:5000

# Docker
docker build -t datathon-passos-magicos .
docker run -p 8000:8000 datathon-passos-magicos
```

---

## 10. Convenções e Regras Importantes

### Código
- **Logging:** Usar `logging` (nunca `print()`). Formato: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`
- **Imports:** Módulos src são importados como `from src.module import function`
- **sys.path:** Cada módulo adiciona o project root via `sys.path.append(os.path.join(os.path.dirname(__file__), '..'))`
- **Idioma:** Código e docstrings em inglês. README e documentação em português.

### ML
- **⚠️ NUNCA** usar colunas de resultado 2024 como features (leakage). Lista completa em `select_features()`.
- **Serialização:** Sempre joblib (não pickle direto).
- **Pipeline:** Usar sklearn Pipeline com ColumnTransformer.
- **Métricas:** Sempre reportar accuracy, F1, precision, recall e ROC-AUC.

### Testes
- **Cobertura mínima:** 80% (atualmente 92%).
- **Fixtures:** Usar fixtures compartilhadas do `conftest.py`.
- **Integrações:** Usar `unittest.mock.patch` para isolar MLflow.
- **Dados reais:** Testes que dependem do Excel devem usar `pytest.skip()` se arquivo não existir.

### API
- **Endpoint `/predict`:** Recebe `{"data": {...}}`, retorna `{"risk_prediction": int, "risk_probability": float}`.
- **Feature alignment:** A API alinha automaticamente as features com o modelo, preenchendo faltantes.
- **Lista de leakage duplicada:** Existe em `app.py` e em `feature_engineering.py`. **Manter sincronizadas.**

---

## 11. Melhorias Pendentes (Opcionais)

| Melhoria | Descrição | Prioridade |
|----------|-----------|-----------|
| Dados multi-ano | Integrar sheets PEDE2022/2023 para features históricas mais ricas | 🟢 Melhoria |
| Deploy cloud | Subir API no GCP Cloud Run, Heroku ou AWS | 🟢 Opcional |
| CI/CD | GitHub Actions para rodar testes automaticamente | 🟢 Melhoria |
| Campos tipados na API | Substituir `data: dict` por campos Pydantic explícitos | 🟢 Melhoria |
| Centralizar leakage_cols | Mover lista de colunas de leakage para um único lugar (config) | 🟢 Melhoria |
| Autenticação endpoints | Proteger `/retrain`, `/promote`, `/discard` com autenticação | 🟡 Segurança |
| Vídeo apresentação | Gravar vídeo de até 5 min em formato gerencial | 🟡 Entregável |

---

## 12. Requisitos do Datathon (Checklist)

| # | Requisito | Status |
|---|-----------|--------|
| 1 | Pipeline completa de treinamento com feature engineering | ✅ |
| 2 | Modularização do código em .py separados | ✅ |
| 3 | API com endpoint /predict (FastAPI) | ✅ |
| 4 | Empacotamento Docker | ✅ |
| 5 | Deploy do modelo (local via Docker) | ✅ |
| 6 | Teste da API | ✅ |
| 7 | Testes unitários ≥80% cobertura | ✅ (92%) |
| 8 | Monitoramento: logs + drift dashboard | ✅ |
| 9 | Documentação completa | ✅ |
