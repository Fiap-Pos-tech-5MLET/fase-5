# 📓 Notebooks — Análise e Modelagem Passos Mágicos

Este diretório contém notebooks Jupyter para análise exploratória de dados (EDA), preparação de dados e modelagem preditiva do projeto **Tech Challenge Fase 5**.

---

## 📂 Estrutura de Arquivos

### Notebooks Principais (Refatorados) ✨

| Notebook | Propósito | Status | Código | Testes |
|----------|-----------|--------|--------|--------|
| **`data_preprocessing_passos_magicos_refactored.ipynb`** | Pipeline de consolidação de dados 2022-2024 | ✅ Prod | `scripts/data_processing.py` | 5+ testes |
| **`eda_passos_magicos_refactored.ipynb`** | Análise exploratória com testes estatísticos | ✅ Prod | `scripts/eda_analysis.py` + `scripts/visualization.py` | Inclusos |
| **`DATATHON-PASSOS-MÁGICOS_refactored.ipynb`** | Exploração e análise de continuidade 2020-2022 | ✅ Prod | `scripts/datathon_cleaning.py` | 38 testes |
| **`EDA_and_Training_refactored.ipynb`** | Pipeline completo: EDA → Treino → Avaliação | ✅ Prod | `src/data_cleaning.py` + `src/feature_engineering.py` + `scripts/visualization.py` | Inclusos |

### Notebooks Legados 📦

| Notebook | Propósito | Status |
|----------|-----------|--------|
| `data_preprocessing_passos_magicos.ipynb` | Versão original (pré-refactor) | 📦 Legado |
| `DATATHON-PASSOS-MÁGICOS.ipynb` | Versão original (pré-refactor) | 📦 Legado |

---

## 🎯 Guideline de Uso

### 1. Para Processamento de Dados
**Use:** `data_preprocessing_passos_magicos_refactored.ipynb`

**Características:**
- ✅ Código modularizado em `scripts/`
- ✅ Funções testadas (`tests/test_data_processing.py`)
- ✅ Documentação completa de lógica e decisões
- ✅ Pipeline reproduzível e auditável

**Saída:** `app/data/processed/dataset_consolidado_2022_2024.parquet`

### 2. Para Análise Exploratória
**Use:** `eda_passos_magicos_refactored.ipynb`

**Características:**
- ✅ Código modularizado em `scripts/eda_analysis.py` e `scripts/visualization.py`
- ✅ Funções testadas (testes inclusos)
- ✅ Testes de normalidade (Shapiro-Wilk, D'Agostino K²)
- ✅ Análise de correlação e coeficiente de variação
- ✅ Validação cruzada estratificada  
- ✅ Feature importance e análise univariada

### 3. Para Análise de Continuidade Estudantil
**Use:** `DATATHON-PASSOS-MÁGICOS_refactored.ipynb`

**Características:**
- ✅ Código modularizado em `scripts/datathon_cleaning.py`
- ✅ 38 testes unitários em `tests/scripts/test_datathon_cleaning.py`
- Análise de continuidade estudantil 2020-2022
- Filtragem e limpeza de datasets
- Contagem de novos alunos e taxas de permanência

### 4. Para Treinamento de Modelos
**Use:** `EDA_and_Training_refactored.ipynb`

**Características:**
- ✅ Usa módulos `src/data_cleaning.py`, `src/feature_engineering.py`, `src/model.py`
- ✅ Pipeline end-to-end modularizado
- ✅ Métricas completas: CM, ROC, Feature Importance

---

## 🔄 Padrão de Refatoração Aplicado

### Antes (Código Inline)
```python
# No notebook (mistura lógica com apresentação)
def padronizar_colunas_ano(df, ano):
    # 50 linhas de código...
    return df_padronizado

df_2022 = padronizar_colunas_ano(df_2022, 2022)
df_2023 = padronizar_colunas_ano(df_2023, 2023)
```

### Depois (Código Modularizado)
```python
# Notebook: apenas chamadas
from scripts.data_processing import padronizar_colunas_ano

df_2022 = padronizar_colunas_ano(df_2022, 2022)
df_2023 = padronizar_colunas_ano(df_2023, 2023)
```

```python
# scripts/data_processing.py: lógica testável
def padronizar_colunas_ano(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Padroniza nomenclatura com sufixo de ano."""
    # ... implementação ...
```

```python
# tests/test_data_processing.py: garantia de qualidade
def test_padronizacao_basica():
    df = pd.DataFrame({"Nome": ["Ana"]})
    resultado = padronizar_colunas_ano(df, 2022)
    assert "NOME_22" in resultado.columns
```

**Benefícios:**
- ✅ Notebooks mais limpos e focados em insights
- ✅ Código reutilizável em scripts Python
- ✅ Testabilidade e qualidade garantidas
- ✅ Manutenção facilitada

---

## 📊 Fluxo de Trabalho Recomendado

```mermaid
flowchart LR
    A[Dados Brutos] --> B[data_preprocessing_refactored.ipynb]
    B --> C[dataset_consolidado.parquet]
    C --> D[eda_passos_magicos_refactored.ipynb]
    D --> E[Insights e Features]
    E --> F[EDA_and_Training_refactored.ipynb]
    F --> G[Modelo Treinado]
```

### Sequência de Execução

1. **Preparação de Dados**
   ```bash
   jupyter notebook data_preprocessing_passos_magicos_refactored.ipynb
   ```
   Saída: `app/data/processed/dataset_consolidado_2022_2024.parquet`

2. **Análise Exploratória**
   ```bash
   jupyter notebook eda_passos_magicos_refactored.ipynb
   ```
   Saída: Visualizações, testes estatísticos, feature selection

3. **Treinamento de Modelo**
   ```bash
   jupyter notebook EDA_and_Training_refactored.ipynb
   ```
   Saída: `app/models/model.pkl`, métricas, artefatos

---

## 🧪 Testes Relacionados

Funções extraídas dos notebooks possuem testes unitários:

```bash
# Teste funções de processamento de dados
pytest tests/test_data_processing.py -v

# Teste funções de limpeza de dados (DATATHON)
pytest tests/scripts/test_datathon_cleaning.py -v

# Teste funções de feature engineering
pytest tests/test_notebook_feature_engineering.py -v

# Teste com cobertura completa
pytest tests/scripts/ --cov=scripts.datathon_cleaning
pytest tests/test_data_processing.py --cov=scripts.data_processing
```

---

## 📚 Módulos de Suporte

### scripts/
- **`data_processing.py`** — ETL, padronização, análise de qualidade
- **`datathon_cleaning.py`** — Limpeza e análise de continuidade estudantil (filter_columns, cleaning_dataset, create_annual_datasets, analyze_student_continuity)
- **`notebook_feature_engineering.py`** — Transformações de FASE, TURMA, flags derivadas
- **`eda_analysis.py`** — Testes estatísticos, validação cruzada
- **`visualization.py`** — Gráficos e visualizações

### src/
- **`data_cleaning.py`** — Limpeza e preparação para modelagem
- **`feature_engineering.py`** — Engenharia de features avançada
- **`model.py`** — Treinamento e avaliação de modelos