# 📓 Notebooks — Análise e Modelagem Passos Mágicos

Este diretório contém notebooks Jupyter para análise exploratória de dados (EDA), preparação de dados e modelagem preditiva do projeto **Tech Challenge Fase 5**.

---

## 📂 Estrutura de Arquivos

### Notebooks Principais (Refatorados) ✨

| Notebook | Propósito | Status | Código |
|----------|-----------|--------|--------|
| **`data_preprocessing_passos_magicos_refactored.ipynb`** | Pipeline de consolidação de dados 2022-2024 | ✅ Prod | `scripts/data_processing.py` |
| **`eda_passos_magicos.ipynb`** | Análise exploratória detalhada com testes estatísticos | 🔄 Legado | Código inline |
| **`EDA_and_Training.ipynb`** | Pipeline completo: EDA → Treino → Avaliação | ✅ Prod | `src/*` |

### Notebooks Legados 📦

| Notebook | Propósito | Status |
|----------|-----------|--------|
| `DATATHON-PASSOS-MÁGICOS.ipynb` | Exploração inicial 2020-2022 | 📦 Arquivo |
| `data_preprocessing_passos_magicos.ipynb` | Versão original (pré-refactor) | 📦 Legado |

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
**Use:** `eda_passos_magicos.ipynb`

**Características:**
- Testes de normalidade (Shapiro-Wilk, D'Agostino K²)
- Análise de correlação e coeficiente de variação
- Validação cruzada estratificada  
- Feature importance e análise univariada

**Potencial de Refatoração:** Alto (funções podem ser movidas para `scripts/eda_analysis.py`)

### 3. Para Treinamento de Modelos
**Use:** `EDA_and_Training.ipynb`

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
    C --> D[eda_passos_magicos.ipynb]
    D --> E[Insights e Features]
    E --> F[EDA_and_Training.ipynb]
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
   jupyter notebook eda_passos_magicos.ipynb
   ```
   Saída: Visualizações, testes estatísticos, feature selection

3. **Treinamento de Modelo**
   ```bash
   jupyter notebook EDA_and_Training.ipynb
   ```
   Saída: `app/models/model.pkl`, métricas, artefatos

---

## 🧪 Testes Relacionados

Funções extraídas dos notebooks possuem testes unitários:

```bash
# Teste funções de processamento de dados
pytest tests/test_data_processing.py -v

# Teste funções de feature engineering
pytest tests/test_notebook_feature_engineering.py -v

# Teste com cobertura
pytest tests/test_data_processing.py --cov=scripts.data_processing
```

---

## 📚 Módulos de Suporte

### scripts/
- **`data_processing.py`** — ETL, padronização, análise de qualidade
- **`notebook_feature_engineering.py`** — Transformações de FASE, TURMA, flags derivadas
- **`eda_analysis.py`** — Testes estatísticos, validação cruzada
- **`visualization.py`** — Gráficos e visualizações

### src/
- **`data_cleaning.py`** — Limpeza e preparação para modelagem
- **`feature_engineering.py`** — Engenharia de features avançada
- **`model.py`** — Treinamento e avaliação de modelos

---

## 🎓 Boas Práticas

### Para Notebooks
- ✅ **Documentação inicial**: Explique lógica, decisões e contexto
- ✅ **Código modularizado**: Importe funções de `scripts/` ao invés de código inline
- ✅ **Células focadas**: Cada célula deve ter propósito claro (carga, transformação, visualização)
- ✅ **Outputs limpos**: Remove outputs desnecessários antes de commit

### Para Scripts
- ✅ **Type hints**: Use anotações de tipo em parâmetros e retornos
- ✅ **Docstrings**: Documente função, parâmetros, retorno e exemplos (formato Google)
- ✅ **Testes**: Cada função deve ter testes correspondentes em `tests/`
- ✅ **Imutabilidade**: Retorne cópias (`df.copy()`) ao invés de modificar in-place

---

## 🔗 Referências

- **README Principal**: [../README.md](../README.md)
- **Guia de Testes**: [../TESTING.md](../TESTING.md)
- **Estratégia de Testes**: [../TESTING_STRATEGY.md](../TESTING_STRATEGY.md)
- **Deployment**: [../DEPLOYMENT.md](../DEPLOYMENT.md)

---

## 🚀 Próximos Passos

### Curto Prazo
- [ ] Refatorar `eda_passos_magicos.ipynb` movendo funções para `scripts/eda_analysis.py`
- [ ] Adicionar testes para funções de `visualization.py`
- [ ] Criar notebook de exemplo para uso das funções

### Médio Prazo
- [ ] Integrar notebooks no pipeline CI/CD (executar como smoke tests)
- [ ] Criar versão HTML dos notebooks para documentação
- [ ] Adicionar badges de status nos notebooks

---

**Última atualização:** 05/03/2026  
**Responsável**: Equipe 5MLET — Tech Challenge Fase 5
