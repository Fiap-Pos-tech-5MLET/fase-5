# Notebooks - Associação Passos Mágicos

Este diretório contém Jupyter Notebooks para análise exploratória, experimentação e documentação do projeto de ML para predição de defasagem escolar.

## 📚 Estrutura de Notebooks

### 01_exploratory_data_analysis.ipynb
**Objetivo**: Análise exploratória dos dados educacionais (2022-2024)

**Conteúdo**:
- Carregamento e visão geral dos dados
- Estatísticas descritivas
- Identificação de missing values e outliers
- Distribuição de variáveis
- Correlações entre features
- Visualizações (histogramas, boxplots, heatmaps)

**Outputs**: 
- Insights sobre qualidade dos dados
- Features mais relevantes
- Necessidades de pré-processamento

---

### 02_feature_engineering_experiments.ipynb
**Objetivo**: Experimentação com engenharia de features

**Conteúdo**:
- Criação de novas features
- Transformações (log, sqrt, polinomiais)
- Encoding de variáveis categóricas
- Normalização e scaling
- Seleção de features (RFE, feature importance)

**Outputs**:
- Conjunto final de features
- Transformações aplicadas
- Feature importance ranking

---

### 03_model_training_experiments.ipynb
**Objetivo**: Experimentação com diferentes modelos

**Conteúdo**:
- Baseline models (Decision Tree, Random Forest, Logistic Regression)
- Advanced models (XGBoost, LightGBM, Neural Networks)
- Hyperparameter tuning
- Cross-validation
- Comparação de modelos

**Outputs**:
- Modelo selecionado
- Melhores hiperparâmetros
- Métricas de validação

---

### 04_model_evaluation.ipynb
**Objetivo**: Avaliação detalhada do modelo final

**Conteúdo**:
- Métricas de performance (Accuracy, Precision, Recall, F1, AUC-ROC)
- Matriz de confusão
- Curva ROC
- Análise de erros
- Feature importance
- SHAP values para interpretabilidade

**Outputs**:
- Relatório de performance
- Análise de interpretabilidade
- Recomendações de uso

---

### 05_api_usage_examples.ipynb
**Objetivo**: Exemplos de uso da API

**Conteúdo**:
- Como fazer predições via API
- Exemplos de requests e responses
- Casos de uso reais
- Integração com o sistema

**Outputs**:
- Guia prático de uso
- Código exemplo para integração

---

## 🚀 Como Usar

### Pré-requisitos
```bash
# Instalar Jupyter
pip install jupyter notebook

# Instalar dependências
pip install -r ../requirements-dev.txt
```

### Executar Notebooks
```bash
# No diretório raiz do projeto
jupyter notebook notebooks/
```

### Ordem Recomendada
1. `01_exploratory_data_analysis.ipynb` - Entender os dados
2. `02_feature_engineering_experiments.ipynb` - Criar features
3. `03_model_training_experiments.ipynb` - Treinar modelos
4. `04_model_evaluation.ipynb` - Avaliar performance
5. `05_api_usage_examples.ipynb` - Usar a API

---

## 📊 Datasets

Os datasets devem estar em:
```
../data/raw/passos_magicos_2022_2024.csv
```

Para processar os dados:
```python
import sys
sys.path.append('..')
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# Carregar
df = load_data('../data/raw/passos_magicos_2022_2024.csv')

# Pré-processar
df_processed = preprocess_data(df)
```

---

## 🔬 Experimentação

Use MLflow para rastrear experimentos:

```python
import mlflow

# Iniciar experimento
mlflow.set_experiment("passos-magicos-experiments")

with mlflow.start_run(run_name="modelo-baseline"):
    # Seu código de treinamento
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", 0.85)
    mlflow.sklearn.log_model(model, "model")
```

Visualizar experimentos:
```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

## 📝 Boas Práticas

1. **Versionamento**: Salve versões importantes dos notebooks
2. **Limpeza**: Limpe outputs antes de commits
3. **Documentação**: Documente decisões e insights
4. **Reprodutibilidade**: Use seeds fixas para experimentos
5. **Modularização**: Mova código reutilizável para `src/`

---

## 🤝 Contribuindo

Ao adicionar novos notebooks:
1. Nomeie seguindo o padrão `NN_descriptive_name.ipynb`
2. Adicione descrição neste README
3. Documente objetivo e outputs esperados
4. Use markdown cells para explicar o código

---

**Última Atualização**: 2026-02-08  
**Mantenedores**: Equipe 5MLET
