# Testing Guide - Associação Passos Mágicos ML Project

Este documento descreve a estratégia de testes, cobertura de código e como executar os testes do projeto de análise e predição de desenvolvimento educacional.

## Índice
- [Estrutura de Testes](#estrutura-de-testes)
- [Cobertura de Código](#cobertura-de-código)
- [Executar Testes](#executar-testes)
- [Verificação de Qualidade](#verificação-de-qualidade)
- [CI/CD Pipeline](#cicd-pipeline)

---

# Estratégia de Testes

## 📊 Visão Geral

Este documento descreve a estratégia de testes atual do projeto para o Datathon.
O foco está na engenharia de features e nos componentes já implementados.

---

## 🗂️ Estrutura de Testes

```
tests/
└── test_feature_engineering.py
```

---

## ✅ Módulos Cobertos

### 1. src/feature_engineering.py

Testes de:
- Criação de features derivadas.
- Tratamento de ausência de colunas esperadas.
- Seleção de features e remoção de vazamentos.

---

## 🧪 Como Executar os Testes

```
pytest tests/test_feature_engineering.py -v
```

---

## 📈 Cobertura Atual

Atualmente, a cobertura está limitada ao módulo de engenharia de features.
A meta de 80% de cobertura global exigida pelo desafio ainda não foi atingida.
```bash
pytest tests/ -vv --tb=long
```

#### 6. Rodar Apenas Testes Rápidos
```bash
pytest tests/ -v -m "not slow"
```

### Usando Make

```bash
# Ver todos os comandos disponíveis
make help

# Rodar testes
make test

# Rodar com cobertura
make coverage

# Gerar relatório HTML
make coverage-html

# Qualidade completa (lint + type + test + coverage)
make quality

# Qualidade rápida
make quick-quality
```

---

## Verificação de Qualidade

### 1. Linting

```bash
# Pylint
pylint src/ app/

# Flake8
flake8 src/ app/ --max-line-length=100

# Ambos com Make
make lint
```

### 2. Formatação de Código

```bash
# Verificar formatação
black --check src/ app/ tests/

# Aplicar formatação
black src/ app/ tests/

# Ordenar imports
isort src/ app/ tests/

# Com Make
make format
```

### 3. Type Checking

```bash
# MyPy
mypy src/ app/

# Com Make
make type-check
```

### 4. Security

```bash
# Bandit (security analysis)
bandit -r src/ app/

# Com Make
make security
```

### 5. Executar Todos os Checks

```bash
# Com Make (recomendado)
make quality

# Ou manualmente
pytest tests/ --cov=src --cov-report=term-missing
pylint src/ app/ --exit-zero
black --check src/ app/ tests/
mypy src/ app/ --exit-zero
bandit -r src/ app/ -v
```

---

## CI/CD Pipeline

### Estrutura do Pipeline

O pipeline GitHub Actions (`.github/workflows/ci-cd-pipeline.yml`) executa:

1. **Code Quality Check** - Linting e formatting
2. **Build** - Verifica sintaxe e imports
3. **Tests** - Testes unitários com cobertura >= 90%
4. **Integration Tests** - Testes de integração
5. **Train Model** - Treina modelo (branch main apenas)
6. **Security** - Análise de segurança
7. **Documentation** - Verifica documentação
8. **Report** - Gera relatório resumido

### Triggering Pipeline

O pipeline é acionado por:
- `push` em `main` ou `develop`
- `pull_request` em `main` ou `develop`

### Visualizar Pipeline

1. Ir para: `https://github.com/Fiap-Pos-tech-5MLET/fase-4/actions`
2. Clicar na run desejada para ver detalhes

---

## Estratégia de Testes

### Estrutura AAA (Arrange-Act-Assert)

Todos os testes seguem este padrão:

```python
def test_something():
    # Arrange - Prepara dados de teste
    model = LSTMModel()
    x = torch.randn(32, 10, 1)
    
    # Act - Executa o que está sendo testado
    output = model.forward(x)
    
    # Assert - Verifica o resultado
    assert output.shape == (32, 1)
```

### Categorias de Testes

#### Unit Tests
- Testam funções/métodos isoladamente
- Sem dependências externas (mocks quando necessário)
- Rápidos (< 1 segundo cada)

#### Integration Tests
- Testam integração entre componentes
- Simulam pipeline real
- Podem ser mais lentos

#### Marks (Marcadores)

```python
@pytest.mark.unit          # Teste unitário
@pytest.mark.slow          # Teste lento
@pytest.mark.gpu           # Requer GPU
@pytest.mark.skipif(...)   # Skip condicional
```

### Fixtures

Fixtures disponíveis em `tests/conftest.py`:

```python
@pytest.fixture
def lstm_model():
    """Modelo LSTM básico"""

@pytest.fixture
def pytorch_device():
    """Dispositivo (CPU ou CUDA)"""

@pytest.fixture
def minmax_scaler():
    """Scaler normalizado"""

@pytest.fixture
def sample_dataloader():
    """DataLoader de exemplo"""
```

---

## Boas Práticas

### ✓ Fazer
- Usar fixtures para dados compartilhados
- Testes independentes e determinísticos
- Nomes descritivos para testes
- Assert com mensagens úteis
- Utilizar `pytest.raises()` para exceções

### ✗ Evitar
- Testes dependentes de ordem de execução
- Testes que modificam arquivos sem cleanup
- Testes com múltiplas responsabilidades
- Assertions vagas

---

## Troubleshooting

### Problema: Testes falham por "Module not found"
```bash
# Solução: Adicione o projeto ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Problema: Testes CUDA falham em máquina sem GPU
```bash
# Solução: Testes com skip automático
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    ...
```

### Problema: Testes lentos
```bash
# Solução: Rodar em paralelo
pip install pytest-xdist
pytest tests/ -n auto
```

### Problema: Cache de importação
```bash
# Solução: Limpar cache
rm -rf __pycache__ .pytest_cache
pytest --cache-clear tests/
```

---

## Métricas de Sucesso

✓ **Tests**: Todos passam (`pytest tests/`)
✓ **Coverage**: >= 90% (`pytest --cov=src`)
✓ **Linting**: Sem warnings (`pylint src/`)
✓ **Type**: Sem erros (`mypy src/`)
✓ **Security**: Sem vulnerabilidades (`bandit -r src/`)
✓ **Format**: Código formatado (`black --check src/`)

---

## Referências

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)
- [GitHub Actions](https://docs.github.com/en/actions)

---

**Última atualização**: Janeiro 2026
**Maintainer**: Equipe 5MLET
