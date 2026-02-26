# Testes e Qualidade

Runbook técnico para execução local e validação no CI.

## 1) Objetivo

- Garantir regressão funcional da API e pipeline ML.
- Garantir qualidade estática (lint, tipos, segurança).
- Manter cobertura mínima do projeto em **85%**.

## 2) Comandos essenciais

### Testes

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov=app --cov-report=term-missing -v
pytest tests/ --cov=src --cov=app --cov-report=html
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

Os testes do diretório `tests/` cobrem os módulos centrais de:

- limpeza e preparação de dados;
- engenharia de features;
- treinamento e artefatos;
- rotas e contratos da API;
- fluxo de monitoramento e regressão básica.

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

---

Referência estratégica: [TESTING_STRATEGY.md](TESTING_STRATEGY.md).
