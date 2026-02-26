# Guia de Contribuição

Guia objetivo para contribuição técnica no projeto.

## 1) Fluxo de branches

- `feature/*` e `bugfix/*` → PR para `develop`
- `develop` → PR para `main`
- `main` é protegida e só recebe merge aprovado

## 2) Requisitos mínimos antes do PR

```bash
make format
make lint
make type-check
make security
make test
```

Se possível, execute também:

```bash
docker build -f Dockerfile .
```

## 3) Padrões de código

- Python com type hints e docstrings em português.
- Nomenclatura PEP8 (`snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`).
- Sem segredos hardcoded.
- Em produção, usar `logging` no lugar de `print`.

## 4) Segurança

- Nunca commitar credenciais, tokens ou `.env` real.
- Validar mudanças com `bandit` e `detect-secrets` (via workflow/local).

## 5) Gestão de incidentes e rollback (GitOps)

- Rollback de produção é **exclusivamente via GitOps**.
- Estratégia padrão:
  1. Ajustar `app/models/champion_run_id.txt` para uma versão estável.
  2. Abrir PR com justificativa do incidente.
  3. Fazer merge e deixar CI/CD aplicar a mudança.
- É proibido rollback manual direto no servidor para evitar drift operacional.

## 6) Checklist de PR

- Objetivo da mudança descrito de forma clara.
- Evidências de validação local (comandos e resultado resumido).
- Impacto em API, modelo e infraestrutura identificado.
- Mudanças em documentação incluídas quando aplicável.

## 7) Workflows oficiais

- `.github/workflows/feature-pipeline.yml`
- `.github/workflows/develop-pipeline.yml`
- `.github/workflows/main-pipeline.yml`

---

Dúvidas: abra uma issue no repositório.
