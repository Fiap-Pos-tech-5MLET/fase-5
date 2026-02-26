# Governança GitFlow e Proteção da Branch `main`

Recomendações para configurar no GitHub (Settings > Branches):

## Regra para `main`

- Exigir Pull Request antes de merge.
- Exigir aprovação de pelo menos 1 reviewer.
- Exigir status checks obrigatórios do workflow `ci-cd-pipeline.yml`.
- Bloquear push direto em `main`.
- Permitir merge apenas de `release/*` e `hotfix/*` (validado no workflow).

## Regra para `develop`

- Exigir Pull Request antes de merge.
- Exigir status checks obrigatórios do workflow `ci-cd-pipeline.yml`.
- Permitir merge apenas de `feature/*` e `bugfix/*` (validado no workflow).

## Fluxo GitFlow adotado

1. `feature/*` ou `bugfix/*` -> PR para `develop`.
2. `release/*` -> PR para `main` quando release aprovada.
3. `hotfix/*` -> PR para `main` para correções críticas.
4. Merge em `main` com sucesso no CI dispara deploy automático no Render.
