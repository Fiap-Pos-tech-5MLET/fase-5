# Governança GitFlow e Proteção da Branch `main`

Recomendações para configurar no GitHub (Settings > Branches):

## Regra para `main`

- Exigir Pull Request antes de merge.
- Exigir aprovação de pelo menos 1 reviewer.
- Exigir status checks obrigatórios do workflow `main-pipeline.yml`.
- Bloquear push direto em `main`.
- Permitir merge apenas de `develop` (validado no workflow).

## Regra para `develop`

- Exigir Pull Request antes de merge.
- Exigir status checks obrigatórios do workflow `develop-pipeline.yml`.
- Permitir merge apenas de `feature/*` e `bugfix/*` (validado no workflow).

## Regra para `feature/*` e `bugfix/*`

- Exigir status checks obrigatórios do workflow `feature-pipeline.yml`.
- Não permitir merge direto em `main`.

## Fluxo GitFlow adotado

1. `feature/*` ou `bugfix/*` -> PR para `develop`.
2. `develop` -> PR para `main` quando release aprovada.
3. Merge em `main` com sucesso no CI dispara deploy automático no Render.
