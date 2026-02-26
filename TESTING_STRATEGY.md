# Estratégia de Testes

Documento estratégico para avaliação da qualidade técnica do projeto.

## 1) Princípios

- **Confiabilidade**: cada mudança deve manter comportamento esperado da API e do pipeline ML.
- **Reprodutibilidade**: execução de testes e checks deve ser simples e padronizada.
- **Governança**: merge só ocorre com aprovação dos gates definidos em CI/CD.

## 2) Metas de qualidade

- Cobertura mínima de referência: **85%**.
- Zero falha crítica em lint, tipagem e segurança.
- Build Docker válido nas etapas críticas de CI.

## 3) Pirâmide de testes

1. **Unitários**: lógica de limpeza, features e utilitários.
2. **Integração**: contratos das rotas FastAPI e fluxo de treino/promoção.
3. **Smoke**: checagens essenciais em `main` para liberar deploy.

## 4) Política de regressão

- Mudanças em contrato de API exigem atualização de testes de rota.
- Mudanças em features de modelo exigem revalidação de inferência.
- Mudanças de infraestrutura exigem validação do build Docker.

## 5) Qualidade por branch (GitFlow)

- `feature/*|bugfix/*`: qualidade + testes rápidos + build Docker.
- `develop`: qualidade + testes completos + segurança + build Docker.
- `main`: smoke tests + build Docker + deploy.

## 6) Critérios de aceite para banca

- Projeto reproduzível com `docker compose up --build`.
- Evidência de CI/CD com gates e relatórios por etapa.
- Evidência de testes automatizados e cobertura.

---

Execução prática e comandos: [TESTING.md](TESTING.md).
