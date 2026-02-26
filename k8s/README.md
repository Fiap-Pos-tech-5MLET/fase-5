# Base Kubernetes

Manifestos iniciais para demonstrar prontidão de orquestração:

- `api-deployment.yaml`: deployment da API FastAPI com probes e recursos.
- `api-service.yaml`: service interno para expor a API dentro do cluster.

## Aplicação

```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```
