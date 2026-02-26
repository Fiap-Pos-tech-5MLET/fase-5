# Kubernetes (Baseline)

Esta pasta contém manifestos mínimos para demonstrar prontidão de orquestração.

## Arquivos

- `api-deployment.yaml`: deployment da API FastAPI (probes e recursos básicos).
- `api-service.yaml`: service interno para exposição da API no cluster.

## Aplicação

```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```

## Escopo

Este baseline não cobre ambiente produtivo completo (Ingress, autoscaling, secrets e observabilidade). Use como ponto de partida para evolução.
