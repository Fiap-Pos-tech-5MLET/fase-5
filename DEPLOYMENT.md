# Deploy para Render - Container Único

## ?? Arquitetura

Este projeto utiliza um **container único** gerenciado pelo **Supervisor** para rodar todos os serviços:

```
+-------------------------------------------------+
¦  Render (Expõe porta 80)                        ¦
¦  +-------------------------------------------+  ¦
¦  ¦   Docker Container                        ¦  ¦
¦  ¦   +-----------------------------------+   ¦  ¦
¦  ¦   ¦   Supervisor (gerencia processos) ¦   ¦  ¦
¦  ¦   ¦   +- Nginx (porta 80)             ¦   ¦  ¦
¦  ¦   ¦   +- FastAPI (porta 8000)         ¦   ¦  ¦
¦  ¦   ¦   +- Streamlit (porta 8501)       ¦   ¦  ¦
¦  ¦   ¦   +- MLflow (porta 5000)          ¦   ¦  ¦
¦  ¦   +-----------------------------------+   ¦  ¦
¦  +-------------------------------------------+  ¦
+-------------------------------------------------+
```

### Roteamento

- **`http://seu-app.onrender.com/`** ? Landing page (index.html)
- **`http://seu-app.onrender.com/api/docs`** ? API Documentation (Swagger)
- **`http://seu-app.onrender.com/dashboard/`** ? Streamlit Dashboard
- **`http://seu-app.onrender.com/mlflow/`** ? MLflow Tracking UI

## ?? Como Fazer Deploy

### Opção 1: Via render.yaml (Recomendada)

1. **Commitar o código**:
```bash
git add .
git commit -m "Deploy: Container único com Supervisor"
git push origin main
```

2. **Criar Web Service no Render**:
   - Acesse [render.com](https://render.com)
   - Clique em **"New +" ? "Web Service"**
   - Conecte seu repositório GitHub
   - O Render detectará automaticamente o `render.yaml`

3. **Aguardar Deploy** (5-10 minutos na primeira vez)

## ?? Arquivos Modificados

### 1. `Dockerfile`
- Instalado Nginx e Supervisor
- Criado configuração do Supervisor inline
- Todos os serviços rodam no mesmo container
- Porta 80 exposta (Nginx)
- Health check via curl

### 2. `nginx.conf`
- Removidos `upstream` blocks
- Proxy para `http://127.0.0.1:PORTA` (localhost)
- Roteamento para API, Dashboard e MLflow

### 3. `app/main.py`
- Adicionado `root_path="/api"` quando ENVIRONMENT=production

### 4. `app/dashboard/config.py`
- API_URL usa `http://127.0.0.1:8000` em produção

### 5. `.streamlit/config.toml`
- `baseUrlPath = "/dashboard"` para rodar atrás do Nginx

## ?? Testar Localmente

### Com Docker (simula produção):

```bash
# Construir imagem
docker build -t datathon-app .

# Rodar container
docker run -p 80:80 --name datathon datathon-app

# Testar endpoints
curl http://localhost/              # Landing page
curl http://localhost/api/docs      # API docs
```

## ?? Logs e Debugging

### Ver logs do Render:
1. Acesse seu serviço no Dashboard do Render
2. Clique na aba **"Logs"**
3. Logs mostrarão:
   - Supervisor iniciando
   - Nginx, API, Streamlit e MLflow rodando

### Logs importantes:

```
INFO supervisord started
INFO spawned: 'nginx' with pid XXX
INFO spawned: 'api' with pid XXX
INFO spawned: 'dashboard' with pid XXX
INFO spawned: 'mlflow' with pid XXX
```

---

**Desenvolvido para Datathon Passos Mágicos - FIAP Pós-Tech 5MLET**
