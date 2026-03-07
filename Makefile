.PHONY: help install install-dev test test-fast coverage coverage-html coverage-check lint format type-check security quality quick-quality clean run-api run-streamlit train docker-build docker-run ci pre-commit

PYTHON := python
PIP := pip
DOCKER_IMAGE := passos-magicos-ml-api:latest

help:
	@echo "Development Setup:"
	@echo "  make install          - Instala dependências de produção"
	@echo "  make install-dev      - Instala dependências + ferramentas de qualidade"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             - Executa testes"
	@echo "  make coverage         - Executa testes com cobertura"
	@echo "  make coverage-html    - Gera relatório HTML de cobertura"
	@echo "  make format           - Formata com Ruff"
	@echo "  make lint             - Lint com Ruff"
	@echo "  make type-check       - Type checking com MyPy"
	@echo "  make security         - Segurança com Bandit + detect-secrets"
	@echo "  make quality          - Pipeline local completo"
	@echo ""
	@echo "Run:"
	@echo "  make run-api          - Sobe API FastAPI"
	@echo "  make run-streamlit    - Sobe dashboard Streamlit"
	@echo "  make train            - Executa treinamento"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build da imagem Docker"
	@echo "  make docker-run       - Executa container local"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install ruff mypy pytest pytest-cov bandit detect-secrets pre-commit

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -q --maxfail=1

coverage:
	pytest tests/ --cov=src --cov=app --cov-report=term-missing --cov-report=xml -v

coverage-html:
	pytest tests/ --cov=src --cov=app --cov-report=html -v

coverage-check:
	coverage report --fail-under=85

format:
	ruff format app/ src/ tests/ scripts/

lint:
	ruff check app/ src/ tests/ scripts/

type-check:
	mypy src/ app/utils/ app/routes/ scripts/ --ignore-missing-imports

security:
	@echo "🔒 Executando análise de segurança..."
	@echo "🔍 Bandit - Análise de vulnerabilidades..."
	bandit -r app/ src/
	@echo ""
	@echo "🔑 Detect-Secrets - Detecção de segredos..."
	detect-secrets scan --all-files --baseline .secrets.baseline > .secrets.scan
	@python scripts/check_secrets.py

quality: format lint type-check test security

quick-quality: lint type-check test

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','htmlcov','dist','build'] if pathlib.Path(p).exists()]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

run-api:
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

run-streamlit:
	streamlit run app/dashboard.py --server.port=8501 --server.address=127.0.0.1

train:
	$(PYTHON) scripts/train.py

docker-build:
	docker build -f Dockerfile -t $(DOCKER_IMAGE) .

docker-run:
	docker run -p 8080:8080 $(DOCKER_IMAGE)

ci: quality

pre-commit:
	pre-commit run --all-files

.DEFAULT_GOAL := help
