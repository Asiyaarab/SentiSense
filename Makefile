.PHONY: help install install-dev install-api test lint format clean train-sample train-full api run-app docker

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps
	pip install -r requirements.txt

install-dev: ## Install runtime + dev deps
	pip install -e ".[dev]"

install-api: ## Install runtime + API deps
	pip install -r requirements-api.txt

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=sentisense --cov-report=term-missing

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Format with black + ruff fix
	black src/ tests/ train.py
	ruff check --fix src/ tests/

train-sample: ## Quick smoke test (built-in tiny CSV, ~30s)
	python train.py --sample

train-full: ## Full IMDB 50K training
	python train.py

train-svm: ## Full training with LinearSVC (~1-2% better F1)
	python train.py --model svm

api: ## Run FastAPI server on :8000
	uvicorn sentisense.app.api:app --reload --port 8000

run-app: ## Run Streamlit app on :8501
	streamlit run src/sentisense/app/streamlit_app.py

docker: ## Build Docker image
	docker build -t sentisense:latest .

clean: ## Remove built artifacts (but keep data/)
	rm -rf models/*.pkl models/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
