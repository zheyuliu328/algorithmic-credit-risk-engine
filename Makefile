.PHONY: help install build test lint format clean docker-build docker-run demo quickstart

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install with dev dependencies
	pip install -e ".[dev]"

build: ## Build package
	python -m build

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=credit_risk_engine --cov-report=html

lint: ## Run linters
	ruff check .

format: ## Format code
	black .
	ruff check --fix .

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t credit-risk-engine:latest .

docker-run: ## Run Docker container
	docker run -p 8501:8501 credit-risk-engine:latest

docker-run-bash: ## Run Docker container with bash
	docker run -it credit-risk-engine:latest bash

demo: ## Run demo with synthetic data (safe, no API key needed)
	python run.py demo

validate: ## Run model validation
	python run.py validate

validate-dry: ## Run validation (dry run, no side effects)
	python run.py validate --dry-run

dashboard: ## Launch Streamlit dashboard
	python run.py dashboard

quickstart: ## Quick start: install + demo (1 minute)
	$(MAKE) install
	$(MAKE) demo
	@echo ""
	@echo "✅ Quick start complete!"
	@echo "Next: Run 'make dashboard' to launch UI or 'make validate' for full validation"
