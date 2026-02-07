.PHONY: help install build test lint format clean docker-build docker-run demo

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
	mypy .

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

demo: ## Run demo (model validation)
	python model_validation.py

validate: ## Run full validation suite
	python model_validation.py
	python psi_monitoring.py

dashboard: ## Launch Streamlit dashboard
	streamlit run app.py

quickstart: ## Quick start: install, validate, and launch
	$(MAKE) install
	$(MAKE) demo
	@echo "✅ Demo complete! Run 'make dashboard' to launch UI."
