PYTHON ?= python3

.PHONY: help install install-dev build test test-cov lint format clean \
        config-check demo quickstart run-real verify \
        docker-build docker-run release

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

build: ## Build package
	$(PYTHON) -m build

test: ## Run tests
	$(PYTHON) -m pytest

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest --cov=src --cov-report=html

lint: ## Run linters
	$(PYTHON) -m ruff check .

format: ## Format code
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

config-check: ## Check configuration
	$(PYTHON) config/validator.py config/config.yaml

demo: ## Run demo with sample data
	$(PYTHON) src/credit_one/run.py demo

quickstart: ## Quick start (default offline)
	$(MAKE) demo

run-real: ## Run with real data (usage: make run-real CSV=path/to/data.csv)
	@if [ -z "$(CSV)" ]; then \
		echo "Usage: make run-real CSV=path/to/data.csv"; \
		exit 1; \
	fi
	$(PYTHON) scripts/run_real.py $(CSV) --output artifacts

verify: ## Run full verification suite
	$(PYTHON) scripts/verify.py

docker-build: ## Build Docker image
	docker build -t $(shell basename $(PWD)):latest .

docker-run: ## Run Docker container
	docker run -v $(PWD)/data:/app/data $(shell basename $(PWD)):latest

release: ## Create a new release (requires VERSION)
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=x.y.z"; \
		exit 1; \
	fi
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)
