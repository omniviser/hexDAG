.PHONY: help install test lint format clean docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	poetry install
	poetry run pre-commit install

test: ## Run tests
	poetry run pytest --cov=hexai --cov=pipelines

test-fast: ## Run tests without coverage
	poetry run pytest -x -v

lint: ## Run linting tools
	poetry run flake8 hexai pipelines
	poetry run mypy hexai pipelines
	poetry run bandit -r hexai pipelines

format: ## Format code
	poetry run black .
	poetry run isort .

clean: ## Clean up build artifacts
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	poetry run mkdocs build

docs-serve: ## Serve documentation locally
	poetry run mkdocs serve

pre-commit: ## Run all pre-commit hooks
	poetry run pre-commit run --all-files

build: ## Build package
	poetry build

publish: ## Publish to PyPI
	poetry publish --build
