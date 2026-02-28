.PHONY: help install dev lint format typecheck test test-cov build clean docker

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format code
	ruff format src/ tests/

typecheck:  ## Run mypy
	mypy src/ --ignore-missing-imports

test:  ## Run tests
	pytest -v

test-cov:  ## Run tests with coverage
	pytest --cov=src/pyaccelerate --cov-report=html --cov-report=term

build:  ## Build wheel + sdist
	python -m build

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache htmlcov

docker:  ## Build Docker image
	docker build -t pyaccelerate:latest .

docker-gpu:  ## Build GPU Docker image
	docker build -f Dockerfile.gpu -t pyaccelerate:gpu .
