# Content Performance Predictor Makefile

.PHONY: help install test lint format clean run-api run-dashboard build docker-build docker-run

# Default target
help:
	@echo "Content Performance Predictor - Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean generated files"
	@echo "  run-api      - Run the API server"
	@echo "  run-dashboard - Run the dashboard"
	@echo "  build        - Build the package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest tests/ -v

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	black --check src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Run the API server
run-api:
	uvicorn src.api.prediction_api:app --host 0.0.0.0 --port 8000 --reload

# Run the dashboard
run-dashboard:
	python dash_app/app.py

# Build the package
build:
	python setup.py sdist bdist_wheel

# Build Docker image
docker-build:
	docker build -t content-performance-predictor .

# Run with Docker Compose
docker-run:
	docker-compose up --build

# Run with Docker Compose (detached)
docker-run-detached:
	docker-compose up -d --build

# Stop Docker Compose
docker-stop:
	docker-compose down

# Run MLflow server
run-mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root file:/mlflow

# Create necessary directories
setup-dirs:
	mkdir -p logs models data/raw data/processed data/external mlflow

# Initialize project
init: setup-dirs install
	@echo "Project initialized successfully!"

# Development setup
dev-setup: init
	@echo "Development environment setup complete!"
	@echo "Run 'make run-api' to start the API server"
	@echo "Run 'make run-dashboard' to start the dashboard"
	@echo "Run 'make run-mlflow' to start MLflow tracking server"