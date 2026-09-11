.PHONY: help up down test lint bench verify clean

help:
	@echo "Available commands:"
	@echo "  make up       - Start Redis and API microservice with Docker Compose"
	@echo "  make down     - Stop all Docker containers"
	@echo "  make test     - Run test suite (14 automated unit & integration tests)"
	@echo "  make verify   - Run E2E customer query verification script"
	@echo "  make bench    - Run 3-architecture baseline benchmark on Golden Set"
	@echo "  make lint     - Verify code syntax and hygiene across all files"

up:
	docker compose up --build

down:
	docker compose down

test:
	python -m pytest tests/ -v --tb=short

verify:
	python scripts/verify_e2e.py

bench:
	python scripts/evaluate_baselines.py

lint:
	python -m compileall src/ scripts/ tests/
