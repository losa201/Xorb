# Xorb developer Makefile

.PHONY: help api orchestrator ptass up down test lint fmt token security-scan precommit-install sanitize-history integration-test integration-up integration-down

help:
	@echo "make up         - docker compose up for dev"
	@echo "make down       - docker compose down"
	@echo "make api        - run API locally (requires venv)"
	@echo "make orchestrator - run orchestrator locally (requires venv)"
	@echo "make test       - run pytest for API"

up:
	docker compose -f deploy/configs/docker-compose.dev.yml up --build

down:
	docker compose -f deploy/configs/docker-compose.dev.yml down

api:
	cd src/api && uvicorn app.main:app --reload --port 8000

orchestrator:
	cd src/orchestrator && python main.py

test:
	pytest -q

token:
	@echo "Requesting dev token (ensure DEV_MODE=true)"
	@curl -s -X POST http://localhost:8000/auth/dev-token | jq -r .access_token

precommit-install: ## Install pre-commit hooks
	@pre-commit install -f

security-scan: ## Run gitleaks scan locally
	@gitleaks detect --source . --no-banner --redact --config .gitleaks.toml || true

sanitize-history: ## Rewrite git history to remove secrets
	@bash tools/secrets/remediate_git_history.sh

# --- Integration Test Targets ---
integration-test: ## Run integration tests
	@pytest -xvs tests/integration/

integration-up: ## Start local NATS JetStream server for integration tests (placeholder)
	@echo "Starting local NATS JetStream server..."
	@echo "Note: This target is a placeholder. The test suite manages NATS lifecycle."
	@echo "To run tests, use 'make integration-test'."

integration-down: ## Stop local NATS JetStream server for integration tests (placeholder)
	@echo "Stopping local NATS JetStream server..."
	@echo "Note: This target is a placeholder. The test suite manages NATS lifecycle."
