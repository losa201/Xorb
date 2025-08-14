# Xorb developer Makefile

.PHONY: help api orchestrator ptass up down test lint fmt token security-scan precommit-install sanitize-history integration-test integration-up integration-down

help:
	@echo "Development Commands:"
	@echo "make up         - docker compose up for dev"
	@echo "make down       - docker compose down"
	@echo "make api        - run API locally (requires venv)"
	@echo "make orchestrator - run orchestrator locally (requires venv)"
	@echo "make test       - run pytest for API"
	@echo ""
	@echo "Security & Compliance:"
	@echo "make guardrails-verify  - run all security guardrails checks"
	@echo "make sbom               - generate Software Bill of Materials"
	@echo "make sign               - sign artifacts (dry-run)"
	@echo "make slis-serve         - start SLI metrics server"
	@echo "make security-full      - run comprehensive security scan"

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

# --- Guardrails and Security Targets ---
guardrails-verify: ## Run all security guardrails checks
	@echo "🔒 Running repository guardrails verification..."
	@echo "📋 Checking pre-commit hooks..."
	@pre-commit run --all-files || { echo "❌ Pre-commit hooks failed"; exit 1; }
	@echo "🔍 Running gitleaks secret detection..."
	@gitleaks detect --source . --no-banner --redact || { echo "❌ Secrets detected"; exit 1; }
	@echo "🛡️ Checking for HS256 usage..."
	@! grep -r --include="*.py" --include="*.js" --include="*.ts" --include="*.json" --include="*.yaml" --include="*.yml" "\bHS256\b" . --exclude-dir=.git --exclude-dir=venv --exclude-dir=.venv --exclude-dir=node_modules || { echo "❌ HS256 usage detected"; exit 1; }
	@echo "📡 Checking for Redis bus usage..."
	@! grep -r --include="*.py" --include="*.js" --include="*.ts" "\b(redis\.(pubsub|subscribe|psubscribe)|redis\.client\.PubSub)\b" . --exclude-dir=.git --exclude-dir=venv --exclude-dir=.venv --exclude-dir=node_modules || { echo "❌ Redis bus usage detected"; exit 1; }
	@echo "🔐 Checking TLS configurations..."
	@! grep -r --include="*.conf" --include="*.yaml" --include="*.yml" --include="*.json" --include="*.py" "(ssl_protocols|tls_version|TLSv1\.[0-2])" . --exclude-dir=.git --exclude-dir=venv --exclude-dir=.venv || { echo "❌ Legacy TLS detected"; exit 1; }
	@echo "✅ All guardrails checks passed"

sbom: ## Generate Software Bill of Materials
	@echo "📋 Generating SBOM..."
	@./tools/scripts/syft_cosign.sh sbom

sign: ## Sign artifacts with Cosign (dry-run by default)
	@echo "✍️ Signing artifacts..."
	@DRY_RUN=true ./tools/scripts/syft_cosign.sh full

slis-serve: ## Start SLI metrics server
	@echo "📊 Starting SLI metrics server..."
	@cd src/xorb_platform/observability && python3 metrics.py --simulate --host 0.0.0.0 --port 9090

# --- Security scanning targets ---
security-full: ## Run comprehensive security scan
	@echo "🔍 Running comprehensive security scan..."
	@./tools/scripts/security-scan.sh || echo "⚠️ Security scan completed with warnings"
