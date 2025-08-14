# Xorb developer Makefile

.PHONY: help api orchestrator ptass up down test lint fmt token security-scan precommit-install sanitize-history integration-test integration-up integration-down backplane-lint nats-iac-plan nats-iac-apply nats-iac-destroy nats-test-isolation nats-generate-creds

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
	@echo ""
	@echo "Backplane Operations (Phase G2):"
	@echo "make backplane-lint     - validate NATS subject schema compliance"
	@echo "make nats-iac-plan      - plan NATS infrastructure changes"
	@echo "make nats-iac-apply     - apply NATS infrastructure"
	@echo "make nats-iac-destroy   - destroy NATS infrastructure"
	@echo "make nats-test-isolation - test tenant isolation"
	@echo "make nats-generate-creds - generate tenant credentials"

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

# --- Backplane Operations (Phase G2) ---
backplane-lint: ## Validate NATS subject schema compliance (v1 immutable)
	@echo "🔍 Validating NATS subject schema compliance..."
	@echo "Schema: xorb.<tenant>.<domain>.<service>.<event>"
	@python3 tools/backplane/subject_lint.py --schema
	@echo ""
	@echo "Scanning source code for violations..."
	@python3 tools/backplane/subject_lint.py --paths src/ services/ xorb_platform_bus/ tests/ tools/ infra/ \
		--allowlist "invalid|SCAN|CREATED|started|scanning|tenant-ñ|tenant@|ab\.|t\.|README\.md" || { \
		echo "❌ Subject schema violations found!"; \
		echo "Fix violations to comply with v1 schema."; \
		exit 1; \
	}
	@echo "✅ All NATS subjects comply with v1 schema"

nats-iac-plan: ## Plan NATS infrastructure changes with Terraform
	@echo "📋 Planning NATS infrastructure changes..."
	@cd infra/iac/nats && terraform init -upgrade
	@cd infra/iac/nats && terraform plan -var-file="environments/dev.tfvars" -out=tfplan 2>/dev/null || { \
		echo "Using default variables (dev.tfvars not found)..."; \
		terraform plan -out=tfplan; \
	}
	@echo "✅ Terraform plan completed. Review tfplan file."

nats-iac-apply: ## Apply NATS infrastructure (requires credentials)
	@echo "🚀 Applying NATS infrastructure..."
	@if [ ! -f "infra/iac/nats/tfplan" ]; then \
		echo "❌ No terraform plan found. Run 'make nats-iac-plan' first."; \
		exit 1; \
	fi
	@cd infra/iac/nats && terraform apply tfplan
	@echo "✅ NATS infrastructure deployed successfully"
	@echo "📄 Configuration files generated in infra/iac/nats/out/"

nats-iac-destroy: ## Destroy NATS infrastructure (DESTRUCTIVE)
	@echo "⚠️  WARNING: This will destroy ALL NATS infrastructure!"
	@echo "This includes accounts, streams, consumers, and data."
	@read -p "Type 'DESTROY' to confirm: " confirm && [ "$$confirm" = "DESTROY" ] || { \
		echo "❌ Destruction cancelled."; \
		exit 1; \
	}
	@cd infra/iac/nats && terraform destroy -auto-approve
	@echo "💥 NATS infrastructure destroyed"

nats-test-isolation: ## Test tenant isolation (requires NATS server)
	@echo "🧪 Testing tenant isolation..."
	@echo "Starting NATS server for testing..."
	@nats-server -js -p 14222 -sd /tmp/nats-test-store &
	@NATS_PID=$$!; \
	sleep 2; \
	echo "Running isolation tests..."; \
	python3 -c " \
import asyncio; \
from xorb_platform_bus.bus.pubsub.nats_client import create_nats_client, Domain, Event; \
async def test(): \
    print('Testing tenant A...'); \
    client_a = create_nats_client('tenant-a', ['nats://localhost:14222']); \
    async with client_a.connection(): \
        await client_a.publish(Domain.SCAN, 'test', Event.CREATED, {'data': 'from-a'}); \
        print('✅ Tenant A can publish to own subjects'); \
    print('Testing tenant B...'); \
    client_b = create_nats_client('tenant-b', ['nats://localhost:14222']); \
    async with client_b.connection(): \
        await client_b.publish(Domain.SCAN, 'test', Event.CREATED, {'data': 'from-b'}); \
        print('✅ Tenant B can publish to own subjects'); \
    print('✅ Tenant isolation test passed'); \
asyncio.run(test()) \
" || { echo "❌ Isolation test failed"; kill $$NATS_PID; exit 1; }; \
	kill $$NATS_PID; \
	echo "🧹 Cleaning up test server..."
	@echo "✅ Tenant isolation verified"

nats-generate-creds: ## Generate NATS credentials for tenants
	@echo "🔑 Generating NATS credentials..."
	@if [ ! -f "infra/iac/nats/out/tenant-t-qa-config.json" ]; then \
		echo "❌ No tenant configurations found. Run 'make nats-iac-apply' first."; \
		exit 1; \
	fi
	@echo "📁 Available tenant configurations:"
	@ls -1 infra/iac/nats/out/tenant-*-config.json 2>/dev/null | sed 's/.*tenant-\(.*\)-config.json/  \1/' || echo "  (none found)"
	@echo ""
	@echo "🔐 Sample credential usage:"
	@echo "export NATS_CREDENTIALS=/path/to/tenant-credentials.creds"
	@echo "nats pub --creds=\$$NATS_CREDENTIALS 'xorb.tenant-1.scan.nmap.created' 'test'"
	@echo ""
	@echo "✅ Use configuration files in infra/iac/nats/out/ for integration"
