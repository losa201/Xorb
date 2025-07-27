# Xorb 2.0 EPYC-Optimized Platform - Enhanced Makefile
# Developer UX improvements with hot-reload, EPYC optimization, and comprehensive tooling

SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help

# Project configuration
PROJECT_NAME := xorb
VERSION := 2.0.0
PYTHON_VERSION := 3.12
PY?=python3

# Directories
SRC_DIR := packages/xorb_core
TESTS_DIR := tests
DOCS_DIR := docs
VENV_DIR := venv
RESULTS_DIR := optimization_results

# EPYC Configuration
EPYC_CORES := 64
NUMA_NODES := 2
EPYC_MODEL := 7702

# Kubernetes configuration
K8S_NAMESPACE := xorb-prod
KUBECTL := kubectl
HELM := helm

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Production deployment configuration
COMPOSE_FILE_PROD := docker-compose.vps.yml
SECRETS_DIR := .secrets
VPS_IP ?= localhost

# ============================================================================
# Help System
# ============================================================================

help: ## Show this help message
	@echo "$(CYAN)Xorb 2.0 Security Intelligence Platform - Developer Tools$(RESET)"
	@echo "$(CYAN)=========================================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)🚀 Development Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "dev-|setup|install|clean" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)🔒 Production Deployment:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "prod-|deploy-|hardening-" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)🔍 Security & Monitoring:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "security-|monitor-|verify-" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)🔧 EPYC Optimization:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "epyc-|numa-|optimize" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)🐛 Testing & Quality:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "test|lint|format|check" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)☸️  Kubernetes & Deployment:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "deploy|k8s-|helm-" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)📊 Monitoring & Analysis:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "monitor|metrics|benchmark|profile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo "$(YELLOW)🔐 Security:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | grep -E "security|audit|scan" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $1, $2}'
	@echo ""
	@echo -e "$(YELLOW)Environment:$(RESET) $(ENV)"
	@echo -e "$(YELLOW)Namespace:$(RESET)   $(NAMESPACE)"
	@echo -e "$(YELLOW)Context:$(RESET)     $(KUBE_CONTEXT)"
	@echo -e "$(YELLOW)Version:$(RESET)     $(VERSION)"
	@echo ""
	@echo -e "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-20s$(RESET) %s\n", $1, $2}' $(MAKEFILE_LIST)

.PHONY: info
info: ## Show detailed environment information
	@echo -e "$(CYAN)=== Xorb 2.0 Environment Information ===$(RESET)"
	@echo -e "$(YELLOW)Project:$(RESET)        $(PROJECT_NAME)"
	@echo -e "$(YELLOW)Version:$(RESET)        $(VERSION)"
	@echo -e "$(YELLOW)Environment:$(RESET)    $(ENV)"
	@echo -e "$(YELLOW)Namespace:$(RESET)      $(NAMESPACE)"
	@echo -e "$(YELLOW)Registry:$(RESET)       $(REGISTRY)"
	@echo -e "$(YELLOW)Build Date:$(RESET)     $(BUILD_DATE)"
	@echo -e "$(YELLOW)Build Commit:$(RESET)   $(BUILD_COMMIT)"
	@echo -e "$(YELLOW)Build Branch:$(RESET)   $(BUILD_BRANCH)"
	@echo ""
	@echo -e "$(CYAN)=== System Information ===$(RESET)"
	@echo -e "$(YELLOW)Docker:$(RESET)         $(docker --version 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Kubectl:$(RESET)        $(kubectl version --client --short 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Helm:$(RESET)           $(helm version --short 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Python:$(RESET)         $(python --version 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Poetry:$(RESET)         $(poetry --version 2>/dev/null || echo 'Not available')"
	@echo ""
	@echo -e "$(CYAN)=== Kubernetes Context ===$(RESET)"
	@kubectl config get-contexts | grep -E "(CURRENT|$(KUBE_CONTEXT))" || echo "Context not available"

# =============================================================================
# DEVELOPMENT SETUP
# =============================================================================

install-dev: ## Install development dependencies
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

.PHONY: setup
setup: ## Initial development environment setup
	@echo -e "$(GREEN)Setting up Xorb 2.0 development environment...$(RESET)"
	@if [ ! -f .env ]; then \
		echo -e "$(YELLOW)Creating .env from template...$(RESET)"; \
		cp .env.$(ENV) .env; \
	fi
	@if [ ! -d $(VENV_DIR) ]; then \
		echo -e "$(YELLOW)Creating virtual environment...$(RESET)"; \
		python3 -m venv $(VENV_DIR); \
	fi; \
	source $(VENV_DIR)/bin/activate; \
	if grep -q "\[tool\\.poetry\]" pyproject.toml && command -v poetry >/dev/null 2>&1; then \
		echo -e "$(YELLOW)Installing Python dependencies with Poetry...$(RESET)"; \
		poetry install --with dev; \
	else \
		echo -e "$(YELLOW)Installing Python dependencies with pip...$(RESET)"; \
		pip install -r requirements.txt; \
	fi
	@if [ -f .pre-commit-config.yaml ]; then \
		echo -e "$(YELLOW)Setting up pre-commit hooks...$(RESET)"; \
		pre-commit install; \
	fi
	@echo -e "$(GREEN)Development environment setup complete!$(RESET)"

.PHONY: deps
deps: ## Install/update project dependencies
	@echo -e "$(GREEN)Installing/updating dependencies...$(RESET)"
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install --with dev; \
		poetry update; \
	else \
		pip install -r requirements.txt; \
		if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi; \
	fi

.PHONY: clean
clean: ## Clean up build artifacts and cache
	@echo -e "$(GREEN)Cleaning up build artifacts...$(RESET)"
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -delete 2>/dev/null || true
	@find . -type f -name ".coverage.*" -delete 2>/dev/null || true
	@rm -rf build/ dist/ .tox/ .nox/ 2>/dev/null || true
	@docker system prune -f 2>/dev/null || true
	@echo -e "$(GREEN)Cleanup complete!$(RESET)"

# =============================================================================
# CODE QUALITY & TESTING
# =============================================================================

.PHONY: format
format: ## Format code with black and isort
	@echo -e "$(GREEN)Formatting code...$(RESET)"
	@black .
	@isort .
	@echo -e "$(GREEN)Code formatting complete!$(RESET)"

.PHONY: lint
lint: ## Run linting with flake8, mypy, and bandit
	@echo -e "$(GREEN)Running linting...$(RESET)"
	@echo -e "$(YELLOW)Running flake8...$(RESET)"
	@flake8 . || true
	@echo -e "$(YELLOW)Running mypy...$(RESET)"
	@mypy . || true
	@echo -e "$(YELLOW)Running bandit security check...$(RESET)"
	@bandit -r . -f json -o bandit-report.json || true
	@echo -e "$(GREEN)Linting complete!$(RESET)"

.PHONY: test
test: ## Run pytest tests
	@echo -e "$(GREEN)Running tests...$(RESET)"
	@pytest tests/ -v --tb=short --cov=. --cov-report=html --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Run tests without coverage (faster)
	@echo -e "$(GREEN)Running fast tests...$(RESET)"
	@pytest tests/ -x -v --tb=short

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo -e "$(GREEN)Running integration tests...$(RESET)"
	@pytest tests/integration/ -v --tb=short

.PHONY: security-scan
security-scan: ## Run security scanning
	@echo -e "$(GREEN)Running security scans...$(RESET)"
	@echo -e "$(YELLOW)Scanning Python dependencies...$(RESET)"
	@safety check || true
	@echo -e "$(YELLOW)Scanning for secrets...$(RESET)"
	@detect-secrets scan --all-files || true
	@echo -e "$(YELLOW)Running bandit security analysis...$(RESET)"
	@bandit -r . || true

.PHONY: quality
quality: format lint test security-scan ## Run all code quality checks

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

.PHONY: build
build: ## Build Docker images
	@echo -e "$(GREEN)Building Docker images...$(RESET)"
	@DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker-compose -f $(DOCKER_COMPOSE_FILE) build \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
		--build-arg BUILD_BRANCH="$(BUILD_BRANCH)"
	@echo -e "$(GREEN)Docker build complete!$(RESET)"

.PHONY: build-prod
build-prod: ## Build production Docker images
	@echo -e "$(GREEN)Building production Docker images...$(RESET)"
	@DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker-compose -f $(COMPOSE_FILE_PROD) build \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg BUILD_COMMIT="$(BUILD_COMMIT)" \
		--build-arg BUILD_BRANCH="$(BUILD_BRANCH)"
	@docker tag xorb-api:latest $(REGISTRY)/xorb-api:$(VERSION)
	@docker tag xorb-worker:latest $(REGISTRY)/xorb-worker:$(VERSION)
	@docker tag xorb-orchestrator:latest $(REGISTRY)/xorb-orchestrator:$(VERSION)

.PHONY: push
push: build-prod ## Push Docker images to registry
	@echo -e "$(GREEN)Pushing Docker images to registry...$(RESET)"
	@docker push $(REGISTRY)/xorb-api:$(VERSION)
	@docker push $(REGISTRY)/xorb-worker:$(VERSION)
	@docker push $(REGISTRY)/xorb-orchestrator:$(VERSION)
	@docker tag $(REGISTRY)/xorb-api:$(VERSION) $(REGISTRY)/xorb-api:latest
	@docker tag $(REGISTRY)/xorb-worker:$(VERSION) $(REGISTRY)/xorb-worker:latest
	@docker tag $(REGISTRY)/xorb-orchestrator:$(VERSION) $(REGISTRY)/xorb-orchestrator:latest
	@docker push $(REGISTRY)/xorb-api:latest
	@docker push $(REGISTRY)/xorb-worker:latest
	@docker push $(REGISTRY)/xorb-orchestrator:latest

.PHONY: up
up: ## Start local development environment
	@echo -e "$(GREEN)Starting development environment...$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) up -d
	@echo -e "$(GREEN)Services started! Access points:$(RESET)"
	@echo -e "  $(CYAN)API:$(RESET)         http://localhost:8000"
	@echo -e "  $(CYAN)Temporal Web:$(RESET) http://localhost:8233"
	@echo -e "  $(CYAN)Neo4j Browser:$(RESET) http://localhost:7474"

.PHONY: down
down: ## Stop local development environment
	@echo -e "$(GREEN)Stopping development environment...$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) down

.PHONY: restart
restart: down up ## Restart local development environment

.PHONY: logs
logs: ## Show logs from local development environment
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

# =============================================================================
# KUBERNETES OPERATIONS
# =============================================================================

.PHONY: k8s-apply
k8s-apply: ## Apply Kubernetes manifests
	@echo -e "$(GREEN)Applying Kubernetes manifests for $(ENV) environment...$(RESET)"
	@kubectl apply -k gitops/overlays/$(ENV)
	@echo -e "$(GREEN)Manifests applied to namespace $(NAMESPACE)$(RESET)"

.PHONY: k8s-delete
k8s-delete: ## Delete Kubernetes resources
	@echo -e "$(YELLOW)Deleting Kubernetes resources for $(ENV) environment...$(RESET)"
	@kubectl delete -k gitops/overlays/$(ENV) --ignore-not-found=true
	@echo -e "$(GREEN)Resources deleted from namespace $(NAMESPACE)$(RESET)"

.PHONY: k8s-status
k8s-status: ## Show Kubernetes deployment status
	@echo -e "$(CYAN)=== Kubernetes Status ($(NAMESPACE)) ===$(RESET)"
	@kubectl get pods,svc,deploy -n $(NAMESPACE) -o wide
	@echo ""
	@echo -e "$(CYAN)=== Recent Events ===$(RESET)"
	@kubectl get events -n $(NAMESPACE) --sort-by='.metadata.creationTimestamp' | tail -10

.PHONY: k8s-logs
k8s-logs: ## Show Kubernetes pod logs
	@echo -e "$(GREEN)Showing logs for $(NAMESPACE)...$(RESET)"
	@kubectl logs -f -l app.kubernetes.io/part-of=xorb -n $(NAMESPACE) --max-log-requests=10

.PHONY: k8s-port-forward
k8s-port-forward: ## Set up port forwarding for development
	@echo -e "$(GREEN)Setting up port forwarding...$(RESET)"
	@kubectl port-forward -n $(NAMESPACE) svc/xorb-api 8000:8000 &
	@kubectl port-forward -n xorb-infra-$(ENV) svc/temporal-web 8233:8233 &
	@kubectl port-forward -n xorb-infra-$(ENV) svc/neo4j 7474:7474 &
	@kubectl port-forward -n xorb-infra-$(ENV) svc/grafana 3001:3000 &
	@echo -e "$(GREEN)Port forwarding started in background$(RESET)"

# =============================================================================
# HELM OPERATIONS
# =============================================================================

.PHONY: helm-install
helm-install: ## Install Helm chart
	@echo -e "$(GREEN)Installing Helm chart for $(ENV) environment...$(RESET)"
	@helm upgrade --install xorb-$(ENV) gitops/helm/xorb-core \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--values gitops/helm/xorb-core/values.yaml \
		--values gitops/helm/xorb-core/values-$(ENV).yaml \
		--set global.xorb.environment=$(ENV)
	@echo -e "$(GREEN)Helm chart installed successfully!$(RESET)"

.PHONY: helm-uninstall
helm-uninstall: ## Uninstall Helm chart
	@echo -e "$(YELLOW)Uninstalling Helm chart for $(ENV) environment...$(RESET)"
	@helm uninstall xorb-$(ENV) -n $(NAMESPACE)
	@echo -e "$(GREEN)Helm chart uninstalled$(RESET)"

.PHONY: helm-upgrade
helm-upgrade: ## Upgrade Helm chart
	@echo -e "$(GREEN)Upgrading Helm chart for $(ENV) environment...$(RESET)"
	@helm upgrade xorb-$(ENV) gitops/helm/xorb-core \
		--namespace $(NAMESPACE) \
		--values gitops/helm/xorb-core/values.yaml \
		--values gitops/helm/xorb-core/values-$(ENV).yaml \
		--set global.xorb.environment=$(ENV)

.PHONY: helm-template
helm-template: ## Generate Helm templates
	@echo -e "$(GREEN)Generating Helm templates...$(RESET)"
	@helm template xorb-$(ENV) gitops/helm/xorb-core \
		--namespace $(NAMESPACE) \
		--values gitops/helm/xorb-core/values.yaml \
		--values gitops/helm/xorb-core/values-$(ENV).yaml \
		--set global.xorb.environment=$(ENV)

# =============================================================================
# ARGOCD & GITOPS
# =============================================================================

.PHONY: gitops-apply
gitops-apply: ## Apply ArgoCD ApplicationSet
	@echo -e "$(GREEN)Applying ArgoCD ApplicationSet...$(RESET)"
	@kubectl apply -f gitops/argocd/projects.yaml
	@kubectl apply -f gitops/argocd/applicationset.yaml
	@echo -e "$(GREEN)GitOps configuration applied$(RESET)"

.PHONY: gitops-status
gitops-status: ## Show ArgoCD application status
	@echo -e "$(GREEN)ArgoCD Application Status:$(RESET)"
	@argocd app list || echo "ArgoCD CLI not available or not connected"

.PHONY: gitops-sync
gitops-sync: ## Sync ArgoCD applications
	@echo -e "$(GREEN)Syncing ArgoCD applications...$(RESET)"
	@argocd app sync xorb-core-$(ENV) || echo "ArgoCD CLI not available"
	@argocd app sync xorb-infra-$(ENV) || echo "ArgoCD CLI not available"

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

.PHONY: metrics
metrics: ## Show Prometheus metrics from API
	@echo -e "$(GREEN)Fetching API Prometheus metrics...$(RESET)"
	@curl -sf http://localhost:8000/metrics | grep http_requests_total || echo "API metrics endpoint not available"
	@echo ""
	@echo -e "$(GREEN)Fetching Worker Prometheus metrics...$(RESET)"
	@curl -sf http://localhost:9001/metrics | grep xorb_ | head -10 || echo "Worker metrics endpoint not available"

.PHONY: health
health: ## Check health of all services
	@echo -e "$(GREEN)Checking service health...$(RESET)"
	@echo -e "$(YELLOW)API Health:$(RESET)"
	@curl -s http://localhost:8000/health || echo "API not available"
	@echo -e "$(YELLOW)Worker Health:$(RESET)"
	@curl -s http://localhost:9090/health || echo "Worker not available"
	@echo -e "$(YELLOW)Orchestrator Health:$(RESET)"
	@curl -s http://localhost:8080/health || echo "Orchestrator not available"

.PHONY: monitoring-stack
monitoring-stack: ## Deploy monitoring stack
	@echo -e "$(GREEN)Deploying monitoring stack...$(RESET)"
	@kubectl apply -f gitops/monitoring/service-monitors.yaml
	@kubectl apply -f gitops/monitoring/prometheus-rules.yaml
	@echo -e "$(GREEN)Monitoring stack deployed$(RESET)"

# =============================================================================
# LOAD TESTING & PERFORMANCE
# =============================================================================

.PHONY: load-test
load-test: ## Run load tests
	@echo -e "$(GREEN)Running load tests...$(RESET)"
	@if [ -f load-test.js ]; then \
		k6 run load-test.js; \
	else \
		echo "Load test script not found"; \
	fi

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo -e "$(GREEN)Running performance benchmarks...$(RESET)"
	@pytest tests/performance/ -v --benchmark-only || echo "No performance tests found"

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo -e "$(GREEN)Running database migrations...$(RESET)"
	@if [ -f alembic.ini ]; then \
		alembic upgrade head; \
	elif [ -f manage.py ]; then \
		python manage.py migrate; \
	else \
		echo "No migration tool found"; \
	fi

.PHONY: db-seed
db-seed: ## Seed database with test data
	@echo -e "$(GREEN)Seeding database with test data...$(RESET)"
	@python scripts/seed_database.py 2>/dev/null || echo "Seed script not found"

.PHONY: db-backup
db-backup: ## Backup database
	@echo -e "$(GREEN)Creating database backup...$(RESET)"
	@mkdir -p backups
	@pg_dump $(DATABASE_URL) > backups/xorb-$(ENV)-$(shell date +%Y%m%d-%H%M%S).sql
	@echo -e "$(GREEN)Database backup created$(RESET)"

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

.PHONY: shell
shell: ## Open a shell in the API container
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api /bin/bash

.PHONY: python-shell
python-shell: ## Open a Python shell with project imports
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api python -c "\
import sys; sys.path.append('/app'); \
from packages.xorb_core.xorb_core import *; \
print('Xorb 2.0 Python shell ready!'); \
import IPython; IPython.start_ipython(argv=[])\
" 2>/dev/null || python -c "\
import sys; sys.path.append('.'); \
from packages.xorb_core.xorb_core import *; \
print('Xorb 2.0 Python shell ready!')\
"

.PHONY: agent-discovery
agent-discovery: ## Test agent discovery
	@echo -e "$(GREEN)Testing agent discovery...$(RESET)"
	@python -c "\
import asyncio; \
from packages.xorb_core.xorb_core.orchestration.enhanced_orchestrator import AgentRegistry; \
async def test(): \
    registry = AgentRegistry(); \
    agents = await registry.discover_agents(); \
    print(f'Discovered {len(agents)} agents: {list(agents)}'); \
asyncio.run(test())\
"

.PHONY: config-validate
config-validate: ## Validate configuration files
	@echo -e "$(GREEN)Validating configuration files...$(RESET)"
	@echo -e "$(YELLOW)Validating YAML files...$(RESET)"
	@find . -name "*.yaml" -o -name "*.yml" | xargs $(VENV_DIR)/bin/yamllint || echo "yamllint not available"
	@echo -e "$(YELLOW)Validating JSON files...$(RESET)"
	@find . -name "*.json" -print0 | xargs -0 -n1 $(VENV_DIR)/bin/python -m json.tool > /dev/null || echo "JSON validation failed"

.PHONY: docs
docs: ## Generate documentation
	@echo -e "$(GREEN)Generating documentation...$(RESET)"
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs build; \
	elif command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/ docs/_build/; \
	else \
		echo "No documentation generator found"; \
	fi

# =============================================================================
# PRODUCTION DEPLOYMENT HARDENING
# =============================================================================

.PHONY: prod-setup-secrets
prod-setup-secrets: ## Setup production secrets
	@echo -e "$(GREEN)Setting up production secrets...$(RESET)"
	@mkdir -p $(SECRETS_DIR)
	@chmod 700 $(SECRETS_DIR)
	@if [ ! -f $(SECRETS_DIR)/postgres_password ]; then \
		openssl rand -base64 32 > $(SECRETS_DIR)/postgres_password; \
		echo -e "$(YELLOW)Generated PostgreSQL password$(RESET)"; \
	fi
	@if [ ! -f $(SECRETS_DIR)/nvidia_api_key ]; then \
		echo "REPLACE_WITH_YOUR_NVIDIA_API_KEY" > $(SECRETS_DIR)/nvidia_api_key; \
		echo -e "$(RED)Please update $(SECRETS_DIR)/nvidia_api_key with your actual API key$(RESET)"; \
	fi
	@chmod 600 $(SECRETS_DIR)/*
	@echo -e "$(GREEN)Secrets setup complete$(RESET)"

.PHONY: prod-build
prod-build: ## Build hardened production images
	@echo -e "$(GREEN)Building hardened production images...$(RESET)"
	@docker build -f Dockerfile.api.hardened -t ghcr.io/xorb/xorb-api:$(VERSION) .
	@docker build -f Dockerfile.worker.hardened -t ghcr.io/xorb/xorb-worker:$(VERSION) .
	@docker build -f Dockerfile.embedding.hardened -t ghcr.io/xorb/xorb-embedding:$(VERSION) .
	@echo -e "$(GREEN)Production images built successfully$(RESET)"

.PHONY: prod-deploy
prod-deploy: prod-build ## Deploy production environment
	@echo -e "$(GREEN)Deploying hardened production environment...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE_PROD) up -d
	@echo -e "$(GREEN)Production deployment complete$(RESET)"
	@echo -e "$(YELLOW)Run 'make verify-deployment' to validate the deployment$(RESET)"

.PHONY: deploy-edge-worker
deploy-edge-worker: ## Deploy edge worker to Pi 5
	@echo -e "$(GREEN)Deploying edge worker to Pi 5...$(RESET)"
	@if [ -z "$(PI_HOST)" ]; then \
		echo -e "$(RED)Please set PI_HOST environment variable$(RESET)"; \
		exit 1; \
	fi
	@./scripts/deploy_pi.sh $(PI_HOST) $(VPS_IP)

.PHONY: hardening-check
hardening-check: ## Run security hardening verification
	@echo -e "$(GREEN)Running security hardening checks...$(RESET)"
	@python3 scripts/verify_hardened_deploy.py

# =============================================================================
# SECURITY & MONITORING
# =============================================================================

.PHONY: security-scan
security-scan: ## Run security vulnerability scan
	@echo -e "$(GREEN)Running security scans...$(RESET)"
	@echo -e "$(YELLOW)Scanning container images...$(RESET)"
	@if command -v trivy >/dev/null 2>&1; then \
		trivy image --severity CRITICAL,HIGH xorb-api:$(VERSION); \
		trivy image --severity CRITICAL,HIGH xorb-worker:$(VERSION); \
		trivy image --severity CRITICAL,HIGH xorb-embedding:$(VERSION); \
	else \
		echo -e "$(RED)Trivy not installed - install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin$(RESET)"; \
	fi

.PHONY: monitor-status
monitor-status: ## Check monitoring stack status
	@echo -e "$(GREEN)Checking monitoring stack status...$(RESET)"
	@echo -e "$(YELLOW)Prometheus:$(RESET)"
	@curl -s http://localhost:9090/-/healthy || echo "Prometheus not available"
	@echo -e "$(YELLOW)Grafana:$(RESET)"
	@curl -s http://localhost:3000/api/health || echo "Grafana not available"
	@echo -e "$(YELLOW)Tempo:$(RESET)"
	@curl -s http://localhost:3200/ready || echo "Tempo not available"

.PHONY: verify-deployment
verify-deployment: ## Verify hardened deployment
	@echo -e "$(GREEN)Verifying hardened deployment...$(RESET)"
	@python3 scripts/verify_hardened_deploy.py

.PHONY: prod-logs
prod-logs: ## View production logs
	@echo -e "$(GREEN)Viewing production logs...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE_PROD) logs -f --tail=100

.PHONY: prod-status
prod-status: ## Check production service status
	@echo -e "$(GREEN)Checking production service status...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE_PROD) ps
	@echo ""
	@echo -e "$(YELLOW)Service Health Checks:$(RESET)"
	@curl -s http://localhost:8000/health | jq '.' || echo "API health check failed"

.PHONY: prod-backup
prod-backup: ## Create production backup
	@echo -e "$(GREEN)Creating production backup...$(RESET)"
	@./scripts/backup_data.sh

.PHONY: prod-update
prod-update: ## Update production deployment
	@echo -e "$(GREEN)Updating production deployment...$(RESET)"
	@git pull origin main
	@docker-compose -f $(COMPOSE_FILE_PROD) up -d --remove-orphans
	@docker image prune -f
	@echo -e "$(GREEN)Production update complete$(RESET)"

.PHONY: prod-restart
prod-restart: ## Restart production services
	@echo -e "$(GREEN)Restarting production services...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE_PROD) restart

.PHONY: prod-stop
prod-stop: ## Stop production services
	@echo -e "$(YELLOW)Stopping production services...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE_PROD) down

.PHONY: prod-destroy
prod-destroy: ## Destroy production environment (WARNING: destructive)
	@echo -e "$(RED)WARNING: This will destroy all production data!$(RESET)"
	@echo -e "$(RED)Press Ctrl+C to cancel or wait 10 seconds to continue...$(RESET)"
	@sleep 10
	@docker-compose -f $(COMPOSE_FILE_PROD) down -v --remove-orphans
	@docker system prune -af --volumes

# =============================================================================
# EDGE COMPUTING
# =============================================================================

.PHONY: edge-status
edge-status: ## Check edge worker status
	@echo -e "$(GREEN)Checking edge worker status...$(RESET)"
	@if [ -z "$(PI_HOST)" ]; then \
		echo -e "$(RED)Please set PI_HOST environment variable$(RESET)"; \
		exit 1; \
	fi
	@ssh pi@$(PI_HOST) 'sudo systemctl status xorb-edge --no-pager'

.PHONY: edge-logs
edge-logs: ## View edge worker logs
	@echo -e "$(GREEN)Viewing edge worker logs...$(RESET)"
	@if [ -z "$(PI_HOST)" ]; then \
		echo -e "$(RED)Please set PI_HOST environment variable$(RESET)"; \
		exit 1; \
	fi
	@ssh pi@$(PI_HOST) 'sudo journalctl -u xorb-edge -f'

.PHONY: edge-restart
edge-restart: ## Restart edge worker
	@echo -e "$(GREEN)Restarting edge worker...$(RESET)"
	@if [ -z "$(PI_HOST)" ]; then \
		echo -e "$(RED)Please set PI_HOST environment variable$(RESET)"; \
		exit 1; \
	fi
	@ssh pi@$(PI_HOST) 'sudo systemctl restart xorb-edge'

# =============================================================================
# CI/CD PIPELINE COMMANDS
# =============================================================================

.PHONY: ci-test
ci-test: ## Run CI test suite
	@echo -e "$(GREEN)Running CI test suite...$(RESET)"
	@$(MAKE) format lint test security-scan

.PHONY: ci-build
ci-build: ## CI build and push
	@echo -e "$(GREEN)Running CI build...$(RESET)"
	@$(MAKE) build-prod push

.PHONY: ci-deploy
ci-deploy: ## CI deployment
	@echo -e "$(GREEN)Running CI deployment...$(RESET)"
	@$(MAKE) gitops-apply gitops-sync

# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

.PHONY: start
start: up ## Alias for 'up'

.PHONY: stop  
stop: down ## Alias for 'down'

.PHONY: deploy
deploy: helm-install ## Alias for 'helm-install'

.PHONY: status
status: k8s-status ## Alias for 'k8s-status'

.PHONY: watch
watch: ## Watch Kubernetes resources
	@watch kubectl get pods,svc,deploy -n $(NAMESPACE)

# =============================================================================
# DEVELOPMENT WORKFLOW SHORTCUTS
# =============================================================================

.PHONY: dev
dev: setup up ## Complete development setup and start
	@echo -e "$(GREEN)Development environment ready!$(RESET)"
	@echo -e "$(CYAN)API:$(RESET)         http://localhost:8000"
	@echo -e "$(CYAN)Temporal Web:$(RESET) http://localhost:8233"
	@echo -e "$(CYAN)Neo4j Browser:$(RESET) http://localhost:7474"

.PHONY: check
check: quality ## Run all code quality checks

.PHONY: full-deploy
full-deploy: ci-test ci-build ci-deploy ## Full CI/CD pipeline

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

.PHONY: debug
debug: ## Show debugging information
	@echo -e "$(CYAN)=== Debug Information ===$(RESET)"
	@$(MAKE) info
	@echo ""
	@echo -e "$(CYAN)=== Docker Status ===$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) ps || echo "Docker Compose not available"
	@echo ""
	@echo -e "$(CYAN)=== Kubernetes Status ===$(RESET)"
	@$(MAKE) k8s-status || echo "Kubernetes not available"
	@echo ""
	@echo -e "$(CYAN)=== Service Health ===$(RESET)"
	@$(MAKE) health

# =============================================================================
# XORBCTL CLI BUILD (Phase 4.3)
# =============================================================================

# Go variables for xorbctl CLI
BINARY_NAME = xorbctl
MAIN_PATH = ./cmd/xorbctl
BUILD_DIR = ./dist
GO_VERSION = $(shell go version | cut -d' ' -f3)

# xorbctl build flags
CLI_LDFLAGS = -s -w \
	-X github.com/xorb/xorbctl/internal/version.Version=$(VERSION) \
	-X github.com/xorb/xorbctl/internal/version.GitCommit=$(GIT_COMMIT) \
	-X github.com/xorb/xorbctl/internal/version.GitTag=$(GIT_TAG) \
	-X github.com/xorb/xorbctl/internal/version.BuildDate=$(BUILD_DATE) \
	-X github.com/xorb/xorbctl/internal/version.GoVersion=$(GO_VERSION)

.PHONY: build-cli
build-cli: ## Build xorbctl CLI binary
	@echo -e "$(GREEN)Building xorbctl CLI...$(RESET)"
	@mkdir -p $(BUILD_DIR)
	@cd $(MAIN_PATH) && \
		CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
		go build -ldflags "$(CLI_LDFLAGS)" -o ../../$(BUILD_DIR)/$(BINARY_NAME) .
	@echo -e "$(GREEN)xorbctl built: $(BUILD_DIR)/$(BINARY_NAME)$(RESET)"

.PHONY: build-cli-all
build-cli-all: ## Build xorbctl for all platforms
	@echo -e "$(GREEN)Building xorbctl for all platforms...$(RESET)"
	@mkdir -p $(BUILD_DIR)
	@cd $(MAIN_PATH) && \
	for os in linux darwin windows; do \
		for arch in amd64 arm64; do \
			if [ $$os = "windows" ]; then ext=".exe"; else ext=""; fi; \
			echo "Building for $$os/$$arch..."; \
			CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch \
			go build -ldflags "$(CLI_LDFLAGS)" -o ../../$(BUILD_DIR)/$(BINARY_NAME)-$$os-$$arch$$ext .; \
		done; \
	done
	@echo -e "$(GREEN)Cross-platform builds completed$(RESET)"

.PHONY: install-cli
install-cli: build-cli ## Install xorbctl to local system
	@echo -e "$(GREEN)Installing xorbctl...$(RESET)"
	@sudo cp $(BUILD_DIR)/$(BINARY_NAME) /usr/local/bin/
	@sudo chmod +x /usr/local/bin/$(BINARY_NAME)
	@echo -e "$(GREEN)xorbctl installed to /usr/local/bin/$(RESET)"

.PHONY: test-cli
test-cli: ## Test xorbctl CLI
	@echo -e "$(GREEN)Testing xorbctl CLI...$(RESET)"
	@cd $(MAIN_PATH) && go test ./... -v
	@echo -e "$(GREEN)CLI tests completed$(RESET)"

.PHONY: cli-deps
cli-deps: ## Install Go dependencies for xorbctl
	@echo -e "$(GREEN)Installing Go dependencies for xorbctl...$(RESET)"
	@cd $(MAIN_PATH) && go mod tidy && go mod download
	@echo -e "$(GREEN)Go dependencies installed$(RESET)"

# =============================================================================
# BLUE-GREEN DEPLOYMENT COMMANDS
# =============================================================================

.PHONY: deploy-green
deploy-green: ## Deploy to green environment (blue-green strategy)
	@echo -e "$(GREEN)Deploying to green environment...$(RESET)"
	@./scripts/blue_green_deployment.sh deploy green

.PHONY: deploy-blue
deploy-blue: ## Deploy to blue environment (blue-green strategy)
	@echo -e "$(GREEN)Deploying to blue environment...$(RESET)"
	@./scripts/blue_green_deployment.sh deploy blue

.PHONY: switch-green
switch-green: ## Switch traffic to green environment
	@echo -e "$(GREEN)Switching traffic to green environment...$(RESET)"
	@./scripts/blue_green_deployment.sh deploy green

.PHONY: switch-blue
switch-blue: ## Switch traffic to blue environment
	@echo -e "$(GREEN)Switching traffic to blue environment...$(RESET)"
	@./scripts/blue_green_deployment.sh deploy blue

.PHONY: bg-deploy
bg-deploy: ## Deploy using automatic blue-green strategy
	@echo -e "$(GREEN)Starting blue-green deployment...$(RESET)"
	@./scripts/blue_green_deployment.sh deploy

.PHONY: bg-status
bg-status: ## Show blue-green deployment status
	@echo -e "$(GREEN)Checking blue-green deployment status...$(RESET)"
	@./scripts/blue_green_deployment.sh status

.PHONY: bg-rollback
bg-rollback: ## Rollback blue-green deployment
	@echo -e "$(YELLOW)Rolling back deployment...$(RESET)"
	@./scripts/blue_green_deployment.sh rollback

.PHONY: bg-cleanup-blue
bg-cleanup-blue: ## Cleanup blue deployment
	@echo -e "$(YELLOW)Cleaning up blue deployment...$(RESET)"
	@./scripts/blue_green_deployment.sh cleanup blue

.PHONY: bg-cleanup-green
bg-cleanup-green: ## Cleanup green deployment
	@echo -e "$(YELLOW)Cleaning up green deployment...$(RESET)"
	@./scripts/blue_green_deployment.sh cleanup green

.PHONY: zero-downtime-deploy
zero-downtime-deploy: ## Perform zero-downtime deployment with full validation
	@echo -e "$(GREEN)Starting zero-downtime deployment with full validation...$(RESET)"
	@$(MAKE) security-scan
	@$(MAKE) test
	@$(MAKE) bg-deploy
	@$(MAKE) verify-deployment

.PHONY: emergency-rollback
emergency-rollback: ## Emergency rollback with immediate traffic switch
	@echo -e "$(RED)EMERGENCY ROLLBACK - Switching traffic immediately...$(RESET)"
	@./scripts/blue_green_deployment.sh rollback
	@$(MAKE) bg-status

# =============================================================================
# DISTRIBUTED COORDINATION TESTING
# =============================================================================

.PHONY: test-coordination
test-coordination: ## Test distributed campaign coordination
	@echo -e "$(GREEN)Testing distributed coordination...$(RESET)"
	@pytest tests/test_distributed_coordination.py -v --tb=short

.PHONY: demo-coordination
demo-coordination: ## Run distributed coordination demonstration
	@echo -e "$(GREEN)Running distributed coordination demonstration...$(RESET)"
	@python scripts/demo_distributed_coordination.py

.PHONY: test-reporting
test-reporting: ## Test business intelligence and reporting
	@echo -e "$(GREEN)Testing business intelligence and reporting...$(RESET)"
	@python scripts/demo_business_intelligence.py

.PHONY: demo-reporting
demo-reporting: ## Run business intelligence demonstration
	@echo -e "$(GREEN)Running business intelligence demonstration...$(RESET)"
	@python scripts/demo_business_intelligence.py

.PHONY: test-stealth
test-stealth: ## Test advanced stealth and evasion techniques
	@echo -e "$(GREEN)Testing advanced stealth and evasion...$(RESET)"
	@pytest tests/test_advanced_evasion_agent.py -v --tb=short

.PHONY: demo-stealth
demo-stealth: ## Run advanced evasion demonstration
	@echo -e "$(GREEN)Running advanced evasion demonstration...$(RESET)"
	@python scripts/demo_advanced_evasion.py

.PHONY: test-ml
test-ml: ## Test advanced machine learning models
	@echo -e "$(GREEN)Testing advanced ML models...$(RESET)"
	@python scripts/test_ml_models.py