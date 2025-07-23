# Xorb 2.0 Development & Operations Makefile
# EPYC-optimized with GitOps, monitoring, and enhanced developer experience

SHELL := /bin/bash
.DEFAULT_GOAL := help

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project configuration
PROJECT_NAME := xorb
VERSION := 2.0.0
REGISTRY := registry.xorb.ai

# Environment detection
ENV ?= development
NAMESPACE := xorb-$(ENV)
KUBE_CONTEXT ?= $(shell kubectl config current-context)

# Build configuration
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%S)
BUILD_COMMIT := $(shell git rev-parse --short HEAD)
BUILD_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# Docker configuration
DOCKER_BUILDKIT := 1
DOCKER_COMPOSE_FILE := docker-compose.yml
ifeq ($(ENV),development)
	DOCKER_COMPOSE_FILE := docker-compose.yml:docker-compose.dev.yml
endif

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# =============================================================================
# HELP & INFORMATION
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo -e "$(CYAN)Xorb 2.0 - AI-Powered Security Intelligence Platform$(RESET)"
	@echo -e "$(CYAN)===============================================$(RESET)"
	@echo ""
	@echo -e "$(YELLOW)Environment:$(RESET) $(ENV)"
	@echo -e "$(YELLOW)Namespace:$(RESET)   $(NAMESPACE)"
	@echo -e "$(YELLOW)Context:$(RESET)     $(KUBE_CONTEXT)"
	@echo -e "$(YELLOW)Version:$(RESET)     $(VERSION)"
	@echo ""
	@echo -e "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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
	@echo -e "$(YELLOW)Docker:$(RESET)         $$(docker --version 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Kubectl:$(RESET)        $$(kubectl version --client --short 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Helm:$(RESET)           $$(helm version --short 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Python:$(RESET)         $$(python --version 2>/dev/null || echo 'Not available')"
	@echo -e "$(YELLOW)Poetry:$(RESET)         $$(poetry --version 2>/dev/null || echo 'Not available')"
	@echo ""
	@echo -e "$(CYAN)=== Kubernetes Context ===$(RESET)"
	@kubectl config get-contexts | grep -E "(CURRENT|$(KUBE_CONTEXT))" || echo "Context not available"

# =============================================================================
# DEVELOPMENT SETUP
# =============================================================================

.PHONY: setup
setup: ## Initial development environment setup
	@echo -e "$(GREEN)Setting up Xorb 2.0 development environment...$(RESET)"
	@if [ ! -f .env ]; then \
		echo -e "$(YELLOW)Creating .env from template...$(RESET)"; \
		cp .env.$(ENV) .env; \
	fi
	@if command -v poetry >/dev/null 2>&1; then \
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
	@DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker-compose -f docker-compose.yml build \
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
metrics: ## Show Prometheus metrics
	@echo -e "$(GREEN)Fetching Prometheus metrics...$(RESET)"
	@curl -s http://localhost:9090/metrics | grep xorb_ | head -20 || echo "Metrics endpoint not available"

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
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api python -c "
import sys; sys.path.append('/app'); 
from packages.xorb_core.xorb_core import *; 
print('Xorb 2.0 Python shell ready!'); 
import IPython; IPython.start_ipython(argv=[])
" 2>/dev/null || python -c "
import sys; sys.path.append('.'); 
from packages.xorb_core.xorb_core import *; 
print('Xorb 2.0 Python shell ready!')
"

.PHONY: agent-discovery
agent-discovery: ## Test agent discovery
	@echo -e "$(GREEN)Testing agent discovery...$(RESET)"
	@python -c "
import asyncio
from packages.xorb_core.xorb_core.orchestration.enhanced_orchestrator import AgentRegistry
async def test():
    registry = AgentRegistry()
    agents = await registry.discover_agents()
    print(f'Discovered {len(agents)} agents: {list(agents)}')
asyncio.run(test())
"

.PHONY: config-validate
config-validate: ## Validate configuration files
	@echo -e "$(GREEN)Validating configuration files...$(RESET)"
	@echo -e "$(YELLOW)Validating YAML files...$(RESET)"
	@find . -name "*.yaml" -o -name "*.yml" | xargs yamllint || echo "yamllint not available"
	@echo -e "$(YELLOW)Validating JSON files...$(RESET)"
	@find . -name "*.json" | xargs python -m json.tool > /dev/null || echo "JSON validation failed"

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