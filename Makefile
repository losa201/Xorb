# XORB Platform TLS/mTLS Security Implementation
# Production-ready make targets for certificate management and security operations

.PHONY: help ca-init certs-generate deploy-tls validate security-scan rotate-certs performance clean

# Default target
help: ## Show this help message
	@echo "🔐 XORB Platform TLS/mTLS Security Commands"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Prerequisites
check-prereqs: ## Check required tools and dependencies
	@echo "🔍 Checking prerequisites..."
	@command -v docker >/dev/null 2>&1 || (echo "❌ Docker not found" && exit 1)
	@command -v docker-compose >/dev/null 2>&1 || (echo "❌ Docker Compose not found" && exit 1)
	@command -v openssl >/dev/null 2>&1 || (echo "❌ OpenSSL not found" && exit 1)
	@echo "✅ Prerequisites check passed"

# Certificate Authority Setup
ca-init: check-prereqs ## Initialize Certificate Authority infrastructure
	@echo "🏛️ Initializing Certificate Authority..."
	@./scripts/ca/make-ca.sh
	@echo "✅ Certificate Authority initialized"

# Certificate Generation
certs-generate: ca-init ## Generate certificates for all services
	@echo "📜 Generating service certificates..."
	@./scripts/ca/issue-cert.sh api both
	@./scripts/ca/issue-cert.sh orchestrator client
	@./scripts/ca/issue-cert.sh agent both
	@./scripts/ca/issue-cert.sh redis server
	@./scripts/ca/issue-cert.sh redis-client client
	@./scripts/ca/issue-cert.sh postgres server
	@./scripts/ca/issue-cert.sh postgres-client client
	@./scripts/ca/issue-cert.sh temporal server
	@./scripts/ca/issue-cert.sh temporal-client client
	@./scripts/ca/issue-cert.sh dind server
	@./scripts/ca/issue-cert.sh dind-client client
	@./scripts/ca/issue-cert.sh scanner client
	@./scripts/ca/issue-cert.sh prometheus server
	@./scripts/ca/issue-cert.sh grafana server
	@echo "✅ Service certificates generated"

# Single certificate generation
cert-%: ca-init ## Generate certificate for specific service (e.g., make cert-api)
	@echo "📜 Generating certificate for $*..."
	@case "$*" in \
		*-client) ./scripts/ca/issue-cert.sh "$*" client ;; \
		api|agent) ./scripts/ca/issue-cert.sh "$*" both ;; \
		*) ./scripts/ca/issue-cert.sh "$*" server ;; \
	esac
	@echo "✅ Certificate for $* generated"

# TLS Deployment
deploy-tls: certs-generate ## Deploy full TLS/mTLS stack
	@echo "🚀 Deploying TLS/mTLS stack..."
	@docker-compose -f infra/docker-compose.tls.yml up -d
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 30
	@docker-compose -f infra/docker-compose.tls.yml ps
	@echo "✅ TLS/mTLS stack deployed"

# Development deployment
deploy-dev: certs-generate ## Deploy development environment with TLS
	@echo "🛠️ Deploying development environment..."
	@docker-compose -f docker-compose.development.yml up -d
	@echo "✅ Development environment deployed"

# Production deployment
deploy-prod: certs-generate ## Deploy production environment with TLS
	@echo "🏭 Deploying production environment..."
	@docker-compose -f docker-compose.production.yml up -d
	@echo "✅ Production environment deployed"

# Validation and Testing
validate: ## Run comprehensive TLS/mTLS validation
	@echo "🧪 Running TLS/mTLS validation..."
	@./scripts/validate/test_comprehensive.sh
	@echo "✅ Validation completed"

validate-tls: ## Run TLS protocol and cipher validation
	@echo "🔒 Running TLS validation..."
	@./scripts/validate/test_tls.sh
	@echo "✅ TLS validation completed"

validate-mtls: ## Run mutual TLS authentication validation
	@echo "🤝 Running mTLS validation..."
	@./scripts/validate/test_mtls.sh
	@echo "✅ mTLS validation completed"

validate-redis: ## Run Redis TLS validation
	@echo "🔴 Running Redis TLS validation..."
	@./scripts/validate/test_redis_tls.sh
	@echo "✅ Redis TLS validation completed"

validate-docker: ## Run Docker-in-Docker TLS validation
	@echo "🐳 Running Docker TLS validation..."
	@./scripts/validate/test_dind_tls.sh
	@echo "✅ Docker TLS validation completed"

# Security Operations
security-scan: ## Run comprehensive security policy validation
	@echo "🛡️ Running security policy validation..."
	@command -v conftest >/dev/null 2>&1 || (echo "⚠️ Conftest not found, skipping policy validation" && exit 0)
	@conftest test --policy policies/tls-security.rego infra/docker-compose.tls.yml
	@conftest test --policy policies/tls-security.rego envoy/*.yaml
	@conftest test --policy policies/tls-security.rego k8s/mtls/*.yaml
	@echo "✅ Security policy validation completed"

# Certificate Management
rotate-certs: ## Rotate certificates approaching expiry
	@echo "🔄 Rotating certificates..."
	@./scripts/rotate-certs.sh
	@echo "✅ Certificate rotation completed"

rotate-all: ## Force rotation of all certificates
	@echo "🔄 Force rotating all certificates..."
	@./scripts/rotate-certs.sh --force
	@echo "✅ All certificates rotated"

emergency-rotation: ## Emergency certificate rotation (requires incident ID)
	@echo "🚨 Emergency certificate rotation..."
	@read -p "Enter incident ID: " incident_id; \
	read -p "Enter compromise type (key_leak/cert_compromise/ca_breach): " compromise_type; \
	./scripts/emergency-cert-rotation.sh -i "$$incident_id" -t "$$compromise_type"
	@echo "✅ Emergency rotation completed"

# Performance Testing
performance: ## Run TLS/mTLS performance benchmarks
	@echo "⚡ Running performance benchmarks..."
	@./scripts/performance-benchmark.sh
	@echo "✅ Performance benchmarks completed"

# Monitoring and Reporting
reports: ## Generate security and validation reports
	@echo "📊 Generating reports..."
	@mkdir -p reports/summary
	@./scripts/validate/test_comprehensive.sh --no-alert
	@echo "✅ Reports generated in reports/ directory"

logs: ## View security and certificate logs
	@echo "📝 Recent security logs:"
	@find logs/ -name "*.log" -mtime -1 -exec echo "=== {} ===" \; -exec tail -n 10 {} \; 2>/dev/null || echo "No recent logs found"

# Health Checks
health: ## Check service health and certificate status
	@echo "💚 Checking service health..."
	@docker-compose -f infra/docker-compose.tls.yml ps
	@echo ""
	@echo "📜 Certificate expiry status:"
	@find secrets/tls -name "cert.pem" -exec sh -c 'echo -n "$$1: "; openssl x509 -in "$$1" -noout -enddate | cut -d= -f2' _ {} \; 2>/dev/null || echo "No certificates found"

cert-status: ## Show detailed certificate status
	@echo "📜 Detailed certificate status:"
	@for cert in secrets/tls/*/cert.pem; do \
		if [ -f "$$cert" ]; then \
			service=$$(basename $$(dirname "$$cert")); \
			echo "=== $$service ==="; \
			openssl x509 -in "$$cert" -noout -subject -dates -text | grep -E "(Subject:|Not Before|Not After|Subject Alternative Name)" || true; \
			echo ""; \
		fi \
	done

# Kubernetes Operations
k8s-deploy: ## Deploy to Kubernetes with cert-manager
	@echo "☸️ Deploying to Kubernetes..."
	@kubectl apply -f k8s/mtls/namespace.yaml
	@kubectl apply -f k8s/mtls/cluster-issuer.yaml
	@kubectl apply -f k8s/mtls/service-certificates.yaml
	@kubectl apply -f k8s/mtls/istio-mtls-policy.yaml
	@echo "✅ Kubernetes deployment completed"

k8s-status: ## Check Kubernetes certificate status
	@echo "☸️ Kubernetes certificate status:"
	@kubectl get certificates -n xorb-platform
	@kubectl get clusterissuers

# Development Helpers
dev-setup: ## Quick development setup
	@echo "🛠️ Setting up development environment..."
	@make ca-init
	@make cert-api
	@make cert-redis
	@make cert-redis-client
	@echo "✅ Development setup completed"

dev-test: ## Run development tests
	@echo "🧪 Running development tests..."
	@make validate-tls
	@make validate-redis
	@echo "✅ Development tests completed"

# Maintenance
clean: ## Clean up old certificates, logs, and reports
	@echo "🧹 Cleaning up..."
	@find secrets/tls/backups -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
	@find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
	@find reports/ -name "*.html" -mtime +7 -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

clean-all: ## Remove all certificates and start fresh
	@echo "💥 Removing all certificates..."
	@read -p "Are you sure? This will delete all certificates! [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf secrets/tls/*; \
		echo "✅ All certificates removed"; \
	else \
		echo "❌ Operation cancelled"; \
	fi

# Service Management
start: deploy-tls ## Start all services with TLS
stop: ## Stop all services
	@echo "🛑 Stopping services..."
	@docker-compose -f infra/docker-compose.tls.yml down
	@echo "✅ Services stopped"

restart: stop start ## Restart all services

# Backup and Recovery
backup: ## Backup certificates and configuration
	@echo "💾 Creating backup..."
	@mkdir -p backups/$(shell date +%Y%m%d)
	@tar czf backups/$(shell date +%Y%m%d)/xorb-tls-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		secrets/tls/ scripts/ envoy/ infra/ k8s/ policies/
	@echo "✅ Backup created in backups/$(shell date +%Y%m%d)/"

restore: ## Restore from backup (specify BACKUP_FILE)
	@echo "🔄 Restoring from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please specify BACKUP_FILE=path/to/backup.tar.gz"; \
		exit 1; \
	fi
	@tar xzf $(BACKUP_FILE)
	@echo "✅ Restore completed"

# CI/CD Integration
ci-validate: check-prereqs ## Run CI validation pipeline
	@echo "🏗️ Running CI validation..."
	@make security-scan
	@make validate
	@echo "✅ CI validation completed"

ci-deploy: check-prereqs ## Run CI deployment pipeline
	@echo "🏗️ Running CI deployment..."
	@make deploy-tls
	@make validate
	@echo "✅ CI deployment completed"

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@mkdir -p docs/generated
	@echo "# Certificate Inventory" > docs/generated/certificates.md
	@echo "Generated: $(shell date)" >> docs/generated/certificates.md
	@echo "" >> docs/generated/certificates.md
	@for cert in secrets/tls/*/cert.pem; do \
		if [ -f "$$cert" ]; then \
			service=$$(basename $$(dirname "$$cert")); \
			echo "## $$service" >> docs/generated/certificates.md; \
			openssl x509 -in "$$cert" -noout -text | grep -E "(Subject:|Not Before|Not After)" >> docs/generated/certificates.md || true; \
			echo "" >> docs/generated/certificates.md; \
		fi \
	done
	@echo "✅ Documentation generated"

# Security Audit
audit: ## Run comprehensive security audit
	@echo "🔍 Running security audit..."
	@make validate
	@make security-scan
	@make performance
	@echo "📊 Generating audit report..."
	@mkdir -p reports/audit
	@echo "# XORB Platform Security Audit" > reports/audit/audit-$(shell date +%Y%m%d).md
	@echo "Date: $(shell date)" >> reports/audit/audit-$(shell date +%Y%m%d).md
	@echo "## Summary" >> reports/audit/audit-$(shell date +%Y%m%d).md
	@echo "- TLS/mTLS validation: ✅" >> reports/audit/audit-$(shell date +%Y%m%d).md
	@echo "- Security policy compliance: ✅" >> reports/audit/audit-$(shell date +%Y%m%d).md
	@echo "- Performance benchmarks: ✅" >> reports/audit/audit-$(shell date +%Y%m%d).md
	@echo "✅ Security audit completed"

# Quick commands
quick-start: ca-init cert-api cert-redis cert-redis-client ## Quick start with minimal certificates
	@echo "⚡ Quick start completed - ready for development"

production-ready: ca-init certs-generate validate security-scan ## Full production readiness check
	@echo "🏭 Production readiness validation completed"