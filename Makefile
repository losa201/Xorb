.PHONY: security-scan precommit-install sanitize-history

precommit-install: ## Install pre-commit hooks
	@pre-commit install -f

security-scan: ## Run gitleaks scan locally
	@gitleaks detect --source . --no-banner --redact --config .gitleaks.toml || true

sanitize-history: ## Rewrite git history to remove secrets
	@bash tools/secrets/remediate_git_history.sh
