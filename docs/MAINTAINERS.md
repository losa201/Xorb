# XORB Platform - Maintainer's Guide

## Quick Start Development

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for UI development)
- Git with pre-commit hooks

### Initial Setup
```bash
# Clone and setup environment
git clone <repo-url>
cd Xorb
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.lock

# Install pre-commit hooks
pre-commit install

# Validate environment
python tools/scripts/validate_environment.py
```

## Development Workflows

### PTaaS Development
```bash
# Start PTaaS stack (NATS + API)
make ptaas-quickstart

# Run PTaaS end-to-end tests
make ptaas-e2e

# Stop PTaaS services
make ptaas-stop
```

### UI Development
```bash
# Start React development server
cd ui/homepage
npm install
npm run dev

# Access at http://localhost:3000
```

### API Development
```bash
# Start FastAPI development server
cd src/api
source ../../venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API docs at http://localhost:8000/docs
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests
pytest tests/security/     # Security tests

# With coverage
pytest --cov=src --cov-report=html
```

## Operations & Runbooks

### Incident Response
- **Primary Runbook**: `runbooks/RUNBOOK_INCIDENT_RESPONSE.md`
- **Rollback Procedures**: `runbooks/RUNBOOK_ROLLBACK.md`
- **Chaos Testing**: `docs/CHAOS_DRILLS.md`

### Monitoring
```bash
# Check service health
make health

# View certificates status
make cert-status

# Validate Prometheus rules
make ops-alerts-validate
```

### Runbook Access
```bash
# Quick runbook access
make ops-runbooks
```

## Release Process

### Version Tagging
1. Update `VERSION` file with semantic version (e.g., `v2025.08.1`)
2. Generate changelog:
   ```bash
   git log --oneline --since="last-release-date" > CHANGELOG_DRAFT.md
   ```
3. Create release notes in `docs/RELEASE_NOTES.md`
4. Tag and push:
   ```bash
   git tag -a v2025.08.1 -m "Release v2025.08.1"
   git push origin v2025.08.1
   ```

### Security Scanning
```bash
# Full security scan before release
make security-full

# Generate SBOM
make sbom

# Dependency audit
make dependency-audit
```

### Release Checklist
- [ ] All CI checks passing
- [ ] Security scan clean
- [ ] PTaaS E2E tests passing
- [ ] Documentation updated
- [ ] SBOM generated
- [ ] Version tagged
- [ ] Release notes published

## Architecture Guidelines

### ADR Compliance
- **ADR-001**: Canonical repository structure
- **ADR-002**: NATS-only messaging (NO Redis pub/sub)
- **ADR-003**: Authentication & authorization framework
- **ADR-004**: Monitoring & observability standards

### Code Standards
```bash
# Format code
make fmt

# Lint code
make lint

# Run repository doctor
make doctor
```

### Security Requirements
- All secrets via environment variables or Vault
- Rate limiting on all public endpoints
- Comprehensive audit logging
- Zero-trust security model

## Troubleshooting

### Common Issues

#### Pre-commit Hooks Failing
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run specific hook
pre-commit run bandit --all-files
```

#### PTaaS Connection Issues
```bash
# Check NATS connectivity
docker logs xorb-nats-1

# Verify API health
curl http://localhost:8000/api/v1/health
```

#### Certificate Issues
```bash
# Generate new certificates
make certs-generate

# Emergency certificate rotation
make emergency-rotation
```

### Debug Commands
```bash
# Repository health check
make doctor

# Service health dashboard
make health

# View service logs
docker-compose logs -f api
```

## Contribution Guidelines

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature development
- `hotfix/*`: Critical fixes

### Code Review
- All changes require PR review
- Security changes require 2+ approvals
- Breaking changes require architecture review

### ADR Process
1. Propose ADR in `docs/ADRs/`
2. Team discussion and review
3. Implementation with compliance checks
4. Validation and approval

## Emergency Procedures

### Security Incident
1. Follow `runbooks/RUNBOOK_INCIDENT_RESPONSE.md`
2. Rotate certificates: `make emergency-rotation`
3. Check audit logs: `docker logs xorb-api-1 | grep AUDIT`
4. Notify security team

### Service Outage
1. Check service health: `make health`
2. Review monitoring dashboards
3. Follow `runbooks/RUNBOOK_ROLLBACK.md` if needed
4. Post-incident review

### Certificate Expiry
```bash
# Check certificate status
make cert-status

# Emergency rotation
make emergency-rotation

# Validate new certificates
make validate
```

## Useful Make Targets

```bash
make help                    # Show all available targets
make ptaas-quickstart       # Quick PTaaS setup
make ptaas-e2e             # PTaaS end-to-end tests
make ops-runbooks          # Display runbook locations
make security-full         # Comprehensive security scan
make sbom                  # Generate SBOM
make audit                 # Security audit
make doctor                # Repository health check
make health                # Service health check
```

## Contact & Support

- **Repository Issues**: GitHub Issues
- **Security Issues**: security@domain.com
- **Documentation**: `docs/` directory
- **Runbooks**: `runbooks/` directory
