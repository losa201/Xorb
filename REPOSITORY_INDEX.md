# XORB Repository Index

**Last Updated:** 2025-08-15  
**Post-Deduplication Canonical Layout**

## Canonical Directory Structure

```
/root/Xorb/
├── src/                          # Python source code (CANONICAL)
│   ├── api/                      # FastAPI application
│   ├── orchestrator/             # Temporal orchestration service
│   ├── common/                   # Shared utilities (CANONICAL)
│   ├── xorb/                     # Core platform modules
│   └── services/                 # Background services
├── ui/                           # Web interfaces (CANONICAL)
│   └── homepage/                 # Main homepage with i18n
│       ├── index.html            # Language selector
│       └── i18n/
│           ├── en/               # English content
│           └── de/               # German content
├── services/                     # Microservices architecture
│   ├── ptaas/                    # PTaaS service
│   ├── xorb-core/               # Backend platform (with compat shims)
│   └── infrastructure/           # Shared infrastructure
├── packages/                     # Shared libraries (with compat shims)
├── proto/                        # Protocol buffers (CANONICAL)
│   ├── threat/v1/
│   ├── compliance/v1/
│   ├── discovery/v1/
│   ├── audit/v1/
│   └── vuln/v1/
├── tests/                        # Test suite (CANONICAL)
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # End-to-end tests
│   └── security/                 # Security tests
├── docs/                         # Documentation (CANONICAL)
│   ├── audit/                    # Audit reports
│   ├── reports/                  # Generated reports
│   └── QUICK_START.md            # Moved from root
├── tools/                        # Development tools
│   ├── repo_doctor.py            # Duplicate detection
│   ├── repo_audit/               # Analysis outputs
│   └── policies/                 # Repository policies
├── configs/                      # Configuration files
├── scripts/                      # Automation scripts
└── infra/                        # Infrastructure as code
```

## Consolidation Mapping (Post-Deduplication)

### Eliminated Duplicates

| Category | Original Locations | Canonical Location | Status |
|----------|-------------------|-------------------|---------|
| **API Layer** | `services/xorb-core/api/` | `src/api/` | ✅ Consolidated |
| **Orchestrator** | `services/xorb-core/orchestrator/` | `src/orchestrator/` | ✅ Consolidated |
| **Common Utils** | `packages/common/` | `src/common/` | ✅ Consolidated |
| **Unit Tests** | `src/api/tests/`, `services/xorb-core/api/tests/` | `tests/unit/` | ✅ Consolidated |
| **Integration Tests** | Scattered locations | `tests/integration/` | ✅ Consolidated |
| **UI Homepage** | `ui/homepage/homepage/`, `ui/homepage-de/` | `ui/homepage/i18n/` | ✅ Consolidated |
| **Documentation** | Root `*.md` files | `docs/` | ✅ Consolidated |

### Compatibility Shims (Temporary - Remove in P06)

| Deprecated Path | Canonical Path | Shim Location |
|----------------|----------------|---------------|
| `services.xorb-core.api.gateway` | `src.api.gateway` | `services/xorb-core/api/gateway.py` |
| `services.xorb-core.api.db_management` | `src.api.db_management` | `services/xorb-core/api/db_management.py` |
| `services.xorb-core.api.dependencies` | `src.api.dependencies` | `services/xorb-core/api/dependencies.py` |
| `services.xorb-core.orchestrator.main` | `src.orchestrator.main` | `services/xorb-core/orchestrator/main.py` |
| `packages.common.*` | `src.common.*` | `packages/common/__init__.py` |

## File Type Guidelines

### Python Code
- **Location:** `src/`
- **Package Structure:** Follow Python package conventions
- **Import Style:** Absolute imports from `src.`
- **Testing:** Unit tests in `tests/unit/`, integration in `tests/integration/`

### Web Interfaces  
- **Location:** `ui/`
- **Structure:** Feature-based directories
- **Internationalization:** Use `i18n/{lang}/` pattern
- **Assets:** Co-located with components

### Documentation
- **Location:** `docs/`
- **Structure:** Purpose-based organization
- **Reports:** Generated content in `docs/reports/`
- **API Docs:** Auto-generated in `docs/api/`

### Protocol Buffers
- **Location:** `proto/{domain}/v1/`
- **Versioning:** Semantic versioning in path
- **Generated Code:** Not committed to repository
- **Documentation:** Inline comments required

### Tests
- **Location:** `tests/{type}/`
- **Naming:** `test_*.py` pattern
- **Structure:** Mirror source structure
- **Fixtures:** Shared in `tests/fixtures/`

### Configuration
- **Location:** `configs/`
- **Environment:** Separate files per environment
- **Secrets:** Use environment variables or vault
- **Templates:** `.template` suffix for examples

## Quality Gates

### Duplicate Prevention
- **CI Check:** `.github/workflows/ci.yml` includes duplicate detection
- **Pre-commit:** Hooks in `src/api/.pre-commit-config.yaml`
- **Tool:** `tools/repo_audit/near_dup_check.py`
- **Policy:** Allowlist in `tools/policies/near_dup_allowlist.txt`

### Code Quality
- **Linting:** Ruff for Python, ESLint for TypeScript
- **Formatting:** Black for Python, Prettier for web
- **Type Checking:** mypy for Python, TypeScript compiler
- **Security:** Bandit, Gitleaks, ADR compliance checks

### Testing Requirements
- **Coverage:** Minimum 75% for new code
- **Types:** Unit, integration, e2e, security tests required
- **Performance:** Smoke tests in CI, full perf tests on demand
- **Contracts:** Protocol buffer compatibility checking

## Development Workflow

### Adding New Code
1. **Determine canonical location** using guidelines above
2. **Check for existing similar code** using repository doctor
3. **Follow established patterns** in similar modules
4. **Add tests** in appropriate test directory
5. **Update documentation** if adding new features

### Import Conventions
```python
# Preferred: Canonical imports
from src.api.gateway import APIGateway
from src.common.config import Config

# Deprecated: Legacy imports (with warnings)
from services.xorb_core.api.gateway import APIGateway  # Will warn
from packages.common.config import Config             # Will warn
```

### Migration Path (P06)
1. **Update imports** to canonical paths
2. **Remove compatibility shims** 
3. **Update documentation** and examples
4. **Remove deprecation warnings**

## Tools & Automation

### Repository Maintenance
- **`tools/repo_doctor.py`** - Comprehensive duplicate analysis
- **`tools/repo_audit/near_dup_check.py`** - Real-time duplicate prevention
- **`tools/policies/near_dup_allowlist.txt`** - Legitimate duplicate allowlist

### Build & Deploy
- **`make doctor`** - Run repository health check
- **`make test`** - Run all test suites
- **`make lint`** - Code quality checks
- **`make security-scan`** - Security analysis

### Monitoring
- **Daily:** Automated duplicate detection in CI
- **Weekly:** Repository health reports
- **Monthly:** Canonical layout compliance review

---

**Note:** This index reflects the post-deduplication state. All 98 duplicate files have been eliminated and canonical locations established. Compatibility shims provide backward compatibility during migration period.