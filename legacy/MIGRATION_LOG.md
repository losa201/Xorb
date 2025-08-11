# Legacy Migration Log - Phase 1

## Files Moved to Legacy (Date: 2025-01-28)

### Duplicate Docker Configurations
- `docker-compose.development.yml` → `legacy/duplicate-configs/`
- `docker-compose.production.yml` → `legacy/duplicate-configs/`
- `docker-compose.yml` → `legacy/duplicate-configs/`
- `deploy/configs/docker-compose*.yml` → `legacy/duplicate-configs/`

### Deprecated Frontend Structures
- `homepage/` → `legacy/old-structures/`
- `homepage-de/` → `legacy/old-structures/`
- `xorb-cipher-vue/` → `legacy/deprecated-services/`

### Old Framework Attempts
- `xorbfw/` → `legacy/deprecated-services/`
- `ptaas-backend/` → `legacy/old-structures/`
- `workspace/` → `legacy/old-structures/`

### Infrastructure Consolidation
- `infra/` (old scattered structure) → `legacy/old-structures/`
- `src/` (old mixed structure) → `legacy/old-structures/`

### Build Artifacts and Temporary Files
- `runtime/` → `legacy/build-artifacts/`
- `logs/` → `legacy/build-artifacts/`
- `dist/` → `legacy/build-artifacts/`
- `venv/` → `legacy/build-artifacts/`
- `source/` → `legacy/build-artifacts/`

### Scattered Test Files
- `test_*.py` (root level) → `legacy/build-artifacts/`
- `conftest.py` (root level) → `legacy/build-artifacts/`

## Active Structure
- ✅ `services/` - New organized service structure
- ✅ `docs/` - Consolidated documentation
- ✅ `tools/` - Development and deployment tools
- ✅ `tests/` - Organized test suites
- ✅ `packages/` - Shared libraries
- ✅ `docker-compose.enterprise.yml` - New enterprise configuration

## Rationale
This migration maintains backward compatibility while cleaning up redundant structures and establishing a clear enterprise-grade organization.