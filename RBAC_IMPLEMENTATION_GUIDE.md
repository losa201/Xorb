# XORB RBAC Implementation Guide

## Overview

This guide covers the complete Role-Based Access Control (RBAC) system implemented for the XORB PTaaS platform. The RBAC system replaces all placeholder security decorators with a production-ready authorization framework that supports hierarchical roles, fine-grained permissions, and multi-tenant access control.

## Architecture

### Core Components

1. **RBAC Service** (`src/api/app/services/rbac_service.py`)
   - Central authorization engine
   - Permission checking and caching
   - Role and permission management
   - Audit logging integration

2. **Database Models** (`src/api/app/infrastructure/rbac_models.py`)
   - `RBACRole`: Role definitions with hierarchy
   - `RBACPermission`: Granular permission definitions  
   - `RBACUserRole`: User-to-role assignments (tenant-specific)
   - `RBACUserPermission`: Direct user permissions

3. **Dependencies** (`src/api/app/auth/rbac_dependencies.py`)
   - FastAPI dependencies for permission checking
   - Role-based dependencies
   - Decorators for route protection

4. **Middleware** (`src/api/app/middleware/rbac_middleware.py`)
   - Request-level RBAC context creation
   - Security header injection

### Database Schema

The RBAC system uses 5 core tables:

```sql
-- Core role definitions
rbac_roles (id, name, display_name, description, level, parent_role_id, ...)

-- Permission definitions  
rbac_permissions (id, name, display_name, resource, action, ...)

-- Role-to-permission mappings
rbac_role_permissions (role_id, permission_id, granted_at, ...)

-- User-to-role assignments (tenant-specific)
rbac_user_roles (user_id, role_id, tenant_id, expires_at, ...)

-- Direct user permissions
rbac_user_permissions (user_id, permission_id, tenant_id, expires_at, ...)
```

## System Roles

The RBAC system includes pre-defined system roles with hierarchical permissions:

### Role Hierarchy (by level)

1. **super_admin** (Level 100)
   - Full cross-tenant system access
   - All permissions granted

2. **tenant_admin** (Level 90)
   - Full access within tenant
   - User management, PTaaS operations, compliance

3. **security_manager** (Level 80)
   - Security operations and user management
   - PTaaS workflows, intelligence analysis

4. **security_analyst** (Level 70)
   - Security analysis and scanning
   - PTaaS scans, intelligence reading

5. **pentester** (Level 60)
   - Penetration testing operations
   - Scan creation and reporting

6. **compliance_officer** (Level 60)
   - Compliance monitoring and reporting
   - Audit access and compliance management

7. **auditor** (Level 50)
   - Read-only audit access
   - Evidence and compliance viewing

8. **viewer** (Level 30)
   - Basic read-only access
   - Limited evidence and scan viewing

9. **user** (Level 20)
   - Basic authenticated access
   - Minimal read permissions

## Permission System

### Permission Format

Permissions follow the format: `resource:action`

Examples:
- `user:create` - Create new users
- `ptaas:scan:create` - Create PTaaS scans
- `intelligence:analyze` - Analyze threat intelligence

### Permission Categories

1. **User Management**
   - `user:create`, `user:read`, `user:update`, `user:delete`
   - `user:manage_roles`

2. **Organization/Tenant**
   - `organization:create`, `organization:read`, `organization:update`, `organization:delete`

3. **PTaaS Operations**
   - `ptaas:scan:create`, `ptaas:scan:read`, `ptaas:scan:update`, `ptaas:scan:delete`
   - `ptaas:scan:cancel`, `ptaas:workflow:manage`

4. **Intelligence Operations**
   - `intelligence:read`, `intelligence:analyze`, `intelligence:manage`

5. **Agent Management**
   - `agent:read`, `agent:create`, `agent:update`, `agent:delete`, `agent:control`

6. **System Administration**
   - `system:admin`, `system:monitor`, `system:config`

7. **Audit & Compliance**
   - `audit:read`, `audit:export`
   - `compliance:read`, `compliance:manage`

8. **Evidence & Storage**
   - `evidence:read`, `evidence:write`, `evidence:delete`

9. **Jobs & Orchestration**
   - `job:read`, `job:create`, `job:cancel`, `job:priority`

10. **Telemetry & Metrics**
    - `telemetry:read`, `telemetry:write`

## Usage Examples

### 1. Protecting Routes with Permissions

```python
from fastapi import APIRouter, Depends
from src.api.app.auth.rbac_dependencies import require_permission

router = APIRouter()

@router.post("/scans")
async def create_scan(
    scan_data: ScanRequest,
    current_user = Depends(require_permission("ptaas:scan:create"))
):
    # Only users with ptaas:scan:create permission can access
    return await scan_service.create_scan(scan_data)
```

### 2. Role-Based Access

```python
from src.api.app.auth.rbac_dependencies import require_role

@router.get("/admin/users")
async def list_users(
    current_user = Depends(require_role("tenant_admin"))
):
    # Only tenant_admin or higher can access
    return await user_service.list_users()
```

### 3. Multiple Permission Requirements

```python
from src.api.app.auth.rbac_dependencies import require_permissions

@router.post("/compliance/scan")
async def compliance_scan(
    current_user = Depends(require_permissions([
        "ptaas:scan:create", 
        "compliance:read"
    ]))
):
    # Requires BOTH permissions
    return await compliance_service.start_scan()
```

### 4. Decorator-Based Protection

```python
from src.api.app.auth.rbac_dependencies import rbac_decorator

@rbac_decorator(permissions=["intelligence:analyze"])
async def analyze_threats(request: Request):
    # Route protected by decorator
    return await threat_service.analyze()

@rbac_decorator(roles=["security_analyst", "security_manager"], require_any_role=True)
async def security_operation(request: Request):
    # Requires any of the specified roles
    return await security_service.operate()
```

### 5. Programmatic Permission Checking

```python
from src.api.app.services.rbac_service import RBACService, RBACContext
from src.api.app.container import get_container

async def check_user_access(user_id: str, tenant_id: str, permission: str):
    rbac_service = get_container().get(RBACService)
    
    context = RBACContext(
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    result = await rbac_service.check_permission(context, permission)
    return result.granted
```

## Role Management

### Assigning Roles

```python
from src.api.app.services.rbac_service import RBACService

rbac_service = get_container().get(RBACService)

# Assign role to user in specific tenant
await rbac_service.assign_role(
    user_id=user_id,
    role_name="security_analyst",
    granted_by=admin_user_id,
    tenant_id=tenant_id
)

# Assign global role (across all tenants)
await rbac_service.assign_role(
    user_id=user_id,
    role_name="super_admin", 
    granted_by=super_admin_id,
    tenant_id=None
)
```

### Direct Permission Assignment

```python
# Grant specific permission directly
await rbac_service.assign_permission(
    user_id=user_id,
    permission_name="special:operation",
    granted_by=admin_user_id,
    tenant_id=tenant_id,
    expires_at=datetime.utcnow() + timedelta(days=30)
)
```

### Role Revocation

```python
# Revoke role from user
await rbac_service.revoke_role(
    user_id=user_id,
    role_name="security_analyst",
    revoked_by=admin_user_id,
    tenant_id=tenant_id
)
```

## Multi-Tenant Support

The RBAC system supports tenant-specific roles and permissions:

### Tenant-Specific Roles

```python
# User can have different roles in different tenants
await rbac_service.assign_role(user_id, "tenant_admin", admin_id, tenant_a)
await rbac_service.assign_role(user_id, "viewer", admin_id, tenant_b)

# Permission check respects tenant context
context_a = RBACContext(user_id=user_id, tenant_id=tenant_a)
context_b = RBACContext(user_id=user_id, tenant_id=tenant_b)

# Will be granted in tenant A, denied in tenant B
result_a = await rbac_service.check_permission(context_a, "user:create")
result_b = await rbac_service.check_permission(context_b, "user:create")
```

### Global Roles

```python
# Global roles (tenant_id=None) apply across all tenants
await rbac_service.assign_role(user_id, "super_admin", admin_id, tenant_id=None)

# Will be granted in any tenant context
result = await rbac_service.check_permission(context, "system:admin")
```

## Caching and Performance

### Built-in Caching

The RBAC service includes comprehensive caching:

- **User Permissions Cache**: TTL 5 minutes
- **Role Permissions Cache**: TTL 1 hour  
- **Permission Check Cache**: TTL 30 minutes

### Cache Keys

```
rbac:user_permissions:{user_id}:tenant:{tenant_id}
rbac:user:{user_id}:tenant:{tenant_id}:perm:{permission}
rbac:role_permissions:{role_id}
```

### Cache Invalidation

Cache is automatically invalidated when:
- User roles are assigned/revoked
- Direct permissions are assigned/revoked
- Role definitions are modified

## Integration with Existing Auth

### Authentication Flow

1. User authenticates via JWT (PR-004 implementation)
2. Auth middleware validates token and creates user context
3. RBAC middleware creates RBAC context from user context
4. Route dependencies check permissions via RBAC service
5. Request proceeds if authorized, returns 403 if denied

### User Claims Integration

```python
# UserClaims from JWT are automatically used in RBAC context
# No changes needed to existing authentication logic

from src.api.app.auth.dependencies import get_current_user
from src.api.app.auth.rbac_dependencies import require_permission

@router.get("/protected")
async def protected_endpoint(
    current_user = Depends(get_current_user),  # JWT auth
    authorized_user = Depends(require_permission("ptaas:scan:read"))  # RBAC
):
    # Both authentication and authorization required
    return {"user": current_user.username}
```

## Security Features

### 1. Principle of Least Privilege

- Users granted minimal permissions needed
- Role hierarchy enforces access levels
- Explicit permission checks required

### 2. Zero Trust Architecture

- Every request requires authentication
- Every operation requires authorization
- No default permissions granted

### 3. Audit Logging

```python
# All permission checks are logged
{
    "event_type": "permission_check",
    "user_id": "user-uuid",
    "tenant_id": "tenant-uuid", 
    "permission": "ptaas:scan:create",
    "granted": true,
    "reason": "User has role security_analyst",
    "source": "role",
    "ip_address": "192.168.1.100",
    "timestamp": "2025-01-11T10:30:00Z"
}
```

### 4. Time-Limited Permissions

```python
# Permissions can have expiration dates
await rbac_service.assign_permission(
    user_id=user_id,
    permission_name="emergency:access",
    granted_by=admin_id,
    expires_at=datetime.utcnow() + timedelta(hours=4)
)
```

### 5. Hierarchical Permission Inheritance

- Child roles inherit parent permissions
- Role levels determine precedence
- Prevents privilege escalation

## Error Handling

### Permission Denied Responses

```python
# HTTP 403 with detailed error message
{
    "detail": "Permission 'ptaas:scan:create' required. User lacks permission 'ptaas:scan:create'",
    "status_code": 403,
    "error_type": "authorization_error"
}
```

### Role Assignment Errors

```python
# Failed role assignment
{
    "error": "Role 'nonexistent_role' not found",
    "success": false
}
```

## Monitoring and Health Checks

### RBAC Service Health Check

```python
health = await rbac_service.health_check()
{
    "status": "healthy",
    "active_roles": 9,
    "active_permissions": 32,
    "cache_enabled": true,
    "timestamp": "2025-01-11T10:30:00Z"
}
```

### Performance Metrics

The RBAC system exposes metrics for:
- Permission check latency
- Cache hit rates
- Role assignment frequency
- Permission denial rates

## Migration from Placeholder System

### Automatic Migration

The new RBAC system is designed to be backward compatible:

1. **Import Replacement**: All imports automatically resolve to new implementations
2. **Decorator Compatibility**: Old decorator names work with new functionality
3. **Gradual Migration**: Can migrate routes one at a time

### Migration Steps

1. **Database Migration**: Run `alembic upgrade` to create RBAC tables
2. **Role Assignment**: Assign appropriate roles to existing users
3. **Permission Verification**: Test critical endpoints with new system
4. **Cache Warmup**: Pre-populate permission cache for active users

### Testing Migration

```python
# Verify user permissions after migration
user_id = "existing-user-id"
context = RBACContext(user_id=user_id, tenant_id=tenant_id)

# Check critical permissions
critical_permissions = [
    "ptaas:scan:create",
    "intelligence:read", 
    "evidence:read"
]

for permission in critical_permissions:
    result = await rbac_service.check_permission(context, permission)
    print(f"{permission}: {'✓' if result.granted else '✗'}")
```

## Best Practices

### 1. Permission Naming

- Use clear, descriptive names
- Follow `resource:action` format
- Group related permissions by resource

### 2. Role Design

- Create roles based on job functions
- Avoid overly broad permissions
- Use role hierarchy effectively

### 3. Tenant Isolation

- Always specify tenant context
- Avoid global permissions unless necessary
- Test cross-tenant access controls

### 4. Performance Optimization

- Leverage caching for frequent checks
- Batch permission checks when possible
- Monitor cache hit rates

### 5. Security Hardening

- Regularly audit role assignments
- Monitor failed permission checks
- Implement permission expiration for sensitive access

## Troubleshooting

### Common Issues

1. **Permission Denied Unexpectedly**
   ```python
   # Check user roles and permissions
   roles = await rbac_service.get_user_roles(user_id, tenant_id)
   permissions = await rbac_service._get_user_permissions(user_id, tenant_id)
   ```

2. **Cache Issues**
   ```python
   # Clear user cache manually
   await rbac_service._invalidate_user_cache(user_id)
   ```

3. **Role Assignment Failures**
   ```python
   # Verify role exists
   available_roles = await rbac_service.get_available_roles()
   ```

### Debug Logging

Enable debug logging for detailed RBAC operations:

```python
import logging
logging.getLogger('src.api.app.services.rbac_service').setLevel(logging.DEBUG)
```

## API Reference

### RBAC Service Methods

```python
class RBACService:
    async def check_permission(context: RBACContext, permission: str) -> PermissionCheck
    async def check_multiple_permissions(context: RBACContext, permissions: List[str]) -> Dict[str, PermissionCheck]
    async def assign_role(user_id: UUID, role_name: str, granted_by: UUID, tenant_id: UUID = None) -> bool
    async def revoke_role(user_id: UUID, role_name: str, revoked_by: UUID, tenant_id: UUID = None) -> bool
    async def assign_permission(user_id: UUID, permission_name: str, granted_by: UUID, tenant_id: UUID = None) -> bool
    async def get_user_roles(user_id: UUID, tenant_id: UUID = None) -> List[Dict[str, Any]]
    async def get_available_roles() -> List[Dict[str, Any]]
    async def get_available_permissions() -> List[Dict[str, Any]]
    async def health_check() -> Dict[str, Any]
```

### Dependencies

```python
# Permission-based
require_permission(permission: str) -> Callable
require_permissions(permissions: List[str]) -> Callable

# Role-based  
require_role(role: str) -> Callable
require_any_role(roles: List[str]) -> Callable

# Pre-defined shortcuts
require_admin() -> Callable
require_tenant_admin() -> Callable
require_security_analyst() -> Callable

# Advanced decorator
rbac_decorator(permissions: List[str] = None, roles: List[str] = None) -> Callable
```

This comprehensive RBAC system provides enterprise-grade authorization capabilities while maintaining compatibility with existing authentication infrastructure and ensuring security best practices throughout the PTaaS platform.