"""
RBAC Integration Tests
End-to-end tests for Role-Based Access Control system
"""

import pytest
import asyncio
from uuid import uuid4
from datetime import datetime
from httpx import AsyncClient
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app.main import app
from src.api.app.container import get_container
from src.api.app.services.rbac_service import RBACService
from src.api.app.infrastructure.rbac_models import RBACRole, RBACPermission, RBACUserRole
from src.api.app.auth.models import UserClaims


@pytest.fixture
async def async_client():
    """Async HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def rbac_service():
    """Get RBAC service from container"""
    container = get_container()
    return container.get(RBACService)


@pytest.fixture
def test_user():
    """Test user with basic credentials"""
    return {
        "user_id": str(uuid4()),
        "username": "test_user",
        "email": "test@example.com",
        "tenant_id": str(uuid4())
    }


@pytest.fixture
def admin_user():
    """Admin user with elevated privileges"""
    return {
        "user_id": str(uuid4()),
        "username": "admin_user", 
        "email": "admin@example.com",
        "tenant_id": str(uuid4())
    }


@pytest.fixture
async def authenticated_headers(test_user):
    """Generate authenticated headers for test user"""
    # This would use your actual JWT generation logic
    # For testing, we'll mock the token
    token = "test_jwt_token_for_user"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_headers(admin_user):
    """Generate authenticated headers for admin user"""
    token = "test_jwt_token_for_admin"
    return {"Authorization": f"Bearer {token}"}


class TestRBACEndToEnd:
    """End-to-end RBAC integration tests"""
    
    @pytest.mark.asyncio
    async def test_role_assignment_workflow(self, rbac_service, test_user, admin_user):
        """Test complete role assignment workflow"""
        user_id = test_user["user_id"]
        admin_id = admin_user["user_id"]
        
        # Admin assigns security_analyst role to user
        result = await rbac_service.assign_role(
            user_id=user_id,
            role_name="security_analyst", 
            granted_by=admin_id
        )
        assert result is True
        
        # Verify user now has the role
        user_roles = await rbac_service.get_user_roles(user_id)
        role_names = [role['name'] for role in user_roles]
        assert "security_analyst" in role_names
        
        # Verify user can perform security analyst actions
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        # Check permissions that security_analyst should have
        scan_permission = await rbac_service.check_permission(context, "ptaas:scan:create")
        assert scan_permission.granted is True
        
        intel_permission = await rbac_service.check_permission(context, "intelligence:read")
        assert intel_permission.granted is True
        
        # Check permissions that security_analyst should NOT have
        admin_permission = await rbac_service.check_permission(context, "system:admin")
        assert admin_permission.granted is False
        
        # Revoke the role
        revoke_result = await rbac_service.revoke_role(
            user_id=user_id,
            role_name="security_analyst",
            revoked_by=admin_id
        )
        assert revoke_result is True
        
        # Verify permissions are revoked
        scan_permission_after = await rbac_service.check_permission(context, "ptaas:scan:create")
        assert scan_permission_after.granted is False
    
    @pytest.mark.asyncio
    async def test_direct_permission_assignment(self, rbac_service, test_user, admin_user):
        """Test direct permission assignment workflow"""
        user_id = test_user["user_id"]
        admin_id = admin_user["user_id"]
        
        # Assign specific permission directly
        result = await rbac_service.assign_permission(
            user_id=user_id,
            permission_name="audit:read",
            granted_by=admin_id
        )
        assert result is True
        
        # Verify user has the permission
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        permission_check = await rbac_service.check_permission(context, "audit:read")
        assert permission_check.granted is True
        assert permission_check.source == "direct"
    
    @pytest.mark.asyncio
    async def test_hierarchical_permissions(self, rbac_service, test_user, admin_user):
        """Test hierarchical role permissions"""
        user_id = test_user["user_id"] 
        admin_id = admin_user["user_id"]
        
        # Assign tenant_admin role (high level)
        await rbac_service.assign_role(
            user_id=user_id,
            role_name="tenant_admin",
            granted_by=admin_id
        )
        
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        # Verify tenant_admin has multiple high-level permissions
        permissions_to_check = [
            "user:read",
            "user:update", 
            "ptaas:scan:create",
            "ptaas:scan:delete",
            "intelligence:analyze",
            "compliance:read"
        ]
        
        for permission in permissions_to_check:
            result = await rbac_service.check_permission(context, permission)
            assert result.granted is True, f"tenant_admin should have {permission}"
        
        # Verify tenant_admin does NOT have super_admin permissions
        super_admin_permissions = [
            "system:admin",
            "organization:create"
        ]
        
        for permission in super_admin_permissions:
            result = await rbac_service.check_permission(context, permission)
            assert result.granted is False, f"tenant_admin should NOT have {permission}"
    
    @pytest.mark.asyncio 
    async def test_tenant_isolation(self, rbac_service, test_user, admin_user):
        """Test tenant-specific role isolation"""
        user_id = test_user["user_id"]
        admin_id = admin_user["user_id"]
        
        tenant_a = uuid4()
        tenant_b = uuid4()
        
        # Assign role in tenant A only
        await rbac_service.assign_role(
            user_id=user_id,
            role_name="security_analyst",
            granted_by=admin_id,
            tenant_id=tenant_a
        )
        
        # Check permissions in tenant A
        context_a = RBACContext(user_id=user_id, tenant_id=tenant_a)
        result_a = await rbac_service.check_permission(context_a, "ptaas:scan:create")
        assert result_a.granted is True
        
        # Check permissions in tenant B (should be denied)
        context_b = RBACContext(user_id=user_id, tenant_id=tenant_b)
        result_b = await rbac_service.check_permission(context_b, "ptaas:scan:create")
        assert result_b.granted is False
    
    @pytest.mark.asyncio
    async def test_multiple_roles_cumulative_permissions(self, rbac_service, test_user, admin_user):
        """Test that multiple roles provide cumulative permissions"""
        user_id = test_user["user_id"]
        admin_id = admin_user["user_id"]
        
        # Assign multiple roles
        await rbac_service.assign_role(user_id, "security_analyst", admin_id)
        await rbac_service.assign_role(user_id, "auditor", admin_id)
        
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        # Should have permissions from both roles
        # From security_analyst
        analyst_perm = await rbac_service.check_permission(context, "ptaas:scan:create")
        assert analyst_perm.granted is True
        
        # From auditor  
        audit_perm = await rbac_service.check_permission(context, "audit:export")
        assert audit_perm.granted is True
        
        # From both (common permission)
        read_perm = await rbac_service.check_permission(context, "evidence:read")
        assert read_perm.granted is True


class TestRBACAPIEndpoints:
    """Test RBAC through API endpoints"""
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_access_granted(self, async_client, authenticated_headers):
        """Test accessing protected endpoint with valid permissions"""
        # Mock the authentication and RBAC checks for testing
        with patch('src.api.app.auth.dependencies.get_current_user') as mock_auth:
            with patch('src.api.app.services.rbac_service.RBACService.check_permission') as mock_rbac:
                # Mock authenticated user
                mock_auth.return_value = UserClaims(
                    user_id=str(uuid4()),
                    username="test_user",
                    tenant_id=str(uuid4()),
                    roles=["security_analyst"],
                    permissions=["ptaas:scan:read"]
                )
                
                # Mock permission check success
                mock_rbac.return_value = Mock(granted=True)
                
                response = await async_client.get(
                    "/api/v1/ptaas/profiles",
                    headers=authenticated_headers
                )
                
                # Should succeed with proper permissions
                assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_access_denied(self, async_client, authenticated_headers):
        """Test accessing protected endpoint without permissions"""
        with patch('src.api.app.auth.dependencies.get_current_user') as mock_auth:
            with patch('src.api.app.services.rbac_service.RBACService.check_permission') as mock_rbac:
                # Mock authenticated user
                mock_auth.return_value = UserClaims(
                    user_id=str(uuid4()),
                    username="limited_user",
                    tenant_id=str(uuid4()),
                    roles=["viewer"],
                    permissions=["evidence:read"]
                )
                
                # Mock permission check failure
                mock_rbac.return_value = Mock(granted=False)
                
                response = await async_client.post(
                    "/api/v1/ptaas/sessions",
                    headers=authenticated_headers,
                    json={
                        "targets": [{"host": "example.com", "ports": [80]}],
                        "scan_type": "quick"
                    }
                )
                
                # Should fail without proper permissions
                assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_admin_endpoint_access(self, async_client, admin_headers):
        """Test admin-only endpoint access"""
        with patch('src.api.app.auth.dependencies.get_current_user') as mock_auth:
            with patch('src.api.app.services.rbac_service.RBACService.check_permission') as mock_rbac:
                # Mock admin user
                mock_auth.return_value = UserClaims(
                    user_id=str(uuid4()),
                    username="admin_user",
                    tenant_id=str(uuid4()),
                    roles=["super_admin"],
                    permissions=["system:admin"]
                )
                
                # Mock admin permission check success
                mock_rbac.return_value = Mock(granted=True)
                
                response = await async_client.get(
                    "/api/v1/admin/users",
                    headers=admin_headers
                )
                
                # Admin should have access
                assert response.status_code in [200, 404]  # 404 if endpoint doesn't exist yet


class TestRBACCaching:
    """Test RBAC caching behavior"""
    
    @pytest.mark.asyncio
    async def test_permission_caching(self, rbac_service, test_user):
        """Test that permissions are properly cached"""
        user_id = test_user["user_id"]
        
        # Mock cache service to verify caching behavior
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Cache miss first time
        mock_cache.set = AsyncMock()
        
        rbac_service.cache = mock_cache
        
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        # First call should miss cache and set cache
        await rbac_service.check_permission(context, "ptaas:scan:read")
        
        # Verify cache.set was called
        mock_cache.set.assert_called_once()
        
        # Second call should hit cache
        mock_cache.get.return_value = {
            'granted': True,
            'reason': 'cached result',
            'source': 'cache'
        }
        
        result = await rbac_service.check_permission(context, "ptaas:scan:read")
        
        # Should return cached result
        assert result.source == "cache"
        assert result.granted is True


class TestRBACPerformance:
    """Test RBAC system performance"""
    
    @pytest.mark.asyncio
    async def test_bulk_permission_checks(self, rbac_service, test_user):
        """Test performance of bulk permission checks"""
        user_id = test_user["user_id"]
        
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        # List of permissions to check
        permissions = [
            "ptaas:scan:read",
            "ptaas:scan:create", 
            "intelligence:read",
            "evidence:read",
            "audit:read",
            "user:read",
            "organization:read"
        ]
        
        # Measure time for bulk check
        import time
        start_time = time.time()
        
        results = await rbac_service.check_multiple_permissions(context, permissions)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly (under 1 second for 7 permissions)
        assert duration < 1.0
        assert len(results) == len(permissions)
    
    @pytest.mark.asyncio
    async def test_concurrent_permission_checks(self, rbac_service, test_user):
        """Test concurrent permission checks"""
        user_id = test_user["user_id"]
        
        from src.api.app.services.rbac_service import RBACContext
        context = RBACContext(
            user_id=user_id,
            tenant_id=test_user["tenant_id"]
        )
        
        # Create multiple concurrent permission checks
        async def check_permission(permission):
            return await rbac_service.check_permission(context, permission)
        
        permissions = ["ptaas:scan:read", "intelligence:read", "evidence:read"]
        
        # Run concurrent checks
        tasks = [check_permission(perm) for perm in permissions]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == len(permissions)
        for result in results:
            assert hasattr(result, 'granted')
            assert hasattr(result, 'reason')


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_rbac_integration.py -v
    pytest.main([__file__, "-v"])