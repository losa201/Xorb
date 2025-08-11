"""
Tests for RBAC System
Comprehensive test suite for Role-Based Access Control
"""

import pytest
import asyncio
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.api.app.services.rbac_service import RBACService, RBACContext, PermissionCheck
from src.api.app.infrastructure.rbac_models import RBACRole, RBACPermission, RBACUserRole, RBACUserPermission
from src.api.app.auth.rbac_dependencies import require_permission, require_role, rbac_decorator
from src.api.app.auth.models import UserClaims


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    return AsyncMock()


@pytest.fixture
def mock_cache_service():
    """Mock cache service"""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = None
    cache.delete.return_value = None
    return cache


@pytest.fixture
def rbac_service(mock_db_session, mock_cache_service):
    """RBAC service instance for testing"""
    return RBACService(mock_db_session, mock_cache_service)


@pytest.fixture
def sample_user_id():
    """Sample user ID"""
    return uuid4()


@pytest.fixture
def sample_tenant_id():
    """Sample tenant ID"""
    return uuid4()


@pytest.fixture
def rbac_context(sample_user_id, sample_tenant_id):
    """Sample RBAC context"""
    return RBACContext(
        user_id=sample_user_id,
        tenant_id=sample_tenant_id,
        ip_address="192.168.1.100",
        user_agent="Test Agent",
        metadata={"test": True}
    )


class TestRBACService:
    """Test cases for RBAC service"""
    
    @pytest.mark.asyncio
    async def test_check_permission_granted(self, rbac_service, rbac_context, mock_db_session):
        """Test permission check when user has permission"""
        # Mock database query to return permission
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = ["ptaas:scan:create"]
        mock_db_session.execute.return_value = mock_result
        
        result = await rbac_service.check_permission(rbac_context, "ptaas:scan:create")
        
        assert isinstance(result, PermissionCheck)
        assert result.granted is True
        assert result.context == rbac_context
        assert "ptaas:scan:create" in result.reason
    
    @pytest.mark.asyncio
    async def test_check_permission_denied(self, rbac_service, rbac_context, mock_db_session):
        """Test permission check when user lacks permission"""
        # Mock database query to return empty permissions
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        result = await rbac_service.check_permission(rbac_context, "admin:system")
        
        assert isinstance(result, PermissionCheck)
        assert result.granted is False
        assert result.source == "denied"
    
    @pytest.mark.asyncio
    async def test_check_permission_with_cache(self, rbac_service, rbac_context, mock_cache_service):
        """Test permission check with cached result"""
        # Mock cache hit
        mock_cache_service.get.return_value = {
            'granted': True,
            'reason': 'cached permission',
            'source': 'cache'
        }
        
        result = await rbac_service.check_permission(rbac_context, "ptaas:scan:read")
        
        assert result.granted is True
        assert result.source == "cache"
        mock_cache_service.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_multiple_permissions(self, rbac_service, rbac_context, mock_db_session):
        """Test checking multiple permissions"""
        # Mock database to return some permissions
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = ["ptaas:scan:create", "ptaas:scan:read"]
        mock_db_session.execute.return_value = mock_result
        
        permissions = ["ptaas:scan:create", "ptaas:scan:read", "admin:system"]
        results = await rbac_service.check_multiple_permissions(rbac_context, permissions)
        
        assert len(results) == 3
        assert results["ptaas:scan:create"].granted is True
        assert results["ptaas:scan:read"].granted is True
        assert results["admin:system"].granted is False
    
    @pytest.mark.asyncio
    async def test_assign_role(self, rbac_service, sample_user_id, mock_db_session):
        """Test role assignment"""
        # Mock role lookup
        mock_role = Mock()
        mock_role.id = uuid4()
        mock_role.name = "security_analyst"
        
        mock_role_result = Mock()
        mock_role_result.scalar_one_or_none.return_value = mock_role
        
        mock_existing_result = Mock()
        mock_existing_result.scalar_one_or_none.return_value = None
        
        mock_db_session.execute.side_effect = [mock_role_result, mock_existing_result]
        mock_db_session.commit = AsyncMock()
        
        result = await rbac_service.assign_role(
            sample_user_id, 
            "security_analyst", 
            uuid4()
        )
        
        assert result is True
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assign_role_not_found(self, rbac_service, sample_user_id, mock_db_session):
        """Test role assignment with non-existent role"""
        # Mock role lookup returning None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        result = await rbac_service.assign_role(
            sample_user_id,
            "nonexistent_role",
            uuid4()
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_revoke_role(self, rbac_service, sample_user_id, mock_db_session):
        """Test role revocation"""
        # Mock assignment lookup
        mock_assignment = Mock()
        mock_assignment.is_active = True
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_assignment
        mock_db_session.execute.return_value = mock_result
        mock_db_session.commit = AsyncMock()
        
        result = await rbac_service.revoke_role(
            sample_user_id,
            "security_analyst", 
            uuid4()
        )
        
        assert result is True
        assert mock_assignment.is_active is False
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assign_permission(self, rbac_service, sample_user_id, mock_db_session):
        """Test direct permission assignment"""
        # Mock permission lookup
        mock_permission = Mock()
        mock_permission.id = uuid4()
        mock_permission.name = "custom:permission"
        
        mock_perm_result = Mock()
        mock_perm_result.scalar_one_or_none.return_value = mock_permission
        
        mock_existing_result = Mock()
        mock_existing_result.scalar_one_or_none.return_value = None
        
        mock_db_session.execute.side_effect = [mock_perm_result, mock_existing_result]
        mock_db_session.commit = AsyncMock()
        
        result = await rbac_service.assign_permission(
            sample_user_id,
            "custom:permission",
            uuid4()
        )
        
        assert result is True
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_roles(self, rbac_service, sample_user_id, sample_tenant_id, mock_db_session):
        """Test getting user roles"""
        # Mock database query
        mock_role = Mock()
        mock_role.id = uuid4()
        mock_role.name = "security_analyst"
        mock_role.display_name = "Security Analyst"
        mock_role.description = "Security analysis role"
        mock_role.level = 70
        mock_role.is_system_role = True
        
        mock_assignment = Mock()
        mock_assignment.tenant_id = sample_tenant_id
        mock_assignment.granted_at = datetime.utcnow()
        mock_assignment.expires_at = None
        
        mock_result = Mock()
        mock_result.all.return_value = [(mock_role, mock_assignment)]
        mock_db_session.execute.return_value = mock_result
        
        roles = await rbac_service.get_user_roles(sample_user_id, sample_tenant_id)
        
        assert len(roles) == 1
        assert roles[0]['name'] == "security_analyst"
        assert roles[0]['display_name'] == "Security Analyst"
        assert roles[0]['tenant_id'] == str(sample_tenant_id)
    
    @pytest.mark.asyncio
    async def test_get_available_roles(self, rbac_service, mock_db_session):
        """Test getting available roles"""
        # Mock database query
        mock_roles = [
            Mock(
                id=uuid4(),
                name="admin",
                display_name="Administrator", 
                description="Full access",
                level=100,
                is_system_role=True
            ),
            Mock(
                id=uuid4(),
                name="user",
                display_name="User",
                description="Basic access",
                level=20,
                is_system_role=True
            )
        ]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_roles
        mock_db_session.execute.return_value = mock_result
        
        roles = await rbac_service.get_available_roles()
        
        assert len(roles) == 2
        assert roles[0]['name'] == "admin"
        assert roles[1]['name'] == "user"
    
    @pytest.mark.asyncio
    async def test_health_check(self, rbac_service, mock_db_session):
        """Test RBAC service health check"""
        # Mock database queries
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [Mock(), Mock()]  # 2 items
        mock_db_session.execute.return_value = mock_result
        
        health = await rbac_service.health_check()
        
        assert health['status'] == 'healthy'
        assert health['active_roles'] == 2
        assert health['active_permissions'] == 2
        assert 'timestamp' in health


class TestRBACDependencies:
    """Test cases for RBAC dependencies"""
    
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request"""
        request = Mock()
        request.client.host = "192.168.1.100"
        request.headers = {"User-Agent": "Test Agent"}
        request.method = "GET"
        request.url.path = "/test"
        return request
    
    @pytest.fixture
    def mock_user_claims(self, sample_user_id, sample_tenant_id):
        """Mock user claims"""
        return UserClaims(
            user_id=str(sample_user_id),
            username="test_user",
            tenant_id=str(sample_tenant_id),
            roles=["security_analyst"],
            permissions=["ptaas:scan:read"]
        )
    
    @pytest.mark.asyncio
    async def test_require_permission_success(self, mock_request, mock_user_claims):
        """Test require_permission dependency with valid permission"""
        with patch('src.api.app.auth.rbac_dependencies.get_container') as mock_container:
            # Mock RBAC service
            mock_rbac_service = Mock()
            mock_rbac_service.check_permission.return_value = PermissionCheck(
                granted=True,
                reason="Permission granted",
                source="role",
                checked_at=datetime.utcnow(),
                context=Mock()
            )
            
            mock_container.return_value.get.return_value = mock_rbac_service
            
            # Create dependency function
            permission_dep = require_permission("ptaas:scan:read")
            
            # Test the dependency
            result = await permission_dep(mock_request, mock_user_claims, mock_rbac_service)
            
            assert result == mock_user_claims
            mock_rbac_service.check_permission.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_require_permission_denied(self, mock_request, mock_user_claims):
        """Test require_permission dependency with denied permission"""
        with patch('src.api.app.auth.rbac_dependencies.get_container') as mock_container:
            # Mock RBAC service
            mock_rbac_service = Mock()
            mock_rbac_service.check_permission.return_value = PermissionCheck(
                granted=False,
                reason="Permission denied",
                source="denied",
                checked_at=datetime.utcnow(),
                context=Mock()
            )
            
            mock_container.return_value.get.return_value = mock_rbac_service
            
            # Create dependency function
            permission_dep = require_permission("admin:system")
            
            # Test the dependency should raise HTTPException
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await permission_dep(mock_request, mock_user_claims, mock_rbac_service)
            
            assert exc_info.value.status_code == 403
            assert "admin:system" in str(exc_info.value.detail)


class TestRBACDecorator:
    """Test cases for RBAC decorator"""
    
    @pytest.fixture
    def mock_request_with_user(self, sample_user_id, sample_tenant_id):
        """Mock request with user in state"""
        request = Mock()
        request.client.host = "192.168.1.100"
        request.headers = {"User-Agent": "Test Agent"}
        request.method = "POST"
        request.url.path = "/api/test"
        
        # Mock user in request state
        user = Mock()
        user.user_id = str(sample_user_id)
        user.tenant_id = str(sample_tenant_id)
        request.state.user = user
        
        return request
    
    @pytest.mark.asyncio
    async def test_rbac_decorator_with_permissions(self, mock_request_with_user):
        """Test RBAC decorator with permission check"""
        with patch('src.api.app.auth.rbac_dependencies.get_container') as mock_container:
            # Mock RBAC service
            mock_rbac_service = Mock()
            mock_rbac_service.check_multiple_permissions.return_value = {
                "ptaas:scan:create": PermissionCheck(
                    granted=True,
                    reason="Permission granted",
                    source="role",
                    checked_at=datetime.utcnow(),
                    context=Mock()
                )
            }
            
            mock_container.return_value.get.return_value = mock_rbac_service
            
            # Create decorated function
            @rbac_decorator(permissions=["ptaas:scan:create"])
            async def test_function(request):
                return {"success": True}
            
            # Test the decorated function
            result = await test_function(mock_request_with_user)
            
            assert result == {"success": True}
            mock_rbac_service.check_multiple_permissions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rbac_decorator_permission_denied(self, mock_request_with_user):
        """Test RBAC decorator with denied permission"""
        with patch('src.api.app.auth.rbac_dependencies.get_container') as mock_container:
            # Mock RBAC service
            mock_rbac_service = Mock()
            mock_rbac_service.check_multiple_permissions.return_value = {
                "admin:system": PermissionCheck(
                    granted=False,
                    reason="Permission denied",
                    source="denied",
                    checked_at=datetime.utcnow(),
                    context=Mock()
                )
            }
            
            mock_container.return_value.get.return_value = mock_rbac_service
            
            # Create decorated function
            @rbac_decorator(permissions=["admin:system"])
            async def test_function(request):
                return {"success": True}
            
            # Test the decorated function should raise HTTPException
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await test_function(mock_request_with_user)
            
            assert exc_info.value.status_code == 403
            assert "admin:system" in str(exc_info.value.detail)


class TestRBACModels:
    """Test cases for RBAC models"""
    
    def test_rbac_role_creation(self):
        """Test creating RBAC role"""
        role = RBACRole(
            name="test_role",
            display_name="Test Role",
            description="Test role description",
            level=50
        )
        
        assert role.name == "test_role"
        assert role.display_name == "Test Role"
        assert role.level == 50
        assert role.is_system_role is False
        assert role.is_active is True
    
    def test_rbac_permission_creation(self):
        """Test creating RBAC permission"""
        permission = RBACPermission(
            name="test:permission",
            display_name="Test Permission",
            description="Test permission description",
            resource="test",
            action="permission"
        )
        
        assert permission.name == "test:permission"
        assert permission.resource == "test"
        assert permission.action == "permission"
        assert permission.is_system_permission is False
        assert permission.is_active is True
    
    def test_user_role_validity(self):
        """Test user role assignment validity"""
        user_role = RBACUserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            is_active=True
        )
        
        assert user_role.is_valid() is True
        
        # Test expired role
        user_role.expires_at = datetime.utcnow() - timedelta(hours=1)
        assert user_role.is_valid() is False
        
        # Test inactive role
        user_role.expires_at = None
        user_role.is_active = False
        assert user_role.is_valid() is False


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_rbac_system.py -v
    pytest.main([__file__, "-v"])