"""
Comprehensive Tenant Isolation Security Tests
Tests for PR-006: Secure tenant context and prevent SQL injection
"""

import pytest
import asyncio
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.app.core.secure_tenant_context import (
    SecureTenantContextManager, TenantContext, TenantSecurityEvent,
    TenantContextViolationType
)
from src.api.app.middleware.secure_tenant_middleware import (
    SecureTenantMiddleware, require_tenant_context, get_tenant_id
)
from src.api.app.infrastructure.secure_query_builder import (
    SecureQueryBuilder, QueryValidationResult, SecurityLevel
)
from src.api.app.services.secure_tenant_service import SecureTenantService
from src.api.app.auth.models import UserClaims
from src.api.app.domain.tenant_entities import Tenant, TenantCreate, TenantStatus


class TestSecureTenantContextManager:
    """Test secure tenant context manager functionality"""
    
    @pytest.fixture
    def mock_db_session_factory(self):
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        
        async def factory():
            return session
        
        return factory
    
    @pytest.fixture
    def mock_cache_service(self):
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.lpush = AsyncMock()
        return cache
    
    @pytest.fixture
    def tenant_manager(self, mock_db_session_factory, mock_cache_service):
        return SecureTenantContextManager(
            db_session_factory=mock_db_session_factory,
            cache_service=mock_cache_service
        )
    
    @pytest.fixture
    def valid_user_claims(self):
        return UserClaims(
            user_id="test-user-123",
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            roles=["user"],
            permissions=["ptaas:scan:read"]
        )
    
    @pytest.fixture
    def admin_user_claims(self):
        return UserClaims(
            user_id="admin-user-456",
            tenant_id="550e8400-e29b-41d4-a716-446655440001",
            email="admin@example.com",
            roles=["super_admin"],
            permissions=["super_admin", "system:admin"]
        )
    
    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.url.path = "/api/v1/ptaas/scans"
        request.method = "GET"
        request.client.host = "192.168.1.100"
        request.headers = {"User-Agent": "TestClient/1.0"}
        return request
    
    @pytest.mark.asyncio
    async def test_validate_user_tenant_access_same_tenant(
        self, tenant_manager, valid_user_claims
    ):
        """Test user access validation for same tenant"""
        tenant_id = UUID(valid_user_claims.tenant_id)
        
        result = await tenant_manager.validate_user_tenant_access(
            valid_user_claims, tenant_id
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_user_tenant_access_super_admin(
        self, tenant_manager, admin_user_claims
    ):
        """Test super admin can access any tenant"""
        different_tenant = uuid4()
        
        result = await tenant_manager.validate_user_tenant_access(
            admin_user_claims, different_tenant
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_user_tenant_access_cross_tenant_denied(
        self, tenant_manager, valid_user_claims, mock_db_session_factory
    ):
        """Test cross-tenant access is denied"""
        different_tenant = uuid4()
        
        # Mock database check to return no membership
        session = await mock_db_session_factory()
        session.execute.return_value.scalar.return_value = None
        
        result = await tenant_manager.validate_user_tenant_access(
            valid_user_claims, different_tenant, session
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_establish_secure_context_success(
        self, tenant_manager, valid_user_claims, mock_request
    ):
        """Test successful secure context establishment"""
        context = await tenant_manager.establish_secure_context(
            mock_request, valid_user_claims
        )
        
        assert isinstance(context, TenantContext)
        assert context.tenant_id == UUID(valid_user_claims.tenant_id)
        assert context.user_id == valid_user_claims.user_id
        assert context.is_valid()
    
    @pytest.mark.asyncio
    async def test_establish_secure_context_missing_tenant(
        self, tenant_manager, mock_request
    ):
        """Test context establishment fails without tenant"""
        user_claims = UserClaims(
            user_id="test-user",
            tenant_id=None,  # No tenant
            email="test@example.com"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await tenant_manager.establish_secure_context(mock_request, user_claims)
        
        assert exc_info.value.status_code == 400
        assert "Tenant context required" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_establish_secure_context_unauthorized_tenant(
        self, tenant_manager, valid_user_claims, mock_request, mock_db_session_factory
    ):
        """Test context establishment fails for unauthorized tenant"""
        # Mock user trying to access different tenant
        different_tenant = str(uuid4())
        valid_user_claims.tenant_id = different_tenant
        
        # Mock database check to deny access
        session = await mock_db_session_factory()
        session.execute.return_value.scalar.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await tenant_manager.establish_secure_context(mock_request, valid_user_claims)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_security_event_logging(
        self, tenant_manager, valid_user_claims, mock_cache_service
    ):
        """Test security event logging"""
        event = TenantSecurityEvent(
            violation_type=TenantContextViolationType.UNAUTHORIZED_ACCESS,
            user_id=valid_user_claims.user_id,
            tenant_id=UUID(valid_user_claims.tenant_id),
            ip_address="192.168.1.100"
        )
        
        await tenant_manager._log_security_event(event)
        
        # Check event was stored
        assert len(tenant_manager.violation_events) == 1
        assert tenant_manager.violation_events[0].violation_type == event.violation_type
        
        # Check cache was called
        mock_cache_service.lpush.assert_called_once()
    
    def test_tenant_context_validation(self):
        """Test tenant context validation"""
        context = TenantContext(
            tenant_id=uuid4(),
            user_id="test-user",
            validated_at=datetime.utcnow() - timedelta(minutes=35)  # Expired
        )
        
        assert not context.is_valid(max_age_minutes=30)
        
        # Refresh context
        context.refresh()
        assert context.is_valid(max_age_minutes=30)


class TestSecureTenantMiddleware:
    """Test secure tenant middleware functionality"""
    
    @pytest.fixture
    def mock_tenant_manager(self):
        manager = Mock(spec=SecureTenantContextManager)
        manager.establish_secure_context = AsyncMock()
        manager._log_security_event = AsyncMock()
        manager.enable_strict_validation = True
        return manager
    
    @pytest.fixture
    def middleware(self, mock_tenant_manager):
        app = Mock()
        return SecureTenantMiddleware(app, mock_tenant_manager)
    
    @pytest.fixture
    def mock_request_with_auth(self):
        request = Mock(spec=Request)
        request.url.path = "/api/v1/ptaas/scans"
        request.method = "GET"
        request.client.host = "192.168.1.100"
        request.headers = {"User-Agent": "TestClient/1.0"}
        request.state.user = UserClaims(
            user_id="test-user",
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com"
        )
        return request
    
    @pytest.mark.asyncio
    async def test_bypass_paths_skip_tenant_check(self, middleware):
        """Test that bypass paths skip tenant checking"""
        request = Mock(spec=Request)
        request.url.path = "/health"
        
        call_next = AsyncMock(return_value=Mock())
        
        response = await middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        assert not hasattr(request.state, 'tenant_context')
    
    @pytest.mark.asyncio
    async def test_header_manipulation_detection_strict(self, middleware):
        """Test detection of header manipulation in strict mode"""
        request = Mock(spec=Request)
        request.url.path = "/api/v1/ptaas/scans"
        request.method = "GET"
        request.client.host = "192.168.1.100"
        request.headers = {
            "User-Agent": "TestClient/1.0",
            "X-Tenant-ID": "malicious-tenant-id"  # Suspicious header
        }
        request.state.user = UserClaims(user_id="test", tenant_id="valid-tenant")
        
        call_next = AsyncMock()
        
        # Should raise exception in strict mode
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)
        
        assert exc_info.value.status_code == 400
        assert "manipulation via headers" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(self, middleware):
        """Test that unauthenticated requests are rejected"""
        request = Mock(spec=Request)
        request.url.path = "/api/v1/ptaas/scans"
        request.state = Mock()  # No user attribute
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 401
        call_next.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_successful_tenant_context_establishment(
        self, middleware, mock_request_with_auth, mock_tenant_manager
    ):
        """Test successful tenant context establishment"""
        tenant_context = TenantContext(
            tenant_id=uuid4(),
            user_id="test-user"
        )
        mock_tenant_manager.establish_secure_context.return_value = tenant_context
        
        call_next = AsyncMock(return_value=Mock())
        
        response = await middleware.dispatch(mock_request_with_auth, call_next)
        
        # Check context was set
        assert mock_request_with_auth.state.tenant_context == tenant_context
        assert mock_request_with_auth.state.tenant_id == tenant_context.tenant_id
        
        call_next.assert_called_once_with(mock_request_with_auth)


class TestSecureQueryBuilder:
    """Test secure query builder functionality"""
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(
            tenant_id=uuid4(),
            user_id="test-user"
        )
    
    @pytest.fixture
    def query_builder(self, tenant_context):
        return SecureQueryBuilder(tenant_context)
    
    def test_build_select_with_tenant_isolation(self, query_builder):
        """Test SELECT query building with tenant isolation"""
        params = query_builder.build_select(
            table="findings",
            columns=["id", "title", "severity"],
            where_conditions={"status": "open"}
        )
        
        assert "tenant_id = :tenant_id" in params.query
        assert "status = :param_0" in params.query
        assert params.params["tenant_id"] == str(query_builder.tenant_context.tenant_id)
        assert params.params["param_0"] == "open"
    
    def test_build_insert_with_tenant_isolation(self, query_builder):
        """Test INSERT query building with tenant isolation"""
        data = {
            "title": "Test Finding",
            "severity": "high",
            "description": "Test description"
        }
        
        params = query_builder.build_insert("findings", data)
        
        assert "tenant_id" in params.params
        assert params.params["tenant_id"] == str(query_builder.tenant_context.tenant_id)
        assert params.params["title"] == "Test Finding"
    
    def test_build_update_prevents_tenant_modification(self, query_builder):
        """Test UPDATE prevents tenant_id modification"""
        data = {
            "title": "Updated Title",
            "tenant_id": "malicious-tenant-id"  # Should be removed
        }
        
        params = query_builder.build_update(
            "findings", 
            data, 
            {"id": "test-id"}
        )
        
        # tenant_id should not be in SET clause
        assert "set_tenant_id" not in params.params
        assert "set_title" in params.params
        # But WHERE clause should include tenant isolation
        assert params.params["tenant_id"] == str(query_builder.tenant_context.tenant_id)
    
    def test_validate_query_dangerous_patterns(self, query_builder):
        """Test query validation catches dangerous patterns"""
        dangerous_queries = [
            "SELECT * FROM users; DROP TABLE users;",  # Multiple statements
            "SELECT * FROM users WHERE id = 1 OR 1=1",  # Injection
            "SELECT * FROM INFORMATION_SCHEMA.tables",  # System tables
            "EXEC xp_cmdshell 'dir'",  # Command execution
            "SELECT * FROM users UNION SELECT * FROM passwords"  # Union attack
        ]
        
        for query in dangerous_queries:
            result = query_builder.validate_query(query)
            assert not result.is_valid, f"Query should be invalid: {query}"
            assert len(result.errors) > 0
    
    def test_validate_query_missing_tenant_isolation(self, query_builder):
        """Test validation catches missing tenant isolation"""
        query = "SELECT * FROM findings WHERE status = 'open'"
        
        result = query_builder.validate_query(query, SecurityLevel.STRICT)
        
        assert not result.is_valid
        assert any("tenant isolation" in error for error in result.errors)
    
    def test_validate_query_valid_parameterized(self, query_builder):
        """Test validation passes for valid parameterized queries"""
        query = "SELECT * FROM findings WHERE tenant_id = :tenant_id AND status = :status"
        
        result = query_builder.validate_query(query)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_identifier_sanitization(self, query_builder):
        """Test SQL identifier sanitization"""
        # Valid identifiers
        assert query_builder._sanitize_identifier("valid_table") == "valid_table"
        assert query_builder._sanitize_identifier("Table123") == "Table123"
        
        # Invalid identifiers should raise ValueError
        with pytest.raises(ValueError):
            query_builder._sanitize_identifier("table; DROP TABLE users;")
        
        with pytest.raises(ValueError):
            query_builder._sanitize_identifier("table'--")
        
        with pytest.raises(ValueError):
            query_builder._sanitize_identifier("123invalid")


class TestSQLInjectionPrevention:
    """Specific tests for SQL injection prevention"""
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(tenant_id=uuid4(), user_id="test-user")
    
    @pytest.mark.asyncio
    async def test_injection_via_column_names(self, tenant_context):
        """Test injection attempts via column names are blocked"""
        builder = SecureQueryBuilder(tenant_context)
        
        malicious_columns = [
            "id; DROP TABLE users; --",
            "id' OR '1'='1",
            "id UNION SELECT password FROM users",
            "id/*comment*/",
        ]
        
        for column in malicious_columns:
            with pytest.raises(ValueError, match="Invalid identifier"):
                builder.build_select("findings", columns=[column])
    
    @pytest.mark.asyncio
    async def test_injection_via_table_names(self, tenant_context):
        """Test injection attempts via table names are blocked"""
        builder = SecureQueryBuilder(tenant_context)
        
        malicious_tables = [
            "users; DROP TABLE findings; --",
            "users' OR '1'='1",
            "users UNION SELECT * FROM passwords",
        ]
        
        for table in malicious_tables:
            with pytest.raises(ValueError, match="Invalid"):
                builder.build_select(table)
    
    @pytest.mark.asyncio
    async def test_parameterized_queries_prevent_injection(self, tenant_context):
        """Test that parameterized queries prevent injection"""
        builder = SecureQueryBuilder(tenant_context)
        
        # Malicious data should be safely parameterized
        malicious_data = {
            "title": "'; DROP TABLE findings; --",
            "description": "' OR '1'='1",
            "severity": "high' UNION SELECT password FROM users --"
        }
        
        params = builder.build_insert("findings", malicious_data)
        
        # Malicious content should be in parameters, not query string
        assert "DROP TABLE" not in params.query
        assert "UNION SELECT" not in params.query
        assert params.params["title"] == "'; DROP TABLE findings; --"


class TestTenantIsolationEnforcement:
    """Tests for tenant isolation enforcement"""
    
    @pytest.fixture
    def tenant_a_context(self):
        return TenantContext(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            user_id="user-a"
        )
    
    @pytest.fixture
    def tenant_b_context(self):
        return TenantContext(
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            user_id="user-b"
        )
    
    def test_queries_include_tenant_filtering(self, tenant_a_context):
        """Test all tenant-scoped queries include tenant filtering"""
        builder = SecureQueryBuilder(tenant_a_context)
        
        # SELECT
        params = builder.build_select("findings")
        assert "tenant_id = :tenant_id" in params.query
        assert params.params["tenant_id"] == str(tenant_a_context.tenant_id)
        
        # UPDATE
        params = builder.build_update("findings", {"title": "New"}, {"id": "123"})
        assert "tenant_id = :tenant_id" in params.query
        
        # DELETE
        params = builder.build_delete("findings", {"id": "123"})
        assert "tenant_id = :tenant_id" in params.query
    
    def test_cross_tenant_queries_blocked(self, tenant_a_context, tenant_b_context):
        """Test that cross-tenant data access is prevented"""
        builder_a = SecureQueryBuilder(tenant_a_context)
        builder_b = SecureQueryBuilder(tenant_b_context)
        
        # Each builder should only access its own tenant's data
        params_a = builder_a.build_select("findings")
        params_b = builder_b.build_select("findings")
        
        assert params_a.params["tenant_id"] != params_b.params["tenant_id"]
        assert params_a.params["tenant_id"] == str(tenant_a_context.tenant_id)
        assert params_b.params["tenant_id"] == str(tenant_b_context.tenant_id)
    
    def test_tenant_id_cannot_be_overridden_in_updates(self, tenant_a_context):
        """Test tenant_id cannot be modified in update operations"""
        builder = SecureQueryBuilder(tenant_a_context)
        
        malicious_data = {
            "title": "Legitimate update",
            "tenant_id": "550e8400-e29b-41d4-a716-446655440001"  # Different tenant
        }
        
        params = builder.build_update("findings", malicious_data, {"id": "123"})
        
        # tenant_id should not be in SET clause
        assert "set_tenant_id" not in params.params
        # But WHERE clause should still filter by correct tenant
        assert params.params["tenant_id"] == str(tenant_a_context.tenant_id)


@pytest.mark.integration
class TestSecureTenantServiceIntegration:
    """Integration tests for secure tenant service"""
    
    @pytest.fixture
    def mock_session(self):
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session
    
    @pytest.fixture
    def tenant_context(self):
        return TenantContext(
            tenant_id=uuid4(),
            user_id="test-user",
            permissions={"user"}
        )
    
    @pytest.fixture
    def admin_context(self):
        return TenantContext(
            tenant_id=uuid4(),
            user_id="admin-user",
            permissions={"super_admin"}
        )
    
    @pytest.fixture
    def tenant_manager(self):
        return Mock(spec=SecureTenantContextManager)
    
    @pytest.mark.asyncio
    async def test_tenant_access_validation(
        self, mock_session, tenant_context, tenant_manager
    ):
        """Test tenant access validation in service"""
        service = SecureTenantService(mock_session, tenant_context, tenant_manager)
        
        # User should only access their own tenant
        different_tenant = uuid4()
        
        result = await service.get_tenant(different_tenant)
        
        # Should return None for unauthorized access
        assert result is None
    
    @pytest.mark.asyncio
    async def test_admin_cross_tenant_access(
        self, mock_session, admin_context, tenant_manager
    ):
        """Test admin can access other tenants"""
        service = SecureTenantService(mock_session, admin_context, tenant_manager)
        
        # Mock database response
        mock_row = Mock()
        mock_row.id = uuid4()
        mock_row.name = "Test Tenant"
        mock_row.slug = "test-tenant"
        mock_row.status = "ACTIVE"
        mock_row.plan = "PROFESSIONAL"
        mock_row.settings = {}
        mock_row.created_at = datetime.utcnow()
        mock_row.updated_at = None
        mock_row.contact_email = "test@example.com"
        mock_row.contact_name = "Test User"
        
        mock_session.execute.return_value.first.return_value = mock_row
        
        result = await service.get_tenant(mock_row.id)
        
        # Admin should be able to access
        assert result is not None
        assert result.id == mock_row.id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])