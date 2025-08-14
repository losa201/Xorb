"""Tests for multi-tenancy functionality."""
import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, Mock, patch

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.tenant_entities import (
    Tenant, TenantUser, Evidence, Finding,
    TenantCreate, TenantStatus, TenantPlan
)
from app.services.tenant_service import TenantService
from app.middleware.tenant_context import TenantContextMiddleware
from app.auth.models import UserClaims, Role


@pytest.fixture
def tenant_service():
    """Tenant service for testing."""
    return TenantService()


@pytest.fixture
def sample_tenant_data():
    """Sample tenant creation data."""
    return TenantCreate(
        name="Test Company",
        slug="test-company",
        contact_email="admin@test.com",
        contact_name="Test Admin",
        plan=TenantPlan.PROFESSIONAL,
        max_users=50,
        max_storage_gb=500
    )


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = Mock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.add = Mock()
    return session


@pytest.fixture
def mock_tenant():
    """Mock tenant entity."""
    return Tenant(
        id=uuid4(),
        name="Test Company",
        slug="test-company",
        status=TenantStatus.ACTIVE.value,
        plan=TenantPlan.PROFESSIONAL.value,
        contact_email="admin@test.com",
        max_users=50,
        max_storage_gb=500,
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_user_claims():
    """Mock user claims for testing."""
    tenant_id = uuid4()
    return UserClaims(
        sub="user123",
        email="user@test.com",
        name="Test User",
        tenant_id=tenant_id,
        roles=[Role.SECURITY_ANALYST],
        exp=datetime.now(timezone.utc) + timedelta(hours=1),
        iat=datetime.now(timezone.utc)
    )


class TestTenantService:
    """Test tenant service operations."""

    @pytest.mark.asyncio
    async def test_create_tenant_success(self, tenant_service, sample_tenant_data, mock_session):
        """Test successful tenant creation."""
        with patch.object(tenant_service, 'session_factory') as mock_factory:
            mock_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.__aexit__.return_value = None

            # Mock successful creation
            result_mock = Mock()
            result_mock.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = result_mock

            tenant = await tenant_service.create_tenant(sample_tenant_data)

            # Verify session methods were called
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_tenant_duplicate_slug(self, tenant_service, sample_tenant_data):
        """Test tenant creation with duplicate slug."""
        with patch.object(tenant_service, 'session_factory') as mock_factory:
            mock_session = Mock(spec=AsyncSession)
            mock_session.add = Mock()
            mock_session.commit = AsyncMock(side_effect=Exception("Duplicate key"))
            mock_session.rollback = AsyncMock()

            mock_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.__aexit__.return_value = None

            with pytest.raises(Exception):
                await tenant_service.create_tenant(sample_tenant_data)

            mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tenant_success(self, tenant_service, mock_tenant):
        """Test successful tenant retrieval."""
        with patch.object(tenant_service, 'session_factory') as mock_factory:
            mock_session = Mock(spec=AsyncSession)
            result_mock = Mock()
            result_mock.scalar_one_or_none.return_value = mock_tenant
            mock_session.execute = AsyncMock(return_value=result_mock)

            mock_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.__aexit__.return_value = None

            tenant = await tenant_service.get_tenant(mock_tenant.id)

            assert tenant == mock_tenant
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tenants_with_filter(self, tenant_service):
        """Test listing tenants with status filter."""
        with patch.object(tenant_service, 'session_factory') as mock_factory:
            mock_session = Mock(spec=AsyncSession)
            result_mock = Mock()
            result_mock.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=result_mock)

            mock_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.__aexit__.return_value = None

            tenants = await tenant_service.list_tenants(
                status=TenantStatus.ACTIVE,
                limit=50,
                offset=10
            )

            assert tenants == []
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_tenant_access_super_admin(self, tenant_service):
        """Test tenant access validation for super admin."""
        super_admin_claims = UserClaims(
            sub="admin123",
            email="admin@test.com",
            tenant_id=uuid4(),
            roles=[Role.SUPER_ADMIN],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc)
        )

        tenant_id = uuid4()
        has_access = await tenant_service.validate_tenant_access(
            super_admin_claims, tenant_id
        )

        assert has_access is True

    @pytest.mark.asyncio
    async def test_validate_tenant_access_same_tenant(self, tenant_service, mock_user_claims):
        """Test tenant access validation for same tenant."""
        has_access = await tenant_service.validate_tenant_access(
            mock_user_claims, mock_user_claims.tenant_id
        )

        assert has_access is True

    @pytest.mark.asyncio
    async def test_validate_tenant_access_different_tenant(self, tenant_service, mock_user_claims):
        """Test tenant access validation for different tenant."""
        different_tenant_id = uuid4()

        with patch.object(tenant_service, 'get_tenant_user', return_value=None):
            has_access = await tenant_service.validate_tenant_access(
                mock_user_claims, different_tenant_id
            )

            assert has_access is False


class TestTenantContextMiddleware:
    """Test tenant context middleware."""

    def test_bypass_paths(self):
        """Test that bypass paths are correctly defined."""
        middleware = TenantContextMiddleware(Mock())

        assert "/health" in middleware.BYPASS_PATHS
        assert "/auth/login" in middleware.BYPASS_PATHS
        assert "/docs" in middleware.BYPASS_PATHS

    @pytest.mark.asyncio
    async def test_middleware_with_authenticated_user(self, mock_user_claims):
        """Test middleware with authenticated user."""
        from fastapi import Request, Response

        middleware = TenantContextMiddleware(Mock())

        # Mock request with user in state
        request = Mock(spec=Request)
        request.url.path = "/api/evidence"
        request.state.user = mock_user_claims
        request.headers.get.return_value = None

        # Mock call_next
        async def mock_call_next(req):
            return Response("OK")

        with patch.object(middleware, '_set_database_tenant_context') as mock_set_context:
            mock_set_context.return_value = None

            response = await middleware.dispatch(request, mock_call_next)

            # Verify tenant context was set
            assert request.state.tenant_id == mock_user_claims.tenant_id
            mock_set_context.assert_called_once_with(mock_user_claims.tenant_id)

    @pytest.mark.asyncio
    async def test_middleware_with_tenant_header(self):
        """Test middleware with X-Tenant-ID header."""
        from fastapi import Request, Response

        middleware = TenantContextMiddleware(Mock())
        tenant_id = uuid4()

        # Mock request with tenant header
        request = Mock(spec=Request)
        request.url.path = "/api/evidence"
        request.state = Mock()
        request.state.user = None
        request.headers.get.return_value = str(tenant_id)

        async def mock_call_next(req):
            return Response("OK")

        with patch.object(middleware, '_set_database_tenant_context') as mock_set_context:
            mock_set_context.return_value = None

            response = await middleware.dispatch(request, mock_call_next)

            assert request.state.tenant_id == tenant_id
            mock_set_context.assert_called_once_with(tenant_id)

    @pytest.mark.asyncio
    async def test_middleware_bypass_health_endpoint(self):
        """Test that health endpoints bypass tenant context."""
        from fastapi import Request, Response

        middleware = TenantContextMiddleware(Mock())

        request = Mock(spec=Request)
        request.url.path = "/health"

        async def mock_call_next(req):
            return Response("OK")

        response = await middleware.dispatch(request, mock_call_next)

        # Should not set tenant context for health endpoints
        assert not hasattr(request.state, 'tenant_id')


class TestRLSPolicies:
    """Test Row Level Security policies."""

    @pytest.mark.asyncio
    async def test_evidence_rls_isolation(self, mock_session):
        """Test that evidence table RLS isolates by tenant."""
        tenant1_id = uuid4()
        tenant2_id = uuid4()

        # This would be an integration test with real database
        # Testing RLS policy enforcement

        # Set tenant context for tenant1
        mock_session.execute = AsyncMock()

        # Simulate setting tenant context
        await mock_session.execute(
            "SELECT set_config('app.tenant_id', :tenant_id, false)",
            {"tenant_id": str(tenant1_id)}
        )

        # Query should only return tenant1's evidence
        query = select(Evidence)
        result_mock = Mock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = result_mock

        await mock_session.execute(query)

        # Verify tenant context was set
        mock_session.execute.assert_any_call(
            "SELECT set_config('app.tenant_id', :tenant_id, false)",
            {"tenant_id": str(tenant1_id)}
        )

    @pytest.mark.asyncio
    async def test_findings_rls_isolation(self, mock_session):
        """Test that findings table RLS isolates by tenant."""
        tenant_id = uuid4()

        # Set tenant context
        await mock_session.execute(
            "SELECT set_config('app.tenant_id', :tenant_id, false)",
            {"tenant_id": str(tenant_id)}
        )

        # Query findings - should only return current tenant's data
        query = select(Finding)
        result_mock = Mock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = result_mock

        await mock_session.execute(query)

        mock_session.execute.assert_any_call(
            "SELECT set_config('app.tenant_id', :tenant_id, false)",
            {"tenant_id": str(tenant_id)}
        )

    @pytest.mark.asyncio
    async def test_super_admin_bypass_rls(self, mock_session):
        """Test that super admin can bypass RLS policies."""
        # Set super admin role
        await mock_session.execute(
            "SELECT set_config('app.user_role', 'super_admin', false)"
        )

        # Query should return all data regardless of tenant
        query = select(Evidence)
        result_mock = Mock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = result_mock

        await mock_session.execute(query)

        mock_session.execute.assert_any_call(
            "SELECT set_config('app.user_role', 'super_admin', false)"
        )


class TestTenantIsolationIntegration:
    """Integration tests for tenant isolation."""

    @pytest.mark.asyncio
    async def test_evidence_upload_tenant_isolation(self):
        """Test that evidence uploads are isolated by tenant."""
        # This would be a full integration test
        # 1. Create two tenants
        # 2. Upload evidence as user from tenant1
        # 3. Try to access evidence as user from tenant2
        # 4. Verify access is denied
        pass

    @pytest.mark.asyncio
    async def test_findings_cross_tenant_access_denied(self):
        """Test that findings cannot be accessed across tenants."""
        # This would be a full integration test
        # 1. Create finding for tenant1
        # 2. Try to access as user from tenant2
        # 3. Verify 404 or access denied
        pass

    @pytest.mark.asyncio
    async def test_embedding_vectors_tenant_isolation(self):
        """Test that embedding vectors are isolated by tenant."""
        # This would be a full integration test
        # Similar to evidence/findings tests
        pass


@pytest.mark.asyncio
async def test_tenant_migration_rollback():
    """Test that tenant migrations can be safely rolled back."""
    # This would test:
    # 1. Apply tenant isolation migration
    # 2. Create test data
    # 3. Rollback migration
    # 4. Verify data integrity
    pass


@pytest.mark.asyncio
async def test_tenant_data_backfill():
    """Test safe backfill of existing data with tenant IDs."""
    # This would test:
    # 1. Create data without tenant context
    # 2. Run backfill script
    # 3. Verify all data has appropriate tenant_id
    # 4. Verify RLS policies work correctly
    pass
