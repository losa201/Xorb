"""
Global pytest configuration and fixtures for XORB testing framework
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import tempfile
import shutil
import os

from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment variables before importing
os.environ["ENVIRONMENT"] = "test"
os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:3000"
os.environ["JWT_SECRET"] = "test-jwt-secret-key-for-testing-32-characters-minimum"

from src.api.app.main import app
from src.api.app.core.database import Base, get_db_session
from src.api.app.core.cache import CacheConfig, CacheBackend, setup_cache
from src.api.app.core.security import SecurityConfig, setup_security
from src.api.app.core.metrics import MetricConfig, setup_metrics
from src.api.app.core.logging import setup_logging


# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use test database


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_database_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    SessionLocal = async_sessionmaker(
        test_database_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with SessionLocal() as session:
        yield session


@pytest.fixture
def override_get_db_session(test_db_session):
    """Override database dependency for testing"""
    async def get_test_db():
        yield test_db_session
    
    app.dependency_overrides[get_db_session] = get_test_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def test_cache_service():
    """Create test cache service with memory backend"""
    config = CacheConfig(
        backend=CacheBackend.MEMORY,
        memory_max_size=100,
        memory_ttl_seconds=60,
        enable_metrics=False
    )
    
    cache_service = setup_cache(config)
    yield cache_service
    
    # Cleanup
    await cache_service.clear()
    await cache_service.close()


@pytest.fixture
def test_security_config():
    """Create test security configuration"""
    return SecurityConfig(
        jwt_secret_key="test-secret-key-for-testing-only",
        jwt_expiration_minutes=30,
        min_password_length=8,
        max_login_attempts=3,
        lockout_duration_minutes=5,
        enable_field_level_encryption=True
    )


@pytest.fixture
def test_security_service(test_security_config):
    """Create test security service"""
    return setup_security(test_security_config)


@pytest.fixture
def test_metrics_config():
    """Create test metrics configuration"""
    return MetricConfig(
        enable_prometheus=False,
        enable_custom_metrics=True,
        collection_interval=1,
        enable_detailed_metrics=False
    )


@pytest.fixture
async def test_metrics_service(test_metrics_config):
    """Create test metrics service"""
    metrics_service = setup_metrics(test_metrics_config)
    await metrics_service.start()
    yield metrics_service
    await metrics_service.stop()


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    setup_logging(
        log_level="WARNING",  # Reduce log noise in tests
        environment="test",
        enable_json=False,
        enable_colors=False
    )


@pytest.fixture
def test_client(override_get_db_session) -> TestClient:
    """Create test client"""
    return TestClient(app)


@pytest.fixture
async def async_test_client(override_get_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_external_service():
    """Mock external service for testing"""
    mock = Mock()
    mock.get_data = AsyncMock(return_value={"status": "success", "data": "test"})
    mock.post_data = AsyncMock(return_value={"status": "created", "id": "123"})
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecureTestPassword123!",
        "roles": ["user"]
    }


@pytest.fixture
def test_admin_user_data():
    """Test admin user data"""
    return {
        "username": "adminuser",
        "email": "admin@example.com",
        "password": "SecureAdminPassword123!",
        "roles": ["admin", "user"]
    }


@pytest.fixture
def test_organization_data():
    """Test organization data"""
    return {
        "name": "Test Organization",
        "plan_type": "enterprise",
        "settings": {
            "max_users": 100,
            "features": ["ptaas", "compliance", "reporting"]
        }
    }


@pytest.fixture
def test_scan_session_data():
    """Test scan session data"""
    return {
        "targets": [
            {
                "host": "scanme.nmap.org",
                "ports": [22, 80, 443],
                "scan_profile": "quick"
            }
        ],
        "scan_type": "comprehensive",
        "metadata": {
            "requested_by": "test_user",
            "priority": "medium"
        }
    }


@pytest.fixture
def test_jwt_token(test_security_service, test_user_data):
    """Create test JWT token"""
    tokens = test_security_service.create_user_tokens(
        user_id="test-user-id",
        additional_claims={
            "username": test_user_data["username"],
            "roles": test_user_data["roles"]
        }
    )
    return tokens["access_token"]


@pytest.fixture
def auth_headers(test_jwt_token):
    """Create authorization headers"""
    return {"Authorization": f"Bearer {test_jwt_token}"}


@pytest.fixture
async def test_redis_connection():
    """Create test Redis connection"""
    try:
        import redis.asyncio as redis
        
        client = redis.from_url(TEST_REDIS_URL)
        await client.ping()
        
        yield client
        
        # Cleanup test data
        await client.flushdb()
        await client.close()
        
    except Exception:
        # Redis not available, yield None
        yield None


@pytest.fixture
def mock_temporal_client():
    """Mock Temporal workflow client"""
    mock = Mock()
    mock.start_workflow = AsyncMock(return_value=Mock(id="test-workflow-id"))
    mock.get_workflow_handle = Mock(return_value=Mock(
        query=AsyncMock(return_value="running"),
        result=AsyncMock(return_value={"status": "completed"})
    ))
    return mock


@pytest.fixture
def mock_nmap_scanner():
    """Mock Nmap scanner for PTaaS tests"""
    mock = Mock()
    mock.scan = Mock(return_value={
        "scan": {
            "scanme.nmap.org": {
                "hostnames": [{"name": "scanme.nmap.org", "type": "PTR"}],
                "addresses": {"ipv4": "45.33.32.156"},
                "status": {"state": "up", "reason": "syn-ack"},
                "tcp": {
                    22: {"state": "open", "reason": "syn-ack", "name": "ssh"},
                    80: {"state": "open", "reason": "syn-ack", "name": "http"}
                }
            }
        },
        "nmap": {
            "command_line": "nmap -sT -O -p22,80 scanme.nmap.org",
            "scanstats": {"timestr": "Test scan", "elapsed": "1.23", "uphosts": "1"}
        }
    })
    return mock


@pytest.fixture
def mock_nuclei_scanner():
    """Mock Nuclei scanner for vulnerability testing"""
    mock = Mock()
    mock.scan = AsyncMock(return_value=[
        {
            "template-id": "http-missing-security-headers",
            "info": {
                "name": "HTTP Missing Security Headers",
                "severity": "info",
                "tags": ["misconfiguration"]
            },
            "matched-at": "http://scanme.nmap.org",
            "type": "http"
        }
    ])
    return mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to test files in unit directory
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to test files in integration directory
        elif "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker to test files in e2e directory
        elif "tests/e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add security marker to security test files
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        # Add performance marker to performance test files
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# Async test utilities
@pytest_asyncio.fixture
async def async_test_setup():
    """Setup for async tests"""
    # Setup any async resources
    yield
    # Cleanup async resources


# Environment variable overrides for testing
@pytest.fixture(autouse=True)
def test_environment_variables(monkeypatch):
    """Set test environment variables"""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)
    monkeypatch.setenv("JWT_SECRET", "test-secret-key")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("ENVIRONMENT", "test")


# Coverage configuration helpers
def pytest_sessionstart(session):
    """Actions to perform at start of test session"""
    # Ensure test directories exist
    test_dirs = ["tests/unit", "tests/integration", "tests/e2e", "tests/security", "tests/performance"]
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """Actions to perform at end of test session"""
    # Cleanup any global test artifacts
    test_files = ["test.db", "test.db-shm", "test.db-wal"]
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)