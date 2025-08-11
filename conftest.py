"""
Global pytest configuration and fixtures
"""

import asyncio
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, Generator

# Add source directories to Python path for test imports
project_root = Path(__file__).parent
api_dir = project_root / "src" / "api"
src_dir = project_root / "src"

for path in [str(api_dir), str(src_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set test environment
os.environ['ENVIRONMENT'] = 'test'
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test_xorb'
os.environ['REDIS_URL'] = 'redis://localhost:6379/15'  # Use test DB


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    mock.ping.return_value = True
    return mock


@pytest.fixture
async def mock_database():
    """Mock database connection for testing."""
    mock = AsyncMock()
    mock.execute.return_value = None
    mock.fetch.return_value = []
    mock.fetchrow.return_value = None
    mock.fetchval.return_value = None
    return mock


@pytest.fixture
def mock_temporal():
    """Mock Temporal workflow client for testing."""
    mock = MagicMock()
    mock.start_workflow.return_value = AsyncMock()
    mock.get_workflow_handle.return_value = AsyncMock()
    return mock


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        'id': 'test-user-123',
        'username': 'testuser',
        'email': 'test@example.com',
        'roles': ['user'],
        'permissions': {
            'read': True,
            'write': False,
            'admin': False
        }
    }


@pytest.fixture
def sample_vulnerability_data():
    """Sample vulnerability data for testing."""
    return {
        'id': 'CVE-2023-12345',
        'title': 'Test Vulnerability',
        'description': 'A test vulnerability for unit testing',
        'severity': 'HIGH',
        'cvss_score': 8.5,
        'affected_products': ['test-product'],
        'published': '2023-01-01T00:00:00Z',
        'modified': '2023-01-01T00:00:00Z'
    }


@pytest.fixture
def sample_scan_result():
    """Sample scan result data for testing."""
    return {
        'scan_id': 'scan-123',
        'target': '192.168.1.1',
        'status': 'completed',
        'findings': [
            {
                'id': 'finding-1',
                'severity': 'MEDIUM',
                'title': 'Test Finding',
                'description': 'A test security finding'
            }
        ],
        'started_at': '2023-01-01T00:00:00Z',
        'completed_at': '2023-01-01T01:00:00Z'
    }


@pytest.fixture
async def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.app.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {
        'Authorization': 'Bearer test-token-123',
        'Content-Type': 'application/json'
    }


# Test utilities
def skip_if_no_database():
    """Skip test if database is not available."""
    try:
        import asyncpg
        return False
    except ImportError:
        return True


def skip_if_no_redis():
    """Skip test if Redis is not available."""
    try:
        import redis
        return False
    except ImportError:
        return True


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if hasattr(item, 'keywords') and ('sleep' in item.keywords or 'slow' in item.keywords):
            item.add_marker(pytest.mark.slow)