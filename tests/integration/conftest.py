"""Fixtures for the XORB integration test suite."""

import asyncio
import tempfile
import os
import shutil
import logging
from typing import AsyncGenerator, Generator, Tuple
import pytest
import pytest_asyncio

# --- Configure logging for tests ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Import fixtures ---
# Note: These are imported for their side effects (pytest fixtures).
# The linter might complain about unused imports, but they are essential.
# noinspection PyUnresolvedReferences
from tests.integration.fixtures.nats_server import nats_server, nats_url, nats_jetstream
# noinspection PyUnresolvedReferences
from tests.integration.fixtures.uds_ring import uds_ring_path
# noinspection PyUnresolvedReferences
from tests.integration.fixtures.certs import cert_and_key_paths, trusted_ca_path
# noinspection PyUnresolvedReferences
from tests.integration.fixtures.jwt import jwt_token_factory


# --- Session-scoped temporary directory for test data ---
@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for the entire test session."""
    temp_dir = tempfile.mkdtemp(prefix="xorb_integration_")
    logger.info(f"Created session temp dir: {temp_dir}")
    yield temp_dir
    logger.info(f"Cleaning up session temp dir: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)

# --- Per-test temporary directory ---
@pytest.fixture()
def per_test_temp_dir(temp_test_dir: str) -> Generator[str, None, None]:
    """Provide a temporary directory for a single test function."""
    test_dir = tempfile.mkdtemp(dir=temp_test_dir, prefix=f"test_{os.getpid()}_")
    logger.debug(f"Created per-test temp dir: {test_dir}")
    yield test_dir
    # Note: Cleanup is handled by the session-scoped temp_test_dir fixture.

# --- Event loop for async tests ---
# Override the default asyncio event loop to allow for scope="session" fixtures
# that need to interact with the loop.
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()