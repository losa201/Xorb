"""Fixtures for simulating the Tier-1 Local Ring (UDS)."""

import logging
import os
import tempfile
from typing import Generator

import pytest

logger = logging.getLogger(__name__)

# Note: This fixture simulates the *presence* of a UDS path for components that
# need to connect to it. The actual UDS ring implementation is complex and
# outside the scope of this test harness. The test will simulate the behavior
# by directly interacting with NATS, assuming the UDS layer correctly forwards
# messages. A full UDS implementation would require significant setup (e.g., Go services).

@pytest.fixture(scope="session")
def uds_ring_path() -> Generator[str, None, None]:
    """
    Provides a path for the Tier-1 Local Ring (UDS).
    In a real system, this would be a path to a Unix Domain Socket.
    For this test, it's a placeholder to satisfy component interfaces.
    """
    # Create a temporary file path to represent the UDS socket path
    # The file itself is not created, as the socket would be.
    with tempfile.TemporaryDirectory() as tmpdir:
        uds_path = os.path.join(tmpdir, "tier1_ring.sock")
        logger.debug(f"Providing simulated UDS ring path: {uds_path}")
        yield uds_path
    # The directory and path are automatically cleaned up by TemporaryDirectory
