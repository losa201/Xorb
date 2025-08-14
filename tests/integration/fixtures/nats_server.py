"""Fixtures for managing a local NATS JetStream server for integration tests."""

import asyncio
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from typing import Tuple, Generator, Any, AsyncGenerator

import nats
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

# --- Helper to find a free port ---
def get_free_port() -> int:
    """Get a free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

# --- NATS Server Process Management ---
@pytest.fixture(scope="session")
def nats_server() -> Generator[Tuple[str, subprocess.Popen], None, None]:
    """
    Starts a NATS server in JetStream mode for the test session.
    Yields the server URL and the process object.
    Cleans up the process on teardown.
    """
    # Use a unique port for this test session to avoid conflicts
    port = get_free_port()
    # Use a temp directory for the NATS store to ensure clean state
    nats_store_dir = tempfile.mkdtemp(prefix="nats_store_")

    nats_cmd = [
        "nats-server",
        "-js",  # Enable JetStream
        "-sd", nats_store_dir,  # Store directory
        "-p", str(port),  # Port
        "--debug",  # Enable debug logging for tests
        "--trace",  # Enable trace logging for tests
    ]

    logger.info(f"Starting NATS JetStream server with command: {' '.join(nats_cmd)}")

    try:
        # Start the NATS server process
        process = subprocess.Popen(
            nats_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        nats_url = f"nats://localhost:{port}"
        logger.info(f"NATS server started on {nats_url}. PID: {process.pid}")

        # --- Wait for NATS to be ready ---
        timeout = 15  # seconds
        start_time = time.time()
        ready = False
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                # Process has exited, likely an error
                stdout, stderr = process.communicate()
                logger.error(f"NATS server failed to start. STDOUT: {stdout}, STDERR: {stderr}")
                raise RuntimeError(f"NATS server process exited with code {process.returncode}")

            try:
                # Attempt to connect to the server
                nc = nats.connect(nats_url, allow_reconnect=False, connect_timeout=1)
                # If connect succeeds, the server is ready
                nc.close()
                ready = True
                logger.info("NATS server is ready.")
                break
            except Exception as e:
                logger.debug(f"Waiting for NATS server to be ready... ({e})")
                time.sleep(0.5)

        if not ready:
            process.terminate()
            process.wait()
            raise RuntimeError("NATS server did not become ready within the timeout period.")

        yield nats_url, process

    finally:
        # --- Teardown: Stop the NATS server ---
        if 'process' in locals() and process.poll() is None:
            logger.info(f"Terminating NATS server process (PID: {process.pid})...")
            # Try graceful shutdown first
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
                logger.info("NATS server stopped gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("NATS server did not terminate gracefully, killing it.")
                process.kill()
                process.wait()
        # Cleanup the store directory
        import shutil
        shutil.rmtree(nats_store_dir, ignore_errors=True)
        logger.info(f"Cleaned up NATS store directory: {nats_store_dir}")


# --- NATS Client Connection ---
@pytest_asyncio.fixture(scope="session")
async def nats_client(nats_server: Tuple[str, subprocess.Popen]) -> AsyncGenerator[nats.NATS, None]:
    """Provides an async NATS client connected to the test server."""
    nats_url, _ = nats_server
    nc = await nats.connect(nats_url)
    logger.debug(f"Connected NATS client to {nats_url}")
    try:
        yield nc
    finally:
        await nc.close()
        logger.debug("Closed NATS client connection.")


# --- NATS JetStream Context ---
@pytest_asyncio.fixture(scope="session")
async def nats_jetstream(nats_client: nats.NATS) -> AsyncGenerator[Tuple[nats.NATS, nats.js.JetStream], None]:
    """Provides the NATS client and JetStream context."""
    js = nats_client.jetstream()
    logger.debug("Obtained NATS JetStream context.")
    yield nats_client, js
