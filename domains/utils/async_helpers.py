"""
XORB Async Utilities

Optimized async helpers for high-performance operations.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from typing import Any

try:
    import aiofiles
except ImportError:
    aiofiles = None
from contextlib import asynccontextmanager

from domains.core import config

logger = logging.getLogger(__name__)


class AsyncPool:
    """Optimized async execution pool for XORB operations."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or config.orchestration.max_concurrent_agents
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, 8))

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, func, *args, **kwargs)

    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Execute CPU-intensive function in process pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._process_pool, func, *args, **kwargs)

    async def map_concurrent(self, func: Callable, items: list[Any],
                           max_concurrent: int | None = None) -> list[Any]:
        """Execute function concurrently over items with semaphore."""
        max_concurrent = max_concurrent or self.max_workers
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _execute_with_semaphore(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                else:
                    return await self.run_in_thread(func, item)

        tasks = [_execute_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def close(self):
        """Close the execution pools."""
        self._thread_pool.shutdown(wait=False)
        self._process_pool.shutdown(wait=False)


class AsyncBatch:
    """Batched async operations with backpressure control."""

    def __init__(self, batch_size: int = 100, max_concurrent_batches: int = 5):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_batches(self, items: list[Any], processor: Callable) -> list[Any]:
        """Process items in batches with controlled concurrency."""
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        async def _process_batch(batch):
            async with self._semaphore:
                return await processor(batch)

        batch_tasks = [_process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)

        return results


class CircuitBreaker:
    """Async circuit breaker for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset circuit breaker
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e


@asynccontextmanager
async def async_file_batch_writer(file_path: str, batch_size: int = 1000):
    """Async batched file writer for high-throughput logging."""
    if not aiofiles:
        raise ImportError("aiofiles not available for async file operations")

    batch = []

    async def flush_batch():
        if batch:
            async with aiofiles.open(file_path, 'a') as f:
                await f.writelines(batch)
            batch.clear()

    try:
        yield {
            'write': lambda line: batch.append(line),
            'flush': flush_batch,
            'batch_size': batch_size
        }
    finally:
        await flush_batch()


def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Async retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator


class AsyncProfiler:
    """Async performance profiler."""

    def __init__(self):
        self.timings: dict[str, list[float]] = {}

    @asynccontextmanager
    async def profile(self, operation_name: str):
        """Profile an async operation."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(duration)

    def get_stats(self, operation_name: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation_name not in self.timings:
            return {}

        timings = self.timings[operation_name]
        return {
            'count': len(timings),
            'total': sum(timings),
            'average': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings)
        }


# Global instances
async_pool = AsyncPool()
async_profiler = AsyncProfiler()
