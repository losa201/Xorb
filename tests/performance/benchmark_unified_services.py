"""
Performance benchmarks for unified services
Comprehensive benchmarking of authentication, orchestration, and other services
"""

import asyncio
import pytest
import time
import statistics
from typing import List, Dict, Any
import redis.asyncio as redis
from unittest.mock import AsyncMock

from src.common.performance_monitor import (
    PerformanceMonitor, setup_performance_monitoring, track_performance
)
from src.api.app.services.unified_auth_service_consolidated import UnifiedAuthService
from src.orchestrator.unified_orchestrator import UnifiedOrchestrator
from src.common.jwt_manager import JWTManager


class BenchmarkSuite:
    """Comprehensive benchmark suite for unified services"""
    
    def __init__(self):
        self.performance_monitor = setup_performance_monitoring()
        self.results: Dict[str, Any] = {}
    
    async def setup(self):
        """Setup benchmark environment"""
        # Setup Redis client for testing
        self.redis_client = redis.from_url("redis://localhost:6379/2")  # Test DB
        await self.redis_client.flushdb()
        
        # Setup mock repositories
        self.mock_user_repo = AsyncMock()
        self.mock_token_repo = AsyncMock()
        
        # Setup unified auth service
        self.auth_service = UnifiedAuthService(
            user_repository=self.mock_user_repo,
            token_repository=self.mock_token_repo,
            redis_client=self.redis_client,
            secret_key="benchmark-secret-key",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
        
        # Setup orchestrator
        self.orchestrator = UnifiedOrchestrator(redis_client=self.redis_client)
        
        # Setup JWT manager
        self.jwt_manager = JWTManager()
        
        print("✅ Benchmark environment setup complete")
    
    async def teardown(self):
        """Cleanup benchmark environment"""
        await self.redis_client.flushdb()
        await self.redis_client.close()
        
        if hasattr(self.orchestrator, 'running') and self.orchestrator.running:
            await self.orchestrator.shutdown()
        
        print("✅ Benchmark environment cleaned up")
    
    async def benchmark_password_hashing(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark password hashing performance"""
        print(f"🔐 Benchmarking password hashing ({iterations} iterations)...")
        
        password = "TestPassword123!"
        
        # Benchmark hashing
        async def hash_password():
            return await self.auth_service.hash_password(password)
        
        hash_result = await self.performance_monitor.benchmark_function(
            hash_password,
            "password_hashing",
            iterations=iterations
        )
        
        # Benchmark verification
        hashed_password = await self.auth_service.hash_password(password)
        
        async def verify_password():
            return await self.auth_service.verify_password(password, hashed_password)
        
        verify_result = await self.performance_monitor.benchmark_function(
            verify_password,
            "password_verification",
            iterations=iterations
        )
        
        return {
            "hashing": {
                "ops_per_second": hash_result.operations_per_second,
                "avg_time_ms": hash_result.avg_time,
                "p95_time_ms": hash_result.p95_time,
                "success_rate": hash_result.success_rate
            },
            "verification": {
                "ops_per_second": verify_result.operations_per_second,
                "avg_time_ms": verify_result.avg_time,
                "p95_time_ms": verify_result.p95_time,
                "success_rate": verify_result.success_rate
            }
        }
    
    async def benchmark_jwt_operations(self, iterations: int = 10000) -> Dict[str, Any]:
        """Benchmark JWT token operations"""
        print(f"🎫 Benchmarking JWT operations ({iterations} iterations)...")
        
        payload = {
            "user_id": "benchmark-user",
            "username": "benchmarkuser",
            "roles": ["user", "admin"]
        }
        
        # Benchmark token creation
        async def create_token():
            return await self.jwt_manager.create_token(payload, expires_minutes=30)
        
        creation_result = await self.performance_monitor.benchmark_function(
            create_token,
            "jwt_creation",
            iterations=iterations,
            concurrent=True,
            concurrency_level=50
        )
        
        # Create a token for verification benchmarks
        test_token = await self.jwt_manager.create_token(payload, expires_minutes=30)
        
        # Benchmark token verification
        async def verify_token():
            return await self.jwt_manager.verify_token(test_token)
        
        verification_result = await self.performance_monitor.benchmark_function(
            verify_token,
            "jwt_verification",
            iterations=iterations,
            concurrent=True,
            concurrency_level=100
        )
        
        return {
            "creation": {
                "ops_per_second": creation_result.operations_per_second,
                "avg_time_ms": creation_result.avg_time,
                "p95_time_ms": creation_result.p95_time,
                "success_rate": creation_result.success_rate
            },
            "verification": {
                "ops_per_second": verification_result.operations_per_second,
                "avg_time_ms": verification_result.avg_time,
                "p95_time_ms": verification_result.p95_time,
                "success_rate": verification_result.success_rate
            }
        }
    
    async def benchmark_redis_operations(self, iterations: int = 10000) -> Dict[str, Any]:
        """Benchmark Redis operations"""
        print(f"📊 Benchmarking Redis operations ({iterations} iterations)...")
        
        # Benchmark SET operations
        async def redis_set():
            key = f"benchmark:set:{int(time.time() * 1000000)}"
            value = "benchmark_value"
            await self.redis_client.set(key, value, ex=60)
        
        set_result = await self.performance_monitor.benchmark_function(
            redis_set,
            "redis_set",
            iterations=iterations,
            concurrent=True,
            concurrency_level=100
        )
        
        # Setup keys for GET benchmark
        test_keys = []
        for i in range(100):
            key = f"benchmark:get:{i}"
            await self.redis_client.set(key, f"value_{i}", ex=300)
            test_keys.append(key)
        
        # Benchmark GET operations
        async def redis_get():
            import random
            key = random.choice(test_keys)
            await self.redis_client.get(key)
        
        get_result = await self.performance_monitor.benchmark_function(
            redis_get,
            "redis_get",
            iterations=iterations,
            concurrent=True,
            concurrency_level=100
        )
        
        # Benchmark complex operations (like account lockout checks)
        async def complex_redis_operation():
            user_id = f"user_{int(time.time() * 1000000) % 1000}"
            
            # Simulate account lockout check
            attempt_key = f"failed_attempts:{user_id}"
            current_attempts = await self.redis_client.get(attempt_key)
            attempts = int(current_attempts) + 1 if current_attempts else 1
            await self.redis_client.setex(attempt_key, 3600, attempts)
            
            if attempts >= 5:
                lock_key = f"account_lock:{user_id}"
                await self.redis_client.setex(lock_key, 1800, "locked")
        
        complex_result = await self.performance_monitor.benchmark_function(
            complex_redis_operation,
            "redis_complex",
            iterations=iterations // 10,  # Fewer iterations for complex operations
            concurrent=True,
            concurrency_level=50
        )
        
        return {
            "set_operations": {
                "ops_per_second": set_result.operations_per_second,
                "avg_time_ms": set_result.avg_time,
                "p95_time_ms": set_result.p95_time
            },
            "get_operations": {
                "ops_per_second": get_result.operations_per_second,
                "avg_time_ms": get_result.avg_time,
                "p95_time_ms": get_result.p95_time
            },
            "complex_operations": {
                "ops_per_second": complex_result.operations_per_second,
                "avg_time_ms": complex_result.avg_time,
                "p95_time_ms": complex_result.p95_time
            }
        }
    
    async def benchmark_authentication_flow(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark complete authentication flow"""
        print(f"🔑 Benchmarking authentication flow ({iterations} iterations)...")
        
        # Setup mock user
        from src.api.app.domain.entities import User
        test_user = User(
            id="benchmark-user",
            username="benchmarkuser",
            email="benchmark@test.com",
            password_hash=await self.auth_service.hash_password("TestPassword123!"),
            roles=["user"],
            is_active=True
        )
        
        self.mock_user_repo.get_by_username.return_value = test_user
        self.mock_user_repo.get_by_id.return_value = test_user
        self.mock_token_repo.save_token.return_value = True
        
        # Benchmark full authentication
        async def authenticate():
            credentials = {
                "username": "benchmarkuser",
                "password": "TestPassword123!",
                "client_ip": "192.168.1.100"
            }
            result = await self.auth_service.authenticate_user(credentials)
            return result.success
        
        auth_result = await self.performance_monitor.benchmark_function(
            authenticate,
            "full_authentication",
            iterations=iterations,
            concurrent=True,
            concurrency_level=20
        )
        
        # Benchmark token validation
        test_token, _ = self.auth_service.create_access_token(test_user)
        
        async def validate_token():
            return await self.auth_service.validate_token(test_token)
        
        validation_result = await self.performance_monitor.benchmark_function(
            validate_token,
            "token_validation",
            iterations=iterations * 2,
            concurrent=True,
            concurrency_level=50
        )
        
        return {
            "authentication": {
                "ops_per_second": auth_result.operations_per_second,
                "avg_time_ms": auth_result.avg_time,
                "p95_time_ms": auth_result.p95_time,
                "success_rate": auth_result.success_rate
            },
            "token_validation": {
                "ops_per_second": validation_result.operations_per_second,
                "avg_time_ms": validation_result.avg_time,
                "p95_time_ms": validation_result.p95_time,
                "success_rate": validation_result.success_rate
            }
        }
    
    async def benchmark_orchestrator_operations(self, iterations: int = 500) -> Dict[str, Any]:
        """Benchmark orchestrator operations"""
        print(f"🎭 Benchmarking orchestrator operations ({iterations} iterations)...")
        
        await self.orchestrator.initialize()
        
        # Create mock service definition
        from src.orchestrator.unified_orchestrator import ServiceDefinition, ServiceType
        
        service_def = ServiceDefinition(
            service_id="benchmark-service",
            name="Benchmark Service",
            service_type=ServiceType.CORE,
            module_path="tests.performance.benchmark_unified_services",
            class_name="MockBenchmarkService",
            dependencies=[],
            config={}
        )
        
        # Benchmark service registration
        def register_service():
            self.orchestrator.register_service(service_def)
        
        registration_result = await self.performance_monitor.benchmark_function(
            register_service,
            "service_registration",
            iterations=iterations,
            concurrent=False  # Registration should be sequential
        )
        
        # Benchmark metrics collection
        async def collect_metrics():
            await self.orchestrator._update_metrics()
            return self.orchestrator.get_metrics()
        
        metrics_result = await self.performance_monitor.benchmark_function(
            collect_metrics,
            "metrics_collection",
            iterations=iterations,
            concurrent=True,
            concurrency_level=10
        )
        
        return {
            "service_registration": {
                "ops_per_second": registration_result.operations_per_second,
                "avg_time_ms": registration_result.avg_time,
                "success_rate": registration_result.success_rate
            },
            "metrics_collection": {
                "ops_per_second": metrics_result.operations_per_second,
                "avg_time_ms": metrics_result.avg_time,
                "success_rate": metrics_result.success_rate
            }
        }
    
    async def benchmark_concurrent_load(self, concurrent_users: int = 100) -> Dict[str, Any]:
        """Benchmark system under concurrent load"""
        print(f"⚡ Benchmarking concurrent load ({concurrent_users} concurrent users)...")
        
        # Setup test data
        from src.api.app.domain.entities import User
        test_users = []
        for i in range(concurrent_users):
            user = User(
                id=f"load-user-{i}",
                username=f"loaduser{i}",
                email=f"load{i}@test.com",
                password_hash=await self.auth_service.hash_password("LoadTest123!"),
                roles=["user"],
                is_active=True
            )
            test_users.append(user)
        
        # Mock repository to return different users
        async def get_user_by_username(username):
            for user in test_users:
                if user.username == username:
                    return user
            return test_users[0]  # Default fallback
        
        self.mock_user_repo.get_by_username.side_effect = get_user_by_username
        
        # Simulate concurrent user sessions
        async def simulate_user_session(user_id: int):
            username = f"loaduser{user_id}"
            
            # Authenticate
            credentials = {
                "username": username,
                "password": "LoadTest123!",
                "client_ip": f"192.168.1.{user_id % 255}"
            }
            
            auth_result = await self.auth_service.authenticate_user(credentials)
            if not auth_result.success:
                return False
            
            # Validate token multiple times (simulating API calls)
            for _ in range(10):
                user = await self.auth_service.validate_token(auth_result.access_token)
                if not user:
                    return False
                
                # Small delay to simulate processing
                await asyncio.sleep(0.001)
            
            return True
        
        # Run concurrent sessions
        start_time = time.time()
        tasks = [simulate_user_session(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_sessions = sum(1 for r in results if r is True)
        failed_sessions = sum(1 for r in results if r is not True)
        total_time = end_time - start_time
        
        return {
            "concurrent_users": concurrent_users,
            "successful_sessions": successful_sessions,
            "failed_sessions": failed_sessions,
            "success_rate": (successful_sessions / concurrent_users) * 100,
            "total_time_seconds": total_time,
            "sessions_per_second": concurrent_users / total_time,
            "avg_session_time": total_time / concurrent_users
        }
    
    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        print("🚀 Starting comprehensive benchmark suite...")
        
        await self.setup()
        
        try:
            # Start performance monitoring
            await self.performance_monitor.start_monitoring(interval=5)
            
            # Run individual benchmarks
            results = {}
            
            results["password_operations"] = await self.benchmark_password_hashing(1000)
            results["jwt_operations"] = await self.benchmark_jwt_operations(10000)
            results["redis_operations"] = await self.benchmark_redis_operations(10000)
            results["authentication_flow"] = await self.benchmark_authentication_flow(1000)
            results["orchestrator_operations"] = await self.benchmark_orchestrator_operations(500)
            results["concurrent_load"] = await self.benchmark_concurrent_load(100)
            
            # Get overall performance summary
            results["performance_summary"] = self.performance_monitor.get_performance_summary(30)
            results["system_health"] = await self.performance_monitor.health_check()
            
            # Stop monitoring
            await self.performance_monitor.stop_monitoring()
            
            return results
            
        finally:
            await self.teardown()
    
    def print_benchmark_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("="*80)
        
        for category, data in results.items():
            if category in ["performance_summary", "system_health"]:
                continue
                
            print(f"\n🔹 {category.replace('_', ' ').title()}:")
            
            if isinstance(data, dict):
                for subcategory, subdata in data.items():
                    if isinstance(subdata, dict) and "ops_per_second" in subdata:
                        print(f"  {subcategory.replace('_', ' ').title()}:")
                        print(f"    Operations/sec: {subdata['ops_per_second']:.2f}")
                        print(f"    Avg time: {subdata['avg_time_ms']:.2f}ms")
                        if 'p95_time_ms' in subdata:
                            print(f"    P95 time: {subdata['p95_time_ms']:.2f}ms")
                        if 'success_rate' in subdata:
                            print(f"    Success rate: {subdata['success_rate']:.1f}%")
                    else:
                        print(f"  {subcategory}: {subdata}")
        
        # System health
        if "system_health" in results:
            health = results["system_health"]
            print(f"\n🏥 System Health: {health['status'].upper()}")
            for check, data in health["checks"].items():
                print(f"  {check.title()}: {data['status']} ({data['value']:.1f}%)")
        
        print("\n" + "="*80)


class MockBenchmarkService:
    """Mock service for orchestrator benchmarks"""
    
    def __init__(self):
        pass
    
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass
    
    async def health_check(self):
        return True


async def run_benchmarks():
    """Main function to run benchmarks"""
    suite = BenchmarkSuite()
    results = await suite.run_full_benchmark_suite()
    suite.print_benchmark_results(results)
    return results


if __name__ == "__main__":
    # Run benchmarks
    asyncio.run(run_benchmarks())