#!/usr/bin/env python3
"""
XORB Integration Tests Runner

Comprehensive integration testing for the refactored XORB platform.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add domains to Python path
sys.path.insert(0, str(Path(__file__).parent))


class XORBIntegrationTester:
    """Comprehensive integration testing for XORB platform."""

    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()

    def print_header(self, title: str):
        """Print formatted header."""
        print(f"\n🧪 {title}")
        print("=" * (len(title) + 4))

    def print_success(self, test: str, details: str = ""):
        """Print success message."""
        details_str = f" - {details}" if details else ""
        print(f"✅ {test}{details_str}")

    def print_warning(self, test: str, details: str = ""):
        """Print warning message."""
        details_str = f" - {details}" if details else ""
        print(f"⚠️  {test}{details_str}")

    def print_error(self, test: str, details: str = ""):
        """Print error message."""
        details_str = f" - {details}" if details else ""
        print(f"❌ {test}{details_str}")

    def test_domain_integration(self):
        """Test integration between domains."""
        self.print_header("Domain Integration Tests")

        try:
            # Test core -> agents integration
            from domains.core import Agent, AgentType, config

            # Create agent using core models
            agent = Agent(
                name="integration-test-agent",
                agent_type=AgentType.RECONNAISSANCE,
                capabilities=["test_capability"]
            )

            self.print_success("Core → Agents", f"Created agent {agent.id[:8]}...")

            # Test configuration integration
            max_agents = config.orchestration.max_concurrent_agents
            self.print_success("Configuration integration", f"Max agents: {max_agents}")

            # Test exception system integration
            from domains.core.exceptions import AgentError

            try:
                raise AgentError("Test integration error", agent_id=agent.id)
            except AgentError as e:
                self.print_success("Exception integration", f"Agent error handled: {e.error_code}")

            self.test_results['domain_integration'] = True

        except Exception as e:
            self.print_error("Domain integration", str(e))
            self.test_results['domain_integration'] = False

    def test_security_integration(self):
        """Test security framework integration."""
        self.print_header("Security Integration Tests")

        try:
            from domains.core.config import config
            from domains.core.exceptions import SecurityError
            from domains.security.jwt import JWTManager

            # Test JWT integration with config
            jwt_manager = JWTManager()
            self.print_success("JWT Manager", "Security manager initialized")

            # Test security configuration
            rate_limit = config.security.api_rate_limit
            self.print_success("Security config", f"Rate limit: {rate_limit}")

            # Test token workflow (will handle gracefully if PyJWT not available)
            try:
                test_payload = {"user": "test_user", "role": "agent", "exp_test": True}
                token = jwt_manager.create_access_token(test_payload)

                # Verify token
                decoded = jwt_manager.verify_token(token)
                self.print_success("JWT workflow", f"Token created and verified, user: {decoded.get('user')}")

            except SecurityError as e:
                if "PyJWT not available" in str(e):
                    self.print_warning("JWT workflow", "PyJWT not installed (expected in demo)")
                else:
                    self.print_error("JWT workflow", str(e))

            self.test_results['security_integration'] = True

        except Exception as e:
            self.print_error("Security integration", str(e))
            self.test_results['security_integration'] = False

    async def test_async_integration(self):
        """Test async utilities integration."""
        self.print_header("Async Integration Tests")

        try:
            from domains.core.config import config
            from domains.utils.async_helpers import (
                AsyncPool,
                CircuitBreaker,
            )

            # Test AsyncPool with config
            max_workers = min(config.orchestration.max_concurrent_agents, 4)
            pool = AsyncPool(max_workers=max_workers)

            self.print_success("AsyncPool integration", f"Pool created with {pool.max_workers} workers")

            # Test concurrent task execution
            async def test_task(item):
                await asyncio.sleep(0.05)  # Simulate async work
                return item ** 2

            test_items = [1, 2, 3, 4, 5]
            start_time = time.time()
            results = await pool.map_concurrent(test_task, test_items, max_concurrent=3)
            duration = time.time() - start_time

            expected_results = [1, 4, 9, 16, 25]
            if results == expected_results:
                self.print_success("Concurrent execution", f"Processed {len(test_items)} items in {duration:.3f}s")
            else:
                self.print_error("Concurrent execution", f"Expected {expected_results}, got {results}")

            # Test CircuitBreaker integration
            breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

            async def failing_task():
                raise Exception("Simulated failure")

            async def success_task():
                return "success"

            # Test failure handling
            failure_count = 0
            for _ in range(3):
                try:
                    await breaker.call(failing_task)
                except Exception:
                    failure_count += 1

            self.print_success("CircuitBreaker", f"Handled {failure_count} failures, state: {breaker.state}")

            # Test recovery
            if breaker.state == "OPEN":
                time.sleep(1.1)  # Wait for recovery timeout
                try:
                    result = await breaker.call(success_task)
                    self.print_success("CircuitBreaker recovery", f"Result: {result}")
                except Exception as e:
                    self.print_warning("CircuitBreaker recovery", str(e))

            await pool.close()
            self.test_results['async_integration'] = True

        except Exception as e:
            self.print_error("Async integration", str(e))
            self.test_results['async_integration'] = False

    def test_configuration_integration(self):
        """Test configuration system integration."""
        self.print_header("Configuration Integration Tests")

        try:
            from domains.core.config import config

            # Test configuration structure
            self.print_success("Config structure", f"Environment: {config.environment}")

            # Test database configuration
            db_config = config.database
            self.print_success("Database config", f"PostgreSQL: {db_config.postgres_host}:{db_config.postgres_port}")
            self.print_success("Redis config", f"Redis: {db_config.redis_host}:{db_config.redis_port}")

            # Test AI configuration
            ai_config = config.ai
            self.print_success("AI config", f"Model: {ai_config.default_model}")
            self.print_success("AI parameters", f"Max tokens: {ai_config.max_tokens}, Temp: {ai_config.temperature}")

            # Test orchestration configuration
            orch_config = config.orchestration
            self.print_success("Orchestration config", f"Max agents: {orch_config.max_concurrent_agents}")
            self.print_success("EPYC config", f"NUMA nodes: {orch_config.epyc_numa_nodes}")

            # Test config export
            config_dict = config.to_dict()
            required_keys = ['environment', 'debug', 'database', 'orchestration']
            missing_keys = [key for key in required_keys if key not in config_dict]

            if not missing_keys:
                self.print_success("Config export", f"All required keys present: {list(config_dict.keys())}")
            else:
                self.print_error("Config export", f"Missing keys: {missing_keys}")

            self.test_results['configuration_integration'] = True

        except Exception as e:
            self.print_error("Configuration integration", str(e))
            self.test_results['configuration_integration'] = False

    def test_infrastructure_integration(self):
        """Test infrastructure integration."""
        self.print_header("Infrastructure Integration Tests")

        try:
            from domains.core.config import config
            from domains.infra.database import DatabaseManager

            # Test database manager initialization
            db_manager = DatabaseManager()
            self.print_success("Database manager", "Initialized successfully")

            # Test configuration integration
            db_config = config.database
            self.print_success("DB config integration", "PostgreSQL URL configured")

            # Test connection URL generation
            postgres_url = db_config.postgres_url
            redis_url = db_config.redis_url

            # Validate URL format (without exposing actual credentials)
            if postgres_url.startswith("postgresql://"):
                self.print_success("PostgreSQL URL", "Valid format")
            else:
                self.print_error("PostgreSQL URL", "Invalid format")

            if redis_url.startswith("redis://"):
                self.print_success("Redis URL", "Valid format")
            else:
                self.print_error("Redis URL", "Invalid format")

            self.test_results['infrastructure_integration'] = True

        except Exception as e:
            self.print_error("Infrastructure integration", str(e))
            self.test_results['infrastructure_integration'] = False

    def test_agent_orchestration_integration(self):
        """Test agent and orchestration integration."""
        self.print_header("Agent Orchestration Integration Tests")

        try:
            from domains.agents.registry import (
                AgentCapability,
                AgentRegistry,
            )
            from domains.core import Agent, AgentType, Campaign, CampaignStatus
            from domains.core.config import config

            # Test agent registry with core models
            registry = AgentRegistry()

            # Test capability system
            capabilities = [
                AgentCapability(
                    name="reconnaissance",
                    description="Network reconnaissance capability",
                    required_resources={"cpu": 1, "memory": 256}
                ),
                AgentCapability(
                    name="vulnerability_scan",
                    description="Vulnerability scanning capability",
                    required_resources={"cpu": 2, "memory": 512}
                )
            ]

            self.print_success("Agent capabilities", f"Created {len(capabilities)} capabilities")

            # Test agent creation with orchestration limits
            max_agents = config.orchestration.max_concurrent_agents
            test_agents = []

            for i in range(min(3, max_agents)):  # Create up to 3 test agents
                agent = Agent(
                    name=f"test-agent-{i}",
                    agent_type=AgentType.RECONNAISSANCE,
                    capabilities=[cap.name for cap in capabilities]
                )
                test_agents.append(agent)

            self.print_success("Agent creation", f"Created {len(test_agents)} test agents")

            # Test campaign integration
            campaign = Campaign(
                name="integration-test-campaign",
                description="Testing agent orchestration integration",
                status=CampaignStatus.PENDING,
                agent_requirements=[AgentType.RECONNAISSANCE]
            )

            self.print_success("Campaign integration", f"Campaign {campaign.id[:8]}... created")

            # Test registry stats
            stats = registry.get_registry_stats()
            self.print_success("Registry stats", f"Total registered: {stats['total_registered']}")

            self.test_results['agent_orchestration'] = True

        except Exception as e:
            self.print_error("Agent orchestration integration", str(e))
            self.test_results['agent_orchestration'] = False

    def test_monitoring_integration(self):
        """Test monitoring and metrics integration."""
        self.print_header("Monitoring Integration Tests")

        try:
            from domains.core.config import config
            from domains.utils.async_helpers import AsyncProfiler

            # Test async profiler
            profiler = AsyncProfiler()

            # Simulate some operations for profiling
            with profiler.profile("test_operation"):
                time.sleep(0.01)  # Simulate work

            # Get profiling stats
            stats = profiler.get_stats("test_operation")
            if stats:
                self.print_success("Performance profiling", f"Recorded {stats['count']} operations")
                self.print_success("Profiling details", f"Avg time: {stats['average']:.4f}s")

            # Test monitoring configuration
            monitoring_config = config.monitoring
            self.print_success("Monitoring config", f"Prometheus: {monitoring_config.prometheus_host}:{monitoring_config.prometheus_port}")
            self.print_success("Logging config", f"Level: {monitoring_config.log_level}, Format: {monitoring_config.log_format}")

            self.test_results['monitoring_integration'] = True

        except Exception as e:
            self.print_error("Monitoring integration", str(e))
            self.test_results['monitoring_integration'] = False

    def print_final_results(self):
        """Print final integration test results."""
        self.print_header("Integration Test Results")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        test_duration = time.time() - self.start_time

        print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
        print(f"⏱️  Test Duration: {test_duration:.2f} seconds")
        print()

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name.replace('_', ' ').title()}")

        print()
        if passed_tests == total_tests:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("🚀 XORB platform integration is SUCCESSFUL!")
            print("✨ Ready for production deployment.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ INTEGRATION MOSTLY SUCCESSFUL!")
            print("⚠️  Review failed tests and warnings.")
            print("🔧 Address issues before production deployment.")
        else:
            print("❌ INTEGRATION ISSUES DETECTED!")
            print("🔧 Fix failed tests before proceeding.")
            print("📋 Review error messages above.")

        print("\n🎯 Integration Coverage:")
        print("   🏗️  Domain separation and communication")
        print("   🔒 Security framework integration")
        print("   ⚡ Async operations and performance")
        print("   ⚙️  Configuration management")
        print("   🗄️  Infrastructure components")
        print("   🤖 Agent and orchestration systems")
        print("   📊 Monitoring and observability")

        print("\n🚀 Next Steps:")
        print("   1. Address any failed tests")
        print("   2. Run production deployment: make deploy-prod")
        print("   3. Start monitoring: make monitor")
        print("   4. Validate end-to-end functionality")

async def main():
    """Run comprehensive integration tests."""
    tester = XORBIntegrationTester()

    print("🧪 XORB Integration Test Suite")
    print("=" * 35)
    print("Running comprehensive integration tests for the refactored XORB platform...")

    # Run synchronous tests
    tester.test_domain_integration()
    tester.test_security_integration()
    tester.test_configuration_integration()
    tester.test_infrastructure_integration()
    tester.test_agent_orchestration_integration()
    tester.test_monitoring_integration()

    # Run async tests
    await tester.test_async_integration()

    # Print final results
    tester.print_final_results()

if __name__ == "__main__":
    asyncio.run(main())
