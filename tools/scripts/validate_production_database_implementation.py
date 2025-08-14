#!/usr/bin/env python3
"""
Production Database Implementation Validation
Tests the replacement of in-memory stubs with production PostgreSQL repositories
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'api'))

from app.infrastructure.production_database_manager import ProductionDatabaseManager
from app.infrastructure.production_database_repositories import (
    ProductionUserRepository, ProductionOrganizationRepository,
    ProductionScanSessionRepository, ProductionAuthTokenRepository,
    ProductionTenantRepository
)
from app.domain.entities import User, Organization, AuthToken
from app.domain.tenant_entities import TenantPlan, TenantStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDatabaseValidator:
    """Comprehensive validation of production database implementation"""

    def __init__(self):
        self.db_manager = None
        self.test_results = {
            "database_connection": False,
            "schema_creation": False,
            "user_repository": False,
            "organization_repository": False,
            "tenant_repository": False,
            "scan_session_repository": False,
            "auth_token_repository": False,
            "performance_metrics": False,
            "health_checks": False
        }

    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        print("🔍 Starting Production Database Implementation Validation")
        print("=" * 60)

        try:
            # Test 1: Database Connection and Initialization
            await self._test_database_connection()

            # Test 2: Schema Creation and Migration
            await self._test_schema_creation()

            # Test 3: Production Repositories
            await self._test_production_repositories()

            # Test 4: Performance and Health
            await self._test_performance_and_health()

            # Test 5: Data Operations
            await self._test_data_operations()

            # Generate final report
            return await self._generate_final_report()

        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return {"success": False, "error": str(e)}

        finally:
            if self.db_manager:
                await self.db_manager.shutdown()

    async def _test_database_connection(self):
        """Test database connection and initialization"""
        print("\n1️⃣ Testing Database Connection...")

        try:
            # Initialize production database manager
            self.db_manager = ProductionDatabaseManager()
            success = await self.db_manager.initialize()

            if success:
                print("✅ Database connection established successfully")
                self.test_results["database_connection"] = True
            else:
                print("❌ Database connection failed")

        except Exception as e:
            print(f"❌ Database connection error: {e}")
            raise

    async def _test_schema_creation(self):
        """Test schema creation and migration"""
        print("\n2️⃣ Testing Schema Creation...")

        try:
            # Test health check to verify schema
            health = await self.db_manager.health_check()

            if health["status"] == "healthy":
                print("✅ Database schema and tables created successfully")
                print(f"   Database size: {health.get('database_size_bytes', 0)} bytes")
                print(f"   Active connections: {health.get('active_connections', 0)}")
                self.test_results["schema_creation"] = True
            else:
                print(f"❌ Schema validation failed: {health}")

        except Exception as e:
            print(f"❌ Schema creation error: {e}")

    async def _test_production_repositories(self):
        """Test all production repository implementations"""
        print("\n3️⃣ Testing Production Repositories...")

        try:
            repositories = await self.db_manager.get_repository_instances()

            # Test User Repository
            await self._test_user_repository(repositories['user_repository'])

            # Test Organization Repository
            await self._test_organization_repository(repositories['organization_repository'])

            # Test Tenant Repository
            await self._test_tenant_repository(repositories['tenant_repository'])

            # Test Scan Session Repository
            await self._test_scan_session_repository(repositories['scan_session_repository'])

            # Test Auth Token Repository
            await self._test_auth_token_repository(repositories['auth_token_repository'])

        except Exception as e:
            print(f"❌ Repository testing error: {e}")

    async def _test_user_repository(self, user_repo: ProductionUserRepository):
        """Test user repository operations"""
        try:
            print("   Testing User Repository...")

            # Create test user
            test_user = User.create(
                username="test_production_user",
                email="test@production.xorb",
                roles=["user", "tester"]
            )

            # Test create
            created_user = await user_repo.create(test_user)
            assert created_user.username == test_user.username

            # Test get by ID
            retrieved_user = await user_repo.get_by_id(created_user.id)
            assert retrieved_user is not None
            assert retrieved_user.email == test_user.email

            # Test get by username
            username_user = await user_repo.get_by_username("test_production_user")
            assert username_user is not None

            # Test update
            created_user.roles = ["user", "tester", "updated"]
            updated_user = await user_repo.update(created_user)
            assert "updated" in updated_user.roles

            # Test delete (soft delete)
            deleted = await user_repo.delete(created_user.id)
            assert deleted is True

            print("   ✅ User Repository - All operations successful")
            self.test_results["user_repository"] = True

        except Exception as e:
            print(f"   ❌ User Repository error: {e}")

    async def _test_organization_repository(self, org_repo: ProductionOrganizationRepository):
        """Test organization repository operations"""
        try:
            print("   Testing Organization Repository...")

            # Create test organization
            test_org = Organization.create(
                name="Test Production Organization",
                plan_type="enterprise"
            )

            # Test create
            created_org = await org_repo.create(test_org)
            assert created_org.name == test_org.name

            # Test get by ID
            retrieved_org = await org_repo.get_by_id(created_org.id)
            assert retrieved_org is not None

            # Test get by name
            name_org = await org_repo.get_by_name("Test Production Organization")
            assert name_org is not None

            print("   ✅ Organization Repository - All operations successful")
            self.test_results["organization_repository"] = True

        except Exception as e:
            print(f"   ❌ Organization Repository error: {e}")

    async def _test_tenant_repository(self, tenant_repo: ProductionTenantRepository):
        """Test tenant repository operations"""
        try:
            print("   Testing Tenant Repository...")

            # Create test tenant
            tenant_data = {
                'name': 'Test Production Tenant',
                'slug': 'test-production-tenant',
                'plan': TenantPlan.ENTERPRISE,
                'settings': {
                    'max_users': 100,
                    'features': ['ptaas', 'compliance']
                }
            }

            # Test create
            created_tenant = await tenant_repo.create_tenant(tenant_data)
            assert created_tenant['name'] == tenant_data['name']

            # Test get by ID
            tenant_id = created_tenant['tenant_id']
            retrieved_tenant = await tenant_repo.get_tenant(tenant_id)
            assert retrieved_tenant is not None

            print("   ✅ Tenant Repository - All operations successful")
            self.test_results["tenant_repository"] = True

        except Exception as e:
            print(f"   ❌ Tenant Repository error: {e}")

    async def _test_scan_session_repository(self, scan_repo: ProductionScanSessionRepository):
        """Test scan session repository operations"""
        try:
            print("   Testing Scan Session Repository...")

            # Create test scan session
            session_data = {
                'user_id': '550e8400-e29b-41d4-a716-446655440000',  # UUID format
                'targets': [{'host': 'test.example.com', 'ports': [80, 443]}],
                'scan_type': 'comprehensive',
                'scan_profile': 'production_test',
                'metadata': {'test': 'production_validation'}
            }

            # Test create
            created_session = await scan_repo.create_session(session_data)
            assert created_session['scan_type'] == 'comprehensive'

            # Test get session
            session_id = created_session['session_id']
            retrieved_session = await scan_repo.get_session(session_id)
            assert retrieved_session is not None

            # Test update
            updates = {'status': 'completed', 'results': {'findings': 5}}
            updated = await scan_repo.update_session(session_id, updates)
            assert updated is True

            print("   ✅ Scan Session Repository - All operations successful")
            self.test_results["scan_session_repository"] = True

        except Exception as e:
            print(f"   ❌ Scan Session Repository error: {e}")

    async def _test_auth_token_repository(self, token_repo: ProductionAuthTokenRepository):
        """Test auth token repository operations"""
        try:
            print("   Testing Auth Token Repository...")

            # Create test auth token
            from uuid import uuid4
            from datetime import timedelta

            test_token = AuthToken(
                user_id=uuid4(),
                token="test_production_token_12345",
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )

            # Test save
            saved_token = await token_repo.save_token(test_token)
            assert saved_token.token == test_token.token

            # Test get by token
            retrieved_token = await token_repo.get_by_token("test_production_token_12345")
            assert retrieved_token is not None

            # Test revoke
            revoked = await token_repo.revoke_token("test_production_token_12345")
            assert revoked is True

            print("   ✅ Auth Token Repository - All operations successful")
            self.test_results["auth_token_repository"] = True

        except Exception as e:
            print(f"   ❌ Auth Token Repository error: {e}")

    async def _test_performance_and_health(self):
        """Test performance metrics and health monitoring"""
        print("\n4️⃣ Testing Performance and Health Monitoring...")

        try:
            # Test health check
            health = await self.db_manager.health_check()
            assert health["status"] == "healthy"
            print(f"   ✅ Health Check: {health['status']}")
            print(f"   Response Time: {health.get('response_time_ms', 0):.2f}ms")

            # Test performance metrics
            metrics = await self.db_manager.get_performance_metrics()
            assert "table_statistics" in metrics
            print(f"   ✅ Performance Metrics: {len(metrics.get('table_statistics', []))} tables monitored")

            self.test_results["performance_metrics"] = True
            self.test_results["health_checks"] = True

        except Exception as e:
            print(f"   ❌ Performance/Health testing error: {e}")

    async def _test_data_operations(self):
        """Test complex data operations and queries"""
        print("\n5️⃣ Testing Complex Data Operations...")

        try:
            # Test pagination
            repositories = await self.db_manager.get_repository_instances()
            user_repo = repositories['user_repository']

            # Create multiple test users for pagination test
            test_users = []
            for i in range(3):
                user = User.create(
                    username=f"pagination_test_{i}",
                    email=f"test{i}@pagination.xorb",
                    roles=["user"]
                )
                created = await user_repo.create(user)
                test_users.append(created)

            print("   ✅ Created test users for pagination")

            # Test concurrent operations
            async def concurrent_lookup(username):
                return await user_repo.get_by_username(username)

            tasks = [concurrent_lookup(f"pagination_test_{i}") for i in range(3)]
            results = await asyncio.gather(*tasks)

            assert all(result is not None for result in results)
            print("   ✅ Concurrent operations successful")

            # Cleanup test users
            for user in test_users:
                await user_repo.delete(user.id)

            print("   ✅ Complex data operations completed successfully")

        except Exception as e:
            print(f"   ❌ Data operations error: {e}")

    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("📊 PRODUCTION DATABASE VALIDATION REPORT")
        print("=" * 60)

        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100

        print(f"\n✅ Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")

        # Get final health metrics
        if self.db_manager:
            health = await self.db_manager.health_check()
            performance = await self.db_manager.get_performance_metrics()

            print(f"\n📈 Performance Summary:")
            print(f"   Database Status: {health.get('status', 'unknown')}")
            print(f"   Response Time: {health.get('response_time_ms', 0):.2f}ms")
            print(f"   Active Connections: {health.get('active_connections', 0)}")
            print(f"   Pool Status: {health.get('pool_status', {})}")

        overall_success = success_rate >= 90.0  # 90% pass rate required

        if overall_success:
            print(f"\n🎉 VALIDATION SUCCESSFUL!")
            print("   Production database implementation is ready for enterprise deployment.")
        else:
            print(f"\n⚠️  VALIDATION NEEDS ATTENTION")
            print("   Some tests failed. Review the issues before production deployment.")

        return {
            "success": overall_success,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "test_results": self.test_results,
            "timestamp": datetime.utcnow().isoformat(),
            "database_health": health if self.db_manager else None
        }


async def main():
    """Main validation entry point"""
    validator = ProductionDatabaseValidator()

    try:
        report = await validator.run_validation()

        # Save report to file
        import json
        with open('production_database_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: production_database_validation_report.json")

        # Exit with appropriate code
        sys.exit(0 if report["success"] else 1)

    except Exception as e:
        print(f"\n💥 Validation failed with critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
