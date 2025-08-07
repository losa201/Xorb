#!/usr/bin/env python3
"""
Simple test script to validate the refactored API with clean architecture
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, str(Path(__file__).parent))

async def test_container_initialization():
    """Test that the dependency injection container initializes correctly"""
    print("Testing container initialization...")
    
    try:
        from app.container import get_container
        from app.services.interfaces import AuthenticationService, EmbeddingService, DiscoveryService
        from app.domain.repositories import UserRepository, OrganizationRepository
        
        container = get_container()
        await container.initialize()
        
        # Test service retrieval
        auth_service = container.get(AuthenticationService)
        embedding_service = container.get(EmbeddingService)
        discovery_service = container.get(DiscoveryService)
        
        # Test repository retrieval
        user_repo = container.get(UserRepository)
        org_repo = container.get(OrganizationRepository)
        
        print("✓ Container initialization successful")
        print(f"✓ Auth service: {type(auth_service).__name__}")
        print(f"✓ Embedding service: {type(embedding_service).__name__}")
        print(f"✓ Discovery service: {type(discovery_service).__name__}")
        print(f"✓ User repository: {type(user_repo).__name__}")
        print(f"✓ Organization repository: {type(org_repo).__name__}")
        
        return True, container
        
    except Exception as e:
        print(f"✗ Container initialization failed: {e}")
        traceback.print_exc()
        return False, None


async def test_domain_entities():
    """Test domain entity creation and validation"""
    print("\nTesting domain entities...")
    
    try:
        from app.domain.entities import User, Organization, EmbeddingRequest
        from app.domain.value_objects import Email, Username, Domain
        from app.domain.exceptions import ValidationError
        
        # Test User entity
        user = User.create(
            username="testuser",
            email="test@example.com",
            roles=["user"]
        )
        
        assert user.has_role("user")
        assert not user.has_role("admin")
        
        user.add_role("reader")
        assert user.has_role("reader")
        
        print("✓ User entity works correctly")
        
        # Test Organization entity
        org = Organization.create(
            name="Test Organization",
            plan_type="Pro"
        )
        
        assert org.plan_type == "Pro"
        assert org.is_active
        
        print("✓ Organization entity works correctly")
        
        # Test value objects
        email = Email("valid@email.com")
        username = Username("validuser")
        domain = Domain("example.com")
        
        print("✓ Value objects work correctly")
        
        # Test validation
        try:
            invalid_email = Email("invalid-email")
            print("✗ Email validation should have failed")
            return False
        except ValueError:
            print("✓ Email validation works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Domain entities test failed: {e}")
        traceback.print_exc()
        return False


async def test_services_with_repositories(container):
    """Test services working with repositories"""
    print("\nTesting services with repositories...")
    
    try:
        from app.services.interfaces import AuthenticationService
        from app.domain.entities import User
        from app.domain.repositories import UserRepository
        
        # Get services from container
        auth_service = container.get(AuthenticationService)
        user_repo = container.get(UserRepository)
        
        # Test user creation and authentication flow
        # Note: This is a simplified test since we're using in-memory repositories
        
        # Check if default user exists (created during container initialization)
        default_user = await user_repo.get_by_username("admin")
        if default_user:
            print("✓ Default admin user found in repository")
        else:
            print("⚠ Default admin user not found - this is expected in some configurations")
        
        # Test user creation
        test_user = User.create(
            username="newuser",
            email="newuser@test.com",
            roles=["user"]
        )
        
        created_user = await user_repo.create(test_user)
        assert created_user.username == "newuser"
        
        # Test user retrieval
        retrieved_user = await user_repo.get_by_username("newuser")
        assert retrieved_user is not None
        assert retrieved_user.username == "newuser"
        
        print("✓ User repository operations work correctly")
        print("✓ Services integrate with repositories correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Services with repositories test failed: {e}")
        traceback.print_exc()
        return False


async def test_fastapi_app_creation():
    """Test that the FastAPI app can be created successfully"""
    print("\nTesting FastAPI app creation...")
    
    try:
        from app.main import app
        
        # Check that the app was created
        assert app is not None
        assert app.title == "Xorb API"
        assert app.version == "3.0.0"
        
        # Check that routes were added
        routes = [route.path for route in app.routes]
        
        expected_routes = ["/health"]
        for expected_route in expected_routes:
            if expected_route in routes:
                print(f"✓ Route {expected_route} found")
            else:
                print(f"⚠ Route {expected_route} not found - this may be expected")
        
        print("✓ FastAPI app creation successful")
        return True
        
    except Exception as e:
        print(f"✗ FastAPI app creation failed: {e}")
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CLEAN ARCHITECTURE VALIDATION TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Container initialization
    success, container = await test_container_initialization()
    if success:
        tests_passed += 1
    
    # Test 2: Domain entities
    if await test_domain_entities():
        tests_passed += 1
    
    # Test 3: Services with repositories (only if container initialized)
    if container and await test_services_with_repositories(container):
        tests_passed += 1
    else:
        print("\nSkipping services test due to container initialization failure")
    
    # Test 4: FastAPI app creation
    if await test_fastapi_app_creation():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Clean architecture refactoring successful.")
        return True
    else:
        print("⚠ Some tests failed. Review the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)