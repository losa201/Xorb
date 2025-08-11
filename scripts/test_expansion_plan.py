#!/usr/bin/env python3
"""
XORB Test Coverage Expansion Plan
Automates the creation of comprehensive test suites to achieve 60%+ coverage.

This script:
1. Analyzes current test coverage
2. Identifies untested code areas
3. Generates test templates
4. Creates test data factories
5. Sets up CI/CD test automation
"""

import os
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TestGap:
    """Represents an area lacking test coverage"""
    file_path: str
    class_name: str
    method_name: str
    line_start: int
    line_end: int
    complexity: int
    priority: str  # critical, high, medium, low

class TestExpansionPlanner:
    """Plans and generates comprehensive test coverage"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.test_dir = project_root / "tests"
        self.coverage_report = {}
        self.test_gaps: List[TestGap] = []
        
    def analyze_current_coverage(self) -> Dict:
        """Analyze current test coverage"""
        try:
            # Run coverage analysis
            result = subprocess.run([
                "python", "-m", "pytest", "--cov=src", "--cov-report=json:coverage.json", 
                "--quiet"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Read coverage report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    self.coverage_report = json.load(f)
            
            return self.coverage_report
            
        except Exception as e:
            print(f"Coverage analysis failed: {e}")
            return {}
    
    def identify_test_gaps(self) -> List[TestGap]:
        """Identify areas lacking test coverage"""
        gaps = []
        
        # Scan all Python files in src directory
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            try:
                gaps.extend(self._analyze_file_coverage(py_file))
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        self.test_gaps = gaps
        return gaps
    
    def _analyze_file_coverage(self, file_path: Path) -> List[TestGap]:
        """Analyze coverage gaps in a specific file"""
        gaps = []
        
        try:
            with open(file_path) as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class has tests
                    if not self._has_tests_for_class(file_path, node.name):
                        gaps.append(TestGap(
                            file_path=str(file_path.relative_to(self.project_root)),
                            class_name=node.name,
                            method_name="",
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            complexity=self._calculate_complexity(node),
                            priority=self._determine_priority(file_path, node.name)
                        ))
                    
                    # Check individual methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if not self._has_tests_for_method(file_path, node.name, item.name):
                                gaps.append(TestGap(
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    class_name=node.name,
                                    method_name=item.name,
                                    line_start=item.lineno,
                                    line_end=getattr(item, 'end_lineno', item.lineno),
                                    complexity=self._calculate_complexity(item),
                                    priority=self._determine_priority(file_path, node.name, item.name)
                                ))
                
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    # Top-level functions
                    if not self._has_tests_for_function(file_path, node.name):
                        gaps.append(TestGap(
                            file_path=str(file_path.relative_to(self.project_root)),
                            class_name="",
                            method_name=node.name,
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            complexity=self._calculate_complexity(node),
                            priority=self._determine_priority(file_path, function_name=node.name)
                        ))
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return gaps
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a code block"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, 
                                ast.With, ast.AsyncWith, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _determine_priority(self, file_path: Path, class_name: str = "", function_name: str = "") -> str:
        """Determine test priority based on code importance"""
        path_str = str(file_path).lower()
        
        # Critical priority
        if any(keyword in path_str for keyword in ["security", "auth", "crypto", "vault"]):
            return "critical"
        if any(keyword in function_name.lower() for keyword in ["auth", "login", "password", "token"]):
            return "critical"
        
        # High priority
        if any(keyword in path_str for keyword in ["service", "api", "router", "main"]):
            return "high"
        if any(keyword in class_name.lower() for keyword in ["service", "manager", "client"]):
            return "high"
        
        # Medium priority
        if any(keyword in path_str for keyword in ["common", "util", "helper"]):
            return "medium"
        
        return "low"
    
    def _has_tests_for_class(self, file_path: Path, class_name: str) -> bool:
        """Check if tests exist for a class"""
        test_file = self._get_test_file_path(file_path)
        if not test_file.exists():
            return False
        
        try:
            with open(test_file) as f:
                content = f.read()
                # Look for test class or test functions mentioning the class
                return f"Test{class_name}" in content or f"test_{class_name.lower()}" in content
        except:
            return False
    
    def _has_tests_for_method(self, file_path: Path, class_name: str, method_name: str) -> bool:
        """Check if tests exist for a method"""
        test_file = self._get_test_file_path(file_path)
        if not test_file.exists():
            return False
        
        try:
            with open(test_file) as f:
                content = f.read()
                return f"test_{method_name}" in content.lower()
        except:
            return False
    
    def _has_tests_for_function(self, file_path: Path, function_name: str) -> bool:
        """Check if tests exist for a function"""
        test_file = self._get_test_file_path(file_path)
        if not test_file.exists():
            return False
        
        try:
            with open(test_file) as f:
                content = f.read()
                return f"test_{function_name}" in content.lower()
        except:
            return False
    
    def _get_test_file_path(self, source_file: Path) -> Path:
        """Get corresponding test file path"""
        relative_path = source_file.relative_to(self.src_dir)
        test_file_name = f"test_{relative_path.name}"
        return self.test_dir / relative_path.parent / test_file_name
    
    def generate_test_templates(self) -> Dict[str, str]:
        """Generate test file templates for identified gaps"""
        templates = {}
        
        # Group gaps by file
        files_gaps = defaultdict(list)
        for gap in self.test_gaps:
            files_gaps[gap.file_path].append(gap)
        
        for file_path, gaps in files_gaps.items():
            if gaps[0].priority in ["critical", "high"]:  # Focus on high-priority gaps
                template = self._create_test_template(file_path, gaps)
                templates[file_path] = template
        
        return templates
    
    def _create_test_template(self, file_path: str, gaps: List[TestGap]) -> str:
        """Create a test template for a specific file"""
        module_name = file_path.replace("/", ".").replace(".py", "")
        file_name = Path(file_path).name.replace(".py", "")
        
        template = f'''"""
Test suite for {file_path}
Auto-generated test template - customize as needed
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import the module under test
from {module_name} import *

class Test{file_name.title().replace("_", "")}:
    """Test class for {file_name} module"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock common dependencies"""
        return {{
            "redis_client": AsyncMock(),
            "database": AsyncMock(),
            "vault_client": AsyncMock(),
            "config": Mock()
        }}
    
'''
        
        # Add test methods for each gap
        for gap in gaps:
            if gap.priority in ["critical", "high"]:
                template += self._create_test_method(gap)
        
        # Add integration tests for critical components
        if any(gap.priority == "critical" for gap in gaps):
            template += self._create_integration_tests(file_path, gaps)
        
        return template
    
    def _create_test_method(self, gap: TestGap) -> str:
        """Create a test method for a specific gap"""
        if gap.class_name and gap.method_name:
            test_name = f"test_{gap.class_name.lower()}_{gap.method_name}"
            description = f"Test {gap.class_name}.{gap.method_name}"
        elif gap.class_name:
            test_name = f"test_{gap.class_name.lower()}_initialization"
            description = f"Test {gap.class_name} class"
        else:
            test_name = f"test_{gap.method_name}"
            description = f"Test {gap.method_name} function"
        
        priority_comment = f"# Priority: {gap.priority.upper()}"
        if gap.priority == "critical":
            priority_comment += " - SECURITY CRITICAL"
        
        return f'''
    {priority_comment}
    def {test_name}(self, mock_dependencies):
        """
        {description}
        
        Test cases:
        - Happy path scenario
        - Error handling
        - Edge cases
        - Security validations (if applicable)
        """
        # TODO: Implement test logic
        # Arrange
        # Act
        # Assert
        pytest.skip("Test implementation needed")
    
    def {test_name}_error_handling(self, mock_dependencies):
        """Test error handling for {description.lower()}"""
        # TODO: Test error scenarios
        pytest.skip("Error handling test needed")
'''
    
    def _create_integration_tests(self, file_path: str, gaps: List[TestGap]) -> str:
        """Create integration tests for critical components"""
        return '''
    @pytest.mark.integration
    async def test_integration_workflow(self, mock_dependencies):
        """Integration test for complete workflow"""
        # TODO: Implement end-to-end integration test
        pytest.skip("Integration test implementation needed")
    
    @pytest.mark.security
    async def test_security_scenarios(self, mock_dependencies):
        """Test security-related scenarios"""
        # TODO: Implement security tests
        pytest.skip("Security test implementation needed")
'''
    
    def create_test_data_factories(self) -> str:
        """Create test data factories"""
        return '''"""
Test data factories for XORB platform
Provides reusable test data and fixtures
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, Any, List
from unittest.mock import AsyncMock

# Test data factories
class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_user(
        username: str = "testuser",
        email: str = "test@example.com",
        roles: List[str] = None,
        is_active: bool = True
    ) -> Dict[str, Any]:
        """Create test user data"""
        return {
            "id": str(uuid4()),
            "username": username,
            "email": email,
            "roles": roles or ["user"],
            "is_active": is_active,
            "created_at": datetime.utcnow(),
            "password_hash": "$2b$12$test_hash"
        }
    
    @staticmethod
    def create_organization(
        name: str = "Test Organization",
        plan_type: str = "Enterprise"
    ) -> Dict[str, Any]:
        """Create test organization data"""
        return {
            "id": str(uuid4()),
            "name": name,
            "plan_type": plan_type,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
    
    @staticmethod
    def create_auth_token(
        user_id: str = None,
        token_type: str = "access",
        expires_in_minutes: int = 30
    ) -> Dict[str, Any]:
        """Create test auth token data"""
        return {
            "id": str(uuid4()),
            "user_id": user_id or str(uuid4()),
            "token_type": token_type,
            "token": f"test_token_{uuid4().hex[:16]}",
            "expires_at": datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            "created_at": datetime.utcnow(),
            "is_revoked": False
        }

# Pytest fixtures
@pytest.fixture
def test_user():
    """Fixture providing a test user"""
    return TestDataFactory.create_user()

@pytest.fixture
def test_organization():
    """Fixture providing a test organization"""
    return TestDataFactory.create_organization()

@pytest.fixture
def test_auth_token():
    """Fixture providing a test auth token"""
    return TestDataFactory.create_auth_token()

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    return redis_mock

@pytest.fixture
def mock_database():
    """Mock database"""
    db_mock = AsyncMock()
    db_mock.fetch.return_value = []
    db_mock.fetchrow.return_value = None
    db_mock.execute.return_value = None
    return db_mock

@pytest.fixture
def mock_vault_client():
    """Mock Vault client"""
    vault_mock = AsyncMock()
    vault_mock.get_secret.return_value = {"test_secret": "test_value"}
    vault_mock.store_secret.return_value = True
    return vault_mock
'''
    
    def create_test_configuration(self) -> str:
        """Create pytest configuration"""
        return '''# Pytest configuration for XORB platform
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=json:coverage.json",
    "--cov-fail-under=60",
    "-v"
]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "security: Security tests",
    "performance: Performance tests",
    "slow: Slow running tests (> 1s)",
    "asyncio: Async tests"
]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning"
]
'''
    
    def generate_coverage_report(self) -> Dict:
        """Generate comprehensive coverage report"""
        critical_gaps = [gap for gap in self.test_gaps if gap.priority == "critical"]
        high_gaps = [gap for gap in self.test_gaps if gap.priority == "high"]
        
        report = {
            "summary": {
                "total_gaps": len(self.test_gaps),
                "critical_gaps": len(critical_gaps),
                "high_priority_gaps": len(high_gaps),
                "current_coverage": self.coverage_report.get("totals", {}).get("percent_covered", 0),
                "target_coverage": 60
            },
            "priority_breakdown": {
                "critical": len(critical_gaps),
                "high": len(high_gaps),
                "medium": len([gap for gap in self.test_gaps if gap.priority == "medium"]),
                "low": len([gap for gap in self.test_gaps if gap.priority == "low"])
            },
            "files_needing_tests": list(set(gap.file_path for gap in self.test_gaps)),
            "security_critical": [
                gap.file_path for gap in critical_gaps 
                if any(keyword in gap.file_path.lower() for keyword in ["auth", "security", "crypto"])
            ]
        }
        
        return report
    
    def execute_test_expansion(self):
        """Execute the complete test expansion plan"""
        print("🧪 XORB Test Coverage Expansion Plan")
        print("=" * 50)
        
        # Analyze current state
        print("1. Analyzing current test coverage...")
        coverage = self.analyze_current_coverage()
        current_coverage = coverage.get("totals", {}).get("percent_covered", 0)
        print(f"   Current coverage: {current_coverage:.1f}%")
        
        # Identify gaps
        print("2. Identifying test gaps...")
        gaps = self.identify_test_gaps()
        print(f"   Found {len(gaps)} areas needing tests")
        
        # Generate templates
        print("3. Generating test templates...")
        templates = self.generate_test_templates()
        print(f"   Created {len(templates)} test templates")
        
        # Write test files
        for file_path, template in templates.items():
            test_file = self._get_test_file_path(Path(file_path))
            test_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not test_file.exists():  # Don't overwrite existing tests
                with open(test_file, 'w') as f:
                    f.write(template)
                print(f"   Created: {test_file}")
        
        # Create test data factories
        print("4. Creating test data factories...")
        factories_file = self.test_dir / "factories.py"
        if not factories_file.exists():
            with open(factories_file, 'w') as f:
                f.write(self.create_test_data_factories())
            print(f"   Created: {factories_file}")
        
        # Generate report
        report = self.generate_coverage_report()
        with open(self.project_root / "test_expansion_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📊 Test Expansion Summary:")
        print(f"   Target Coverage: 60%")
        print(f"   Critical Gaps: {report['priority_breakdown']['critical']}")
        print(f"   High Priority: {report['priority_breakdown']['high']}")
        print(f"   Security Critical Files: {len(report['security_critical'])}")
        
        print("\n📋 Next Steps:")
        print("   1. Review generated test templates")
        print("   2. Implement critical security tests first")
        print("   3. Add test data and mocking")
        print("   4. Run tests: pytest --cov=src")
        print("   5. Iterate until 60% coverage achieved")

def main():
    project_root = Path(".").absolute()
    planner = TestExpansionPlanner(project_root)
    planner.execute_test_expansion()

if __name__ == "__main__":
    main()
'''