#!/usr/bin/env python3
"""
Migration Validation Script
Validates that the migration to unified architecture was successful
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import importlib.util
from datetime import datetime


class MigrationValidator:
    """Validates migration to unified architecture"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.validation_results = []

    def log_result(self, test_name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Log validation result"""
        result = {
            "test_name": test_name,
            "passed": passed,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.validation_results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name} - {message}")

        if not passed and details:
            for key, value in details.items():
                print(f"    {key}: {value}")

    def validate_file_structure(self) -> bool:
        """Validate that required files exist and legacy files are removed"""
        print("\n🔍 Validating file structure...")

        # Required new files
        required_files = [
            "src/api/app/services/unified_auth_service_consolidated.py",
            "src/common/jwt_manager.py",
            "src/common/unified_config.py",
            "src/orchestrator/unified_orchestrator.py",
            "Dockerfile",
            "docker-compose.yml",
            ".env.template"
        ]

        # Legacy files that should be removed
        legacy_files = [
            "src/api/app/services/auth_security_service.py",
            "src/api/app/security/auth.py",
            "src/api/app/services/auth_service.py",
            "src/xorb/core_platform/auth.py",
            "src/api/Dockerfile",
            "src/api/Dockerfile.production",
            "src/api/Dockerfile.secure",
            "infra/docker-compose.yml",
            "infra/docker-compose.production.yml"
        ]

        all_passed = True

        # Check required files
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        self.log_result(
            "required_files_exist",
            len(missing_files) == 0,
            f"Required unified architecture files",
            {"missing_files": missing_files} if missing_files else {"all_files_present": True}
        )

        if missing_files:
            all_passed = False

        # Check legacy files are removed
        remaining_legacy = []
        for file_path in legacy_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                remaining_legacy.append(file_path)

        self.log_result(
            "legacy_files_removed",
            len(remaining_legacy) == 0,
            f"Legacy files removed",
            {"remaining_legacy_files": remaining_legacy} if remaining_legacy else {"all_legacy_removed": True}
        )

        if remaining_legacy:
            all_passed = False

        return all_passed

    def validate_imports(self) -> bool:
        """Validate that imports have been properly updated"""
        print("\n🔍 Validating import statements...")

        # Check for remaining legacy imports
        legacy_imports = [
            "from .services.auth_security_service import",
            "from .security.auth import",
            "from .services.auth_service import",
            "from common.config import Settings",
            "from api.app.security import SecuritySettings",
            "from xorb.shared.config import PlatformConfig"
        ]

        python_files = list(self.project_root.rglob("*.py"))
        files_with_legacy_imports = {}

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                file_legacy_imports = []

                for legacy_import in legacy_imports:
                    if legacy_import in content:
                        file_legacy_imports.append(legacy_import)

                if file_legacy_imports:
                    files_with_legacy_imports[str(py_file)] = file_legacy_imports

            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

        self.log_result(
            "legacy_imports_removed",
            len(files_with_legacy_imports) == 0,
            "Legacy imports have been updated",
            {"files_with_legacy_imports": files_with_legacy_imports} if files_with_legacy_imports else {}
        )

        return len(files_with_legacy_imports) == 0

    def validate_unified_services(self) -> bool:
        """Validate that unified services can be imported and instantiated"""
        print("\n🔍 Validating unified services...")

        all_passed = True

        # Test unified auth service import
        try:
            sys.path.insert(0, str(self.project_root / "src"))

            # Import unified auth service
            spec = importlib.util.spec_from_file_location(
                "unified_auth_service",
                self.project_root / "src/api/app/services/unified_auth_service_consolidated.py"
            )
            unified_auth_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(unified_auth_module)

            # Check that UnifiedAuthService class exists
            has_unified_auth = hasattr(unified_auth_module, 'UnifiedAuthService')

            self.log_result(
                "unified_auth_service_import",
                has_unified_auth,
                "UnifiedAuthService can be imported",
                {"class_found": has_unified_auth}
            )

            if not has_unified_auth:
                all_passed = False

        except Exception as e:
            self.log_result(
                "unified_auth_service_import",
                False,
                "Failed to import UnifiedAuthService",
                {"error": str(e)}
            )
            all_passed = False

        # Test JWT manager import
        try:
            spec = importlib.util.spec_from_file_location(
                "jwt_manager",
                self.project_root / "src/common/jwt_manager.py"
            )
            jwt_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(jwt_module)

            has_jwt_manager = hasattr(jwt_module, 'JWTManager')

            self.log_result(
                "jwt_manager_import",
                has_jwt_manager,
                "JWTManager can be imported",
                {"class_found": has_jwt_manager}
            )

            if not has_jwt_manager:
                all_passed = False

        except Exception as e:
            self.log_result(
                "jwt_manager_import",
                False,
                "Failed to import JWTManager",
                {"error": str(e)}
            )
            all_passed = False

        # Test unified config import
        try:
            spec = importlib.util.spec_from_file_location(
                "unified_config",
                self.project_root / "src/common/unified_config.py"
            )
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            has_get_config = hasattr(config_module, 'get_config')

            self.log_result(
                "unified_config_import",
                has_get_config,
                "Unified config can be imported",
                {"function_found": has_get_config}
            )

            if not has_get_config:
                all_passed = False

        except Exception as e:
            self.log_result(
                "unified_config_import",
                False,
                "Failed to import unified config",
                {"error": str(e)}
            )
            all_passed = False

        return all_passed

    def validate_docker_configuration(self) -> bool:
        """Validate Docker configuration"""
        print("\n🔍 Validating Docker configuration...")

        all_passed = True

        # Check Dockerfile
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            try:
                content = dockerfile_path.read_text()

                # Check for multi-stage build
                has_multistage = "FROM python:3.11-slim as" in content

                # Check for build targets
                has_dev_target = "as development" in content
                has_prod_target = "as production" in content
                has_secure_target = "as secure" in content

                self.log_result(
                    "dockerfile_structure",
                    has_multistage and has_dev_target and has_prod_target,
                    "Dockerfile has proper multi-stage structure",
                    {
                        "multistage": has_multistage,
                        "development_target": has_dev_target,
                        "production_target": has_prod_target,
                        "secure_target": has_secure_target
                    }
                )

                if not (has_multistage and has_dev_target and has_prod_target):
                    all_passed = False

            except Exception as e:
                self.log_result(
                    "dockerfile_structure",
                    False,
                    "Failed to validate Dockerfile",
                    {"error": str(e)}
                )
                all_passed = False
        else:
            self.log_result(
                "dockerfile_exists",
                False,
                "Unified Dockerfile not found"
            )
            all_passed = False

        # Check docker-compose.yml
        compose_path = self.project_root / "docker-compose.yml"
        if compose_path.exists():
            try:
                content = compose_path.read_text()

                # Check for unified services
                has_api_service = "api:" in content
                has_postgres = "postgres:" in content
                has_redis = "redis:" in content
                has_temporal = "temporal:" in content

                # Check for environment variables
                has_env_vars = "environment:" in content and "DATABASE_URL" in content

                self.log_result(
                    "docker_compose_structure",
                    has_api_service and has_postgres and has_redis,
                    "Docker Compose has required services",
                    {
                        "api_service": has_api_service,
                        "postgres": has_postgres,
                        "redis": has_redis,
                        "temporal": has_temporal,
                        "environment_vars": has_env_vars
                    }
                )

                if not (has_api_service and has_postgres and has_redis):
                    all_passed = False

            except Exception as e:
                self.log_result(
                    "docker_compose_structure",
                    False,
                    "Failed to validate docker-compose.yml",
                    {"error": str(e)}
                )
                all_passed = False
        else:
            self.log_result(
                "docker_compose_exists",
                False,
                "Unified docker-compose.yml not found"
            )
            all_passed = False

        return all_passed

    def validate_environment_template(self) -> bool:
        """Validate environment template"""
        print("\n🔍 Validating environment template...")

        env_template_path = self.project_root / ".env.template"
        if not env_template_path.exists():
            self.log_result(
                "env_template_exists",
                False,
                "Environment template not found"
            )
            return False

        try:
            content = env_template_path.read_text()

            # Check for required environment variables
            required_vars = [
                "ENVIRONMENT=",
                "DATABASE_URL=",
                "REDIS_URL=",
                "JWT_SECRET=",
                "API_HOST=",
                "API_PORT=",
                "CORS_ORIGINS="
            ]

            missing_vars = []
            for var in required_vars:
                if var not in content:
                    missing_vars.append(var.rstrip('='))

            self.log_result(
                "env_template_complete",
                len(missing_vars) == 0,
                "Environment template has required variables",
                {"missing_variables": missing_vars} if missing_vars else {}
            )

            return len(missing_vars) == 0

        except Exception as e:
            self.log_result(
                "env_template_readable",
                False,
                "Failed to read environment template",
                {"error": str(e)}
            )
            return False

    def validate_test_structure(self) -> bool:
        """Validate test structure"""
        print("\n🔍 Validating test structure...")

        # Check for integration tests
        integration_test_path = self.project_root / "tests/integration/test_unified_auth_integration.py"
        has_integration_tests = integration_test_path.exists()

        # Check for security tests
        security_test_path = self.project_root / "tests/security/security_test_framework.py"
        has_security_tests = security_test_path.exists()

        # Check for performance tests
        performance_test_path = self.project_root / "tests/performance/benchmark_unified_services.py"
        has_performance_tests = performance_test_path.exists()

        self.log_result(
            "test_structure",
            has_integration_tests and has_security_tests and has_performance_tests,
            "Test structure is properly organized",
            {
                "integration_tests": has_integration_tests,
                "security_tests": has_security_tests,
                "performance_tests": has_performance_tests
            }
        )

        return has_integration_tests and has_security_tests and has_performance_tests

    def run_syntax_check(self) -> bool:
        """Run Python syntax check on key files"""
        print("\n🔍 Running syntax validation...")

        key_files = [
            "src/api/app/services/unified_auth_service_consolidated.py",
            "src/common/jwt_manager.py",
            "src/common/unified_config.py",
            "src/orchestrator/unified_orchestrator.py"
        ]

        syntax_errors = []

        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "py_compile", str(full_path)],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        syntax_errors.append({
                            "file": file_path,
                            "error": result.stderr
                        })

                except Exception as e:
                    syntax_errors.append({
                        "file": file_path,
                        "error": str(e)
                    })

        self.log_result(
            "syntax_validation",
            len(syntax_errors) == 0,
            "Python syntax validation",
            {"syntax_errors": syntax_errors} if syntax_errors else {}
        )

        return len(syntax_errors) == 0

    def generate_validation_report(self) -> bool:
        """Generate validation report"""
        passed_tests = sum(1 for result in self.validation_results if result["passed"])
        total_tests = len(self.validation_results)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report_content = f"""# Migration Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests}
- **Failed**: {total_tests - passed_tests}
- **Success Rate**: {success_rate:.1f}%

## Test Results

"""

        for result in self.validation_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            report_content += f"### {result['test_name']}\n"
            report_content += f"**Status**: {status}\n"
            report_content += f"**Message**: {result['message']}\n"

            if result["details"]:
                report_content += "**Details**:\n"
                for key, value in result["details"].items():
                    report_content += f"- {key}: {value}\n"

            report_content += "\n"

        # Add recommendations for failed tests
        failed_tests = [r for r in self.validation_results if not r["passed"]]
        if failed_tests:
            report_content += "## Recommendations\n\n"
            report_content += "The following issues need to be addressed:\n\n"

            for result in failed_tests:
                report_content += f"- **{result['test_name']}**: {result['message']}\n"

        report_content += f"""
## Next Steps

{'✅ **Migration validation passed!** You can proceed with testing your application.' if success_rate >= 90 else '❌ **Migration validation failed.** Please address the issues above before proceeding.'}

### Testing Commands
```bash
# Run integration tests
pytest tests/integration/

# Run security tests
python tests/security/security_test_framework.py

# Run performance benchmarks
python tests/performance/benchmark_unified_services.py

# Validate configuration
python -c "from src.common.unified_config import validate_config; validate_config()"
```

### Docker Testing
```bash
# Test development build
docker-compose build

# Test production build
BUILD_TARGET=production docker-compose build

# Start services
docker-compose up -d

# Check health
curl http://localhost:8000/health
```
"""

        report_path = self.project_root / "MIGRATION_VALIDATION_REPORT.md"
        report_path.write_text(report_content)

        print(f"\n📋 Validation report generated: {report_path}")
        return success_rate >= 90

    def run_full_validation(self) -> bool:
        """Run complete migration validation"""
        print("🔍 Starting migration validation...\n")

        # Run all validation tests
        file_structure_ok = self.validate_file_structure()
        imports_ok = self.validate_imports()
        services_ok = self.validate_unified_services()
        docker_ok = self.validate_docker_configuration()
        env_ok = self.validate_environment_template()
        tests_ok = self.validate_test_structure()
        syntax_ok = self.run_syntax_check()

        # Generate report
        overall_success = self.generate_validation_report()

        # Summary
        if overall_success:
            print("\n🎉 Migration validation PASSED!")
            print("Your XORB Platform has been successfully migrated to the unified architecture.")
        else:
            print("\n❌ Migration validation FAILED!")
            print("Please review the validation report and address the issues.")

        return overall_success


def main():
    """Main validation script entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate XORB Platform migration to unified architecture"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory (default: current directory)"
    )

    args = parser.parse_args()

    # Validate project root
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Error: Project root directory not found: {project_root}")
        sys.exit(1)

    # Run validation
    validator = MigrationValidator(str(project_root))
    success = validator.run_full_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
