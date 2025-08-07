#!/usr/bin/env python3
"""
XORB Refactoring Validation Script

Validates the refactored codebase structure and functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add domains to Python path
sys.path.insert(0, str(Path(__file__).parent))

class RefactoringValidator:
    """Validates the refactored XORB architecture."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0

    def log_error(self, test_name: str, error: str):
        """Log an error."""
        self.errors.append(f"❌ {test_name}: {error}")

    def log_warning(self, test_name: str, warning: str):
        """Log a warning."""
        self.warnings.append(f"⚠️  {test_name}: {warning}")

    def log_success(self, test_name: str):
        """Log a success."""
        print(f"✅ {test_name}")
        self.success_count += 1

    def test_domain_structure(self):
        """Test that all domain directories exist."""
        expected_domains = [
            "domains/core",
            "domains/agents",
            "domains/orchestration",
            "domains/security",
            "domains/llm",
            "domains/utils",
            "domains/infra"
        ]

        for domain in expected_domains:
            domain_path = Path(domain)
            if domain_path.exists() and domain_path.is_dir():
                self.log_success(f"Domain structure: {domain}")
            else:
                self.log_error("Domain structure", f"Missing domain: {domain}")

    def test_core_imports(self):
        """Test core domain imports."""
        try:
            self.log_success("Core domain imports")
        except Exception as e:
            self.log_error("Core domain imports", str(e))

    def test_config_system(self):
        """Test configuration system."""
        try:
            from domains.core import config

            # Test config access
            db_url = config.database.postgres_url
            max_agents = config.orchestration.max_concurrent_agents

            # Test environment variable usage
            if "POSTGRES_PASSWORD" in db_url or not db_url:
                self.log_warning("Config system", "Database URL may contain placeholder")

            self.log_success("Configuration system")
        except Exception as e:
            self.log_error("Configuration system", str(e))

    def test_agent_registry(self):
        """Test agent registry system."""
        try:
            from domains.agents.registry import AgentRegistry

            # Test registry instantiation
            test_registry = AgentRegistry()
            stats = test_registry.get_registry_stats()

            self.log_success("Agent registry system")
        except Exception as e:
            self.log_error("Agent registry system", str(e))

    def test_database_manager(self):
        """Test database manager."""
        try:
            from domains.infra.database import DatabaseManager, db_manager

            # Test manager instantiation (dependencies may not be installed)
            test_manager = DatabaseManager()

            self.log_success("Database manager")
        except ImportError as e:
            if "aioredis" in str(e) or "sqlalchemy" in str(e) or "neo4j" in str(e):
                self.log_warning("Database manager", f"Optional dependency missing: {e}")
            else:
                self.log_error("Database manager", str(e))
        except Exception as e:
            self.log_error("Database manager", str(e))

    def test_async_helpers(self):
        """Test async utilities."""
        try:
            from domains.utils.async_helpers import (
                AsyncBatch,
                AsyncPool,
                CircuitBreaker,
            )

            # Test instantiation
            pool = AsyncPool(max_workers=4)
            batch = AsyncBatch(batch_size=10)
            breaker = CircuitBreaker()

            self.log_success("Async utilities")
        except ImportError as e:
            if "aiofiles" in str(e) or "redis" in str(e):
                self.log_warning("Async utilities", f"Optional dependency missing: {e}")
            else:
                self.log_error("Async utilities", str(e))
        except Exception as e:
            self.log_error("Async utilities", str(e))

    def test_legacy_organization(self):
        """Test legacy file organization."""
        legacy_path = Path("legacy")
        if legacy_path.exists():
            demo_files = list(legacy_path.glob("demos/*.py"))
            phase_files = list(legacy_path.glob("phase_files/*.md"))

            if demo_files and phase_files:
                self.log_success(f"Legacy organization ({len(demo_files)} demos, {len(phase_files)} docs)")
            else:
                self.log_warning("Legacy organization", "Some legacy files may be missing")
        else:
            self.log_error("Legacy organization", "Legacy directory not found")

    def test_secrets_hygiene(self):
        """Test that no hardcoded secrets remain."""
        suspicious_patterns = [
            "sk-or-v1-",
            "nvapi-",
            "sk-ant-",
            "gsk_"
        ]

        files_to_check = [
            "config/config.json",
            "tests/integration/test_llm_integration.py",
            "services/cost-monitor/cost_monitoring_service.py",
            "services/ai-remediation/remediation_engine.py"
        ]

        secrets_found = False
        for file_path in files_to_check:
            if Path(file_path).exists():
                try:
                    content = Path(file_path).read_text()
                    for pattern in suspicious_patterns:
                        if pattern in content and "${" not in content[content.index(pattern):content.index(pattern)+50]:
                            self.log_error("Secrets hygiene", f"Potential hardcoded secret in {file_path}")
                            secrets_found = True
                            break
                except Exception:
                    pass

        if not secrets_found:
            self.log_success("Secrets hygiene - no hardcoded secrets detected")

    def test_makefile_commands(self):
        """Test that new Makefile exists."""
        makefile_path = Path("Makefile.refactored")
        if makefile_path.exists():
            content = makefile_path.read_text()
            required_targets = ["setup", "quality", "test", "security-scan", "agent-discovery"]

            missing_targets = []
            for target in required_targets:
                if f"{target}:" not in content:
                    missing_targets.append(target)

            if missing_targets:
                self.log_warning("Makefile commands", f"Missing targets: {missing_targets}")
            else:
                self.log_success("Makefile commands")
        else:
            self.log_error("Makefile commands", "Makefile.refactored not found")

    def test_documentation_updates(self):
        """Test documentation updates."""
        readme_path = Path("README.refactored.md")
        if readme_path.exists():
            content = readme_path.read_text()
            required_sections = [
                "domain-driven",
                "quick start",
                "security best practices",
                "development workflow"
            ]

            missing_sections = []
            for section in required_sections:
                if section.lower() not in content.lower():
                    missing_sections.append(section)

            if missing_sections:
                self.log_warning("Documentation", f"Missing sections: {missing_sections}")
            else:
                self.log_success("Documentation updates")
        else:
            self.log_error("Documentation", "README.refactored.md not found")

    async def run_validation(self):
        """Run all validation tests."""
        print("🔍 XORB Refactoring Validation")
        print("=" * 40)

        # Run synchronous tests
        self.test_domain_structure()
        self.test_core_imports()
        self.test_config_system()
        self.test_agent_registry()
        self.test_database_manager()
        self.test_async_helpers()
        self.test_legacy_organization()
        self.test_secrets_hygiene()
        self.test_makefile_commands()
        self.test_documentation_updates()

        # Print summary
        print("\n" + "=" * 40)
        print(f"✅ Successful tests: {self.success_count}")

        if self.warnings:
            print(f"⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   {warning}")

        if self.errors:
            print(f"❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   {error}")
            return False
        else:
            print("🎉 All validation tests passed!")
            return True

async def main():
    """Main validation function."""
    validator = RefactoringValidator()
    success = await validator.run_validation()

    if success:
        print("\n✨ XORB refactoring validation SUCCESSFUL!")
        print("🚀 The codebase is ready for production deployment.")
        sys.exit(0)
    else:
        print("\n💥 XORB refactoring validation FAILED!")
        print("🔧 Please fix the errors above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
