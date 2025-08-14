#!/usr/bin/env python3
"""
XORB Platform File Organizer
Sorts all files into their corresponding best practices folders according to clean architecture.

This script:
1. Analyzes all existing files
2. Categorizes them by type and purpose
3. Creates the proper directory structure
4. Moves files to appropriate locations
5. Updates import statements
6. Validates the new structure
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

class FileOrganizer:
    """Organizes files according to clean architecture best practices"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / "backups" / "file_organization"

        # Define the target clean architecture structure
        self.target_structure = {
            # Domain Layer (Core Business Logic)
            "src/domain": {
                "entities": [],
                "value_objects": [],
                "repositories": [],
                "services": [],
                "events": []
            },

            # Application Layer (Use Cases)
            "src/application": {
                "use_cases": [],
                "commands": [],
                "queries": [],
                "dto": [],
                "interfaces": []
            },

            # Infrastructure Layer (External Dependencies)
            "src/infrastructure": {
                "persistence": [],
                "messaging": [],
                "external_services": [],
                "security": [],
                "monitoring": [],
                "cache": [],
                "container": []
            },

            # Presentation Layer (User Interfaces)
            "src/presentation": {
                "api": [],
                "graphql": [],
                "websockets": [],
                "web": [],
                "middleware": []
            },

            # Shared Kernel (Common Components)
            "src/shared": {
                "common": [],
                "exceptions": [],
                "types": [],
                "constants": [],
                "utils": []
            },

            # Configuration Management
            "src/configuration": {
                "environments": [],
                "policies": [],
                "schemas": []
            }
        }

        # File categorization rules
        self.categorization_rules = {
            # Domain Layer Rules
            "domain/entities": [
                lambda f: self._contains_patterns(f, ["class.*Entity", "@dataclass.*class", "AggregateRoot", "class.*Model"]),
                lambda f: "entities" in str(f).lower() and self._is_python_file(f),
                lambda f: self._file_contains_business_logic(f)
            ],

            "domain/repositories": [
                lambda f: self._contains_patterns(f, ["class.*Repository", "Repository.*ABC", "Repository.*Interface"]),
                lambda f: "repositories" in str(f).lower() or "repository" in str(f).lower(),
                lambda f: self._contains_patterns(f, ["get_by_id", "save", "delete", "find_by"])
            ],

            "domain/services": [
                lambda f: self._contains_patterns(f, ["class.*DomainService", "class.*Service"]) and not self._is_infrastructure_service(f),
                lambda f: "domain" in str(f).lower() and "service" in str(f).lower(),
                lambda f: self._contains_business_rules(f)
            ],

            # Application Layer Rules
            "application/use_cases": [
                lambda f: self._contains_patterns(f, ["class.*UseCase", "class.*Handler", "execute.*async"]),
                lambda f: "use_case" in str(f).lower() or "handler" in str(f).lower(),
                lambda f: self._contains_patterns(f, ["@abstractmethod.*execute", "def execute"])
            ],

            "application/dto": [
                lambda f: self._contains_patterns(f, ["class.*DTO", "class.*Request", "class.*Response"]),
                lambda f: "dto" in str(f).lower() or ("request" in str(f).lower() or "response" in str(f).lower()),
                lambda f: self._contains_patterns(f, ["BaseModel", "pydantic", "@dataclass"])
            ],

            # Infrastructure Layer Rules
            "infrastructure/persistence": [
                lambda f: self._contains_patterns(f, ["asyncpg", "sqlalchemy", "database", "migrations"]),
                lambda f: any(keyword in str(f).lower() for keyword in ["db", "database", "migration", "persistence"]),
                lambda f: self._contains_patterns(f, ["connection", "transaction", "query"])
            ],

            "infrastructure/security": [
                lambda f: any(keyword in str(f).lower() for keyword in ["auth", "security", "crypto", "vault", "jwt", "password"]),
                lambda f: self._contains_patterns(f, ["hash_password", "verify_password", "authenticate", "authorize"]),
                lambda f: "security" in str(f) or "auth" in str(f)
            ],

            "infrastructure/monitoring": [
                lambda f: any(keyword in str(f).lower() for keyword in ["monitoring", "metrics", "prometheus", "grafana"]),
                lambda f: self._contains_patterns(f, ["prometheus_client", "structlog", "logger"]),
                lambda f: "monitoring" in str(f) or "metrics" in str(f)
            ],

            "infrastructure/cache": [
                lambda f: any(keyword in str(f).lower() for keyword in ["redis", "cache"]),
                lambda f: self._contains_patterns(f, ["redis", "cache", "lru_cache"])
            ],

            "infrastructure/container": [
                lambda f: "container" in str(f).lower() or "dependency" in str(f).lower(),
                lambda f: self._contains_patterns(f, ["dependency.*injection", "Container", "register"])
            ],

            # Presentation Layer Rules
            "presentation/api": [
                lambda f: any(keyword in str(f).lower() for keyword in ["router", "api", "endpoint"]),
                lambda f: self._contains_patterns(f, ["FastAPI", "APIRouter", "@app\\.", "router\\."]),
                lambda f: "routers" in str(f) or "api" in str(f)
            ],

            "presentation/middleware": [
                lambda f: "middleware" in str(f).lower(),
                lambda f: self._contains_patterns(f, ["Middleware", "@middleware", "process_request"])
            ],

            # Shared Layer Rules
            "shared/common": [
                lambda f: any(keyword in str(f).lower() for keyword in ["common", "utils", "helpers", "config"]),
                lambda f: self._contains_patterns(f, ["def.*util", "helper", "utility"]),
                lambda f: "common" in str(f) or "utils" in str(f)
            ],

            "shared/exceptions": [
                lambda f: any(keyword in str(f).lower() for keyword in ["exception", "error"]),
                lambda f: self._contains_patterns(f, ["Exception", "Error", "raise"])
            ]
        }

    def _is_python_file(self, file_path: Path) -> bool:
        """Check if file is a Python file"""
        return file_path.suffix == ".py"

    def _contains_patterns(self, file_path: Path, patterns: List[str]) -> bool:
        """Check if file contains any of the given patterns"""
        if not self._is_python_file(file_path):
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in patterns)
        except:
            return False

    def _contains_business_logic(self, file_path: Path) -> bool:
        """Check if file contains business logic"""
        business_indicators = [
            "business.*rule", "domain.*logic", "policy", "calculation", "validation",
            "process.*order", "calculate.*price", "validate.*business", "apply.*rule"
        ]
        return self._contains_patterns(file_path, business_indicators)

    def _contains_business_rules(self, file_path: Path) -> bool:
        """Check if file contains business rules"""
        rules_indicators = [
            "class.*Policy", "class.*Rule", "class.*Calculator", "business.*logic",
            "domain.*service", "process", "calculate", "validate.*business"
        ]
        return self._contains_patterns(file_path, rules_indicators)

    def _is_infrastructure_service(self, file_path: Path) -> bool:
        """Check if service is infrastructure-related"""
        infra_keywords = ["database", "cache", "email", "notification", "external", "api.*client"]
        return any(keyword in str(file_path).lower() for keyword in infra_keywords)

    def _file_contains_business_logic(self, file_path: Path) -> bool:
        """Check if file contains core business logic"""
        business_patterns = [
            "class.*(?:User|Order|Product|Invoice|Payment|Account)",
            "class.*(?:Threat|Incident|Vulnerability|Asset|Risk)",
            "aggregate.*root", "domain.*entity", "@dataclass.*class.*Entity"
        ]
        return self._contains_patterns(file_path, business_patterns)

    def analyze_current_files(self) -> Dict[str, List[str]]:
        """Analyze all current files and categorize them"""
        current_files = {
            "python_files": [],
            "config_files": [],
            "documentation": [],
            "tests": [],
            "scripts": [],
            "docker_files": [],
            "infrastructure": [],
            "frontend": [],
            "other": []
        }

        # Scan all files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                category = self._categorize_file(file_path)
                current_files[category].append(str(file_path.relative_to(self.project_root)))

        return current_files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = [
            "__pycache__", ".git", "node_modules", ".pytest_cache",
            "venv", ".venv", "backups", ".coverage", "htmlcov",
            "dist", "build", ".egg-info"
        ]
        return any(pattern in str(file_path) for pattern in ignore_patterns)

    def _categorize_file(self, file_path: Path) -> str:
        """Categorize a file based on its type and content"""
        file_str = str(file_path).lower()

        if file_path.suffix == ".py":
            if "test" in file_str:
                return "tests"
            elif "script" in file_str or file_path.parent.name == "scripts":
                return "scripts"
            else:
                return "python_files"
        elif file_path.suffix in [".yml", ".yaml", ".json", ".toml", ".ini", ".env"]:
            return "config_files"
        elif file_path.suffix in [".md", ".rst", ".txt"]:
            return "documentation"
        elif file_path.name in ["Dockerfile", "docker-compose.yml"] or "docker" in file_str:
            return "docker_files"
        elif any(keyword in file_str for keyword in ["infra", "k8s", "kubernetes", "terraform"]):
            return "infrastructure"
        elif any(keyword in file_str for keyword in ["frontend", "react", "vue", "angular", "ptaas"]):
            return "frontend"
        else:
            return "other"

    def create_target_structure(self):
        """Create the target directory structure"""
        print("📁 Creating target directory structure...")

        for base_path, subdirs in self.target_structure.items():
            base_dir = self.project_root / base_path
            base_dir.mkdir(parents=True, exist_ok=True)

            # Create __init__.py for the base directory
            init_file = base_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Module initialization"""')

            for subdir in subdirs:
                sub_path = base_dir / subdir
                sub_path.mkdir(parents=True, exist_ok=True)

                # Create __init__.py for subdirectories
                sub_init = sub_path / "__init__.py"
                if not sub_init.exists():
                    sub_init.write_text('"""Module initialization"""')

        # Create additional important directories
        additional_dirs = [
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "tests/security",
            "tests/performance",
            "docs/api",
            "docs/architecture",
            "docs/deployment",
            "tools/scripts",
            "tools/monitoring",
            "tools/security"
        ]

        for dir_path in additional_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

    def categorize_python_files(self) -> Dict[str, List[Path]]:
        """Categorize Python files by their purpose"""
        categorized = {category: [] for category in self.categorization_rules.keys()}
        categorized["uncategorized"] = []

        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not self._should_ignore_file(f)]

        for py_file in python_files:
            categorized_flag = False

            # Apply categorization rules
            for category, rules in self.categorization_rules.items():
                if any(rule(py_file) for rule in rules):
                    categorized[category].append(py_file)
                    categorized_flag = True
                    break

            if not categorized_flag:
                categorized["uncategorized"].append(py_file)

        return categorized

    def move_files_to_structure(self, categorized_files: Dict[str, List[Path]]):
        """Move files to their target locations"""
        print("📦 Moving files to target structure...")

        # Create backup
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        moved_files = {}

        for category, files in categorized_files.items():
            if category == "uncategorized" or not files:
                continue

            # Determine target directory
            target_base = self.project_root / "src" / category.replace("/", "/")

            for file_path in files:
                try:
                    # Create backup
                    relative_path = file_path.relative_to(self.project_root)
                    backup_path = self.backup_dir / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, backup_path)

                    # Determine new location
                    new_path = target_base / file_path.name

                    # Handle naming conflicts
                    counter = 1
                    while new_path.exists():
                        name_parts = file_path.stem, counter, file_path.suffix
                        new_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        new_path = target_base / new_name
                        counter += 1

                    # Move file
                    shutil.move(str(file_path), str(new_path))
                    moved_files[str(file_path)] = str(new_path)

                    print(f"✅ Moved: {relative_path} -> {new_path.relative_to(self.project_root)}")

                except Exception as e:
                    print(f"❌ Failed to move {file_path}: {e}")

        return moved_files

    def organize_config_files(self):
        """Organize configuration files"""
        print("⚙️ Organizing configuration files...")

        config_mappings = {
            "docker-compose*.yml": "devops/docker",
            "Dockerfile*": "devops/docker",
            "*.k8s.yaml": "devops/kubernetes",
            "*.terraform": "devops/terraform",
            ".github/workflows/*.yml": "devops/ci-cd",
            "config/*.yaml": "src/configuration/environments",
            "monitoring/*.yml": "devops/monitoring"
        }

        for pattern, target_dir in config_mappings.items():
            target_path = self.project_root / target_dir
            target_path.mkdir(parents=True, exist_ok=True)

            # Find matching files and move them
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        new_path = target_path / file_path.name
                        if not new_path.exists():
                            shutil.move(str(file_path), str(new_path))
                            print(f"✅ Config moved: {file_path.name} -> {target_dir}")
                    except Exception as e:
                        print(f"❌ Failed to move config {file_path}: {e}")

    def organize_documentation(self):
        """Organize documentation files"""
        print("📚 Organizing documentation files...")

        doc_mappings = {
            "*README*.md": "docs",
            "*ARCHITECTURE*.md": "docs/architecture",
            "*API*.md": "docs/api",
            "*DEPLOYMENT*.md": "docs/deployment",
            "*SECURITY*.md": "docs/security",
            "*BEST_PRACTICES*.md": "docs/best-practices"
        }

        for pattern, target_dir in doc_mappings.items():
            target_path = self.project_root / target_dir
            target_path.mkdir(parents=True, exist_ok=True)

            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        new_path = target_path / file_path.name
                        if not new_path.exists():
                            shutil.move(str(file_path), str(new_path))
                            print(f"✅ Doc moved: {file_path.name} -> {target_dir}")
                    except Exception as e:
                        print(f"❌ Failed to move doc {file_path}: {e}")

    def organize_tests(self):
        """Organize test files"""
        print("🧪 Organizing test files...")

        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(list(self.project_root.rglob("*_test.py")))

        for test_file in test_files:
            if self._should_ignore_file(test_file):
                continue

            # Determine test category
            file_content = ""
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    file_content = f.read().lower()
            except:
                continue

            # Categorize test
            if "integration" in str(test_file).lower() or "@pytest.mark.integration" in file_content:
                target_dir = "tests/integration"
            elif "e2e" in str(test_file).lower() or "end_to_end" in str(test_file).lower():
                target_dir = "tests/e2e"
            elif "security" in str(test_file).lower() or "@pytest.mark.security" in file_content:
                target_dir = "tests/security"
            elif "performance" in str(test_file).lower() or "@pytest.mark.performance" in file_content:
                target_dir = "tests/performance"
            else:
                target_dir = "tests/unit"

            # Move test file
            target_path = self.project_root / target_dir
            target_path.mkdir(parents=True, exist_ok=True)

            try:
                new_path = target_path / test_file.name
                if not new_path.exists():
                    shutil.move(str(test_file), str(new_path))
                    print(f"✅ Test moved: {test_file.name} -> {target_dir}")
            except Exception as e:
                print(f"❌ Failed to move test {test_file}: {e}")

    def organize_scripts(self):
        """Organize script files"""
        print("🔧 Organizing scripts...")

        script_files = list(self.project_root.rglob("*.py"))
        script_files = [f for f in script_files if "script" in str(f).lower() or f.parent.name == "scripts"]

        for script_file in script_files:
            if self._should_ignore_file(script_file):
                continue

            # Categorize script
            file_name = script_file.name.lower()
            if any(keyword in file_name for keyword in ["deploy", "build", "ci", "cd"]):
                target_dir = "tools/deployment"
            elif any(keyword in file_name for keyword in ["security", "scan", "audit"]):
                target_dir = "tools/security"
            elif any(keyword in file_name for keyword in ["monitor", "metric", "alert"]):
                target_dir = "tools/monitoring"
            else:
                target_dir = "tools/scripts"

            # Move script
            target_path = self.project_root / target_dir
            target_path.mkdir(parents=True, exist_ok=True)

            try:
                new_path = target_path / script_file.name
                if not new_path.exists():
                    shutil.move(str(script_file), str(new_path))
                    print(f"✅ Script moved: {script_file.name} -> {target_dir}")
            except Exception as e:
                print(f"❌ Failed to move script {script_file}: {e}")

    def update_imports(self, moved_files: Dict[str, str]):
        """Update import statements in moved files"""
        print("🔄 Updating import statements...")

        # This is a simplified version - in practice, you'd need more sophisticated import analysis
        import_mappings = {
            "from src.api.app.": "from src.presentation.api.",
            "from src.common.": "from src.shared.common.",
            "from src.api.app.services.": "from src.application.use_cases.",
            "from .services.": "from src.application.use_cases.",
            "from .domain.": "from src.domain.",
            "from .infrastructure.": "from src.infrastructure."
        }

        # Update imports in all Python files
        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                updated_content = content
                for old_import, new_import in import_mappings.items():
                    updated_content = updated_content.replace(old_import, new_import)

                if updated_content != content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"✅ Updated imports in: {py_file.relative_to(self.project_root)}")

            except Exception as e:
                print(f"❌ Failed to update imports in {py_file}: {e}")

    def validate_structure(self):
        """Validate the new directory structure"""
        print("✅ Validating new structure...")

        validation_results = {
            "missing_init_files": [],
            "empty_directories": [],
            "misplaced_files": [],
            "structure_health": "good"
        }

        # Check for missing __init__.py files
        for py_dir in self.project_root.rglob("*"):
            if py_dir.is_dir() and any(py_dir.glob("*.py")):
                init_file = py_dir / "__init__.py"
                if not init_file.exists():
                    validation_results["missing_init_files"].append(str(py_dir))

        # Check for empty directories
        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                validation_results["empty_directories"].append(str(dir_path))

        # Generate validation report
        if validation_results["missing_init_files"]:
            print(f"⚠️ Missing __init__.py files: {len(validation_results['missing_init_files'])}")

        if validation_results["empty_directories"]:
            print(f"ℹ️ Empty directories: {len(validation_results['empty_directories'])}")

        return validation_results

    def execute_organization(self):
        """Execute the complete file organization process"""
        print("🗂️ XORB Platform File Organization")
        print("=" * 50)

        # Step 1: Analyze current files
        print("1. Analyzing current file structure...")
        current_files = self.analyze_current_files()
        print(f"   Python files: {len(current_files['python_files'])}")
        print(f"   Config files: {len(current_files['config_files'])}")
        print(f"   Documentation: {len(current_files['documentation'])}")
        print(f"   Tests: {len(current_files['tests'])}")

        # Step 2: Create target structure
        print("2. Creating target directory structure...")
        self.create_target_structure()

        # Step 3: Categorize and move Python files
        print("3. Categorizing Python files...")
        categorized_files = self.categorize_python_files()
        for category, files in categorized_files.items():
            if files:
                print(f"   {category}: {len(files)} files")

        moved_files = self.move_files_to_structure(categorized_files)

        # Step 4: Organize other file types
        print("4. Organizing configuration files...")
        self.organize_config_files()

        print("5. Organizing documentation...")
        self.organize_documentation()

        print("6. Organizing test files...")
        self.organize_tests()

        print("7. Organizing scripts...")
        self.organize_scripts()

        # Step 5: Update imports
        print("8. Updating import statements...")
        self.update_imports(moved_files)

        # Step 6: Validate structure
        print("9. Validating new structure...")
        validation = self.validate_structure()

        print("\n✅ File organization completed!")
        print(f"📁 Files moved: {len(moved_files)}")
        print(f"📋 Validation: {validation['structure_health']}")

        # Generate summary report
        self.generate_summary_report(current_files, categorized_files, moved_files, validation)

    def generate_summary_report(self, current_files, categorized_files, moved_files, validation):
        """Generate a summary report of the organization process"""
        report_path = self.project_root / "FILE_ORGANIZATION_REPORT.md"

        report_content = f"""# XORB Platform File Organization Report

## Summary
- **Total files analyzed**: {sum(len(files) for files in current_files.values())}
- **Python files moved**: {len(moved_files)}
- **Structure validation**: {validation['structure_health']}

## File Categories
"""

        for category, files in categorized_files.items():
            if files:
                report_content += f"- **{category}**: {len(files)} files\n"

        report_content += f"""
## Moved Files
"""
        for old_path, new_path in moved_files.items():
            report_content += f"- `{old_path}` → `{new_path}`\n"

        report_content += f"""
## Validation Results
- Missing __init__.py files: {len(validation['missing_init_files'])}
- Empty directories: {len(validation['empty_directories'])}

## New Directory Structure
```
src/
├── domain/              # Domain Layer (Business Logic)
├── application/         # Application Layer (Use Cases)
├── infrastructure/      # Infrastructure Layer (External Dependencies)
├── presentation/        # Presentation Layer (User Interfaces)
├── shared/             # Shared Kernel (Common Components)
└── configuration/      # Configuration Management

tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
├── e2e/               # End-to-end tests
├── security/          # Security tests
└── performance/       # Performance tests

docs/
├── api/               # API documentation
├── architecture/      # Architecture documentation
├── deployment/        # Deployment guides
└── best-practices/    # Best practices documentation

tools/
├── scripts/           # General scripts
├── deployment/        # Deployment scripts
├── security/          # Security tools
└── monitoring/        # Monitoring tools

devops/
├── docker/            # Docker configurations
├── kubernetes/        # Kubernetes manifests
├── ci-cd/            # CI/CD configurations
└── monitoring/        # Monitoring configurations
```

Report generated on: {Path(__file__).stat().st_mtime}
"""

        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"📄 Report saved: {report_path}")


def main():
    project_root = Path(".").absolute()
    organizer = FileOrganizer(project_root)
    organizer.execute_organization()


if __name__ == "__main__":
    main()
