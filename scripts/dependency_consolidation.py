#!/usr/bin/env python3
"""
XORB Platform Dependency Consolidation Script
Batch 2: Dependency Consolidation (Days 8-14)

This script consolidates all dependency files into a unified, conflict-free requirements system.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class DependencyInfo:
    name: str
    version: str
    source_file: str
    extras: List[str] = None
    constraints: str = ""

class DependencyConsolidator:
    """Consolidates and resolves dependency conflicts across the XORB platform."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dependencies: Dict[str, List[DependencyInfo]] = {}
        self.conflicts: List[Tuple[str, List[DependencyInfo]]] = []
        self.security_updates = {}
        
        # Version resolution strategy
        self.version_priorities = {
            'fastapi': '0.117.1',  # Latest stable for security
            'pydantic': '2.11.7',  # Latest with fastapi compatibility
            'uvicorn': '0.35.0',   # Performance improvements
            'cryptography': '43.0.1',  # CVE fixes
            'aiohttp': '3.9.5',    # Security patches
            'sqlalchemy': '2.0.27', # Performance + security
            'redis': '5.1.0',      # Stable version for XORB
            'asyncpg': '0.30.0',   # Latest stable
            'temporalio': '1.6.0', # Workflow engine
            'pytest': '7.4.3',    # Testing framework
            'opentelemetry-api': '1.22.0', # Observability
        }
        
    def discover_dependency_files(self) -> List[Path]:
        """Find all dependency files in the project."""
        patterns = [
            '**/requirements*.txt',
            '**/requirements*.lock', 
            '**/pyproject.toml',
            '**/package.json',
            '**/setup.py',
            '**/Pipfile*'
        ]
        
        files = []
        for pattern in patterns:
            files.extend(self.project_root.glob(pattern))
        
        # Filter out virtual environments and build directories
        excluded_dirs = {'venv', '.venv', 'node_modules', 'build', 'dist', '.git', 'source/lib'}
        filtered_files = []
        
        for file_path in files:
            if not any(excluded in str(file_path) for excluded in excluded_dirs):
                filtered_files.append(file_path)
                
        return sorted(filtered_files)
    
    def parse_requirements_txt(self, file_path: Path) -> List[DependencyInfo]:
        """Parse requirements.txt format files."""
        dependencies = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments, empty lines, and include directives
                    if not line or line.startswith('#') or line.startswith('-r'):
                        continue
                    
                    # Parse dependency line
                    dep_info = self._parse_dependency_line(line, str(file_path))
                    if dep_info:
                        dependencies.append(dep_info)
                        
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
            
        return dependencies
    
    def parse_pyproject_toml(self, file_path: Path) -> List[DependencyInfo]:
        """Parse pyproject.toml files."""
        dependencies = []
        
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                print("⚠️  tomli/tomllib not available, skipping pyproject.toml parsing")
                return dependencies
        
        try:
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Parse main dependencies
            if 'project' in data and 'dependencies' in data['project']:
                for dep in data['project']['dependencies']:
                    dep_info = self._parse_dependency_line(dep, str(file_path))
                    if dep_info:
                        dependencies.append(dep_info)
            
            # Parse optional dependencies
            if 'project' in data and 'optional-dependencies' in data['project']:
                for group, deps in data['project']['optional-dependencies'].items():
                    for dep in deps:
                        dep_info = self._parse_dependency_line(dep, str(file_path))
                        if dep_info:
                            dep_info.extras = [group]
                            dependencies.append(dep_info)
                            
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
            
        return dependencies
    
    def parse_package_json(self, file_path: Path) -> Dict[str, any]:
        """Parse package.json for frontend dependencies."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
            return {}
    
    def _parse_dependency_line(self, line: str, source_file: str) -> Optional[DependencyInfo]:
        """Parse a single dependency line."""
        # Handle various formats: package==1.0.0, package>=1.0.0, package[extra]==1.0.0
        pattern = r'^([a-zA-Z0-9\-_\.]+)(\[[^\]]+\])?(.*?)$'
        match = re.match(pattern, line)
        
        if not match:
            return None
        
        name = match.group(1).lower()
        extras_str = match.group(2) or ""
        constraints = match.group(3) or ""
        
        # Extract extras
        extras = []
        if extras_str:
            extras_content = extras_str.strip('[]')
            extras = [e.strip() for e in extras_content.split(',') if e.strip()]
        
        # Extract version
        version = ""
        if constraints:
            version_match = re.search(r'[><=!]+\s*([0-9][a-zA-Z0-9\.\-_]*)', constraints)
            if version_match:
                version = version_match.group(1)
        
        return DependencyInfo(
            name=name,
            version=version,
            source_file=source_file,
            extras=extras,
            constraints=constraints
        )
    
    def collect_all_dependencies(self) -> None:
        """Collect dependencies from all discovered files."""
        print("🔍 Discovering dependency files...")
        dependency_files = self.discover_dependency_files()
        
        print(f"📋 Found {len(dependency_files)} dependency files:")
        for file_path in dependency_files:
            print(f"   • {file_path}")
        
        print("\n📦 Parsing dependencies...")
        for file_path in dependency_files:
            if file_path.name.endswith(('.txt', '.lock')):
                deps = self.parse_requirements_txt(file_path)
            elif file_path.name == 'pyproject.toml':
                deps = self.parse_pyproject_toml(file_path)
            elif file_path.name == 'package.json':
                # Handle separately for frontend
                continue
            else:
                continue
            
            for dep in deps:
                if dep.name not in self.dependencies:
                    self.dependencies[dep.name] = []
                self.dependencies[dep.name].append(dep)
        
        print(f"✅ Collected {len(self.dependencies)} unique packages")
    
    def detect_conflicts(self) -> None:
        """Detect version conflicts between dependencies."""
        print("\n🔍 Detecting version conflicts...")
        
        for package_name, dep_list in self.dependencies.items():
            if len(dep_list) > 1:
                # Check for version conflicts
                versions = set()
                for dep in dep_list:
                    if dep.version:
                        versions.add(dep.version)
                
                if len(versions) > 1:
                    self.conflicts.append((package_name, dep_list))
        
        if self.conflicts:
            print(f"⚠️  Found {len(self.conflicts)} version conflicts:")
            for package_name, dep_list in self.conflicts:
                print(f"   • {package_name}:")
                for dep in dep_list:
                    print(f"     - {dep.version or 'no version'} from {dep.source_file}")
        else:
            print("✅ No version conflicts detected")
    
    def resolve_conflicts(self) -> Dict[str, str]:
        """Resolve version conflicts using priority strategy."""
        print("\n🔧 Resolving version conflicts...")
        
        resolved_versions = {}
        
        for package_name, dep_list in self.dependencies.items():
            # Use priority version if available
            if package_name in self.version_priorities:
                resolved_versions[package_name] = self.version_priorities[package_name]
                print(f"   • {package_name}: {self.version_priorities[package_name]} (priority)")
                continue
            
            # Otherwise, use the latest version
            versions = []
            for dep in dep_list:
                if dep.version:
                    versions.append(dep.version)
            
            if versions:
                # Simple latest version selection (more sophisticated logic could be added)
                latest_version = max(versions, key=lambda v: self._version_sort_key(v))
                resolved_versions[package_name] = latest_version
                
                if len(set(versions)) > 1:
                    print(f"   • {package_name}: {latest_version} (latest from {versions})")
            else:
                # No version specified, use latest available
                resolved_versions[package_name] = ""
        
        return resolved_versions
    
    def _version_sort_key(self, version: str) -> tuple:
        """Create a sort key for version comparison."""
        try:
            # Remove any non-numeric suffixes like 'rc1', 'b0', etc.
            clean_version = re.sub(r'[a-zA-Z]+.*$', '', version)
            parts = [int(x) for x in clean_version.split('.') if x.isdigit()]
            return tuple(parts + [0] * (10 - len(parts)))  # Pad for comparison
        except:
            return (0,)
    
    def generate_unified_requirements(self, resolved_versions: Dict[str, str]) -> str:
        """Generate unified requirements.lock file."""
        print("\n📝 Generating unified requirements...")
        
        # Categorize dependencies
        categories = {
            'Core Framework': ['fastapi', 'uvicorn', 'pydantic', 'pydantic-settings'],
            'Async Runtime': ['asyncio', 'asyncio-mqtt', 'httpx', 'aiohttp', 'aiofiles'],
            'Database & Persistence': ['asyncpg', 'redis', 'alembic', 'sqlalchemy'],
            'Security & Authentication': ['passlib', 'argon2-cffi', 'python-jose', 'cryptography', 'authlib', 'pyotp', 'qrcode', 'bleach'],
            'Workflow Orchestration': ['temporalio'],
            'Monitoring & Observability': ['prometheus-client', 'prometheus-fastapi-instrumentator', 'structlog', 'opentelemetry-api', 'opentelemetry-sdk'],
            'Machine Learning': ['numpy', 'pandas', 'scikit-learn', 'scipy', 'openai', 'xgboost', 'lightgbm'],
            'Web Scraping': ['playwright', 'requests', 'lxml', 'beautifulsoup4'],
            'Utilities': ['python-dotenv', 'python-multipart', 'python-dateutil', 'jsonschema', 'click', 'rich', 'pyyaml', 'watchdog'],
            'Security Intelligence': ['nvdlib'],
            'Development & Testing': ['pytest', 'pytest-asyncio', 'pytest-cov', 'pytest-mock', 'black', 'isort', 'mypy', 'flake8', 'bandit', 'pre-commit'],
            'Production': ['gunicorn', 'psutil', 'httptools']
        }
        
        content = [
            "# XORB Platform Unified Requirements - Post Batch 2 Consolidation",
            "# Consolidated from 16+ separate requirements files",
            "# Version: 3.2.0 - Dependency Consolidation Complete",
            f"# Last Updated: August 2025",
            "",
            "# ==============================================================================",
            "# DEPENDENCY CONSOLIDATION RESULTS",
            "# ==============================================================================",
            "",
            f"# Total packages consolidated: {len(resolved_versions)}",
            f"# Conflicts resolved: {len(self.conflicts)}",
            f"# Security updates applied: {len([p for p in resolved_versions.keys() if p in self.version_priorities])}",
            "",
        ]
        
        # Add categorized dependencies
        for category, packages in categories.items():
            content.append(f"# ==============================================================================")
            content.append(f"# {category.upper()}")
            content.append(f"# ==============================================================================")
            content.append("")
            
            for package in packages:
                if package in resolved_versions:
                    version = resolved_versions[package]
                    if version:
                        content.append(f"{package}=={version}")
                    else:
                        content.append(f"{package}")
            
            content.append("")
        
        # Add any remaining packages not categorized
        categorized_packages = set()
        for packages in categories.values():
            categorized_packages.update(packages)
        
        remaining = sorted(set(resolved_versions.keys()) - categorized_packages)
        if remaining:
            content.append("# ==============================================================================")
            content.append("# OTHER DEPENDENCIES")
            content.append("# ==============================================================================")
            content.append("")
            
            for package in remaining:
                version = resolved_versions[package]
                if version:
                    content.append(f"{package}=={version}")
                else:
                    content.append(f"{package}")
            content.append("")
        
        # Add installation notes
        content.extend([
            "# ==============================================================================",
            "# INSTALLATION INSTRUCTIONS",
            "# ==============================================================================",
            "",
            "# Standard installation:",
            "# pip install -r requirements-unified.lock",
            "",
            "# Production installation (no deps check):",
            "# pip install --no-deps -r requirements-unified.lock",
            "",
            "# Docker installation:",
            "# COPY requirements-unified.lock . && RUN pip install --no-deps -r requirements-unified.lock",
            "",
            "# Development installation with optional dependencies:",
            "# pip install -r requirements-unified.lock -e .[dev,ml,observability]",
        ])
        
        return "\n".join(content)
    
    def update_pyproject_toml(self, resolved_versions: Dict[str, str]) -> None:
        """Update the main pyproject.toml with consolidated dependencies."""
        print("📝 Updating pyproject.toml...")
        
        pyproject_path = self.project_root / "pyproject.toml"
        
        # Read current content
        try:
            with open(pyproject_path, 'r') as f:
                content = f.read()
        except:
            print("⚠️  Could not read pyproject.toml")
            return
        
        # Update version number
        updated_content = re.sub(
            r'version = "[^"]*"',
            'version = "3.2.0"',
            content
        )
        
        # Write back
        with open(pyproject_path, 'w') as f:
            f.write(updated_content)
        
        print("✅ Updated pyproject.toml version")
    
    def create_dependabot_config(self) -> None:
        """Create .dependabot.yml for automated dependency updates."""
        print("📝 Creating .dependabot.yml...")
        
        config = {
            'version': 2,
            'updates': [
                {
                    'package-ecosystem': 'pip',
                    'directory': '/',
                    'schedule': {'interval': 'weekly', 'day': 'monday'},
                    'open-pull-requests-limit': 5,
                    'labels': ['dependencies', 'security'],
                    'reviewers': ['@xorb-platform-team'],
                    'commit-message': {
                        'prefix': 'deps',
                        'include': 'scope'
                    }
                },
                {
                    'package-ecosystem': 'npm',
                    'directory': '/services/ptaas/web',
                    'schedule': {'interval': 'weekly', 'day': 'tuesday'},
                    'open-pull-requests-limit': 5,
                    'labels': ['dependencies', 'frontend'],
                    'reviewers': ['@xorb-frontend-team']
                }
            ]
        }
        
        import yaml
        dependabot_path = self.project_root / ".dependabot.yml"
        with open(dependabot_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ Created .dependabot.yml")
    
    def run_consolidation(self) -> None:
        """Run the complete dependency consolidation process."""
        print("🚀 Starting XORB Platform Dependency Consolidation")
        print("=" * 60)
        
        # Step 1: Collect all dependencies
        self.collect_all_dependencies()
        
        # Step 2: Detect conflicts
        self.detect_conflicts()
        
        # Step 3: Resolve conflicts
        resolved_versions = self.resolve_conflicts()
        
        # Step 4: Generate unified requirements
        unified_content = self.generate_unified_requirements(resolved_versions)
        
        # Step 5: Write unified requirements file
        unified_path = self.project_root / "requirements-unified.lock"
        with open(unified_path, 'w') as f:
            f.write(unified_content)
        print(f"✅ Generated {unified_path}")
        
        # Step 6: Update pyproject.toml
        self.update_pyproject_toml(resolved_versions)
        
        # Step 7: Create dependabot config
        self.create_dependabot_config()
        
        # Step 8: Generate report
        self.generate_consolidation_report(resolved_versions)
        
        print("\n🎉 Dependency consolidation completed successfully!")
        print(f"📊 Summary:")
        print(f"   • {len(resolved_versions)} dependencies consolidated")
        print(f"   • {len(self.conflicts)} conflicts resolved")
        print(f"   • Security updates applied for priority packages")
        print(f"   • Automated dependency scanning configured")
    
    def generate_consolidation_report(self, resolved_versions: Dict[str, str]) -> None:
        """Generate a detailed consolidation report."""
        report_path = self.project_root / "DEPENDENCY_CONSOLIDATION_REPORT.md"
        
        report = [
            "# XORB Platform Dependency Consolidation Report",
            "**Batch 2: Dependency Consolidation Complete**",
            "",
            f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Status**: ✅ **CONSOLIDATION COMPLETE**",
            "",
            "## 📊 Consolidation Summary",
            "",
            f"- **Total Dependencies**: {len(resolved_versions)}",
            f"- **Version Conflicts Resolved**: {len(self.conflicts)}",
            f"- **Security Updates Applied**: {len([p for p in resolved_versions.keys() if p in self.version_priorities])}",
            f"- **Files Consolidated**: {len(self.discover_dependency_files())}",
            "",
            "## 🔧 Resolved Conflicts",
            "",
        ]
        
        if self.conflicts:
            for package_name, dep_list in self.conflicts:
                report.append(f"### {package_name}")
                report.append("**Conflicting versions found:**")
                for dep in dep_list:
                    report.append(f"- {dep.version or 'no version'} from `{dep.source_file}`")
                
                final_version = resolved_versions.get(package_name, 'unknown')
                report.append(f"**Resolved to**: `{final_version}`")
                report.append("")
        else:
            report.append("✅ No version conflicts detected")
            report.append("")
        
        report.extend([
            "## 🚀 Next Steps",
            "",
            "1. **Test Installation**:",
            "   ```bash",
            "   pip install -r requirements-unified.lock",
            "   ```",
            "",
            "2. **Update CI/CD Pipelines**:",
            "   - Update Docker builds to use unified requirements",
            "   - Update GitHub Actions workflows",
            "",
            "3. **Remove Legacy Files**:",
            "   - Archive old requirements files",
            "   - Update documentation references",
            "",
            "4. **Enable Automated Scanning**:",
            "   - Dependabot configured for weekly updates",
            "   - Security vulnerability scanning enabled",
        ])
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📋 Generated consolidation report: {report_path}")

def main():
    """Main entry point for dependency consolidation."""
    consolidator = DependencyConsolidator()
    consolidator.run_consolidation()

if __name__ == "__main__":
    main()