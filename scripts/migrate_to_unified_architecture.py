#!/usr/bin/env python3
"""
Automated Migration Script for XORB Platform Unified Architecture
Helps migrate from legacy services to the new unified architecture
"""

import os
import sys
import json
import shutil
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess


class ArchitectureMigrator:
    """Automated migration tool for unified architecture"""
    
    def __init__(self, project_root: str, backup_dir: Optional[str] = None):
        self.project_root = Path(project_root)
        self.backup_dir = Path(backup_dir) if backup_dir else self.project_root / "migration_backup"
        self.migration_log = []
        self.dry_run = False
        
        # Migration mappings
        self.import_mappings = {
            # Authentication services
            "from .services.auth_security_service import AuthSecurityService": 
                "from .services.unified_auth_service_consolidated import UnifiedAuthService",
            "from .security.auth import XORBAuthenticator": 
                "from .services.unified_auth_service_consolidated import UnifiedAuthService",
            "from .services.auth_service import AuthenticationServiceImpl": 
                "from .services.unified_auth_service_consolidated import UnifiedAuthService",
            "from xorb.core_platform.auth import UnifiedAuthService": 
                "from api.app.services.unified_auth_service_consolidated import UnifiedAuthService",
            
            # Configuration imports
            "from common.config import Settings": 
                "from common.unified_config import get_config",
            "from api.app.security import SecuritySettings": 
                "from common.unified_config import get_config",
            "from xorb.shared.config import PlatformConfig": 
                "from common.unified_config import get_config",
            
            # JWT management
            "import jwt": 
                "from common.jwt_manager import get_jwt_manager",
            "from jose import jwt": 
                "from common.jwt_manager import get_jwt_manager",
            
            # Orchestrator imports
            "from api.app.infrastructure.service_orchestrator import ServiceOrchestrator": 
                "from orchestrator.unified_orchestrator import UnifiedOrchestrator",
            "from orchestrator.workflow_orchestrator import WorkflowOrchestrator": 
                "from orchestrator.unified_orchestrator import UnifiedOrchestrator",
            "from xorb.architecture.fusion_orchestrator import FusionOrchestrator": 
                "from orchestrator.unified_orchestrator import UnifiedOrchestrator",
        }
        
        # Class name mappings
        self.class_mappings = {
            "AuthSecurityService": "UnifiedAuthService",
            "XORBAuthenticator": "UnifiedAuthService",
            "AuthenticationServiceImpl": "UnifiedAuthService",
            "ServiceOrchestrator": "UnifiedOrchestrator",
            "WorkflowOrchestrator": "UnifiedOrchestrator",
            "FusionOrchestrator": "UnifiedOrchestrator",
            "Settings": "get_config()",
            "SecuritySettings": "get_config().security",
            "PlatformConfig": "get_config()",
        }
        
        # Configuration access patterns
        self.config_patterns = [
            # Old: settings.jwt_secret -> New: config.security.jwt_secret
            (r'settings\.jwt_secret', 'config.security.jwt_secret'),
            (r'settings\.database_url', 'config.database.url'),
            (r'settings\.redis_url', 'config.redis.url'),
            (r'SecuritySettings\(\)\.([a-zA-Z_]+)', r'get_config().security.\1'),
            (r'PlatformConfig\.([A-Z_]+)', r'get_config().epyc.\1'),
            
            # Docker environment variables
            (r'JWT_SECRET\s*=\s*os\.getenv\(["\']JWT_SECRET["\'][^)]*\)', 
             '# JWT_SECRET moved to centralized jwt_manager'),
            
            # Password context
            (r'pwd_context\s*=\s*CryptContext\([^)]+\)', 
             '# Password hashing moved to unified auth service'),
        ]
    
    def log(self, message: str, level: str = "INFO"):
        """Log migration message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.migration_log.append(log_entry)
        print(log_entry)
    
    def create_backup(self):
        """Create backup of current codebase"""
        if self.dry_run:
            self.log("DRY RUN: Would create backup", "INFO")
            return
        
        if self.backup_dir.exists():
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_backup = self.backup_dir.parent / f"migration_backup_{backup_timestamp}"
            shutil.move(str(self.backup_dir), str(old_backup))
            self.log(f"Moved existing backup to {old_backup}", "INFO")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy critical files
        critical_dirs = ["src", "tests", "requirements.txt", "docker-compose*.yml", "Dockerfile*"]
        
        for pattern in critical_dirs:
            for item in self.project_root.glob(pattern):
                if item.is_file():
                    shutil.copy2(item, self.backup_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, self.backup_dir / item.name, ignore_dangling_symlinks=True)
        
        self.log(f"Created backup in {self.backup_dir}", "SUCCESS")
    
    def analyze_codebase(self) -> Dict[str, List[str]]:
        """Analyze codebase for migration requirements"""
        self.log("Analyzing codebase for migration requirements...", "INFO")
        
        analysis = {
            "auth_service_usage": [],
            "config_usage": [],
            "jwt_usage": [],
            "orchestrator_usage": [],
            "docker_files": [],
            "requirements_files": [],
            "test_files": []
        }
        
        # Find Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for auth service usage
                if any(pattern in content for pattern in [
                    "AuthSecurityService", "XORBAuthenticator", "AuthenticationServiceImpl"
                ]):
                    analysis["auth_service_usage"].append(str(py_file))
                
                # Check for config usage
                if any(pattern in content for pattern in [
                    "from common.config import", "SecuritySettings", "PlatformConfig"
                ]):
                    analysis["config_usage"].append(str(py_file))
                
                # Check for JWT usage
                if any(pattern in content for pattern in [
                    "JWT_SECRET", "jwt.encode", "jwt.decode"
                ]):
                    analysis["jwt_usage"].append(str(py_file))
                
                # Check for orchestrator usage
                if any(pattern in content for pattern in [
                    "ServiceOrchestrator", "WorkflowOrchestrator", "FusionOrchestrator"
                ]):
                    analysis["orchestrator_usage"].append(str(py_file))
                
                # Check for test files
                if "test_" in py_file.name or py_file.parent.name in ["tests", "test"]:
                    analysis["test_files"].append(str(py_file))
                    
            except Exception as e:
                self.log(f"Error analyzing {py_file}: {e}", "WARNING")
        
        # Find Docker files
        docker_files = list(self.project_root.rglob("Dockerfile*")) + \
                      list(self.project_root.rglob("docker-compose*.yml"))
        analysis["docker_files"] = [str(f) for f in docker_files]
        
        # Find requirements files
        req_files = list(self.project_root.rglob("requirements*.txt"))
        analysis["requirements_files"] = [str(f) for f in req_files]
        
        self.log(f"Analysis complete: Found {len(analysis['auth_service_usage'])} files with auth usage", "INFO")
        return analysis
    
    def migrate_imports(self, file_path: Path) -> bool:
        """Migrate imports in a Python file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            modified = False
            
            # Apply import mappings
            for old_import, new_import in self.import_mappings.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    modified = True
                    self.log(f"Updated import in {file_path}: {old_import} -> {new_import}", "INFO")
            
            # Apply regex patterns for configuration
            for pattern, replacement in self.config_patterns:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    self.log(f"Updated pattern in {file_path}: {pattern}", "INFO")
            
            # Update class instantiations
            for old_class, new_class in self.class_mappings.items():
                # Handle class instantiation
                class_pattern = rf'\b{old_class}\s*\('
                if re.search(class_pattern, content):
                    content = re.sub(class_pattern, f'{new_class}(', content)
                    modified = True
                    self.log(f"Updated class usage in {file_path}: {old_class} -> {new_class}", "INFO")
            
            if modified and not self.dry_run:
                file_path.write_text(content, encoding='utf-8')
                return True
            elif modified and self.dry_run:
                self.log(f"DRY RUN: Would update {file_path}", "INFO")
                return True
                
        except Exception as e:
            self.log(f"Error migrating {file_path}: {e}", "ERROR")
            
        return False
    
    def migrate_docker_files(self) -> bool:
        """Migrate Docker configuration"""
        self.log("Migrating Docker configuration...", "INFO")
        
        # Remove old Docker files
        old_docker_files = [
            "src/api/Dockerfile",
            "src/api/Dockerfile.production", 
            "src/api/Dockerfile.secure",
            "infra/docker-compose.yml",
            "infra/docker-compose.production.yml",
            "docker-compose.monitoring.yml",
            "docker-compose.security.yml"
        ]
        
        removed_files = []
        for docker_file in old_docker_files:
            file_path = self.project_root / docker_file
            if file_path.exists():
                if not self.dry_run:
                    file_path.unlink()
                removed_files.append(docker_file)
                self.log(f"Removed old Docker file: {docker_file}", "INFO")
        
        # Update docker-compose references
        compose_files = list(self.project_root.rglob("docker-compose*.yml"))
        for compose_file in compose_files:
            try:
                content = compose_file.read_text()
                
                # Update build contexts to use new unified Dockerfile
                updated_content = re.sub(
                    r'dockerfile:\s*src/api/Dockerfile[.\w]*',
                    'dockerfile: Dockerfile',
                    content
                )
                
                # Update build targets
                updated_content = re.sub(
                    r'target:\s*(development|production|secure)',
                    r'target: \1',
                    updated_content
                )
                
                if updated_content != content and not self.dry_run:
                    compose_file.write_text(updated_content)
                    self.log(f"Updated Docker Compose file: {compose_file}", "INFO")
                    
            except Exception as e:
                self.log(f"Error updating {compose_file}: {e}", "ERROR")
        
        return len(removed_files) > 0
    
    def migrate_requirements(self) -> bool:
        """Migrate requirements files"""
        self.log("Migrating requirements files...", "INFO")
        
        # Check if service-specific requirements are just redirects
        service_req_files = [
            "src/api/requirements.txt",
            "src/orchestrator/requirements.txt", 
            "src/services/worker/requirements.txt"
        ]
        
        redirected_files = []
        for req_file in service_req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                try:
                    content = file_path.read_text().strip()
                    if content.startswith("-r ../../requirements"):
                        redirected_files.append(req_file)
                        self.log(f"Requirements file {req_file} already redirected", "INFO")
                    else:
                        # Convert to redirect
                        if not self.dry_run:
                            redirect_content = "# This file redirects to the unified requirements\n"
                            redirect_content += "# Migrated to unified requirements management\n"
                            redirect_content += "-r ../../requirements.txt\n"
                            file_path.write_text(redirect_content)
                        self.log(f"Converted {req_file} to redirect to unified requirements", "INFO")
                        redirected_files.append(req_file)
                        
                except Exception as e:
                    self.log(f"Error processing {req_file}: {e}", "ERROR")
        
        return len(redirected_files) > 0
    
    def update_tests(self, analysis: Dict[str, List[str]]) -> bool:
        """Update test files for new architecture"""
        self.log("Updating test files...", "INFO")
        
        updated_count = 0
        
        for test_file in analysis["test_files"]:
            file_path = Path(test_file)
            
            try:
                content = file_path.read_text(encoding='utf-8')
                modified = False
                
                # Update test imports
                test_specific_mappings = {
                    "from app.services.auth_security_service import AuthSecurityService":
                        "from app.services.unified_auth_service_consolidated import UnifiedAuthService",
                    
                    "AuthSecurityService(": "UnifiedAuthService(",
                    "auth_security_service": "unified_auth_service",
                    
                    # Update mock patterns
                    "mock_auth_security_service": "mock_unified_auth_service",
                    "@pytest.fixture\ndef auth_security_service": "@pytest.fixture\ndef unified_auth_service",
                }
                
                for old_pattern, new_pattern in test_specific_mappings.items():
                    if old_pattern in content:
                        content = content.replace(old_pattern, new_pattern)
                        modified = True
                
                # Update test function names
                content = re.sub(
                    r'def test_auth_security_([a-z_]+)',
                    r'def test_unified_auth_\1',
                    content
                )
                
                if modified:
                    if not self.dry_run:
                        file_path.write_text(content, encoding='utf-8')
                    updated_count += 1
                    self.log(f"Updated test file: {test_file}", "INFO")
                    
            except Exception as e:
                self.log(f"Error updating test {test_file}: {e}", "ERROR")
        
        return updated_count > 0
    
    def create_env_template(self):
        """Create environment template with new configuration"""
        env_template_path = self.project_root / ".env.template"
        
        env_template_content = """# XORB Platform Environment Configuration
# Copy this file to .env and update with your values

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://xorb:xorb_password@localhost:5432/xorb
DB_POOL_SIZE=20
DB_MAX_CONNECTIONS=50

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=20

# Security Configuration
JWT_SECRET=your-jwt-secret-change-in-production
JWT_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Password Security
PASSWORD_MIN_LENGTH=12
PASSWORD_REQUIRE_SPECIAL=true
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_BURST=50

# SSO Configuration (Optional)
AZURE_SSO_ENABLED=false
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
AZURE_TENANT=common

GOOGLE_SSO_ENABLED=false
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

OKTA_SSO_ENABLED=false
OKTA_CLIENT_ID=
OKTA_CLIENT_SECRET=
OKTA_DOMAIN=

GITHUB_SSO_ENABLED=false
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
TRACING_ENABLED=true

# External Services
TEMPORAL_HOST=localhost:7233
SCANNER_SERVICE_URL=http://localhost:8001
COMPLIANCE_SERVICE_URL=http://localhost:8002

# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=100
S3_ENABLED=false
S3_BUCKET=
S3_REGION=us-east-1

# EPYC Optimization (Optional)
NUMA_NODES=2
CPU_CORES=16
MEMORY_GB=32
ENABLE_NUMA_OPTIMIZATION=true
"""
        
        if not self.dry_run:
            env_template_path.write_text(env_template_content)
        
        self.log(f"Created environment template: {env_template_path}", "SUCCESS")
    
    def run_migration(self, analysis: Dict[str, List[str]]) -> Dict[str, bool]:
        """Run the complete migration process"""
        self.log("Starting unified architecture migration...", "INFO")
        
        results = {
            "imports_migrated": False,
            "docker_migrated": False,
            "requirements_migrated": False,
            "tests_updated": False,
            "env_template_created": False
        }
        
        # Migrate imports in Python files
        files_to_migrate = set()
        files_to_migrate.update(analysis["auth_service_usage"])
        files_to_migrate.update(analysis["config_usage"])
        files_to_migrate.update(analysis["jwt_usage"])
        files_to_migrate.update(analysis["orchestrator_usage"])
        
        import_changes = 0
        for file_path_str in files_to_migrate:
            file_path = Path(file_path_str)
            if self.migrate_imports(file_path):
                import_changes += 1
        
        results["imports_migrated"] = import_changes > 0
        
        # Migrate Docker configuration
        results["docker_migrated"] = self.migrate_docker_files()
        
        # Migrate requirements
        results["requirements_migrated"] = self.migrate_requirements()
        
        # Update tests
        results["tests_updated"] = self.update_tests(analysis)
        
        # Create environment template
        self.create_env_template()
        results["env_template_created"] = True
        
        return results
    
    def generate_migration_report(self, analysis: Dict[str, List[str]], results: Dict[str, bool]):
        """Generate migration report"""
        report_path = self.project_root / "MIGRATION_REPORT.md"
        
        report_content = f"""# XORB Platform Migration Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Migration Summary

### Files Analyzed
- Authentication service usage: {len(analysis['auth_service_usage'])} files
- Configuration usage: {len(analysis['config_usage'])} files  
- JWT usage: {len(analysis['jwt_usage'])} files
- Orchestrator usage: {len(analysis['orchestrator_usage'])} files
- Docker files: {len(analysis['docker_files'])} files
- Requirements files: {len(analysis['requirements_files'])} files
- Test files: {len(analysis['test_files'])} files

### Migration Results
- ✅ Imports migrated: {results['imports_migrated']}
- ✅ Docker configuration migrated: {results['docker_migrated']}
- ✅ Requirements migrated: {results['requirements_migrated']}
- ✅ Tests updated: {results['tests_updated']}
- ✅ Environment template created: {results['env_template_created']}

## Post-Migration Steps

1. **Review and test the migrated code**
   ```bash
   # Run tests to ensure everything works
   pytest tests/
   
   # Run security tests
   python tests/security/security_test_framework.py
   
   # Run performance benchmarks
   python tests/performance/benchmark_unified_services.py
   ```

2. **Update environment configuration**
   ```bash
   # Copy the template and update with your values
   cp .env.template .env
   # Edit .env with your actual configuration
   ```

3. **Build and test Docker containers**
   ```bash
   # Build development image
   docker-compose build
   
   # Test development environment
   docker-compose up -d
   
   # Build production image
   BUILD_TARGET=production docker-compose build
   ```

4. **Update CI/CD pipelines**
   - Update build scripts to use new Dockerfile
   - Update test commands to use new test structure
   - Update deployment scripts for new unified architecture

## Files Modified

### Python Files with Import Changes
{chr(10).join([f"- {f}" for f in analysis['auth_service_usage'][:10]])}
{f"... and {len(analysis['auth_service_usage'])-10} more files" if len(analysis['auth_service_usage']) > 10 else ""}

### Docker Files Removed/Updated
{chr(10).join([f"- {f}" for f in analysis['docker_files']])}

### Requirements Files Updated
{chr(10).join([f"- {f}" for f in analysis['requirements_files']])}

## Migration Log
{chr(10).join(self.migration_log)}

## Troubleshooting

If you encounter issues after migration:

1. **Import errors**: Check the import mappings in the migration log
2. **Configuration errors**: Verify your .env file matches the new format
3. **Docker build errors**: Ensure you're using the new unified Dockerfile
4. **Test failures**: Update test imports and fixture names

For assistance, refer to the CLEANUP_MIGRATION_GUIDE.md file.
"""
        
        if not self.dry_run:
            report_path.write_text(report_content)
        
        self.log(f"Generated migration report: {report_path}", "SUCCESS")
    
    def run_full_migration(self, dry_run: bool = False) -> bool:
        """Run the complete migration process"""
        self.dry_run = dry_run
        
        if dry_run:
            self.log("Running migration in DRY RUN mode - no files will be modified", "INFO")
        
        try:
            # Create backup
            if not dry_run:
                self.create_backup()
            
            # Analyze codebase
            analysis = self.analyze_codebase()
            
            # Run migration
            results = self.run_migration(analysis)
            
            # Generate report
            self.generate_migration_report(analysis, results)
            
            # Summary
            total_results = sum(results.values())
            self.log(f"Migration completed: {total_results}/5 tasks successful", "SUCCESS")
            
            if not dry_run:
                self.log(f"Backup created in: {self.backup_dir}", "INFO")
                self.log("Please review the migration report and test your application", "INFO")
            
            return total_results >= 4  # At least 4/5 tasks should succeed
            
        except Exception as e:
            self.log(f"Migration failed: {e}", "ERROR")
            return False


def main():
    """Main migration script entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate XORB Platform to unified architecture"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory (default: current directory)"
    )
    parser.add_argument(
        "--backup-dir", 
        help="Custom backup directory path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run migration analysis without making changes"
    )
    
    args = parser.parse_args()
    
    # Validate project root
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"Error: Project root directory not found: {project_root}")
        sys.exit(1)
    
    # Check for key files to confirm this is an XORB project
    key_files = ["src/api", "src/orchestrator", "requirements.txt"]
    if not all((project_root / f).exists() for f in key_files):
        print("Warning: This doesn't appear to be an XORB project root")
        print("Expected to find: src/api, src/orchestrator, requirements.txt")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run migration
    migrator = ArchitectureMigrator(
        project_root=str(project_root),
        backup_dir=args.backup_dir
    )
    
    success = migrator.run_full_migration(dry_run=args.dry_run)
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("📋 Please review MIGRATION_REPORT.md for next steps")
        sys.exit(0)
    else:
        print("\n❌ Migration encountered errors")
        print("📋 Check the migration log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()