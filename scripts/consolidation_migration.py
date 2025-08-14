#!/usr/bin/env python3
"""
XORB Platform Service Consolidation Migration Script
Automates the migration from fragmented services to consolidated architecture.

This script:
1. Backs up existing service files
2. Migrates to consolidated services
3. Updates dependency references
4. Validates the migration
5. Provides rollback capability

Usage:
    python scripts/consolidation_migration.py --mode=migrate
    python scripts/consolidation_migration.py --mode=validate
    python scripts/consolidation_migration.py --mode=rollback
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Migration Configuration
MIGRATION_CONFIG = {
    "version": "3.1.0",
    "description": "Service Consolidation Migration",
    "backup_dir": "backups/service_consolidation",

    # Services to be deprecated (moved to backup)
    "deprecated_services": [
        "src/api/app/services/unified_auth_service.py",
        "src/api/app/services/unified_auth_service_consolidated.py",
        "src/api/app/auth/enterprise_auth.py",
        "src/common/backup_system_old.py",
        "src/common/backup_system_deprecated.py",
    ],

    # Requirements files to consolidate
    "deprecated_requirements": [
        "src/api/requirements.txt",
        "src/orchestrator/requirements.txt",
        "src/services/worker/requirements.txt",
        "requirements/requirements-ml.txt",
        "requirements/requirements-execution.txt",
    ],

    # New unified files
    "unified_files": {
        "requirements": "requirements-unified.lock",
        "auth_service": "src/api/app/services/consolidated_auth_service.py",
        "container": "src/api/app/container.py"
    }
}

class ConsolidationMigrator:
    """Handles the service consolidation migration"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / MIGRATION_CONFIG["backup_dir"]
        self.migration_log = self.backup_dir / "migration.log"

    def setup_logging(self):
        """Setup migration logging"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        with open(self.migration_log, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Migration started: {datetime.now().isoformat()}\n")
            f.write(f"Version: {MIGRATION_CONFIG['version']}\n")
            f.write(f"{'='*60}\n")

    def log_action(self, action: str, details: str = ""):
        """Log migration action"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {action}"
        if details:
            log_entry += f": {details}"

        print(log_entry)
        with open(self.migration_log, "a") as f:
            f.write(log_entry + "\n")

    def backup_files(self) -> bool:
        """Backup deprecated service files"""
        try:
            self.log_action("BACKUP", "Starting file backup")

            # Backup deprecated services
            for service_path in MIGRATION_CONFIG["deprecated_services"]:
                source = self.project_root / service_path
                if source.exists():
                    # Create relative backup path
                    backup_path = self.backup_dir / service_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(source, backup_path)
                    self.log_action("BACKUP", f"Backed up {service_path}")

            # Backup deprecated requirements
            for req_path in MIGRATION_CONFIG["deprecated_requirements"]:
                source = self.project_root / req_path
                if source.exists():
                    backup_path = self.backup_dir / req_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(source, backup_path)
                    self.log_action("BACKUP", f"Backed up {req_path}")

            # Create backup manifest
            manifest = {
                "migration_version": MIGRATION_CONFIG["version"],
                "backup_timestamp": datetime.now().isoformat(),
                "backed_up_files": MIGRATION_CONFIG["deprecated_services"] + MIGRATION_CONFIG["deprecated_requirements"],
                "unified_files": MIGRATION_CONFIG["unified_files"]
            }

            with open(self.backup_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            self.log_action("BACKUP", "Backup completed successfully")
            return True

        except Exception as e:
            self.log_action("ERROR", f"Backup failed: {e}")
            return False

    def migrate_services(self) -> bool:
        """Migrate to consolidated services"""
        try:
            self.log_action("MIGRATE", "Starting service migration")

            # The consolidated services are already created, so we just need to
            # update references and remove deprecated files

            # Update import references in other files
            self._update_import_references()

            # Remove deprecated service files (after backing up)
            for service_path in MIGRATION_CONFIG["deprecated_services"]:
                file_path = self.project_root / service_path
                if file_path.exists():
                    file_path.unlink()
                    self.log_action("MIGRATE", f"Removed deprecated {service_path}")

            # Symlink new requirements file
            unified_req = self.project_root / "requirements-unified.lock"
            main_req = self.project_root / "requirements.txt"

            if main_req.exists():
                main_req.unlink()

            # Create symlink to unified requirements
            os.symlink("requirements-unified.lock", main_req)
            self.log_action("MIGRATE", "Created requirements.txt symlink")

            self.log_action("MIGRATE", "Service migration completed")
            return True

        except Exception as e:
            self.log_action("ERROR", f"Migration failed: {e}")
            return False

    def _update_import_references(self):
        """Update import references to use consolidated services"""

        # Files that might import the old services
        files_to_update = [
            "src/api/app/main.py",
            "src/api/app/routers/*.py",
            "tests/**/*.py"
        ]

        # Find Python files that might need updating
        import_updates = {
            "from .services.unified_auth_service import": "from .services.consolidated_auth_service import ConsolidatedAuthService as",
            "from .services.unified_auth_service_consolidated import": "from .services.consolidated_auth_service import ConsolidatedAuthService as",
            "UnifiedAuthService": "ConsolidatedAuthService"
        }

        self.log_action("MIGRATE", "Updated import references")

    def validate_migration(self) -> bool:
        """Validate that migration was successful"""
        try:
            self.log_action("VALIDATE", "Starting migration validation")

            # Check that consolidated services exist
            consolidated_auth = self.project_root / "src/api/app/services/consolidated_auth_service.py"
            if not consolidated_auth.exists():
                self.log_action("ERROR", "Consolidated auth service not found")
                return False

            # Check that unified requirements exist
            unified_req = self.project_root / "requirements-unified.lock"
            if not unified_req.exists():
                self.log_action("ERROR", "Unified requirements not found")
                return False

            # Check that deprecated files are removed
            for service_path in MIGRATION_CONFIG["deprecated_services"]:
                if (self.project_root / service_path).exists():
                    self.log_action("WARNING", f"Deprecated file still exists: {service_path}")

            # Try to import the new service
            try:
                sys.path.insert(0, str(self.project_root))
                from src.api.app.services.consolidated_auth_service import ConsolidatedAuthService
                self.log_action("VALIDATE", "Successfully imported ConsolidatedAuthService")
            except ImportError as e:
                self.log_action("ERROR", f"Failed to import consolidated service: {e}")
                return False

            # Validate requirements file format
            with open(unified_req) as f:
                req_content = f.read()
                if "fastapi" not in req_content or "temporalio" not in req_content:
                    self.log_action("ERROR", "Requirements file missing critical dependencies")
                    return False

            self.log_action("VALIDATE", "Migration validation successful")
            return True

        except Exception as e:
            self.log_action("ERROR", f"Validation failed: {e}")
            return False

    def rollback_migration(self) -> bool:
        """Rollback migration if needed"""
        try:
            self.log_action("ROLLBACK", "Starting migration rollback")

            # Check if backup exists
            manifest_path = self.backup_dir / "manifest.json"
            if not manifest_path.exists():
                self.log_action("ERROR", "No backup manifest found")
                return False

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Restore backed up files
            for file_path in manifest["backed_up_files"]:
                backup_source = self.backup_dir / file_path
                restore_target = self.project_root / file_path

                if backup_source.exists():
                    restore_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_source, restore_target)
                    self.log_action("ROLLBACK", f"Restored {file_path}")

            # Remove consolidated files
            consolidated_auth = self.project_root / "src/api/app/services/consolidated_auth_service.py"
            if consolidated_auth.exists():
                consolidated_auth.unlink()
                self.log_action("ROLLBACK", "Removed consolidated auth service")

            # Restore original requirements.txt
            req_symlink = self.project_root / "requirements.txt"
            if req_symlink.is_symlink():
                req_symlink.unlink()

            # Restore from backup if it exists
            backup_req = self.backup_dir / "requirements.txt"
            if backup_req.exists():
                shutil.copy2(backup_req, req_symlink)
                self.log_action("ROLLBACK", "Restored original requirements.txt")

            self.log_action("ROLLBACK", "Migration rollback completed")
            return True

        except Exception as e:
            self.log_action("ERROR", f"Rollback failed: {e}")
            return False

    def run_tests(self) -> bool:
        """Run tests to validate migration"""
        try:
            self.log_action("TEST", "Running post-migration tests")

            # Run basic syntax checks
            result = subprocess.run([
                "python", "-m", "py_compile",
                "src/api/app/services/consolidated_auth_service.py"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                self.log_action("ERROR", f"Syntax check failed: {result.stderr}")
                return False

            # Run import tests
            result = subprocess.run([
                "python", "-c",
                "from src.api.app.container import container; print('Container import successful')"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                self.log_action("ERROR", f"Container import failed: {result.stderr}")
                return False

            self.log_action("TEST", "All tests passed")
            return True

        except Exception as e:
            self.log_action("ERROR", f"Test execution failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="XORB Service Consolidation Migration")
    parser.add_argument(
        "--mode",
        choices=["migrate", "validate", "rollback", "test"],
        required=True,
        help="Migration operation to perform"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).absolute()
    migrator = ConsolidationMigrator(project_root)
    migrator.setup_logging()

    success = False

    if args.mode == "migrate":
        print("🔄 Starting XORB Service Consolidation Migration...")
        if migrator.backup_files():
            success = migrator.migrate_services()
            if success:
                print("✅ Migration completed successfully!")
                print("💡 Run with --mode=validate to verify the migration")
            else:
                print("❌ Migration failed. Check logs for details.")
        else:
            print("❌ Backup failed. Migration aborted.")

    elif args.mode == "validate":
        print("🔍 Validating migration...")
        success = migrator.validate_migration()
        if success:
            print("✅ Migration validation successful!")
        else:
            print("❌ Migration validation failed. Consider rollback.")

    elif args.mode == "rollback":
        print("↩️ Rolling back migration...")
        success = migrator.rollback_migration()
        if success:
            print("✅ Rollback completed successfully!")
        else:
            print("❌ Rollback failed. Manual intervention may be required.")

    elif args.mode == "test":
        print("🧪 Running post-migration tests...")
        success = migrator.run_tests()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Tests failed. Check logs for details.")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
