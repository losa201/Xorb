#!/usr/bin/env python3
"""
Requirements Migration Script
Migrates all services from scattered requirements files to unified pyproject.toml
"""

import os
import shutil
from pathlib import Path
from typing import List


class RequirementsMigrator:
    """Handles migration of requirements across the platform"""
    
    def __init__(self, root_dir: str = "/root/Xorb"):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / "legacy_requirements_backup"
        
        # Files to migrate
        self.old_requirements = [
            "requirements.txt",
            "src/api/requirements.txt", 
            "src/orchestrator/requirements.txt",
            "src/services/worker/requirements.txt",
            "requirements/requirements-ml.txt",
            "requirements/requirements-execution.txt"
        ]
        
        # Dockerfiles to update
        self.dockerfiles = [
            "src/api/Dockerfile",
            "src/api/Dockerfile.secure",
            "src/orchestrator/Dockerfile",
            "src/services/worker/Dockerfile",
            "infra/docker-compose.yml",
            "infra/docker-compose.production.yml"
        ]
    
    def backup_old_requirements(self):
        """Backup existing requirements files"""
        print("🔄 Backing up existing requirements files...")
        
        self.backup_dir.mkdir(exist_ok=True)
        
        for req_file in self.old_requirements:
            file_path = self.root_dir / req_file
            if file_path.exists():
                backup_path = self.backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                print(f"   ✅ Backed up {req_file} -> {backup_path}")
    
    def update_dockerfiles(self):
        """Update Dockerfiles to use new requirements structure"""
        print("🐳 Updating Dockerfiles...")
        
        for dockerfile_path in self.dockerfiles:
            file_path = self.root_dir / dockerfile_path
            if not file_path.exists():
                print(f"   ⚠️ Dockerfile not found: {dockerfile_path}")
                continue
                
            # Read current content
            content = file_path.read_text()
            
            # Replace old requirements patterns
            replacements = {
                "COPY requirements.txt .": "COPY requirements.lock .",
                "RUN pip install -r requirements.txt": "RUN pip install -r requirements.lock",
                "COPY src/api/requirements.txt .": "COPY requirements.lock .",
                "COPY src/orchestrator/requirements.txt .": "COPY requirements.lock .",
                "COPY requirements/requirements-*.txt .": "# Legacy requirements removed",
                "-r requirements.txt": "-r requirements.lock",
            }
            
            updated = False
            for old, new in replacements.items():
                if old in content:
                    content = content.replace(old, new)
                    updated = True
            
            if updated:
                file_path.write_text(content)
                print(f"   ✅ Updated {dockerfile_path}")
            else:
                print(f"   ℹ️ No changes needed: {dockerfile_path}")
    
    def create_service_requirements_redirects(self):
        """Create redirect files for services that expect local requirements.txt"""
        print("🔗 Creating requirement redirects for services...")
        
        redirect_content = """# This file redirects to the unified requirements
# Migrated to unified requirements management
-r ../../requirements.lock
"""
        
        service_dirs = [
            "src/api",
            "src/orchestrator", 
            "src/services/worker"
        ]
        
        for service_dir in service_dirs:
            service_path = self.root_dir / service_dir
            if service_path.exists():
                requirements_file = service_path / "requirements.txt"
                requirements_file.write_text(redirect_content)
                print(f"   ✅ Created redirect: {requirements_file}")
    
    def update_documentation(self):
        """Update documentation to reflect new requirements structure"""
        print("📚 Updating documentation...")
        
        # Update CLAUDE.md
        claude_md = self.root_dir / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()
            
            # Update installation commands
            old_install = "pip install -r requirements.txt"
            new_install = "pip install -r requirements.lock  # or pip install -e ."
            
            if old_install in content:
                content = content.replace(old_install, new_install)
                claude_md.write_text(content)
                print("   ✅ Updated CLAUDE.md installation commands")
        
        # Update README if exists
        readme_files = ["README.md", "src/api/README.md"]
        for readme_path in readme_files:
            readme_file = self.root_dir / readme_path
            if readme_file.exists():
                content = readme_file.read_text()
                if "requirements.txt" in content:
                    content = content.replace("requirements.txt", "requirements.lock")
                    readme_file.write_text(content)
                    print(f"   ✅ Updated {readme_path}")
    
    def validate_migration(self):
        """Validate the migration was successful"""
        print("✅ Validating migration...")
        
        # Check new files exist
        required_files = [
            "requirements.lock",
            "pyproject.toml"
        ]
        
        for file_name in required_files:
            file_path = self.root_dir / file_name
            if file_path.exists():
                print(f"   ✅ {file_name} exists")
            else:
                print(f"   ❌ {file_name} missing!")
        
        # Check backup was created
        if self.backup_dir.exists() and list(self.backup_dir.glob("*.txt")):
            print(f"   ✅ Backup created: {self.backup_dir}")
        else:
            print("   ⚠️ No backup files found")
        
        # Check service redirects
        service_requirements = [
            "src/api/requirements.txt",
            "src/orchestrator/requirements.txt",
            "src/services/worker/requirements.txt"
        ]
        
        for req_file in service_requirements:
            file_path = self.root_dir / req_file
            if file_path.exists():
                content = file_path.read_text()
                if "requirements.lock" in content:
                    print(f"   ✅ Redirect created: {req_file}")
                else:
                    print(f"   ⚠️ Redirect incomplete: {req_file}")
    
    def run_migration(self):
        """Execute the complete migration"""
        print("🚀 Starting Requirements Migration")
        print("=" * 50)
        
        try:
            # Step 1: Backup existing files
            self.backup_old_requirements()
            
            # Step 2: Update Dockerfiles  
            self.update_dockerfiles()
            
            # Step 3: Create service redirects
            self.create_service_requirements_redirects()
            
            # Step 4: Update documentation
            self.update_documentation()
            
            # Step 5: Validate migration
            self.validate_migration()
            
            print("\n" + "=" * 50)
            print("✅ Requirements Migration Completed Successfully!")
            print("\nNext Steps:")
            print("1. Test installation: pip install -r requirements.lock")
            print("2. Test with ML features: pip install -e .[ml]")
            print("3. Update CI/CD pipelines to use new requirements")
            print("4. Remove old requirements files after validation")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print("Please check the error and try again")
            return False
        
        return True


if __name__ == "__main__":
    migrator = RequirementsMigrator()
    success = migrator.run_migration()
    exit(0 if success else 1)