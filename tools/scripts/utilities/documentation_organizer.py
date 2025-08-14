#!/usr/bin/env python3
"""
XORB Documentation Organizer
Identifies and organizes documentation files to reduce sprawl and improve maintainability
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime


class DocumentationOrganizer:
    """Organizes XORB documentation files"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.legacy_patterns = [
            r".*IMPLEMENTATION.*COMPLETE.*",
            r".*STATUS.*COMPLETE.*",
            r".*MIGRATION.*COMPLETE.*",
            r".*PRINCIPAL_AUDITOR.*",
            r".*ENHANCEMENT.*COMPLETE.*",
            r".*DEPLOYMENT.*COMPLETE.*",
            r".*PRODUCTION.*COMPLETE.*",
            r".*STRATEGIC.*COMPLETE.*",
            r".*SOPHISTICATED.*COMPLETE.*"
        ]
        self.duplicate_patterns = [
            r".*README.*\d+.*",  # Numbered READMEs
            r".*IMPLEMENTATION.*\d+.*",  # Numbered implementation docs
            r".*_\d{8}_\d{6}.*"  # Timestamped files
        ]

    def scan_documentation(self) -> Dict[str, List[Path]]:
        """Scan all markdown files and categorize them"""
        categories = {
            "core": [],
            "legacy": [],
            "duplicates": [],
            "api": [],
            "architecture": [],
            "deployment": [],
            "security": [],
            "enterprise": []
        }

        # Find all markdown files
        md_files = list(self.root_path.rglob("*.md"))

        for file_path in md_files:
            file_name = file_path.name
            relative_path = file_path.relative_to(self.root_path)

            # Skip files in legacy directory
            if "legacy" in str(relative_path):
                continue

            # Categorize files
            if self._is_legacy_file(file_name):
                categories["legacy"].append(file_path)
            elif self._is_duplicate_file(file_name):
                categories["duplicates"].append(file_path)
            elif self._is_api_file(file_name, relative_path):
                categories["api"].append(file_path)
            elif self._is_architecture_file(file_name, relative_path):
                categories["architecture"].append(file_path)
            elif self._is_deployment_file(file_name, relative_path):
                categories["deployment"].append(file_path)
            elif self._is_security_file(file_name, relative_path):
                categories["security"].append(file_path)
            elif self._is_enterprise_file(file_name, relative_path):
                categories["enterprise"].append(file_path)
            else:
                categories["core"].append(file_path)

        return categories

    def _is_legacy_file(self, file_name: str) -> bool:
        """Check if file matches legacy patterns"""
        for pattern in self.legacy_patterns:
            if re.match(pattern, file_name, re.IGNORECASE):
                return True
        return False

    def _is_duplicate_file(self, file_name: str) -> bool:
        """Check if file appears to be a duplicate"""
        for pattern in self.duplicate_patterns:
            if re.match(pattern, file_name, re.IGNORECASE):
                return True
        return False

    def _is_api_file(self, file_name: str, relative_path: Path) -> bool:
        """Check if file is API documentation"""
        api_keywords = ["api", "endpoint", "rest", "swagger", "openapi"]
        return (
            "api" in str(relative_path).lower() or
            any(keyword in file_name.lower() for keyword in api_keywords)
        )

    def _is_architecture_file(self, file_name: str, relative_path: Path) -> bool:
        """Check if file is architecture documentation"""
        arch_keywords = ["architecture", "design", "structure", "service"]
        return (
            "architecture" in str(relative_path).lower() or
            any(keyword in file_name.lower() for keyword in arch_keywords)
        )

    def _is_deployment_file(self, file_name: str, relative_path: Path) -> bool:
        """Check if file is deployment documentation"""
        deploy_keywords = ["deployment", "deploy", "install", "setup", "docker", "kubernetes"]
        return (
            "deployment" in str(relative_path).lower() or
            any(keyword in file_name.lower() for keyword in deploy_keywords)
        )

    def _is_security_file(self, file_name: str, relative_path: Path) -> bool:
        """Check if file is security documentation"""
        security_keywords = ["security", "compliance", "audit", "policy"]
        return any(keyword in file_name.lower() for keyword in security_keywords)

    def _is_enterprise_file(self, file_name: str, relative_path: Path) -> bool:
        """Check if file is enterprise documentation"""
        enterprise_keywords = ["enterprise", "production", "operational", "runbook"]
        return (
            "enterprise" in str(relative_path).lower() or
            any(keyword in file_name.lower() for keyword in enterprise_keywords)
        )

    def generate_report(self, categories: Dict[str, List[Path]]) -> str:
        """Generate a detailed report of documentation organization"""
        report = []
        report.append("# XORB Documentation Organization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary statistics
        total_files = sum(len(files) for files in categories.values())
        report.append("## Summary Statistics")
        report.append(f"- **Total Documentation Files**: {total_files}")
        report.append("")

        for category, files in categories.items():
            if files:
                report.append(f"- **{category.title()} Files**: {len(files)}")
        report.append("")

        # Detailed breakdown
        for category, files in categories.items():
            if files:
                report.append(f"## {category.title()} Files ({len(files)})")
                report.append("")

                for file_path in sorted(files):
                    relative_path = file_path.relative_to(self.root_path)
                    report.append(f"- `{relative_path}`")

                report.append("")

        return "\n".join(report)

    def archive_legacy_files(self, legacy_files: List[Path], dry_run: bool = True) -> List[str]:
        """Archive legacy documentation files"""
        actions = []
        legacy_dir = self.root_path / "legacy" / "archived_documentation"

        if not dry_run:
            legacy_dir.mkdir(parents=True, exist_ok=True)

        for file_path in legacy_files:
            relative_path = file_path.relative_to(self.root_path)
            destination = legacy_dir / relative_path.name

            if dry_run:
                actions.append(f"WOULD MOVE: {relative_path} -> {destination.relative_to(self.root_path)}")
            else:
                try:
                    shutil.move(str(file_path), str(destination))
                    actions.append(f"MOVED: {relative_path} -> {destination.relative_to(self.root_path)}")
                except Exception as e:
                    actions.append(f"ERROR: Failed to move {relative_path}: {e}")

        return actions

    def identify_duplicates(self, categories: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """Identify potential duplicate files based on content similarity"""
        duplicates = {}
        all_files = []

        # Collect all files except legacy
        for category, files in categories.items():
            if category != "legacy":
                all_files.extend(files)

        # Group files by similar names (simplified duplicate detection)
        name_groups = {}
        for file_path in all_files:
            # Extract base name without numbers/timestamps
            base_name = re.sub(r'_\d+|_\d{8}_\d{6}|\d+', '', file_path.stem).lower()
            if base_name not in name_groups:
                name_groups[base_name] = []
            name_groups[base_name].append(file_path)

        # Find groups with multiple files
        for base_name, files in name_groups.items():
            if len(files) > 1:
                duplicates[base_name] = files

        return duplicates

    def suggest_consolidation(self, categories: Dict[str, List[Path]]) -> List[str]:
        """Suggest consolidation actions"""
        suggestions = []

        # Check for oversized categories
        if len(categories["core"]) > 20:
            suggestions.append("Consider organizing core files into more specific categories")

        # Check for scattered API docs
        api_files = categories["api"]
        if api_files:
            suggestions.append(f"Consider consolidating {len(api_files)} API files into docs/api/")

        # Check for multiple README files
        readme_files = [f for f in categories["core"] if "readme" in f.name.lower()]
        if len(readme_files) > 1:
            suggestions.append(f"Consider consolidating {len(readme_files)} README files")

        return suggestions


def main():
    """Main execution function"""
    organizer = DocumentationOrganizer()

    print("🔍 Scanning XORB documentation files...")
    categories = organizer.scan_documentation()

    print("📊 Generating organization report...")
    report = organizer.generate_report(categories)

    # Save report
    report_file = Path("documentation_organization_report.md")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"✅ Report saved to: {report_file}")

    # Identify duplicates
    duplicates = organizer.identify_duplicates(categories)
    if duplicates:
        print("\n⚠️  Potential duplicates found:")
        for base_name, files in duplicates.items():
            print(f"  {base_name}: {len(files)} files")

    # Generate suggestions
    suggestions = organizer.suggest_consolidation(categories)
    if suggestions:
        print("\n💡 Consolidation suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

    # Ask about archiving legacy files
    legacy_files = categories["legacy"]
    if legacy_files:
        print(f"\n📚 Found {len(legacy_files)} legacy files")
        response = input("Archive legacy files? (y/N): ").strip().lower()

        if response == 'y':
            actions = organizer.archive_legacy_files(legacy_files, dry_run=False)
            print("📦 Archive actions:")
            for action in actions[:10]:  # Show first 10
                print(f"  {action}")
            if len(actions) > 10:
                print(f"  ... and {len(actions) - 10} more")
        else:
            # Show what would be archived (dry run)
            actions = organizer.archive_legacy_files(legacy_files, dry_run=True)
            print("📋 Would archive (dry run):")
            for action in actions[:5]:  # Show first 5
                print(f"  {action}")
            if len(actions) > 5:
                print(f"  ... and {len(actions) - 5} more")

    print("\n✨ Documentation organization analysis complete!")


if __name__ == "__main__":
    main()
