#!/usr/bin/env python3
"""
XORB Platform Documentation Generator
Automated documentation generation and management tools
"""

import os
import re
import json
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import subprocess
import shutil

class DocumentationGenerator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.docs_dir = self.base_dir / "docs"
        self.templates_dir = self.docs_dir / "templates"
        self.reports_dir = self.base_dir / "reports" / "docs"
        
        # Create directories if they don't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Documentation metadata
        self.metadata = {}
        self.load_metadata()
    
    def load_metadata(self) -> None:
        """Load existing documentation metadata"""
        metadata_file = self.docs_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "files": {},
                "last_scan": None,
                "total_files": 0,
                "categories": {},
                "tags": {}
            }
    
    def save_metadata(self) -> None:
        """Save documentation metadata"""
        metadata_file = self.docs_dir / "metadata.json"
        self.metadata["last_scan"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_frontmatter(self, file_path: Path) -> Dict:
        """Extract YAML frontmatter from markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for frontmatter
        if not content.startswith('---'):
            return {}
        
        # Extract frontmatter
        try:
            end_marker = content.find('---', 3)
            if end_marker == -1:
                return {}
            
            frontmatter_text = content[3:end_marker].strip()
            return yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError:
            return {}
    
    def analyze_content(self, file_path: Path) -> Dict:
        """Analyze markdown content for metadata"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count various elements
        analysis = {
            "word_count": len(content.split()),
            "line_count": len(content.splitlines()),
            "heading_count": len(re.findall(r'^#+\s', content, re.MULTILINE)),
            "code_block_count": len(re.findall(r'```', content)) // 2,
            "link_count": len(re.findall(r'\[.*?\]\(.*?\)', content)),
            "image_count": len(re.findall(r'!\[.*?\]\(.*?\)', content)),
            "table_count": len(re.findall(r'^\|.*\|', content, re.MULTILINE)),
            "todo_count": len(re.findall(r'TODO|FIXME|XXX', content, re.IGNORECASE))
        }
        
        # Extract headings
        headings = re.findall(r'^(#+)\s+(.+)', content, re.MULTILINE)
        analysis["headings"] = [{"level": len(h[0]), "text": h[1]} for h in headings]
        
        # Estimate reading time (average 200 words per minute)
        analysis["estimated_reading_time"] = max(1, analysis["word_count"] // 200)
        
        return analysis
    
    def scan_documentation(self) -> None:
        """Scan all documentation files and collect metadata"""
        print("🔍 Scanning documentation files...")
        
        total_files = 0
        categories = {}
        tags = {}
        
        # Find all markdown files
        for md_file in self.base_dir.rglob("*.md"):
            # Skip certain directories
            if any(skip in str(md_file) for skip in ["venv", ".venv", "node_modules"]):
                continue
            
            total_files += 1
            relative_path = str(md_file.relative_to(self.base_dir))
            
            # Extract frontmatter and analyze content
            frontmatter = self.extract_frontmatter(md_file)
            analysis = self.analyze_content(md_file)
            
            # Get file stats
            stat = md_file.stat()
            
            # Combine metadata
            file_metadata = {
                "path": relative_path,
                "frontmatter": frontmatter,
                "analysis": analysis,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "last_scanned": datetime.now().isoformat()
            }
            
            self.metadata["files"][relative_path] = file_metadata
            
            # Collect categories and tags
            if "category" in frontmatter:
                cat = frontmatter["category"]
                categories[cat] = categories.get(cat, 0) + 1
            
            if "tags" in frontmatter and isinstance(frontmatter["tags"], list):
                for tag in frontmatter["tags"]:
                    tags[tag] = tags.get(tag, 0) + 1
        
        # Update metadata
        self.metadata["total_files"] = total_files
        self.metadata["categories"] = categories
        self.metadata["tags"] = tags
        
        print(f"✅ Scanned {total_files} documentation files")
    
    def generate_index(self) -> None:
        """Generate documentation index"""
        print("📚 Generating documentation index...")
        
        index_content = [
            "---",
            "title: \"XORB Platform Documentation Index\"",
            "description: \"Comprehensive index of all platform documentation\"",
            "category: \"Index\"",
            f"generated: \"{datetime.now().isoformat()}\"",
            "---",
            "",
            "# 📚 XORB Platform Documentation Index",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Total Documents**: {self.metadata['total_files']}  ",
            f"**Categories**: {len(self.metadata['categories'])}  ",
            f"**Tags**: {len(self.metadata['tags'])}  ",
            "",
            "## 📊 Documentation Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Files | {self.metadata['total_files']} |",
            f"| Categories | {len(self.metadata['categories'])} |",
            f"| Unique Tags | {len(self.metadata['tags'])} |",
            "",
            "## 📂 Categories",
            ""
        ]
        
        # Add categories
        for category, count in sorted(self.metadata["categories"].items()):
            index_content.append(f"### {category} ({count} documents)")
            index_content.append("")
            
            # Find files in this category
            category_files = [
                (path, meta) for path, meta in self.metadata["files"].items()
                if meta["frontmatter"].get("category") == category
            ]
            
            # Sort by title or filename
            category_files.sort(key=lambda x: x[1]["frontmatter"].get("title", Path(x[0]).stem))
            
            for path, meta in category_files:
                title = meta["frontmatter"].get("title", Path(path).stem)
                desc = meta["frontmatter"].get("description", "")
                
                # Add reading time if available
                reading_time = meta["analysis"].get("estimated_reading_time", 0)
                time_str = f" ({reading_time} min read)" if reading_time > 0 else ""
                
                index_content.append(f"- [{title}]({path}){time_str}")
                if desc:
                    index_content.append(f"  - {desc}")
            
            index_content.append("")
        
        # Add tag cloud
        index_content.extend([
            "## 🏷️ Tag Cloud",
            "",
            "Popular tags across all documentation:",
            ""
        ])
        
        # Sort tags by frequency
        sorted_tags = sorted(self.metadata["tags"].items(), key=lambda x: x[1], reverse=True)
        tag_links = []
        
        for tag, count in sorted_tags[:20]:  # Top 20 tags
            tag_links.append(f"[{tag}](?tag={tag}) ({count})")
        
        # Format tags in rows
        for i in range(0, len(tag_links), 5):
            index_content.append("| " + " | ".join(tag_links[i:i+5]) + " |")
        
        # Add recent updates
        index_content.extend([
            "",
            "## 🕒 Recently Updated",
            "",
            "Documents updated in the last 30 days:",
            ""
        ])
        
        # Find recently updated files
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_files = []
        
        for path, meta in self.metadata["files"].items():
            modified_date = datetime.fromisoformat(meta["modified"])
            if modified_date > thirty_days_ago:
                recent_files.append((path, meta, modified_date))
        
        # Sort by modification date
        recent_files.sort(key=lambda x: x[2], reverse=True)
        
        for path, meta, mod_date in recent_files[:10]:  # Top 10 recent
            title = meta["frontmatter"].get("title", Path(path).stem)
            date_str = mod_date.strftime("%Y-%m-%d")
            index_content.append(f"- [{title}]({path}) - {date_str}")
        
        # Write index file
        index_file = self.docs_dir / "INDEX.md"
        with open(index_file, 'w') as f:
            f.write('\n'.join(index_content))
        
        print(f"✅ Generated documentation index: {index_file}")
    
    def create_from_template(self, template_name: str, output_path: str, **kwargs) -> None:
        """Create new documentation from template"""
        template_file = self.templates_dir / f"{template_name}-template.md"
        
        if not template_file.exists():
            print(f"❌ Template not found: {template_file}")
            return
        
        with open(template_file, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders
        for key, value in kwargs.items():
            placeholder = f"[{key.upper()}]"
            template_content = template_content.replace(placeholder, str(value))
        
        # Update metadata
        today = datetime.now().strftime("%Y-%m-%d")
        template_content = template_content.replace("YYYY-MM-DD", today)
        
        # Write new file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(template_content)
        
        print(f"✅ Created documentation: {output_file}")
    
    def validate_frontmatter(self) -> List[str]:
        """Validate frontmatter across all documents"""
        issues = []
        required_fields = ["title", "description", "category", "last_updated"]
        
        for path, meta in self.metadata["files"].items():
            frontmatter = meta["frontmatter"]
            
            if not frontmatter:
                issues.append(f"{path}: Missing frontmatter")
                continue
            
            for field in required_fields:
                if field not in frontmatter:
                    issues.append(f"{path}: Missing required field '{field}'")
        
        return issues
    
    def generate_health_report(self) -> None:
        """Generate documentation health report"""
        print("📊 Generating documentation health report...")
        
        # Collect metrics
        total_files = self.metadata["total_files"]
        files_with_frontmatter = sum(1 for meta in self.metadata["files"].values() 
                                   if meta["frontmatter"])
        
        total_words = sum(meta["analysis"].get("word_count", 0) 
                         for meta in self.metadata["files"].values())
        
        # Find issues
        frontmatter_issues = self.validate_frontmatter()
        
        # Generate report
        report_content = [
            "# Documentation Health Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
            "## 📊 Overview",
            "",
            f"- **Total Files**: {total_files}",
            f"- **Files with Frontmatter**: {files_with_frontmatter} ({100*files_with_frontmatter//total_files}%)",
            f"- **Total Words**: {total_words:,}",
            f"- **Average Words per Document**: {total_words//total_files if total_files > 0 else 0}",
            "",
            "## 🎯 Quality Metrics",
            "",
            f"- **Frontmatter Coverage**: {100*files_with_frontmatter//total_files}%",
            f"- **Documentation Issues**: {len(frontmatter_issues)}",
            "",
            "## 📂 Category Distribution",
            ""
        ]
        
        # Add category chart
        for category, count in sorted(self.metadata["categories"].items()):
            percentage = 100 * count // total_files
            bar = "█" * (percentage // 2)
            report_content.append(f"- **{category}**: {count} files {bar} {percentage}%")
        
        # Add issues section
        if frontmatter_issues:
            report_content.extend([
                "",
                "## ⚠️ Issues Found",
                ""
            ])
            
            for issue in frontmatter_issues[:20]:  # Limit to first 20
                report_content.append(f"- {issue}")
            
            if len(frontmatter_issues) > 20:
                report_content.append(f"- ... and {len(frontmatter_issues) - 20} more issues")
        
        # Add recommendations
        report_content.extend([
            "",
            "## 💡 Recommendations",
            ""
        ])
        
        if files_with_frontmatter < total_files:
            missing = total_files - files_with_frontmatter
            report_content.append(f"- Add frontmatter to {missing} documents")
        
        if len(frontmatter_issues) > 0:
            report_content.append(f"- Fix {len(frontmatter_issues)} frontmatter issues")
        
        if total_words // total_files < 100:
            report_content.append("- Consider expanding documentation content")
        
        # Write report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"health_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"✅ Generated health report: {report_file}")
        
        # Also create a JSON report for automation
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_files": total_files,
                "files_with_frontmatter": files_with_frontmatter,
                "frontmatter_coverage": 100 * files_with_frontmatter // total_files,
                "total_words": total_words,
                "average_words": total_words // total_files if total_files > 0 else 0,
                "issues_count": len(frontmatter_issues)
            },
            "categories": self.metadata["categories"],
            "tags": self.metadata["tags"],
            "issues": frontmatter_issues
        }
        
        json_file = self.reports_dir / f"health_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
    
    def update_search_index(self) -> None:
        """Generate search index for documentation"""
        print("🔍 Updating search index...")
        
        search_index = {
            "documents": [],
            "generated": datetime.now().isoformat(),
            "total_documents": 0
        }
        
        for path, meta in self.metadata["files"].items():
            # Create search document
            doc = {
                "id": path,
                "title": meta["frontmatter"].get("title", Path(path).stem),
                "description": meta["frontmatter"].get("description", ""),
                "category": meta["frontmatter"].get("category", "Uncategorized"),
                "tags": meta["frontmatter"].get("tags", []),
                "path": path,
                "word_count": meta["analysis"].get("word_count", 0),
                "reading_time": meta["analysis"].get("estimated_reading_time", 0),
                "headings": [h["text"] for h in meta["analysis"].get("headings", [])],
                "modified": meta["modified"]
            }
            
            search_index["documents"].append(doc)
        
        search_index["total_documents"] = len(search_index["documents"])
        
        # Write search index
        search_file = self.docs_dir / "search_index.json"
        with open(search_file, 'w') as f:
            json.dump(search_index, f, indent=2)
        
        print(f"✅ Updated search index: {search_file}")

def main():
    parser = argparse.ArgumentParser(description="XORB Documentation Generator")
    parser.add_argument("--scan", action="store_true", help="Scan all documentation")
    parser.add_argument("--index", action="store_true", help="Generate documentation index")
    parser.add_argument("--health", action="store_true", help="Generate health report")
    parser.add_argument("--search", action="store_true", help="Update search index")
    parser.add_argument("--create", help="Create from template: user-guide|api-reference|architecture")
    parser.add_argument("--output", help="Output path for new document")
    parser.add_argument("--title", help="Title for new document")
    parser.add_argument("--description", help="Description for new document")
    parser.add_argument("--category", help="Category for new document")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DocumentationGenerator()
    
    if args.all or args.scan:
        generator.scan_documentation()
        generator.save_metadata()
    
    if args.all or args.index:
        generator.generate_index()
    
    if args.all or args.health:
        generator.generate_health_report()
    
    if args.all or args.search:
        generator.update_search_index()
    
    if args.create:
        if not args.output:
            print("❌ --output required when creating from template")
            return
        
        kwargs = {}
        if args.title:
            kwargs["task description"] = args.title
            kwargs["title"] = args.title
        if args.description:
            kwargs["description"] = args.description
        if args.category:
            kwargs["category"] = args.category
        
        generator.create_from_template(args.create, args.output, **kwargs)
    
    if not any([args.scan, args.index, args.health, args.search, args.create, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()