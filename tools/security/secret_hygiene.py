#!/usr/bin/env python3
import os
import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configuration
PATTERNS_FILE = "patterns.yml"
REPORT_JSON = "secret_report.json"
REPORT_MD = "secret_report.md"
IGNORE_DIRS = {"gen", "node_modules", "venv", "archive"}


def load_patterns() -> Dict[str, str]:
    """Load regex patterns from YAML configuration file"""
    try:
        with open(PATTERNS_FILE, 'r') as f:
            patterns = yaml.safe_load(f)
        return patterns.get('patterns', {})
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return {}


def should_ignore(path: str) -> bool:
    """Check if path should be ignored"""
    return any(ignored in path for ignored in IGNORE_DIRS)


def scan_file(file_path: str, patterns: Dict[str, str]) -> List[Dict[str, Any]]:
    """Scan file for patterns and return matches"""
    matches = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                for category, pattern in patterns.items():
                    for match in re.finditer(pattern, line):
                        matches.append({
                            "category": category,
                            "file": file_path,
                            "line": line_num,
                            "match": match.group(),
                            "context": line.strip()
                        })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return matches


def generate_json_report(all_matches: List[Dict[str, Any]]) -> None:
    """Generate JSON format report"""
    with open(REPORT_JSON, 'w') as f:
        json.dump({
            "total_matches": len(all_matches),
            "matches": all_matches
        }, f, indent=2)


def generate_markdown_report(all_matches: List[Dict[str, Any]]) -> None:
    """Generate Markdown format report"""
    # Count matches by category
    category_counts = {}
    for match in all_matches:
        category = match["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    # Sort categories by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    with open(REPORT_MD, 'w') as f:
        f.write("# XORB Secret Hygiene Report\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Category | Count |\n")
        f.write("|---------|-------|\n")
        for category, count in sorted_categories:
            f.write(f"| {category} | {count} |\n")

        # Detailed findings
        f.write("\n## Detailed Findings\n\n")
        for category, _ in sorted_categories:
            category_matches = [m for m in all_matches if m["category"] == category]
            f.write(f"### {category} ({len(category_matches)})\n\n")
            for match in category_matches[:5]:  # Show top 5 offenders
                f.write(f"- **{match['file']}:{match['line']}**: `{match['match']}`\n")
                f.write(f"  ```{match['context']}\n")
            if len(category_matches) > 5:
                f.write(f"  ... and {len(category_matches) - 5} more\n")
            f.write("\n")


def main() -> None:
    """Main function"""
    patterns = load_patterns()
    if not patterns:
        print("No patterns loaded. Exiting.")
        return

    all_matches = []
    for root, dirs, files in os.walk("."):
        # Remove ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            if should_ignore(file_path):
                continue

            matches = scan_file(file_path, patterns)
            all_matches.extend(matches)

    generate_json_report(all_matches)
    generate_markdown_report(all_matches)

    print(f"Scan complete. Found {len(all_matches)} potential secrets.")
    print(f"Reports generated: {REPORT_JSON}, {REPORT_MD}")


if __name__ == "__main__":
    main()
    # Always exit with code 0 as per requirements
    exit(0)
