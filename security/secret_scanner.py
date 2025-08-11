#!/usr/bin/env python3
"""
Secret Scanner for XORB
Detects hardcoded secrets, API keys, and other sensitive information
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

@dataclass
class SecretMatch:
    """Represents a detected secret"""
    file_path: str
    line_number: int
    line_content: str
    secret_type: str
    severity: str
    rule_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number, 
            "content": self.line_content[:100] + "..." if len(self.line_content) > 100 else self.line_content,
            "type": self.secret_type,
            "severity": self.severity,
            "rule": self.rule_name
        }


class SecretScanner:
    """Scans for hardcoded secrets in source code"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.excluded_paths = {
            ".git/", "node_modules/", "__pycache__/", ".pytest_cache/",
            "venv/", ".venv/", "dist/", "build/", ".next/",
            "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll"
        }
        
    def _load_patterns(self) -> List[Dict[str, Any]]:
        """Load secret detection patterns"""
        return [
            {
                "name": "Hardcoded Password",
                "pattern": r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']',
                "type": "password",
                "severity": "high"
            },
            {
                "name": "API Key",
                "pattern": r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']([A-Za-z0-9_\-]{20,})["\']',
                "type": "api_key", 
                "severity": "high"
            },
            {
                "name": "JWT Secret",
                "pattern": r'(?i)(jwt[_-]?secret|secret[_-]?key)\s*[=:]\s*["\']([A-Za-z0-9_\-+/=]{32,})["\']',
                "type": "jwt_secret",
                "severity": "critical"
            },
            {
                "name": "Database URL with Credentials",
                "pattern": r'postgresql://[^:]+:[^@]+@[^/]+/\w+',
                "type": "database_url",
                "severity": "critical"
            },
            {
                "name": "AWS Access Key",
                "pattern": r'AKIA[0-9A-Z]{16}',
                "type": "aws_key",
                "severity": "critical"
            },
            {
                "name": "RSA Private Key",
                "pattern": r'-----BEGIN (RSA )?PRIVATE KEY-----',
                "type": "private_key",
                "severity": "critical"
            },
            {
                "name": "Generic Secret Pattern",
                "pattern": r'(?i)(secret|token|key|password)\s*[=:]\s*["\']([A-Za-z0-9_\-+/=]{16,})["\']',
                "type": "generic_secret",
                "severity": "medium"
            },
            {
                "name": "OAuth Client Secret",
                "pattern": r'(?i)(client[_-]?secret|oauth[_-]?secret)\s*[=:]\s*["\']([A-Za-z0-9_\-]{20,})["\']',
                "type": "oauth_secret",
                "severity": "high"
            }
        ]
        
    def scan_file(self, file_path: Path) -> List[SecretMatch]:
        """Scan a single file for secrets"""
        matches = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern_def in self.patterns:
                        pattern = pattern_def["pattern"]
                        matches_found = re.finditer(pattern, line)
                        
                        for match in matches_found:
                            # Skip if it looks like a template or example
                            if self._is_template_value(match.group()):
                                continue
                                
                            secret_match = SecretMatch(
                                file_path=str(file_path),
                                line_number=line_num,
                                line_content=line.strip(),
                                secret_type=pattern_def["type"],
                                severity=pattern_def["severity"],
                                rule_name=pattern_def["name"]
                            )
                            matches.append(secret_match)
                            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            
        return matches
        
    def _is_template_value(self, value: str) -> bool:
        """Check if value appears to be a template/placeholder"""
        template_indicators = [
            "your-", "example", "replace", "change", "update",
            "xxx", "yyy", "zzz", "abc", "123", "placeholder",
            "template", "sample", "demo", "test"
        ]
        
        value_lower = value.lower()
        return any(indicator in value_lower for indicator in template_indicators)
        
    def scan_directory(self, directory: Path) -> List[SecretMatch]:
        """Scan directory recursively for secrets"""
        all_matches = []
        
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Skip excluded paths
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            # Skip binary files
            if self._is_binary_file(file_path):
                continue
                
            matches = self.scan_file(file_path)
            all_matches.extend(matches)
            
        return all_matches
        
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except:
            return True
            
    def generate_report(self, matches: List[SecretMatch]) -> Dict[str, Any]:
        """Generate a summary report"""
        if not matches:
            return {
                "status": "clean",
                "total_secrets": 0,
                "message": "No hardcoded secrets detected"
            }
            
        # Group by severity
        by_severity = {}
        by_type = {}
        by_file = {}
        
        for match in matches:
            # By severity
            if match.severity not in by_severity:
                by_severity[match.severity] = []
            by_severity[match.severity].append(match)
            
            # By type
            if match.secret_type not in by_type:
                by_type[match.secret_type] = []
            by_type[match.secret_type].append(match)
            
            # By file
            if match.file_path not in by_file:
                by_file[match.file_path] = []
            by_file[match.file_path].append(match)
            
        critical_count = len(by_severity.get("critical", []))
        high_count = len(by_severity.get("high", []))
        
        return {
            "status": "secrets_detected",
            "total_secrets": len(matches),
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": len(by_severity.get("medium", [])),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "by_type": {k: len(v) for k, v in by_type.items()},
            "affected_files": len(by_file),
            "secrets": [match.to_dict() for match in matches]
        }


def main():
    parser = argparse.ArgumentParser(description="Scan for hardcoded secrets")
    parser.add_argument("path", nargs="?", default=".", help="Path to scan")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--severity", choices=["critical", "high", "medium"], help="Minimum severity")
    
    args = parser.parse_args()
    
    scanner = SecretScanner()
    scan_path = Path(args.path)
    
    print(f"🔍 Scanning for secrets in: {scan_path}")
    
    if scan_path.is_file():
        matches = scanner.scan_file(scan_path)
    else:
        matches = scanner.scan_directory(scan_path)
        
    # Filter by severity if specified
    if args.severity:
        severity_levels = {"medium": 1, "high": 2, "critical": 3}
        min_level = severity_levels[args.severity]
        matches = [m for m in matches if severity_levels.get(m.severity, 0) >= min_level]
    
    report = scanner.generate_report(matches)
    
    if args.format == "json":
        output = json.dumps(report, indent=2)
    else:
        # Text format
        if report["status"] == "clean":
            output = "✅ No hardcoded secrets detected!"
        else:
            lines = [
                f"🚨 {report['total_secrets']} secrets detected!",
                f"   Critical: {report['critical_count']}",
                f"   High: {report['high_count']}",
                f"   Medium: {report['medium_count']}",
                f"   Files affected: {report['affected_files']}",
                "",
                "Detected secrets:"
            ]
            
            for secret in report["secrets"]:
                lines.append(f"  📄 {secret['file']}:{secret['line']}")
                lines.append(f"     Type: {secret['type']} | Severity: {secret['severity']}")
                lines.append(f"     Rule: {secret['rule']}")
                lines.append(f"     Content: {secret['content']}")
                lines.append("")
                
        output = "\n".join(lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)
        
    # Exit with error code if secrets found
    exit_code = min(report.get("critical_count", 0) + report.get("high_count", 0), 1)
    exit(exit_code)


if __name__ == "__main__":
    main()