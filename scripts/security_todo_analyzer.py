#!/usr/bin/env python3
"""
XORB Security TODO Analyzer
Analyzes and categorizes TODO comments for security implications.

This script:
1. Scans all TODO/FIXME/HACK comments in the codebase
2. Categorizes them by security risk level
3. Provides actionable recommendations
4. Generates a security review report
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class SecurityRisk(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

@dataclass
class TodoItem:
    file_path: str
    line_number: int
    comment_type: str  # TODO, FIXME, HACK, etc.
    text: str
    security_risk: SecurityRisk
    category: str
    recommendation: str

class SecurityTodoAnalyzer:
    """Analyzes TODO comments for security implications"""
    
    # Security-related keywords that indicate higher risk
    SECURITY_KEYWORDS = {
        "critical": ["password", "secret", "key", "token", "auth", "security", "encrypt", "decrypt", "hash", "salt"],
        "high": ["permission", "access", "login", "session", "oauth", "jwt", "crypto", "ssl", "tls"],
        "medium": ["validate", "sanitize", "escape", "input", "output", "config", "env"],
        "low": ["log", "debug", "trace", "monitor", "audit"]
    }
    
    # Patterns that indicate specific security issues
    SECURITY_PATTERNS = {
        r"(?i)hardcoded.*(?:password|key|secret|token)": (SecurityRisk.CRITICAL, "Hardcoded credentials"),
        r"(?i)temporary.*(?:auth|security|password)": (SecurityRisk.HIGH, "Temporary security bypass"),
        r"(?i)disable.*(?:auth|security|validation)": (SecurityRisk.HIGH, "Security feature disabled"),
        r"(?i)(?:fix|implement).*(?:auth|security)": (SecurityRisk.MEDIUM, "Security feature incomplete"),
        r"(?i)(?:validate|sanitize|escape)": (SecurityRisk.MEDIUM, "Input validation needed"),
        r"(?i)logging.*(?:password|secret|key)": (SecurityRisk.HIGH, "Sensitive data in logs"),
        r"(?i)development.*only": (SecurityRisk.MEDIUM, "Development-only code"),
        r"(?i)production.*(?:change|fix|implement)": (SecurityRisk.HIGH, "Production security issue"),
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.todos: List[TodoItem] = []
    
    def scan_files(self) -> List[TodoItem]:
        """Scan all Python files for TODO comments"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip virtual environments and external packages
            if any(part in str(file_path) for part in ["venv", "node_modules", "site-packages", ".git"]):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    todos = self._extract_todos(line)
                    for todo_type, todo_text in todos:
                        todo_item = TodoItem(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            comment_type=todo_type,
                            text=todo_text.strip(),
                            security_risk=SecurityRisk.NONE,
                            category="",
                            recommendation=""
                        )
                        
                        # Analyze security implications
                        self._analyze_security_risk(todo_item)
                        self.todos.append(todo_item)
                        
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
        
        return self.todos
    
    def _extract_todos(self, line: str) -> List[Tuple[str, str]]:
        """Extract TODO-style comments from a line"""
        todos = []
        
        # Pattern to match TODO, FIXME, HACK, XXX, BUG comments
        pattern = r'#\s*(TODO|FIXME|HACK|XXX|BUG)\s*:?\s*(.+)'
        matches = re.finditer(pattern, line, re.IGNORECASE)
        
        for match in matches:
            todo_type = match.group(1).upper()
            todo_text = match.group(2).strip()
            todos.append((todo_type, todo_text))
        
        return todos
    
    def _analyze_security_risk(self, todo: TodoItem):
        """Analyze the security risk level of a TODO comment"""
        text_lower = todo.text.lower()
        
        # Check for specific security patterns
        for pattern, (risk, category) in self.SECURITY_PATTERNS.items():
            if re.search(pattern, todo.text):
                todo.security_risk = risk
                todo.category = category
                todo.recommendation = self._get_recommendation(risk, category)
                return
        
        # Check for security keywords
        for risk_level, keywords in self.SECURITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if risk_level == "critical":
                        todo.security_risk = SecurityRisk.CRITICAL
                    elif risk_level == "high":
                        todo.security_risk = SecurityRisk.HIGH
                    elif risk_level == "medium":
                        todo.security_risk = SecurityRisk.MEDIUM
                    else:
                        todo.security_risk = SecurityRisk.LOW
                    
                    todo.category = f"Security-related: {keyword}"
                    todo.recommendation = self._get_recommendation(todo.security_risk, keyword)
                    return
        
        # Default categorization based on comment type
        if todo.comment_type in ["HACK", "FIXME"]:
            todo.security_risk = SecurityRisk.MEDIUM
            todo.category = "Code quality issue"
            todo.recommendation = "Review and properly implement solution"
        else:
            todo.security_risk = SecurityRisk.LOW
            todo.category = "General improvement"
            todo.recommendation = "Address when convenient"
    
    def _get_recommendation(self, risk: SecurityRisk, category: str) -> str:
        """Get specific recommendation based on risk level and category"""
        recommendations = {
            SecurityRisk.CRITICAL: "🚨 IMMEDIATE ACTION REQUIRED - This poses a critical security risk",
            SecurityRisk.HIGH: "⚠️ HIGH PRIORITY - Address before production deployment",
            SecurityRisk.MEDIUM: "⚡ MEDIUM PRIORITY - Should be addressed in current sprint",
            SecurityRisk.LOW: "💡 LOW PRIORITY - Address during maintenance cycles"
        }
        
        base_recommendation = recommendations.get(risk, "Review and assess")
        
        # Add specific guidance based on category
        if "password" in category.lower() or "secret" in category.lower():
            base_recommendation += "\n  • Use environment variables or secure vault"
            base_recommendation += "\n  • Never hardcode sensitive credentials"
        elif "auth" in category.lower():
            base_recommendation += "\n  • Implement proper authentication flow"
            base_recommendation += "\n  • Use established security libraries"
        elif "validation" in category.lower():
            base_recommendation += "\n  • Implement input validation and sanitization"
            base_recommendation += "\n  • Use Pydantic models for validation"
        
        return base_recommendation
    
    def generate_report(self) -> Dict:
        """Generate comprehensive security TODO report"""
        risk_counts = {risk: 0 for risk in SecurityRisk}
        categories = {}
        
        for todo in self.todos:
            risk_counts[todo.security_risk] += 1
            
            if todo.category not in categories:
                categories[todo.category] = []
            categories[todo.category].append(todo)
        
        # Sort todos by risk level (critical first)
        sorted_todos = sorted(
            self.todos, 
            key=lambda x: list(SecurityRisk).index(x.security_risk)
        )
        
        report = {
            "summary": {
                "total_todos": len(self.todos),
                "by_risk_level": {risk.value: count for risk, count in risk_counts.items()},
                "security_related": sum(risk_counts[risk] for risk in [SecurityRisk.CRITICAL, SecurityRisk.HIGH, SecurityRisk.MEDIUM]),
                "immediate_action_required": risk_counts[SecurityRisk.CRITICAL] + risk_counts[SecurityRisk.HIGH]
            },
            "todos_by_risk": {
                risk.value: [
                    {
                        "file": todo.file_path,
                        "line": todo.line_number,
                        "type": todo.comment_type,
                        "text": todo.text,
                        "category": todo.category,
                        "recommendation": todo.recommendation
                    }
                    for todo in sorted_todos if todo.security_risk == risk
                ]
                for risk in SecurityRisk
            },
            "categories": {
                category: len(todos) for category, todos in categories.items()
            }
        }
        
        return report
    
    def print_security_summary(self):
        """Print a security-focused summary"""
        report = self.generate_report()
        
        print("🔒 XORB Security TODO Analysis Report")
        print("=" * 50)
        print(f"Total TODOs found: {report['summary']['total_todos']}")
        print(f"Security-related: {report['summary']['security_related']}")
        print(f"Immediate action required: {report['summary']['immediate_action_required']}")
        print()
        
        print("Risk Level Breakdown:")
        for risk in SecurityRisk:
            count = report['summary']['by_risk_level'][risk.value]
            if count > 0:
                emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "💡", "none": "📝"}
                print(f"  {emoji.get(risk.value, '•')} {risk.value.upper()}: {count}")
        print()
        
        # Show critical and high priority items
        critical_items = report['todos_by_risk']['critical']
        high_items = report['todos_by_risk']['high']
        
        if critical_items:
            print("🚨 CRITICAL SECURITY ISSUES (Immediate Action Required):")
            for item in critical_items[:5]:  # Show first 5
                print(f"  📁 {item['file']}:{item['line']}")
                print(f"     {item['text']}")
                print(f"     Category: {item['category']}")
                print()
        
        if high_items:
            print("⚠️ HIGH PRIORITY SECURITY ISSUES:")
            for item in high_items[:5]:  # Show first 5
                print(f"  📁 {item['file']}:{item['line']}")
                print(f"     {item['text']}")
                print(f"     Category: {item['category']}")
                print()
        
        # Show recommendations
        print("📋 RECOMMENDED ACTIONS:")
        actions = [
            "1. Address all CRITICAL items immediately",
            "2. Review HIGH priority items before production",
            "3. Create GitHub issues for MEDIUM priority items",
            "4. Schedule LOW priority items for maintenance cycles",
            "5. Add security review to PR process"
        ]
        for action in actions:
            print(f"  {action}")

def main():
    project_root = Path(".").absolute()
    analyzer = SecurityTodoAnalyzer(project_root)
    
    print("🔍 Scanning for TODO comments...")
    todos = analyzer.scan_files()
    
    print(f"✅ Found {len(todos)} TODO comments")
    print()
    
    # Generate and display summary
    analyzer.print_security_summary()
    
    # Save detailed report
    report = analyzer.generate_report()
    report_path = project_root / "security_todo_report.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()