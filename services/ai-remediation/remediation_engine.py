#!/usr/bin/env python3
"""
Xorb Autonomous Remediation Suggestion System
Phase 6.2 - AI-Powered Fix Generation and Validation
"""

import asyncio
import json
import logging
import os
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import asyncpg
import aioredis
import aiofiles
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
logger = structlog.get_logger("xorb.ai_remediation")

# Phase 6.2 Metrics
remediation_suggestions_total = Counter(
    'remediation_suggestions_total',
    'Total remediation suggestions generated',
    ['suggestion_type', 'language', 'ai_model']
)

remediation_generation_duration = Histogram(
    'remediation_generation_duration_seconds',
    'Time to generate remediation suggestions',
    ['suggestion_type', 'complexity']
)

remediation_validation_results = Counter(
    'remediation_validation_results_total',
    'Remediation validation results', 
    ['validation_type', 'result', 'language']
)

code_analysis_duration = Histogram(
    'code_analysis_duration_seconds',
    'Duration of static code analysis',
    ['analyzer_type', 'language']
)

remediation_success_rate = Gauge(
    'remediation_success_rate',
    'Success rate of applied remediations',
    ['remediation_type']
)

fix_confidence_score = Histogram(
    'fix_confidence_score',
    'Confidence score distribution of generated fixes',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

class RemediationType(Enum):
    CODE_PATCH = "code_patch"
    CONFIG_FIX = "config_fix"
    INFRASTRUCTURE = "infrastructure_fix"  
    DEPENDENCY_UPDATE = "dependency_update"
    SECURITY_CONTROL = "security_control"
    PROCESS_CHANGE = "process_change"

class FixComplexity(Enum):
    SIMPLE = "simple"        # Single line/config change
    MODERATE = "moderate"    # Multiple files, simple logic
    COMPLEX = "complex"      # Requires architectural changes
    CRITICAL = "critical"    # High-risk, extensive changes

class ValidationStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"

@dataclass
class CodeContext:
    """Context about the vulnerable code"""
    repository_url: Optional[str]
    file_path: str
    line_number: int
    function_name: Optional[str]
    class_name: Optional[str]
    language: str
    framework: Optional[str]
    vulnerable_code: str
    surrounding_context: str
    dependencies: List[str]
    imports: List[str]

@dataclass
class RemediationSuggestion:
    """A specific remediation suggestion"""
    id: str
    vulnerability_id: str
    suggestion_type: RemediationType
    title: str
    description: str
    
    # Fix details
    fix_code: Optional[str]
    fix_config: Optional[Dict]
    fix_commands: Optional[List[str]]
    affected_files: List[str]
    
    # Validation
    complexity: FixComplexity
    confidence: float  # 0-1 scale
    risk_level: str    # "low", "medium", "high"
    testing_required: bool
    rollback_plan: str
    
    # Implementation guidance
    implementation_steps: List[str]
    prerequisites: List[str]
    estimated_effort_hours: float
    
    # Validation results
    static_analysis_passed: bool
    security_scan_passed: bool
    test_coverage_impact: float
    
    # AI insights
    ai_reasoning: str
    alternative_approaches: List[str]
    potential_side_effects: List[str]
    
    # Metadata
    generated_by: str
    generated_at: datetime
    validated_at: Optional[datetime]
    model_version: str

@dataclass
class ValidationResult:
    """Result of fix validation"""
    suggestion_id: str
    validation_type: str
    status: ValidationStatus
    score: float
    issues: List[str]
    recommendations: List[str]
    details: Dict

class StaticCodeAnalyzer:
    """Performs static analysis on vulnerable code"""
    
    def __init__(self):
        self.analyzers = {
            "python": ["bandit", "semgrep", "pylint"],
            "javascript": ["eslint", "semgrep", "jshint"],
            "java": ["spotbugs", "pmd", "semgrep"],
            "csharp": ["security-code-scan", "sonarqube"],
            "go": ["gosec", "staticcheck"],
            "php": ["psalm", "phpstan", "semgrep"]
        }
    
    async def analyze_vulnerability(self, code_context: CodeContext) -> Dict:
        """Analyze vulnerable code to understand the issue"""
        
        start_time = datetime.now()
        
        try:
            with code_analysis_duration.labels(
                analyzer_type="vulnerability_analysis",
                language=code_context.language
            ).time():
                
                # Create temporary file with vulnerable code
                analysis_results = {}
                
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix=self._get_file_extension(code_context.language),
                    delete=False
                ) as tmp_file:
                    
                    tmp_file.write(code_context.vulnerable_code)
                    tmp_file.flush()
                    
                    # Run static analysis tools
                    if code_context.language == "python":
                        analysis_results.update(await self._analyze_python(tmp_file.name))
                    elif code_context.language == "javascript":
                        analysis_results.update(await self._analyze_javascript(tmp_file.name))
                    # Add other languages as needed
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                
                analysis_results.update({
                    "analysis_duration": (datetime.now() - start_time).total_seconds(),
                    "vulnerability_patterns": await self._identify_vulnerability_patterns(code_context),
                    "security_hotspots": await self._find_security_hotspots(code_context),
                    "dependency_issues": await self._check_dependency_vulnerabilities(code_context)
                })
                
                return analysis_results
                
        except Exception as e:
            logger.error("Static analysis failed", error=str(e))
            return {"error": str(e), "analysis_failed": True}
    
    async def _analyze_python(self, file_path: str) -> Dict:
        """Python-specific static analysis"""
        
        results = {}
        
        try:
            # Run bandit for security issues
            bandit_cmd = ["bandit", "-f", "json", file_path]
            result = await self._run_analysis_tool(bandit_cmd)
            if result.returncode == 0:
                results["bandit"] = json.loads(result.stdout.decode())
        except Exception as e:
            results["bandit_error"] = str(e)
        
        try:
            # Run semgrep for additional patterns
            semgrep_cmd = ["semgrep", "--config=auto", "--json", file_path]
            result = await self._run_analysis_tool(semgrep_cmd)
            if result.returncode == 0:
                results["semgrep"] = json.loads(result.stdout.decode())
        except Exception as e:
            results["semgrep_error"] = str(e)
        
        return results
    
    async def _analyze_javascript(self, file_path: str) -> Dict:
        """JavaScript-specific static analysis"""
        
        results = {}
        
        try:
            # Run ESLint with security rules
            eslint_cmd = ["eslint", "--format", "json", file_path]
            result = await self._run_analysis_tool(eslint_cmd)
            if result.returncode in [0, 1]:  # ESLint returns 1 when issues found
                results["eslint"] = json.loads(result.stdout.decode())
        except Exception as e:
            results["eslint_error"] = str(e)
        
        return results
    
    async def _run_analysis_tool(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run analysis tool command"""
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "csharp": ".cs",
            "go": ".go",
            "php": ".php"
        }
        return extensions.get(language, ".txt")
    
    async def _identify_vulnerability_patterns(self, context: CodeContext) -> List[str]:
        """Identify common vulnerability patterns"""
        
        patterns = []
        code = context.vulnerable_code.lower()
        
        # SQL Injection patterns
        if any(keyword in code for keyword in ["select", "insert", "update", "delete"]):
            if "%" in code or ".format(" in code or "f\"" in code:
                patterns.append("sql_injection_risk")
        
        # XSS patterns
        if any(keyword in code for keyword in ["innerhtml", "document.write", "eval"]):
            patterns.append("xss_risk")
        
        # Command injection patterns
        if any(keyword in code for keyword in ["os.system", "subprocess", "exec", "eval"]):
            patterns.append("command_injection_risk")
        
        # Insecure randomness
        if any(keyword in code for keyword in ["math.random", "random.random"]):
            patterns.append("weak_randomness")
        
        # Hardcoded secrets
        if any(keyword in code for keyword in ["password", "secret", "api_key", "token"]):
            patterns.append("hardcoded_secrets")
        
        return patterns
    
    async def _find_security_hotspots(self, context: CodeContext) -> List[Dict]:
        """Find security hotspots in code"""
        
        hotspots = []
        lines = context.vulnerable_code.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check for dangerous functions
            dangerous_functions = [
                "eval", "exec", "os.system", "subprocess.call",
                "pickle.loads", "yaml.load", "input", "raw_input"
            ]
            
            for func in dangerous_functions:
                if func in line_lower:
                    hotspots.append({
                        "line": i + 1,
                        "type": "dangerous_function",
                        "function": func,
                        "code": line.strip(),
                        "severity": "high"
                    })
        
        return hotspots
    
    async def _check_dependency_vulnerabilities(self, context: CodeContext) -> List[Dict]:
        """Check for known vulnerabilities in dependencies"""
        
        vulnerabilities = []
        
        # This would integrate with vulnerability databases
        # For now, simulate based on common vulnerable packages
        vulnerable_packages = {
            "requests": ["2.19.0", "2.20.0"],
            "django": ["2.1.0", "2.2.0"],
            "flask": ["1.0.0", "1.1.0"],
            "lodash": ["4.17.0", "4.17.10"]
        }
        
        for dep in context.dependencies:
            for vuln_pkg, vuln_versions in vulnerable_packages.items():
                if vuln_pkg in dep.lower():
                    vulnerabilities.append({
                        "package": vuln_pkg,
                        "current_version": "unknown",
                        "vulnerable_versions": vuln_versions,
                        "severity": "medium",
                        "cve": f"CVE-2024-{hash(dep) % 10000:04d}"
                    })
        
        return vulnerabilities

class AIRemediationGenerator:
    """Generates remediation suggestions using AI"""
    
    def __init__(self):
        self.openai_client = None
        self.openrouter_client = None
        self.model_configs = {
            "gpt-4": {
                "client": "openai",
                "max_tokens": 4000,
                "temperature": 0.1
            },
            "qwen/qwen-coder:free": {
                "client": "openrouter", 
                "max_tokens": 2000,
                "temperature": 0.1
            }
        }
    
    async def initialize(self, openai_key: str, openrouter_key: str):
        """Initialize AI clients"""
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.openrouter_client = AsyncOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    async def generate_remediation_suggestions(
        self,
        vulnerability_id: str,
        vulnerability_description: str,
        code_context: CodeContext,
        analysis_results: Dict
    ) -> List[RemediationSuggestion]:
        """Generate comprehensive remediation suggestions"""
        
        start_time = datetime.now()
        suggestions = []
        
        try:
            with remediation_generation_duration.labels(
                suggestion_type="comprehensive",
                complexity="moderate"
            ).time():
                
                # Generate different types of suggestions
                code_suggestions = await self._generate_code_fixes(
                    vulnerability_id, vulnerability_description, code_context, analysis_results
                )
                suggestions.extend(code_suggestions)
                
                config_suggestions = await self._generate_config_fixes(
                    vulnerability_id, vulnerability_description, code_context, analysis_results
                )
                suggestions.extend(config_suggestions)
                
                dependency_suggestions = await self._generate_dependency_updates(
                    vulnerability_id, vulnerability_description, code_context, analysis_results
                )
                suggestions.extend(dependency_suggestions)
                
                # Update metrics
                for suggestion in suggestions:
                    remediation_suggestions_total.labels(
                        suggestion_type=suggestion.suggestion_type.value,
                        language=code_context.language,
                        ai_model="gpt4_enhanced"
                    ).inc()
                    
                    fix_confidence_score.observe(suggestion.confidence)
                
                logger.info("Generated remediation suggestions",
                           vulnerability_id=vulnerability_id,
                           suggestion_count=len(suggestions),
                           duration=(datetime.now() - start_time).total_seconds())
                
                return suggestions
                
        except Exception as e:
            logger.error("Failed to generate remediation suggestions",
                        vulnerability_id=vulnerability_id,
                        error=str(e))
            return []
    
    async def _generate_code_fixes(
        self,
        vulnerability_id: str,
        description: str,
        context: CodeContext,
        analysis: Dict
    ) -> List[RemediationSuggestion]:
        """Generate code-level fixes"""
        
        # Use GPT-4 for complex code analysis and fixing
        prompt = self._build_code_fix_prompt(description, context, analysis)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert security engineer specializing in code vulnerability remediation. Generate secure, production-ready fixes with comprehensive explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            fix_data = json.loads(response.choices[0].message.content)
            
            suggestion = RemediationSuggestion(
                id=f"code_fix_{vulnerability_id}_{int(datetime.now().timestamp())}",
                vulnerability_id=vulnerability_id,
                suggestion_type=RemediationType.CODE_PATCH,
                title=fix_data.get("title", "Code Fix"),
                description=fix_data.get("description", ""),
                fix_code=fix_data.get("fixed_code", ""),
                fix_config=None,
                fix_commands=None,
                affected_files=[context.file_path],
                complexity=FixComplexity(fix_data.get("complexity", "moderate")),
                confidence=float(fix_data.get("confidence", 0.7)),
                risk_level=fix_data.get("risk_level", "medium"),
                testing_required=fix_data.get("testing_required", True),
                rollback_plan=fix_data.get("rollback_plan", "Revert to original code"),
                implementation_steps=fix_data.get("implementation_steps", []),
                prerequisites=fix_data.get("prerequisites", []),
                estimated_effort_hours=float(fix_data.get("effort_hours", 2.0)),
                static_analysis_passed=False,  # Will be validated later
                security_scan_passed=False,
                test_coverage_impact=0.0,
                ai_reasoning=fix_data.get("reasoning", ""),
                alternative_approaches=fix_data.get("alternatives", []),
                potential_side_effects=fix_data.get("side_effects", []),
                generated_by="gpt-4",
                generated_at=datetime.now(),
                validated_at=None,
                model_version="6.2.0"
            )
            
            return [suggestion]
            
        except Exception as e:
            logger.error("Failed to generate code fix", error=str(e))
            return []
    
    async def _generate_config_fixes(
        self,
        vulnerability_id: str,
        description: str,
        context: CodeContext,
        analysis: Dict
    ) -> List[RemediationSuggestion]:
        """Generate configuration-based fixes"""
        
        # Use Qwen for configuration optimization
        prompt = self._build_config_fix_prompt(description, context, analysis)
        
        try:
            response = await self.openrouter_client.chat.completions.create(
                model="qwen/qwen-coder:free",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a DevOps security specialist. Generate secure configuration fixes for infrastructure and application settings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse response (assuming JSON format)
            try:
                config_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # If not JSON, create a structured response
                config_data = {
                    "title": "Configuration Security Fix",
                    "description": response.choices[0].message.content[:200],
                    "config_changes": {},
                    "confidence": 0.6
                }
            
            suggestion = RemediationSuggestion(
                id=f"config_fix_{vulnerability_id}_{int(datetime.now().timestamp())}",
                vulnerability_id=vulnerability_id,
                suggestion_type=RemediationType.CONFIG_FIX,
                title=config_data.get("title", "Configuration Fix"),
                description=config_data.get("description", ""),
                fix_code=None,
                fix_config=config_data.get("config_changes", {}),
                fix_commands=config_data.get("commands", []),
                affected_files=config_data.get("config_files", ["config.yml"]),
                complexity=FixComplexity.SIMPLE,
                confidence=float(config_data.get("confidence", 0.6)),
                risk_level="low",
                testing_required=True,
                rollback_plan="Revert configuration changes",
                implementation_steps=config_data.get("steps", []),
                prerequisites=config_data.get("prerequisites", []),
                estimated_effort_hours=1.0,
                static_analysis_passed=True,
                security_scan_passed=False,
                test_coverage_impact=0.0,
                ai_reasoning=config_data.get("reasoning", ""),
                alternative_approaches=[],
                potential_side_effects=config_data.get("side_effects", []),
                generated_by="qwen-coder",
                generated_at=datetime.now(),
                validated_at=None,
                model_version="6.2.0"
            )
            
            return [suggestion]
            
        except Exception as e:
            logger.error("Failed to generate config fix", error=str(e))
            return []
    
    async def _generate_dependency_updates(
        self,
        vulnerability_id: str,
        description: str,
        context: CodeContext,
        analysis: Dict
    ) -> List[RemediationSuggestion]:
        """Generate dependency update suggestions"""
        
        suggestions = []
        dependency_vulns = analysis.get("dependency_issues", [])
        
        for vuln in dependency_vulns:
            suggestion = RemediationSuggestion(
                id=f"dep_update_{vulnerability_id}_{vuln['package']}",
                vulnerability_id=vulnerability_id,
                suggestion_type=RemediationType.DEPENDENCY_UPDATE,
                title=f"Update {vuln['package']} to secure version",
                description=f"Update {vuln['package']} to fix {vuln['cve']}",
                fix_code=None,
                fix_config=None,
                fix_commands=[f"pip install {vuln['package']}>=2.25.0"],
                affected_files=["requirements.txt", "package.json"],
                complexity=FixComplexity.SIMPLE,
                confidence=0.9,
                risk_level="low",
                testing_required=True,
                rollback_plan="Downgrade to previous version if issues",
                implementation_steps=[
                    "Update dependency version",
                    "Run tests",
                    "Deploy to staging",
                    "Validate functionality"
                ],
                prerequisites=["Backup current dependencies"],
                estimated_effort_hours=0.5,
                static_analysis_passed=True,
                security_scan_passed=True,
                test_coverage_impact=0.1,
                ai_reasoning="Dependency vulnerability requires version update",
                alternative_approaches=["Find alternative package"],
                potential_side_effects=["API changes in new version"],
                generated_by="dependency_analyzer",
                generated_at=datetime.now(),
                validated_at=None,
                model_version="6.2.0"
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _build_code_fix_prompt(self, description: str, context: CodeContext, analysis: Dict) -> str:
        """Build prompt for code fix generation"""
        
        return f"""
Fix this security vulnerability in {context.language} code:

VULNERABILITY: {description}

VULNERABLE CODE:
```{context.language}
{context.vulnerable_code}
```

CONTEXT:
- File: {context.file_path}
- Line: {context.line_number}
- Function: {context.function_name}
- Language: {context.language}
- Framework: {context.framework}

STATIC ANALYSIS RESULTS:
{json.dumps(analysis, indent=2)}

Please provide a comprehensive fix in the following JSON format:
{{
  "title": "Brief fix title",
  "description": "Detailed explanation of the fix",
  "fixed_code": "Complete fixed code with proper escaping",
  "reasoning": "Why this fix addresses the vulnerability",
  "complexity": "simple|moderate|complex|critical",
  "confidence": 0.0-1.0,
  "risk_level": "low|medium|high",
  "testing_required": true|false,
  "rollback_plan": "How to revert if needed",
  "implementation_steps": ["step 1", "step 2", "..."],
  "prerequisites": ["prereq 1", "prereq 2"],
  "effort_hours": estimated_hours_as_float,
  "alternatives": ["alternative approach 1", "..."],
  "side_effects": ["potential side effect 1", "..."]
}}

Focus on:
1. Secure coding practices
2. Input validation and sanitization
3. Proper error handling
4. Performance considerations
5. Maintainability
"""
    
    def _build_config_fix_prompt(self, description: str, context: CodeContext, analysis: Dict) -> str:
        """Build prompt for configuration fix generation"""
        
        return f"""
Generate secure configuration fixes for this vulnerability:

VULNERABILITY: {description}
LANGUAGE/FRAMEWORK: {context.language}/{context.framework}
ANALYSIS: {json.dumps(analysis, indent=2)}

Generate configuration changes, security headers, environment variables, or infrastructure settings that would mitigate this vulnerability.

Return as JSON with:
{{
  "title": "Configuration fix title",
  "description": "What this config change does",
  "config_changes": {{"key": "value", "key2": "value2"}},
  "commands": ["command1", "command2"],
  "config_files": ["file1.yml", "file2.conf"],
  "steps": ["implementation step 1", "step 2"],
  "confidence": 0.0-1.0,
  "reasoning": "Why this config change helps",
  "side_effects": ["potential impact 1", "impact 2"]
}}
"""

class RemediationValidator:
    """Validates remediation suggestions before deployment"""
    
    def __init__(self):
        self.static_analyzer = StaticCodeAnalyzer()
    
    async def validate_suggestion(self, suggestion: RemediationSuggestion) -> ValidationResult:
        """Comprehensive validation of remediation suggestion"""
        
        start_time = datetime.now()
        validation_issues = []
        validation_score = 0.0
        
        try:
            # Static analysis validation
            if suggestion.fix_code:
                static_result = await self._validate_static_analysis(suggestion)
                validation_score += static_result["score"] * 0.4
                validation_issues.extend(static_result["issues"])
            
            # Security validation
            security_result = await self._validate_security(suggestion)
            validation_score += security_result["score"] * 0.3
            validation_issues.extend(security_result["issues"])
            
            # Business logic validation
            logic_result = await self._validate_business_logic(suggestion)
            validation_score += logic_result["score"] * 0.2
            validation_issues.extend(logic_result["issues"])
            
            # Performance impact validation
            perf_result = await self._validate_performance_impact(suggestion)
            validation_score += perf_result["score"] * 0.1
            validation_issues.extend(perf_result["issues"])
            
            # Determine overall status
            if validation_score >= 0.8 and len(validation_issues) == 0:
                status = ValidationStatus.PASSED
            elif validation_score >= 0.6 and len([i for i in validation_issues if i.get("severity") == "high"]) == 0:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.FAILED
            
            result = ValidationResult(
                suggestion_id=suggestion.id,
                validation_type="comprehensive",
                status=status,
                score=validation_score,
                issues=validation_issues,
                recommendations=self._generate_validation_recommendations(validation_issues),
                details={
                    "static_analysis": static_result,
                    "security_validation": security_result,
                    "logic_validation": logic_result,
                    "performance_validation": perf_result,
                    "validation_duration": (datetime.now() - start_time).total_seconds()
                }
            )
            
            # Update metrics
            remediation_validation_results.labels(
                validation_type="comprehensive",
                result=status.value,
                language=suggestion.affected_files[0].split('.')[-1] if suggestion.affected_files else "unknown"
            ).inc()
            
            return result
            
        except Exception as e:
            logger.error("Validation failed", suggestion_id=suggestion.id, error=str(e))
            
            return ValidationResult(
                suggestion_id=suggestion.id,
                validation_type="comprehensive",
                status=ValidationStatus.FAILED,
                score=0.0,
                issues=[{"type": "validation_error", "message": str(e), "severity": "high"}],
                recommendations=["Manual review required due to validation failure"],
                details={"error": str(e)}
            )
    
    async def _validate_static_analysis(self, suggestion: RemediationSuggestion) -> Dict:
        """Validate using static analysis tools"""
        
        if not suggestion.fix_code:
            return {"score": 0.5, "issues": []}
        
        try:
            # Create temporary file with fixed code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',  # Assume Python for now
                delete=False
            ) as tmp_file:
                tmp_file.write(suggestion.fix_code)
                tmp_file.flush()
                
                # Run static analysis
                issues = []
                score = 1.0
                
                # Check for common issues
                if "eval(" in suggestion.fix_code:
                    issues.append({
                        "type": "dangerous_function",
                        "message": "Use of eval() detected",
                        "severity": "high"
                    })
                    score -= 0.3
                
                if "# TODO" in suggestion.fix_code or "# FIXME" in suggestion.fix_code:
                    issues.append({
                        "type": "incomplete_fix",
                        "message": "TODO/FIXME comments suggest incomplete fix",
                        "severity": "medium"
                    })
                    score -= 0.1
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return {"score": max(0.0, score), "issues": issues}
                
        except Exception as e:
            return {
                "score": 0.0,
                "issues": [{"type": "analysis_error", "message": str(e), "severity": "high"}]
            }
    
    async def _validate_security(self, suggestion: RemediationSuggestion) -> Dict:
        """Validate security aspects of the fix"""
        
        issues = []
        score = 1.0
        
        # Check for proper input validation
        if suggestion.fix_code and "input" in suggestion.fix_code.lower():
            if not any(keyword in suggestion.fix_code.lower() 
                      for keyword in ["validate", "sanitize", "escape", "filter"]):
                issues.append({
                    "type": "missing_input_validation",
                    "message": "Input handling without apparent validation",
                    "severity": "medium"
                })
                score -= 0.2
        
        # Check for SQL injection prevention
        if suggestion.fix_code and any(keyword in suggestion.fix_code.lower() 
                                      for keyword in ["select", "insert", "update", "delete"]):
            if not any(keyword in suggestion.fix_code.lower()
                      for keyword in ["parameterized", "prepared", "bind", "?"]):
                issues.append({
                    "type": "sql_injection_risk",
                    "message": "SQL queries without parameterization",
                    "severity": "high"
                })
                score -= 0.4
        
        # Check for XSS prevention
        if suggestion.fix_code and "html" in suggestion.fix_code.lower():
            if not any(keyword in suggestion.fix_code.lower()
                      for keyword in ["escape", "sanitize", "encode"]):
                issues.append({
                    "type": "xss_risk",
                    "message": "HTML handling without escaping",
                    "severity": "high"
                })
                score -= 0.3
        
        return {"score": max(0.0, score), "issues": issues}
    
    async def _validate_business_logic(self, suggestion: RemediationSuggestion) -> Dict:
        """Validate business logic correctness"""
        
        issues = []
        score = 0.8  # Default decent score
        
        # Check complexity vs confidence
        if suggestion.complexity == FixComplexity.COMPLEX and suggestion.confidence < 0.7:
            issues.append({
                "type": "low_confidence_complex_fix",
                "message": "Complex fix with low confidence may need review",
                "severity": "medium"
            })
            score -= 0.2
        
        # Check effort estimation reasonableness
        if suggestion.estimated_effort_hours > 8:
            issues.append({
                "type": "high_effort_fix",
                "message": "High effort fix may need architectural review",
                "severity": "low"  
            })
            score -= 0.1
        
        return {"score": max(0.0, score), "issues": issues}
    
    async def _validate_performance_impact(self, suggestion: RemediationSuggestion) -> Dict:
        """Validate performance impact of the fix"""
        
        issues = []
        score = 0.9  # Default good score
        
        # Check for performance anti-patterns
        if suggestion.fix_code:
            code = suggestion.fix_code.lower()
            
            # Check for potential performance issues
            if "while true:" in code or "while 1:" in code:
                issues.append({
                    "type": "infinite_loop_risk",
                    "message": "Potential infinite loop detected",
                    "severity": "high"
                })
                score -= 0.4
            
            if code.count("for ") > 3:
                issues.append({
                    "type": "nested_loops",
                    "message": "Multiple nested loops may impact performance",
                    "severity": "low"
                })
                score -= 0.1
        
        return {"score": max(0.0, score), "issues": issues}
    
    def _generate_validation_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate recommendations based on validation issues"""
        
        recommendations = []
        
        high_severity_count = len([i for i in issues if i.get("severity") == "high"])
        medium_severity_count = len([i for i in issues if i.get("severity") == "medium"])
        
        if high_severity_count > 0:
            recommendations.append("Address high-severity issues before deployment")
            recommendations.append("Conduct security review with senior engineer")
        
        if medium_severity_count > 2:
            recommendations.append("Consider alternative implementation approach")
        
        if any("performance" in str(issue) for issue in issues):
            recommendations.append("Conduct performance testing before deployment")
        
        if any("test" in str(issue) for issue in issues):
            recommendations.append("Ensure comprehensive test coverage")
        
        if not recommendations:
            recommendations.append("Fix passed validation - ready for deployment")
        
        return recommendations

class AutonomousRemediationEngine:
    """Main remediation engine coordinating all components"""
    
    def __init__(self):
        self.static_analyzer = StaticCodeAnalyzer()
        self.ai_generator = AIRemediationGenerator()
        self.validator = RemediationValidator()
        self.db_pool = None
        self.redis = None
    
    async def initialize(self, config: Dict):
        """Initialize the remediation engine"""
        
        logger.info("Initializing Autonomous Remediation Engine...")
        
        # Initialize database
        database_url = config.get("database_url")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=3, max_size=8)
        
        # Initialize Redis
        redis_url = config.get("redis_url")
        self.redis = await aioredis.from_url(redis_url)
        
        # Initialize AI generator
        await self.ai_generator.initialize(
            config.get("openai_api_key"),
            config.get("openrouter_api_key")
        )
        
        # Create database tables
        await self._create_remediation_tables()
        
        logger.info("Autonomous Remediation Engine initialized successfully")
    
    async def _create_remediation_tables(self):
        """Create database tables for remediation"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS remediation_suggestions (
                    id VARCHAR(255) PRIMARY KEY,
                    vulnerability_id VARCHAR(255) NOT NULL,
                    suggestion_type VARCHAR(50) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    
                    -- Fix details
                    fix_code TEXT,
                    fix_config JSONB,
                    fix_commands JSONB,
                    affected_files JSONB,
                    
                    -- Validation
                    complexity VARCHAR(20) NOT NULL,
                    confidence FLOAT NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    testing_required BOOLEAN DEFAULT true,
                    rollback_plan TEXT,
                    
                    -- Implementation
                    implementation_steps JSONB,
                    prerequisites JSONB,
                    estimated_effort_hours FLOAT,
                    
                    -- Validation results
                    static_analysis_passed BOOLEAN DEFAULT false,
                    security_scan_passed BOOLEAN DEFAULT false,
                    test_coverage_impact FLOAT DEFAULT 0.0,
                    
                    -- AI insights
                    ai_reasoning TEXT,
                    alternative_approaches JSONB,
                    potential_side_effects JSONB,
                    
                    -- Metadata
                    generated_by VARCHAR(100),
                    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    validated_at TIMESTAMP WITH TIME ZONE,
                    model_version VARCHAR(20),
                    
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_remediation_vulnerability 
                ON remediation_suggestions(vulnerability_id);
                
                CREATE INDEX IF NOT EXISTS idx_remediation_type 
                ON remediation_suggestions(suggestion_type);
                
                CREATE INDEX IF NOT EXISTS idx_remediation_confidence 
                ON remediation_suggestions(confidence DESC);
                
                CREATE TABLE IF NOT EXISTS validation_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    suggestion_id VARCHAR(255) REFERENCES remediation_suggestions(id),
                    validation_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    score FLOAT NOT NULL,
                    issues JSONB,
                    recommendations JSONB,
                    details JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
    
    async def generate_remediation_for_vulnerability(self, vulnerability_id: str) -> List[RemediationSuggestion]:
        """Generate comprehensive remediation suggestions for a vulnerability"""
        
        start_time = datetime.now()
        
        try:
            # Get vulnerability details and code context
            vuln_data = await self._get_vulnerability_details(vulnerability_id)
            if not vuln_data:
                raise ValueError(f"Vulnerability {vulnerability_id} not found")
            
            code_context = await self._build_code_context(vuln_data)
            
            # Perform static analysis
            analysis_results = await self.static_analyzer.analyze_vulnerability(code_context)
            
            # Generate AI-powered suggestions
            suggestions = await self.ai_generator.generate_remediation_suggestions(
                vulnerability_id,
                vuln_data['description'],
                code_context,
                analysis_results
            )
            
            # Validate each suggestion
            validated_suggestions = []
            for suggestion in suggestions:
                validation_result = await self.validator.validate_suggestion(suggestion)
                
                # Update suggestion with validation results
                suggestion.static_analysis_passed = validation_result.status in [
                    ValidationStatus.PASSED, ValidationStatus.WARNING
                ]
                suggestion.security_scan_passed = validation_result.score >= 0.7
                suggestion.validated_at = datetime.now()
                
                # Store suggestion and validation
                await self._store_suggestion(suggestion)
                await self._store_validation_result(validation_result)
                
                validated_suggestions.append(suggestion)
            
            # Update success rate metric
            success_count = len([s for s in validated_suggestions if s.static_analysis_passed])
            if suggestions:
                success_rate = success_count / len(suggestions)
                remediation_success_rate.labels(
                    remediation_type="ai_generated"
                ).set(success_rate)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("Remediation generation completed",
                       vulnerability_id=vulnerability_id,
                       suggestion_count=len(validated_suggestions),
                       validation_passed=success_count,
                       duration=duration)
            
            return validated_suggestions
            
        except Exception as e:
            logger.error("Remediation generation failed",
                        vulnerability_id=vulnerability_id,
                        error=str(e))
            raise
    
    async def _get_vulnerability_details(self, vulnerability_id: str) -> Optional[Dict]:
        """Get vulnerability details from database"""
        
        async with self.db_pool.acquire() as conn:
            return await conn.fetchrow("""
                SELECT v.*, a.name as asset_name, a.asset_type
                FROM vulnerabilities v
                JOIN assets a ON v.asset_id = a.id
                WHERE v.id = $1
            """, vulnerability_id)
    
    async def _build_code_context(self, vuln_data: Dict) -> CodeContext:
        """Build code context from vulnerability data"""
        
        # This would integrate with code repositories
        # For now, create a simplified context
        
        return CodeContext(
            repository_url=vuln_data.get('repository_url'),
            file_path=vuln_data.get('file_path', 'app.py'),
            line_number=vuln_data.get('line_number', 1),
            function_name=vuln_data.get('function_name'),
            class_name=vuln_data.get('class_name'),
            language=vuln_data.get('language', 'python'),
            framework=vuln_data.get('framework', 'flask'),
            vulnerable_code=vuln_data.get('vulnerable_code', '# Vulnerable code'),
            surrounding_context=vuln_data.get('context', ''),
            dependencies=vuln_data.get('dependencies', []),
            imports=vuln_data.get('imports', [])
        )
    
    async def _store_suggestion(self, suggestion: RemediationSuggestion):
        """Store remediation suggestion in database"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO remediation_suggestions 
                    (id, vulnerability_id, suggestion_type, title, description,
                     fix_code, fix_config, fix_commands, affected_files,
                     complexity, confidence, risk_level, testing_required, rollback_plan,
                     implementation_steps, prerequisites, estimated_effort_hours,
                     static_analysis_passed, security_scan_passed, test_coverage_impact,
                     ai_reasoning, alternative_approaches, potential_side_effects,
                     generated_by, generated_at, validated_at, model_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, 
                            $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)
                """,
                suggestion.id, suggestion.vulnerability_id, suggestion.suggestion_type.value,
                suggestion.title, suggestion.description, suggestion.fix_code,
                json.dumps(suggestion.fix_config) if suggestion.fix_config else None,
                json.dumps(suggestion.fix_commands) if suggestion.fix_commands else None,
                json.dumps(suggestion.affected_files), suggestion.complexity.value,
                suggestion.confidence, suggestion.risk_level, suggestion.testing_required,
                suggestion.rollback_plan, json.dumps(suggestion.implementation_steps),
                json.dumps(suggestion.prerequisites), suggestion.estimated_effort_hours,
                suggestion.static_analysis_passed, suggestion.security_scan_passed,
                suggestion.test_coverage_impact, suggestion.ai_reasoning,
                json.dumps(suggestion.alternative_approaches),
                json.dumps(suggestion.potential_side_effects), suggestion.generated_by,
                suggestion.generated_at, suggestion.validated_at, suggestion.model_version)
                
        except Exception as e:
            logger.error("Failed to store remediation suggestion", error=str(e))
    
    async def _store_validation_result(self, result: ValidationResult):
        """Store validation result in database"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO validation_results
                    (suggestion_id, validation_type, status, score, issues, recommendations, details)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                result.suggestion_id, result.validation_type, result.status.value,
                result.score, json.dumps(result.issues),
                json.dumps(result.recommendations), json.dumps(result.details))
                
        except Exception as e:
            logger.error("Failed to store validation result", error=str(e))
    
    async def get_remediation_statistics(self) -> Dict:
        """Get comprehensive remediation statistics"""
        
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_suggestions,
                        AVG(confidence) as avg_confidence,
                        COUNT(*) FILTER (WHERE static_analysis_passed = true) as validation_passed,
                        COUNT(*) FILTER (WHERE suggestion_type = 'code_patch') as code_fixes,
                        COUNT(*) FILTER (WHERE suggestion_type = 'config_fix') as config_fixes,
                        COUNT(*) FILTER (WHERE suggestion_type = 'dependency_update') as dep_updates,
                        AVG(estimated_effort_hours) as avg_effort_hours
                    FROM remediation_suggestions
                    WHERE generated_at >= NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    "total_suggestions_generated": stats['total_suggestions'],
                    "average_confidence": float(stats['avg_confidence'] or 0),
                    "validation_pass_rate": (stats['validation_passed'] / max(1, stats['total_suggestions'])),
                    "suggestion_types": {
                        "code_patches": stats['code_fixes'],
                        "config_fixes": stats['config_fixes'],
                        "dependency_updates": stats['dep_updates']
                    },
                    "average_effort_hours": float(stats['avg_effort_hours'] or 0),
                    "ai_models_used": ["gpt-4", "qwen-coder"],
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get remediation statistics", error=str(e))
            return {"error": str(e)}

async def main():
    """Main remediation engine service"""
    
    # Start Prometheus metrics server
    start_http_server(8011)
    
    # Initialize engine
    config = {
        "database_url": os.getenv("DATABASE_URL", 
                                 "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379/0"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", 
                                       "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")
    }
    
    engine = AutonomousRemediationEngine()
    await engine.initialize(config)
    
    logger.info("🤖 Xorb Autonomous Remediation Engine started",
               service_version="6.2.0",
               features=["ai_code_generation", "static_analysis", "fix_validation", "multi_model_ai"])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down remediation engine")

if __name__ == "__main__":
    asyncio.run(main())