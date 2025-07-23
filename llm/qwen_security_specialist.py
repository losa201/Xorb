#!/usr/bin/env python3
"""
Qwen 2.5 235B Security Specialist
Optimized prompts and workflows for the Qwen model's capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .intelligent_client import IntelligentLLMClient, LLMRequest, TaskType
from .payload_generator import PayloadCategory, TargetContext, GeneratedPayload

logger = logging.getLogger(__name__)

class QwenSecuritySpecialist:
    """Specialized security testing using Qwen 2.5 235B model"""
    
    def __init__(self, llm_client: IntelligentLLMClient):
        self.llm_client = llm_client
        self.preferred_model = "qwen/qwen3-235b-a22b-07-25:free"
    
    async def generate_advanced_payloads(
        self,
        category: PayloadCategory,
        target_context: TargetContext,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate advanced payloads using Qwen's superior reasoning"""
        
        # Craft specialized prompt for Qwen 2.5's capabilities
        prompt = self._build_qwen_security_prompt(category, target_context, count)
        
        request = LLMRequest(
            task_type=TaskType.PAYLOAD_GENERATION,
            prompt=prompt,
            target_info={
                "url": target_context.url,
                "tech_stack": target_context.technology_stack,
                "os": target_context.operating_system
            },
            max_tokens=3000,  # Leverage larger context
            temperature=0.7,
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            return self._parse_qwen_response(response, category)
        except Exception as e:
            logger.error(f"Qwen payload generation failed: {e}")
            return []
    
    def _build_qwen_security_prompt(
        self,
        category: PayloadCategory,
        target_context: TargetContext,
        count: int
    ) -> str:
        """Build optimized prompt for Qwen 2.5 235B model"""
        
        # Base context
        context_info = f"""
TARGET ANALYSIS:
- URL: {target_context.url or 'Not specified'}
- Technology Stack: {', '.join(target_context.technology_stack or ['Unknown'])}
- Operating System: {target_context.operating_system or 'Unknown'}
- Web Server: {target_context.web_server or 'Unknown'}
- Input Fields: {', '.join(target_context.input_fields or ['Unknown'])}
- Parameters: {', '.join(target_context.parameters or ['Unknown'])}
"""
        
        # Category-specific advanced prompts
        category_prompts = {
            PayloadCategory.XSS: f"""
TASK: Generate {count} advanced Cross-Site Scripting (XSS) payloads for authorized security testing.

REQUIREMENTS:
1. Bypass modern WAF filters (CloudFlare, AWS WAF, ModSecurity)
2. Work with CSP restrictions (Content Security Policy)
3. Target specific input contexts (HTML, JavaScript, CSS, URL)
4. Include both reflected and stored XSS variants
5. Demonstrate advanced techniques (DOM-based, mutation XSS, CSP bypass)

TECHNICAL FOCUS:
- Filter evasion techniques (encoding, obfuscation, polyglots)
- Context-aware payloads (attribute, script, HTML context)
- Modern browser exploitation (Chrome, Firefox, Safari)
- Social engineering vectors (credential harvesting, keylogging)

OUTPUT FORMAT (JSON):
{{
  "payloads": [
    {{
      "payload": "actual payload string",
      "technique": "bypass method used",
      "context": "where to inject (parameter, header, etc)",
      "explanation": "detailed technical explanation",
      "evasion_score": 1-10,
      "impact_level": "low/medium/high/critical",
      "detection_difficulty": 1-10,
      "remediation": "specific fix recommendations"
    }}
  ]
}}
""",
            
            PayloadCategory.SQL_INJECTION: f"""
TASK: Generate {count} advanced SQL Injection payloads for authorized penetration testing.

REQUIREMENTS:
1. Multi-database compatibility (MySQL, PostgreSQL, SQLite, MSSQL, Oracle)
2. Bypass input validation and prepared statement weaknesses
3. Include time-based, union-based, boolean-based, and error-based techniques
4. Target different injection points (WHERE, ORDER BY, INSERT, UPDATE)
5. Advanced exploitation (file read/write, RCE, privilege escalation)

TECHNICAL FOCUS:
- WAF bypass techniques (comment variations, encoding, case manipulation)
- Database-specific functions and syntax
- Blind injection with optimized payloads
- Second-order injection vectors
- NoSQL injection for MongoDB, CouchDB

OUTPUT FORMAT (JSON):
{{
  "payloads": [
    {{
      "payload": "SQL injection string",
      "database_type": "target database",
      "injection_type": "union/boolean/time/error-based",
      "injection_point": "WHERE/ORDER BY/INSERT/etc",
      "explanation": "attack methodology",
      "stealth_level": 1-10,
      "data_extraction": "what data can be extracted",
      "privilege_escalation": "potential for escalation",
      "remediation": "prevention measures"
    }}
  ]
}}
""",
            
            PayloadCategory.SSRF: f"""
TASK: Generate {count} advanced Server-Side Request Forgery (SSRF) payloads.

REQUIREMENTS:
1. Cloud metadata service targeting (AWS, Azure, GCP)
2. Internal network reconnaissance and port scanning
3. Protocol smuggling and bypass techniques
4. File system access via file:// protocol
5. Chain with other vulnerabilities (RCE, authentication bypass)

TECHNICAL FOCUS:
- URL parsing bypass (different encodings, IP representations)
- Cloud environment exploitation (EC2 metadata, Lambda functions)
- Internal service discovery (Redis, MongoDB, Elasticsearch)
- DNS rebinding and time-of-check vulnerabilities
- Gopher protocol exploitation

OUTPUT FORMAT (JSON):
{{
  "payloads": [
    {{
      "payload": "SSRF URL/payload",
      "target_service": "what service to target",
      "bypass_technique": "how it evades filters",
      "explanation": "attack vector description",
      "cloud_platform": "AWS/Azure/GCP if applicable",
      "internal_impact": "what internal resources accessible",
      "chaining_potential": "other vulns it can chain with",
      "remediation": "mitigation strategies"
    }}
  ]
}}
""",
            
            PayloadCategory.RCE: f"""
TASK: Generate {count} advanced Remote Code Execution (RCE) payloads.

REQUIREMENTS:
1. Multi-platform compatibility (Linux, Windows, macOS)
2. Language-specific exploitation (PHP, Python, Node.js, Java)
3. Bypass command filtering and input sanitization
4. File upload and deserialization vectors
5. Memory corruption and buffer overflow techniques

TECHNICAL FOCUS:
- Command injection in different shells (bash, PowerShell, cmd)
- Deserialization attacks (Java, Python pickle, PHP unserialize)
- Template injection (Jinja2, Twig, Freemarker)
- File inclusion to RCE escalation
- Web shell deployment and persistence

OUTPUT FORMAT (JSON):
{{
  "payloads": [
    {{
      "payload": "RCE command/code",
      "platform": "target OS/platform",
      "language": "programming language if applicable",
      "vector": "injection point (command, deserialization, etc)",
      "explanation": "how the exploit works",
      "persistence": "how to maintain access",
      "privilege_level": "user/admin/root achievable",
      "stealth_techniques": "evasion methods",
      "remediation": "prevention measures"
    }}
  ]
}}
"""
        }
        
        base_prompt = f"""
You are an elite cybersecurity researcher specializing in authorized penetration testing and vulnerability research. Your expertise includes advanced exploitation techniques, evasion methods, and defensive countermeasures.

ETHICAL GUIDELINES:
- All payloads are for AUTHORIZED security testing only
- Include defensive recommendations with each payload
- Focus on educational value and responsible disclosure
- Emphasize detection and prevention methods

{context_info}

{category_prompts.get(category, f"Generate advanced {category.value} payloads for security testing.")}

ADDITIONAL REQUIREMENTS:
- Prioritize novel and creative techniques
- Include detailed technical explanations
- Provide accurate severity assessments
- Consider real-world deployment scenarios
- Balance offensive capabilities with defensive insights

Generate payloads that demonstrate your advanced understanding of modern security testing while maintaining ethical standards and educational value.
"""
        
        return base_prompt
    
    def _parse_qwen_response(self, response, category: PayloadCategory) -> List[Dict[str, Any]]:
        """Parse Qwen's structured response"""
        try:
            import json
            
            content = response.content.strip()
            
            # Try to extract JSON from the response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            
            # Parse JSON response
            if content.startswith('{'):
                data = json.loads(content)
                if "payloads" in data:
                    return data["payloads"]
                else:
                    return [data]  # Single payload object
            
            # Fallback text parsing for non-JSON responses
            return self._extract_payloads_from_text(content, category)
            
        except Exception as e:
            logger.error(f"Failed to parse Qwen response: {e}")
            return self._extract_payloads_from_text(response.content, category)
    
    def _extract_payloads_from_text(self, content: str, category: PayloadCategory) -> List[Dict[str, Any]]:
        """Extract payloads from text response"""
        import re
        
        payloads = []
        
        # Define patterns for different payload types
        patterns = {
            PayloadCategory.XSS: [
                r'<script[^>]*>.*?</script>',
                r'<[^>]+on\w+[^>]*>',
                r'javascript:[^"\'\s]*',
                r'data:[^"\'\s]*'
            ],
            PayloadCategory.SQL_INJECTION: [
                r"'[^']*(?:UNION|SELECT|OR|AND)[^']*",
                r'--[^\n]*',
                r';[^;]*(?:DROP|INSERT|UPDATE|DELETE)[^;]*'
            ],
            PayloadCategory.SSRF: [
                r'https?://(?:127\.0\.0\.1|localhost|169\.254\.169\.254)[^\s]*',
                r'file://[^\s]*',
                r'gopher://[^\s]*',
                r'dict://[^\s]*'
            ],
            PayloadCategory.RCE: [
                r'[;&|`$()]\s*\w+',
                r'exec\([^)]*\)',
                r'eval\([^)]*\)',
                r'system\([^)]*\)'
            ]
        }
        
        if category in patterns:
            for pattern in patterns[category]:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    payloads.append({
                        "payload": match,
                        "technique": "pattern_extracted",
                        "explanation": f"Extracted {category.value} payload from text",
                        "confidence": 0.6
                    })
        
        return payloads[:10]  # Limit to 10 payloads
    
    async def analyze_vulnerability_with_qwen(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Use Qwen 2.5 for advanced vulnerability analysis"""
        
        analysis_prompt = f"""
VULNERABILITY ANALYSIS TASK:

You are a senior security consultant analyzing a vulnerability finding. Provide comprehensive analysis including business impact, technical details, and strategic recommendations.

FINDING DETAILS:
{json.dumps(finding, indent=2) if isinstance(finding, dict) else str(finding)}

ANALYSIS REQUIREMENTS:

1. TECHNICAL ASSESSMENT:
   - Root cause analysis
   - Attack vector evaluation
   - Exploitability assessment
   - Technical complexity rating

2. BUSINESS IMPACT:
   - Data confidentiality impact
   - System integrity impact
   - Service availability impact
   - Regulatory compliance implications

3. RISK SCORING:
   - CVSS v3.1 base score calculation
   - Environmental score adjustments
   - Temporal score considerations
   - Overall risk rating (Critical/High/Medium/Low)

4. EXPLOITATION SCENARIOS:
   - Step-by-step attack scenarios
   - Chaining with other vulnerabilities
   - Persistence and lateral movement potential
   - Data exfiltration possibilities

5. REMEDIATION STRATEGY:
   - Immediate containment measures
   - Short-term fixes (1-2 weeks)
   - Long-term architectural improvements
   - Validation and testing approach

6. STRATEGIC RECOMMENDATIONS:
   - Security control enhancements
   - Process improvements
   - Training and awareness needs
   - Monitoring and detection capabilities

OUTPUT FORMAT (JSON):
{{
  "vulnerability_id": "unique identifier",
  "severity": "Critical/High/Medium/Low",
  "cvss_score": 0.0-10.0,
  "cvss_vector": "CVSS:3.1/...",
  "technical_analysis": {{
    "root_cause": "detailed explanation",
    "attack_complexity": "Low/Medium/High",
    "privileges_required": "None/Low/High",
    "user_interaction": "None/Required"
  }},
  "business_impact": {{
    "confidentiality": "None/Low/High",
    "integrity": "None/Low/High", 
    "availability": "None/Low/High",
    "financial_impact": "estimated cost"
  }},
  "exploitation": {{
    "scenarios": ["scenario 1", "scenario 2"],
    "prerequisites": ["requirement 1", "requirement 2"],
    "chaining_potential": "description"
  }},
  "remediation": {{
    "immediate": ["action 1", "action 2"],
    "short_term": ["fix 1", "fix 2"],
    "long_term": ["improvement 1", "improvement 2"],
    "validation": "testing approach"
  }},
  "recommendations": {{
    "controls": ["control 1", "control 2"],
    "processes": ["process 1", "process 2"],
    "monitoring": ["detection 1", "detection 2"]
  }}
}}

Provide thorough, actionable analysis that demonstrates deep security expertise and practical business understanding.
"""
        
        request = LLMRequest(
            task_type=TaskType.VULNERABILITY_ANALYSIS,
            prompt=analysis_prompt,
            max_tokens=3500,
            temperature=0.3,  # Lower temperature for analysis
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            # Parse structured analysis
            import json
            content = response.content.strip()
            
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            
            if content.startswith('{'):
                return json.loads(content)
            else:
                return {
                    "analysis": content,
                    "model_used": response.model_used,
                    "confidence": response.confidence_score
                }
                
        except Exception as e:
            logger.error(f"Qwen vulnerability analysis failed: {e}")
            return {"error": str(e)}
    
    async def generate_campaign_strategy(self, targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive testing strategy using Qwen's planning capabilities"""
        
        strategy_prompt = f"""
CYBERSECURITY CAMPAIGN STRATEGY DEVELOPMENT:

You are a lead penetration tester designing a comprehensive security assessment strategy. Create a detailed, phased approach for testing multiple targets while maintaining ethical standards and operational efficiency.

TARGET PORTFOLIO:
{json.dumps(targets, indent=2)}

STRATEGY REQUIREMENTS:

1. RECONNAISSANCE PHASE:
   - OSINT gathering methodologies
   - Target prioritization matrix
   - Technology stack analysis
   - Attack surface mapping

2. VULNERABILITY DISCOVERY:
   - Automated scanning strategy
   - Manual testing priorities
   - Custom payload development
   - Zero-day research opportunities

3. EXPLOITATION PLANNING:
   - Attack vector prioritization
   - Chaining vulnerabilities
   - Privilege escalation paths
   - Persistence mechanisms

4. POST-EXPLOITATION:
   - Data collection strategies
   - Lateral movement planning
   - Evidence preservation
   - Clean exit procedures

5. REPORTING & REMEDIATION:
   - Finding classification
   - Business impact assessment
   - Remediation roadmap
   - Executive presentation

6. OPERATIONAL CONSIDERATIONS:
   - Timeline and resource allocation
   - Risk management and safety controls
   - Legal and compliance requirements
   - Communication protocols

OUTPUT FORMAT (JSON):
{{
  "campaign_overview": {{
    "duration": "estimated timeframe",
    "complexity": "Low/Medium/High",
    "resource_requirements": "team size and skills",
    "success_probability": 0.0-1.0
  }},
  "phases": [
    {{
      "phase_name": "reconnaissance",
      "duration": "timeframe",
      "objectives": ["objective 1", "objective 2"],
      "methodologies": ["method 1", "method 2"],
      "deliverables": ["deliverable 1", "deliverable 2"],
      "success_criteria": "how to measure success"
    }}
  ],
  "target_prioritization": [
    {{
      "target": "target identifier",
      "priority": 1-10,
      "rationale": "why this priority",
      "estimated_effort": "hours/days",
      "success_probability": 0.0-1.0
    }}
  ],
  "risk_assessment": {{
    "operational_risks": ["risk 1", "risk 2"],
    "mitigation_strategies": ["strategy 1", "strategy 2"],
    "contingency_plans": ["plan A", "plan B"]
  }},
  "resource_allocation": {{
    "personnel": "team composition",
    "tools": ["tool 1", "tool 2"],
    "budget": "estimated cost",
    "timeline": "milestone schedule"
  }},
  "success_metrics": {{
    "quantitative": ["metric 1", "metric 2"],
    "qualitative": ["indicator 1", "indicator 2"],
    "reporting": "how to measure and report progress"
  }}
}}

Design a strategy that maximizes security findings while minimizing operational risk and maintaining ethical standards throughout the engagement.
"""
        
        request = LLMRequest(
            task_type=TaskType.EXPLOITATION_STRATEGY,
            prompt=strategy_prompt,
            max_tokens=4000,
            temperature=0.5,
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            import json
            content = response.content.strip()
            
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            
            if content.startswith('{'):
                strategy = json.loads(content)
                strategy["generated_by"] = "qwen-235b"
                strategy["confidence"] = response.confidence_score
                return strategy
            else:
                return {
                    "strategy": content,
                    "model_used": response.model_used,
                    "confidence": response.confidence_score
                }
                
        except Exception as e:
            logger.error(f"Qwen strategy generation failed: {e}")
            return {"error": str(e)}

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get information about Qwen 2.5 235B capabilities"""
        return {
            "model_name": "Qwen 2.5 235B",
            "context_window": 32768,
            "max_tokens": 4096,
            "strengths": [
                "Advanced reasoning and analysis",
                "Large context window for complex scenarios",
                "Strong coding and technical knowledge",
                "Excellent structured output generation",
                "Superior vulnerability analysis capabilities"
            ],
            "optimal_use_cases": [
                "Complex payload generation",
                "Multi-step exploitation strategies", 
                "Detailed vulnerability analysis",
                "Campaign planning and orchestration",
                "Technical report generation"
            ],
            "limitations": [
                "Rate limits on free tier",
                "Requires careful prompt engineering",
                "May be overkill for simple tasks"
            ]
        }