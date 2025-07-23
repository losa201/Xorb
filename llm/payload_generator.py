#!/usr/bin/env python3
"""
LLM-Powered Payload Generator for XORB Supreme
Generates context-aware security payloads using AI
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from .intelligent_client import IntelligentLLMClient, LLMRequest, TaskType, LLMResponse

logger = logging.getLogger(__name__)

class PayloadCategory(Enum):
    XSS = "xss"
    SQL_INJECTION = "sql_injection"
    SSRF = "ssrf"
    RCE = "remote_code_execution"
    LFI = "local_file_inclusion"
    XXE = "xml_external_entity"
    CSRF = "csrf"
    DIRECTORY_TRAVERSAL = "directory_traversal"
    COMMAND_INJECTION = "command_injection"
    AUTHENTICATION_BYPASS = "auth_bypass"

class PayloadComplexity(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CUSTOM = "custom"

@dataclass
class TargetContext:
    """Context information about the target system"""
    url: Optional[str] = None
    technology_stack: List[str] = None
    operating_system: Optional[str] = None
    web_server: Optional[str] = None
    application_type: Optional[str] = None
    input_fields: List[str] = None
    known_endpoints: List[str] = None
    security_headers: Dict[str, str] = None
    cookies: Dict[str, str] = None
    parameters: List[str] = None

class GeneratedPayload(BaseModel):
    category: PayloadCategory
    complexity: PayloadComplexity
    payload: str
    description: str
    target_parameter: Optional[str] = None
    expected_result: str
    detection_difficulty: int = Field(ge=1, le=5)  # 1=easy to detect, 5=hard
    success_probability: float = Field(ge=0.0, le=1.0)
    remediation: str
    references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PayloadGenerator:
    def __init__(self, llm_client: IntelligentLLMClient):
        self.llm_client = llm_client
        self.payload_templates = self._load_payload_templates()
        
    def _load_payload_templates(self) -> Dict[PayloadCategory, Dict[str, Any]]:
        """Load base payload templates for each category"""
        return {
            PayloadCategory.XSS: {
                "basic_templates": [
                    "<script>alert('XSS')</script>",
                    "<img src=x onerror=alert(1)>",
                    "<svg onload=alert(1)>"
                ],
                "context_patterns": ["input", "textarea", "url_param", "header"],
                "encoding_variants": ["url", "html", "unicode", "base64"]
            },
            PayloadCategory.SQL_INJECTION: {
                "basic_templates": [
                    "' OR 1=1--",
                    "' UNION SELECT 1,2,3--",
                    "'; DROP TABLE users--"
                ],
                "context_patterns": ["login", "search", "id_param", "order_by"],
                "database_types": ["mysql", "postgresql", "mssql", "sqlite", "oracle"]
            },
            PayloadCategory.SSRF: {
                "basic_templates": [
                    "http://127.0.0.1:80",
                    "http://localhost:22",
                    "http://metadata.google.internal/"
                ],
                "protocols": ["http", "https", "ftp", "file", "gopher"],
                "targets": ["localhost", "169.254.169.254", "metadata", "internal"]
            },
            PayloadCategory.RCE: {
                "basic_templates": [
                    "; whoami",
                    "$(whoami)",
                    "`id`",
                    "&& ls -la"
                ],
                "os_variants": ["linux", "windows", "unix"],
                "injection_points": ["parameter", "header", "file_upload", "api"]
            }
        }
    
    async def generate_contextual_payloads(
        self, 
        category: PayloadCategory,
        target_context: TargetContext,
        complexity: PayloadComplexity = PayloadComplexity.INTERMEDIATE,
        count: int = 5
    ) -> List[GeneratedPayload]:
        """Generate context-aware payloads using LLM"""
        
        # Build detailed prompt for LLM
        prompt = self._build_payload_prompt(category, target_context, complexity, count)
        
        request = LLMRequest(
            task_type=TaskType.PAYLOAD_GENERATION,
            prompt=prompt,
            target_info=self._serialize_target_context(target_context),
            max_tokens=2000,
            temperature=0.8,  # Higher creativity for payload generation
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            payloads = self._parse_llm_payloads(response, category, complexity)
            
            # Enhance with static knowledge if needed
            if len(payloads) < count:
                static_payloads = self._generate_static_fallback(category, target_context, count - len(payloads))
                payloads.extend(static_payloads)
            
            return payloads[:count]
            
        except Exception as e:
            logger.error(f"LLM payload generation failed: {e}")
            # Fallback to template-based generation
            return self._generate_static_fallback(category, target_context, count)
    
    def _build_payload_prompt(
        self,
        category: PayloadCategory,
        target_context: TargetContext,
        complexity: PayloadComplexity,
        count: int
    ) -> str:
        """Build detailed prompt for LLM payload generation"""
        
        # Base prompt structure
        prompt_parts = [
            f"Generate {count} {complexity.value} {category.value.replace('_', ' ')} payloads for security testing.",
            "",
            "REQUIREMENTS:",
            "- Payloads must be for AUTHORIZED testing only",
            "- Include detailed explanations of how each payload works",
            "- Provide detection difficulty assessment (1-5 scale)",
            "- Include success probability estimation",
            "- Suggest remediation steps",
            "",
            "TARGET CONTEXT:"
        ]
        
        # Add target context details
        if target_context.url:
            prompt_parts.append(f"- URL: {target_context.url}")
        if target_context.technology_stack:
            prompt_parts.append(f"- Technology Stack: {', '.join(target_context.technology_stack)}")
        if target_context.operating_system:
            prompt_parts.append(f"- Operating System: {target_context.operating_system}")
        if target_context.web_server:
            prompt_parts.append(f"- Web Server: {target_context.web_server}")
        if target_context.input_fields:
            prompt_parts.append(f"- Input Fields: {', '.join(target_context.input_fields)}")
        if target_context.parameters:
            prompt_parts.append(f"- Parameters: {', '.join(target_context.parameters)}")
        
        # Add category-specific guidance
        category_guidance = {
            PayloadCategory.XSS: "Focus on bypassing common filters, CSP, and WAF detection.",
            PayloadCategory.SQL_INJECTION: "Consider database type, query context, and blind vs error-based techniques.",
            PayloadCategory.SSRF: "Target cloud metadata services, internal services, and protocol smuggling.",
            PayloadCategory.RCE: "Consider OS command injection, code execution, and file upload vectors."
        }
        
        if category in category_guidance:
            prompt_parts.extend(["", "SPECIFIC GUIDANCE:", f"- {category_guidance[category]}"])
        
        # Request structured output
        prompt_parts.extend([
            "",
            "OUTPUT FORMAT:",
            "Provide a JSON array with objects containing:",
            "- payload: the actual payload string",
            "- description: detailed explanation",
            "- target_parameter: which parameter/field to use",
            "- expected_result: what should happen when executed",
            "- detection_difficulty: 1-5 scale (1=easy to detect, 5=hard)",
            "- success_probability: 0.0-1.0 (likelihood of success)",
            "- remediation: how to fix/prevent this vulnerability",
            "- references: relevant CVE, CWE, or documentation links"
        ])
        
        return "\n".join(prompt_parts)
    
    def _serialize_target_context(self, context: TargetContext) -> Dict[str, Any]:
        """Convert target context to serializable format"""
        return {
            "url": context.url,
            "technology_stack": context.technology_stack or [],
            "operating_system": context.operating_system,
            "web_server": context.web_server,
            "application_type": context.application_type,
            "input_fields": context.input_fields or [],
            "known_endpoints": context.known_endpoints or [],
            "security_headers": context.security_headers or {},
            "parameters": context.parameters or []
        }
    
    def _parse_llm_payloads(
        self,
        response: LLMResponse,
        category: PayloadCategory,
        complexity: PayloadComplexity
    ) -> List[GeneratedPayload]:
        """Parse LLM response into structured payloads"""
        payloads = []
        
        try:
            # Try to parse as JSON first
            if response.content.strip().startswith('[') or response.content.strip().startswith('{'):
                data = json.loads(response.content)
                if isinstance(data, list):
                    payload_data = data
                else:
                    payload_data = [data]
            else:
                # Parse text-based response
                payload_data = self._parse_text_payloads(response.content)
            
            for item in payload_data:
                if isinstance(item, dict) and "payload" in item:
                    payload = GeneratedPayload(
                        category=category,
                        complexity=complexity,
                        payload=item.get("payload", ""),
                        description=item.get("description", "LLM-generated payload"),
                        target_parameter=item.get("target_parameter"),
                        expected_result=item.get("expected_result", "Unknown"),
                        detection_difficulty=int(item.get("detection_difficulty", 3)),
                        success_probability=float(item.get("success_probability", 0.5)),
                        remediation=item.get("remediation", "Implement input validation"),
                        references=item.get("references", []),
                        metadata={
                            "llm_model": response.model_used,
                            "confidence": response.confidence_score,
                            "generated_at": response.generated_at.isoformat()
                        }
                    )
                    payloads.append(payload)
        
        except Exception as e:
            logger.error(f"Failed to parse LLM payloads: {e}")
            # Fallback: extract payloads from text
            payloads = self._extract_payloads_from_text(response.content, category, complexity)
        
        return payloads
    
    def _parse_text_payloads(self, content: str) -> List[Dict[str, Any]]:
        """Parse payloads from text-based LLM response"""
        payloads = []
        
        # Look for payload patterns in text
        payload_blocks = re.findall(r'(?:payload|example):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        description_blocks = re.findall(r'(?:description|explanation):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        
        for i, payload in enumerate(payload_blocks):
            description = description_blocks[i] if i < len(description_blocks) else "Generated payload"
            payloads.append({
                "payload": payload.strip(),
                "description": description.strip(),
                "detection_difficulty": 3,
                "success_probability": 0.6
            })
        
        return payloads
    
    def _extract_payloads_from_text(
        self,
        content: str,
        category: PayloadCategory,
        complexity: PayloadComplexity
    ) -> List[GeneratedPayload]:
        """Extract payloads from unstructured text response"""
        payloads = []
        
        # Use regex patterns to find potential payloads
        patterns = {
            PayloadCategory.XSS: [
                r'<script[^>]*>.*?</script>',
                r'<[^>]+on\w+=[^>]*>',
                r'javascript:[^"\']*'
            ],
            PayloadCategory.SQL_INJECTION: [
                r"'[^']*(?:OR|UNION|SELECT)[^']*",
                r'--[^\n]*',
                r';[^;]*(?:DROP|INSERT|UPDATE)[^;]*'
            ],
            PayloadCategory.SSRF: [
                r'https?://(?:127\.0\.0\.1|localhost|169\.254\.169\.254)[^\s]*',
                r'file://[^\s]*',
                r'gopher://[^\s]*'
            ]
        }
        
        if category in patterns:
            for pattern in patterns[category]:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    payload = GeneratedPayload(
                        category=category,
                        complexity=complexity,
                        payload=match,
                        description=f"Extracted {category.value} payload",
                        expected_result="Vulnerability exploitation",
                        detection_difficulty=3,
                        success_probability=0.5,
                        remediation="Implement proper input validation and output encoding"
                    )
                    payloads.append(payload)
        
        return payloads
    
    def _generate_static_fallback(
        self,
        category: PayloadCategory,
        target_context: TargetContext,
        count: int
    ) -> List[GeneratedPayload]:
        """Generate fallback payloads using static templates"""
        payloads = []
        
        if category not in self.payload_templates:
            return payloads
        
        templates = self.payload_templates[category]
        basic_payloads = templates.get("basic_templates", [])
        
        for i, template in enumerate(basic_payloads[:count]):
            payload = GeneratedPayload(
                category=category,
                complexity=PayloadComplexity.BASIC,
                payload=template,
                description=f"Static {category.value} payload template",
                expected_result="Basic vulnerability test",
                detection_difficulty=2,
                success_probability=0.4,
                remediation="Implement input validation and output encoding",
                metadata={"source": "static_template", "template_index": i}
            )
            payloads.append(payload)
        
        return payloads
    
    async def generate_chained_payloads(
        self,
        categories: List[PayloadCategory],
        target_context: TargetContext
    ) -> List[GeneratedPayload]:
        """Generate chained exploitation payloads"""
        
        prompt = f"""
        Generate a chained exploitation sequence using these vulnerability types:
        {', '.join([cat.value for cat in categories])}
        
        Create a realistic attack chain that:
        1. Starts with initial foothold
        2. Escalates privileges or access
        3. Achieves final objective
        
        Target context: {json.dumps(self._serialize_target_context(target_context), indent=2)}
        
        Provide step-by-step payloads with:
        - Order of execution
        - Dependencies between steps
        - Expected intermediate results
        - Final objective achievement
        """
        
        request = LLMRequest(
            task_type=TaskType.EXPLOITATION_STRATEGY,
            prompt=prompt,
            target_info=self._serialize_target_context(target_context),
            max_tokens=3000,
            temperature=0.7
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            return self._parse_chained_payloads(response, categories)
        except Exception as e:
            logger.error(f"Chained payload generation failed: {e}")
            return []
    
    def _parse_chained_payloads(
        self,
        response: LLMResponse,
        categories: List[PayloadCategory]
    ) -> List[GeneratedPayload]:
        """Parse chained payload response"""
        # Implementation would parse the LLM response for chained payloads
        # This is a simplified version
        return []
    
    async def enhance_payload_with_context(
        self,
        base_payload: GeneratedPayload,
        additional_context: Dict[str, Any]
    ) -> GeneratedPayload:
        """Enhance existing payload with additional context"""
        
        prompt = f"""
        Enhance this security payload based on additional context:
        
        Original Payload: {base_payload.payload}
        Description: {base_payload.description}
        Category: {base_payload.category.value}
        
        Additional Context:
        {json.dumps(additional_context, indent=2)}
        
        Provide an enhanced version that:
        - Adapts to the new context
        - Increases success probability
        - Reduces detection likelihood
        - Maintains ethical testing principles
        """
        
        request = LLMRequest(
            task_type=TaskType.PAYLOAD_GENERATION,
            prompt=prompt,
            target_info=additional_context,
            max_tokens=1500,
            temperature=0.6
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            enhanced_payloads = self._parse_llm_payloads(response, base_payload.category, base_payload.complexity)
            
            if enhanced_payloads:
                enhanced = enhanced_payloads[0]
                enhanced.metadata.update({
                    "enhanced_from": base_payload.payload,
                    "enhancement_context": additional_context
                })
                return enhanced
            
        except Exception as e:
            logger.error(f"Payload enhancement failed: {e}")
        
        return base_payload
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get payload generation statistics"""
        return {
            "supported_categories": len(PayloadCategory),
            "template_categories": len(self.payload_templates),
            "llm_client_stats": self.llm_client.get_usage_stats()
        }