#!/usr/bin/env python3
"""
Enhanced Multi-Provider LLM Client for XORB Supreme
Supports paid APIs with intelligent routing, cost controls, and fallback systems
"""

import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from pydantic import BaseModel, Field, validator
import hashlib

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL_FREE = "local_free"

class TaskComplexity(Enum):
    SIMPLE = "simple"      # Basic payloads, simple analysis
    MODERATE = "moderate"  # Context-aware payloads, detailed analysis
    COMPLEX = "complex"    # Creative exploitation, professional reports
    EXPERT = "expert"      # Chained attacks, business logic analysis

@dataclass
class ModelCapabilities:
    name: str
    provider: LLMProvider
    cost_per_1k_tokens: float
    max_tokens: int
    context_window: int
    best_for_tasks: List[str]
    supports_structured_output: bool = False
    supports_function_calling: bool = False
    creativity_score: float = 0.7  # 0-1, higher = more creative
    reliability_score: float = 0.8  # 0-1, higher = more reliable
    rate_limit_rpm: int = 60

class EnhancedLLMRequest(BaseModel):
    task_type: str
    prompt: str
    target_info: Optional[Dict[str, Any]] = None
    complexity: TaskComplexity = TaskComplexity.MODERATE
    max_tokens: int = 2000
    temperature: float = 0.7
    structured_output: bool = False
    use_paid_api: bool = True
    budget_limit_usd: float = 0.50  # Max spend per request
    priority: int = 1  # 1-5, higher = more important
    creativity_required: bool = False
    chain_with_previous: Optional[str] = None  # Chain with previous request ID

class LLMResponse(BaseModel):
    content: str
    model_used: str
    provider: LLMProvider
    tokens_used: int
    cost_usd: float
    confidence_score: float
    generated_at: datetime
    request_id: str
    structured_data: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    was_fallback: bool = False

class EnhancedMultiProviderClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cost tracking
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.budget_limits = {
            "daily": config.get("daily_budget_limit", 10.0),
            "monthly": config.get("monthly_budget_limit", 100.0),
            "per_request": config.get("per_request_limit", 2.0)
        }
        
        # Rate limiting and caching
        self.request_cache: Dict[str, LLMResponse] = {}
        self.request_counts: Dict[str, List[datetime]] = {}
        self.request_history: List[Dict[str, Any]] = []
        
        # Model configurations
        self.models = self._initialize_enhanced_models()
        
        # Prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
    def _initialize_enhanced_models(self) -> Dict[str, ModelCapabilities]:
        """Initialize enhanced model configurations with capabilities"""
        return {
            # Premium OpenAI Models
            "gpt-4o": ModelCapabilities(
                name="gpt-4o",
                provider=LLMProvider.OPENAI,
                cost_per_1k_tokens=0.005,
                max_tokens=4096,
                context_window=128000,
                best_for_tasks=["payload_generation", "report_writing", "exploitation_chains"],
                supports_structured_output=True,
                supports_function_calling=True,
                creativity_score=0.9,
                reliability_score=0.95,
                rate_limit_rpm=500
            ),
            
            "gpt-4-turbo": ModelCapabilities(
                name="gpt-4-turbo",
                provider=LLMProvider.OPENAI,
                cost_per_1k_tokens=0.01,
                max_tokens=4096,
                context_window=128000,
                best_for_tasks=["complex_analysis", "business_logic", "professional_reports"],
                supports_structured_output=True,
                creativity_score=0.85,
                reliability_score=0.9,
                rate_limit_rpm=300
            ),
            
            # Claude Models via OpenRouter
            "claude-3-opus": ModelCapabilities(
                name="anthropic/claude-3-opus",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.015,
                max_tokens=4096,
                context_window=200000,
                best_for_tasks=["professional_reports", "ethical_analysis", "complex_reasoning"],
                supports_structured_output=True,
                creativity_score=0.8,
                reliability_score=0.95,
                rate_limit_rpm=100
            ),
            
            "claude-3-sonnet": ModelCapabilities(
                name="anthropic/claude-3-sonnet",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.003,
                max_tokens=4096,
                context_window=200000,
                best_for_tasks=["payload_generation", "vulnerability_analysis"],
                supports_structured_output=True,
                creativity_score=0.75,
                reliability_score=0.9,
                rate_limit_rpm=200
            ),
            
            # Gemini Models
            "gemini-1.5-pro": ModelCapabilities(
                name="google/gemini-pro-1.5",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0025,
                max_tokens=8192,
                context_window=1000000,  # Massive context
                best_for_tasks=["large_context_analysis", "document_processing"],
                supports_structured_output=True,
                creativity_score=0.7,
                reliability_score=0.85,
                rate_limit_rpm=100
            ),
            
            # Free tier fallbacks (existing)
            "qwen-free": ModelCapabilities(
                name="qwen/qwen3-235b-a22b-07-25:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,
                max_tokens=4096,
                context_window=32768,
                best_for_tasks=["basic_payloads", "simple_analysis"],
                creativity_score=0.6,
                reliability_score=0.7,
                rate_limit_rpm=20
            )
        }
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load optimized prompt templates for different tasks"""
        return {
            "creative_payload_generation": """
You are an elite red team operator specializing in creative, advanced payload development for authorized security testing.

TARGET CONTEXT:
{target_context}

TASK: Generate {count} highly creative {vulnerability_type} payloads that:
1. Bypass modern security controls (WAF, CSP, input validation)
2. Demonstrate advanced techniques beyond basic payloads  
3. Are tailored to the specific technology stack
4. Include detailed explanations of bypass techniques
5. Consider real-world deployment scenarios

CREATIVITY REQUIREMENTS:
- Use novel encoding/obfuscation techniques
- Chain multiple attack vectors when possible
- Leverage edge cases in parsers/interpreters
- Consider timing-based and logic-based approaches
- Include polymorphic variations

OUTPUT FORMAT (JSON):
{{
  "payloads": [
    {{
      "payload": "actual payload string",
      "technique": "bypass method used",
      "creativity_score": 1-10,
      "complexity": "low/medium/high",
      "explanation": "detailed technical explanation",
      "target_context": "where/how to use",
      "evasion_methods": ["method1", "method2"],
      "chaining_potential": "how to chain with other attacks",
      "detection_difficulty": 1-10,
      "impact_assessment": "potential business impact"
    }}
  ]
}}

Focus on advanced, creative techniques that demonstrate deep understanding of security mechanisms.
""",

            "exploitation_chain_analysis": """
You are a senior penetration tester specializing in complex attack chain development.

VULNERABILITY SET:
{vulnerabilities}

TARGET ENVIRONMENT:
{target_info}

TASK: Design sophisticated exploitation chains that:
1. Combine multiple vulnerabilities for maximum impact
2. Escalate privileges and maintain persistence
3. Demonstrate realistic attack scenarios
4. Consider detection evasion throughout the chain
5. Provide step-by-step execution guidance

CHAIN REQUIREMENTS:
- Logical progression from initial access to final objective
- Multiple fallback paths if primary route fails
- Stealth considerations at each stage
- Data exfiltration and cleanup procedures
- Risk assessment for each step

OUTPUT FORMAT (JSON):
{{
  "exploitation_chains": [
    {{
      "chain_name": "descriptive name",
      "initial_vector": "entry point vulnerability",
      "steps": [
        {{
          "step": 1,
          "technique": "what to do",
          "payload": "specific payload/command",
          "expected_result": "what should happen",
          "fallback": "if this fails, do X",
          "stealth_level": 1-10,
          "risk_level": "low/medium/high"
        }}
      ],
      "final_objective": "ultimate goal achieved",
      "cleanup_required": "steps to clean up traces",
      "detection_points": ["where detection might occur"],
      "mitigation_bypasses": ["how to avoid common defenses"],
      "business_impact": "real-world consequences"
    }}
  ]
}}

Design chains that demonstrate sophisticated, multi-stage attack scenarios.
""",

            "professional_report_generation": """
You are a senior cybersecurity consultant writing executive-level security assessment reports.

ASSESSMENT DATA:
{assessment_data}

TASK: Generate a comprehensive, professional security report including:
1. Executive summary with business impact analysis
2. Technical findings with CVSS scoring
3. Detailed proof-of-concept documentation
4. Strategic remediation roadmap
5. Risk prioritization matrix

REPORT REQUIREMENTS:
- Executive-friendly language with technical depth
- Clear risk communication with business context
- Actionable remediation recommendations
- Compliance considerations (GDPR, SOX, PCI-DSS)
- Cost-benefit analysis for fixes

OUTPUT FORMAT (Structured Markdown):
# Security Assessment Report

## Executive Summary
[High-level findings and business impact]

## Risk Assessment Matrix
[Prioritized vulnerabilities with CVSS scores]

## Technical Findings
[Detailed vulnerability analysis with PoCs]

## Remediation Roadmap
[Phased approach to security improvements]

## Strategic Recommendations  
[Long-term security architecture guidance]

## Appendices
[Technical details, tools used, methodology]

Write as a seasoned consultant addressing both technical and business stakeholders.
""",

            "business_logic_analysis": """
You are a specialized security researcher focusing on business logic vulnerabilities and fraud scenarios.

TARGET APPLICATION:
{application_context}

BUSINESS MODEL:
{business_model}

TASK: Identify sophisticated business logic flaws that:
1. Exploit application workflow weaknesses
2. Bypass business rule enforcement
3. Enable financial fraud or data manipulation
4. Demonstrate realistic abuse scenarios
5. Consider regulatory and compliance implications

ANALYSIS REQUIREMENTS:
- Map critical business workflows
- Identify trust boundaries and assumptions
- Model attacker economic incentives
- Consider insider threat scenarios
- Evaluate fraud detection bypass techniques

OUTPUT FORMAT (JSON):
{{
  "business_logic_flaws": [
    {{
      "flaw_name": "descriptive name",
      "business_process": "affected workflow",
      "vulnerability_description": "detailed explanation",
      "attack_scenario": "step-by-step abuse case",  
      "economic_impact": "financial consequences",
      "fraud_potential": "monetization methods",
      "detection_difficulty": 1-10,
      "regulatory_implications": ["compliance issues"],
      "proof_of_concept": "demonstration steps",
      "business_risk": "operational impact",
      "remediation_complexity": "effort to fix"
    }}
  ]
}}

Focus on high-impact business logic vulnerabilities specific to the target industry.
"""
        }
    
    async def start(self):
        """Initialize the enhanced client"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
                headers={"User-Agent": "XORB-Supreme-Enhanced/3.0"}
            )
            logger.info("Enhanced multi-provider LLM client started")
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def select_optimal_model(self, request: EnhancedLLMRequest) -> str:
        """Select the best model based on request requirements and budget"""
        
        # Check budget constraints
        if not self._check_budget_limits(request.budget_limit_usd):
            logger.warning("Budget limits exceeded, falling back to free tier")
            request.use_paid_api = False
        
        # Filter models based on requirements
        suitable_models = []
        
        for name, model in self.models.items():
            # Skip paid models if not requested or budget exceeded
            if not request.use_paid_api and model.cost_per_1k_tokens > 0:
                continue
            
            # Check task compatibility
            if any(task in model.best_for_tasks for task in [request.task_type]):
                suitable_models.append((name, model))
        
        if not suitable_models:
            # Fallback to any available model
            suitable_models = [(name, model) for name, model in self.models.items()
                             if model.cost_per_1k_tokens == 0 or request.use_paid_api]
        
        # Score and select best model
        best_model = None
        best_score = -1
        
        for name, model in suitable_models:
            if not self._check_rate_limit(name):
                continue
            
            # Calculate model score
            score = 0
            
            # Cost efficiency (lower cost = higher score)
            if model.cost_per_1k_tokens == 0:
                score += 2  # Free models get bonus
            else:
                score += (1.0 / model.cost_per_1k_tokens) * 0.1
            
            # Creativity bonus if required
            if request.creativity_required:
                score += model.creativity_score * 2
            
            # Reliability bonus
            score += model.reliability_score
            
            # Complexity matching
            complexity_bonus = {
                TaskComplexity.SIMPLE: 0.5,
                TaskComplexity.MODERATE: 1.0,
                TaskComplexity.COMPLEX: 1.5,
                TaskComplexity.EXPERT: 2.0
            }
            score += complexity_bonus.get(request.complexity, 1.0)
            
            # Structured output bonus
            if request.structured_output and model.supports_structured_output:
                score += 1.0
            
            if score > best_score:
                best_score = score
                best_model = name
        
        return best_model or "qwen-free"  # Ultimate fallback
    
    async def generate_enhanced_payload(self, request: EnhancedLLMRequest) -> LLMResponse:
        """Generate payloads using optimal model selection and enhanced prompts"""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.request_cache:
            cached = self.request_cache[cache_key]
            logger.info(f"Returning cached response for {request.task_type}")
            return cached
        
        # Select optimal model
        model_name = self.select_optimal_model(request)
        model_config = self.models[model_name]
        
        # Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(request)
        
        try:
            # Make API call
            if model_config.provider == LLMProvider.OPENAI:
                response = await self._call_openai(model_name, enhanced_prompt, request)
            elif model_config.provider == LLMProvider.OPENROUTER:
                response = await self._call_openrouter(model_name, enhanced_prompt, request)
            elif model_config.provider == LLMProvider.CLAUDE:
                response = await self._call_claude_direct(model_name, enhanced_prompt, request)
            elif model_config.provider == LLMProvider.GEMINI:
                response = await self._call_gemini_direct(model_name, enhanced_prompt, request)
            else:
                # Free tier fallback
                response = await self._call_free_tier(model_name, enhanced_prompt, request)
            
            # Add execution time
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Track costs and usage
            self._track_enhanced_usage(model_name, response)
            
            # Cache successful responses
            self.request_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced payload generation failed: {e}")
            
            # Try fallback to free tier
            if request.use_paid_api and model_config.cost_per_1k_tokens > 0:
                logger.info("Attempting free tier fallback")
                fallback_request = request.copy()
                fallback_request.use_paid_api = False
                return await self.generate_enhanced_payload(fallback_request)
            
            raise
    
    def _build_enhanced_prompt(self, request: EnhancedLLMRequest) -> str:
        """Build enhanced prompt using templates and context"""
        
        # Get base template
        template_key = self._map_task_to_template(request.task_type)
        base_template = self.prompt_templates.get(template_key, request.prompt)
        
        # Build context variables
        context_vars = {
            "target_context": json.dumps(request.target_info or {}, indent=2),
            "vulnerability_type": request.task_type,
            "count": 5,  # Default count
            "vulnerabilities": request.target_info or {},
            "target_info": request.target_info or {},
            "assessment_data": request.target_info or {},
            "application_context": request.target_info or {},
            "business_model": request.target_info or {}
        }
        
        # Format template with context
        try:
            formatted_prompt = base_template.format(**context_vars)
        except KeyError as e:
            logger.warning(f"Template formatting failed: {e}, using base prompt")
            formatted_prompt = request.prompt
        
        # Add ethical constraints
        ethical_prefix = """
IMPORTANT: This analysis is for AUTHORIZED SECURITY TESTING ONLY.
- Only test systems you own or have explicit written permission to test
- Follow responsible disclosure practices
- Include defensive recommendations with findings
- Focus on educational and defensive value
- Respect all legal and ethical boundaries

"""
        
        return ethical_prefix + formatted_prompt
    
    def _map_task_to_template(self, task_type: str) -> str:
        """Map task types to prompt templates"""
        mapping = {
            "payload_generation": "creative_payload_generation",
            "creative_payloads": "creative_payload_generation", 
            "exploitation_chains": "exploitation_chain_analysis",
            "chain_analysis": "exploitation_chain_analysis",
            "professional_report": "professional_report_generation",
            "report_generation": "professional_report_generation",
            "business_logic": "business_logic_analysis",
            "logic_flaws": "business_logic_analysis"
        }
        return mapping.get(task_type, "creative_payload_generation")
    
    def _check_budget_limits(self, request_limit: float) -> bool:
        """Check if request is within budget limits"""
        if self.daily_spend + request_limit > self.budget_limits["daily"]:
            return False
        if self.monthly_spend + request_limit > self.budget_limits["monthly"]:
            return False  
        if request_limit > self.budget_limits["per_request"]:
            return False
        return True
    
    def _check_rate_limit(self, model_name: str) -> bool:
        """Enhanced rate limit checking"""
        now = datetime.utcnow()
        model_config = self.models[model_name]
        
        if model_name not in self.request_counts:
            self.request_counts[model_name] = []
        
        # Remove old requests outside the window
        cutoff = now - timedelta(minutes=1)
        self.request_counts[model_name] = [
            req_time for req_time in self.request_counts[model_name]
            if req_time > cutoff
        ]
        
        return len(self.request_counts[model_name]) < model_config.rate_limit_rpm
    
    def _generate_cache_key(self, request: EnhancedLLMRequest) -> str:
        """Generate cache key for request deduplication"""
        cache_data = {
            "task_type": request.task_type,
            "prompt_hash": hashlib.md5(request.prompt.encode()).hexdigest(),
            "target_info": request.target_info,
            "complexity": request.complexity.value,
            "creativity_required": request.creativity_required
        }
        return hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _track_enhanced_usage(self, model_name: str, response: LLMResponse):
        """Enhanced usage tracking with budget monitoring"""
        now = datetime.utcnow()
        
        # Update counters
        if model_name not in self.request_counts:
            self.request_counts[model_name] = []
        self.request_counts[model_name].append(now)
        
        # Update spend tracking
        self.daily_spend += response.cost_usd
        self.monthly_spend += response.cost_usd
        
        # Add to detailed history
        self.request_history.append({
            "timestamp": now.isoformat(),
            "model": model_name,
            "provider": response.provider.value,
            "tokens": response.tokens_used,
            "cost": response.cost_usd,
            "task_type": "unknown",  # Would be passed from request
            "execution_time_ms": response.execution_time_ms,
            "was_fallback": response.was_fallback
        })
        
        # Cleanup old history
        if len(self.request_history) > 10000:
            self.request_history = self.request_history[-5000:]
    
    async def _call_openai(self, model: str, prompt: str, request: EnhancedLLMRequest) -> LLMResponse:
        """Call OpenAI API with enhanced features"""
        headers = {
            "Authorization": f"Bearer {self.config.get('openai_api_key')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert cybersecurity researcher specializing in ethical security testing and advanced payload development."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        if request.structured_output:
            payload["response_format"] = {"type": "json_object"}
        
        async with self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"OpenAI API error {resp.status}: {error_text}")
            
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            return LLMResponse(
                content=content,
                model_used=model,
                provider=LLMProvider.OPENAI,
                tokens_used=tokens_used,
                cost_usd=self._calculate_cost(model, tokens_used),
                confidence_score=0.85,
                generated_at=datetime.utcnow(),
                request_id=f"openai_{int(time.time())}"
            )
    
    async def _call_openrouter(self, model: str, prompt: str, request: EnhancedLLMRequest) -> LLMResponse:
        """Enhanced OpenRouter API call"""
        headers = {
            "Authorization": f"Bearer {self.config.get('openrouter_api_key')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://xorb-supreme.ai",
            "X-Title": "XORB Supreme Security Platform"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an elite cybersecurity expert specializing in advanced payload development and security research."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        if request.structured_output and self.models[model].supports_structured_output:
            payload["response_format"] = {"type": "json_object"}
        
        async with self.session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"OpenRouter API error {resp.status}: {error_text}")
            
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            return LLMResponse(
                content=content,
                model_used=model,
                provider=LLMProvider.OPENROUTER,
                tokens_used=tokens_used,
                cost_usd=self._calculate_cost(model, tokens_used),
                confidence_score=0.8,
                generated_at=datetime.utcnow(),
                request_id=f"openrouter_{int(time.time())}"
            )
    
    async def _call_claude_direct(self, model: str, prompt: str, request: EnhancedLLMRequest) -> LLMResponse:
        """Direct Claude API call (placeholder - implement based on Anthropic SDK)"""
        # This would use the official Anthropic SDK
        raise NotImplementedError("Direct Claude API integration pending")
    
    async def _call_gemini_direct(self, model: str, prompt: str, request: EnhancedLLMRequest) -> LLMResponse:
        """Direct Gemini API call (placeholder - implement based on Google AI SDK)"""
        # This would use the official Google AI SDK
        raise NotImplementedError("Direct Gemini API integration pending")
    
    async def _call_free_tier(self, model: str, prompt: str, request: EnhancedLLMRequest) -> LLMResponse:
        """Fallback to free tier models"""
        # Use existing free tier implementation
        return await self._call_openrouter(model, prompt, request)
    
    def _calculate_cost(self, model_name: str, tokens_used: int) -> float:
        """Calculate API cost"""
        model_config = self.models[model_name]
        return (tokens_used / 1000) * model_config.cost_per_1k_tokens
    
    def get_enhanced_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        return {
            "total_requests": len(self.request_history),
            "daily_spend": self.daily_spend,
            "monthly_spend": self.monthly_spend,
            "budget_limits": self.budget_limits,
            "cache_hit_rate": len(self.request_cache) / max(len(self.request_history), 1),
            "models_used": list(set(h["model"] for h in self.request_history)),
            "average_execution_time": sum(h.get("execution_time_ms", 0) for h in self.request_history) / max(len(self.request_history), 1),
            "fallback_rate": sum(1 for h in self.request_history if h.get("was_fallback", False)) / max(len(self.request_history), 1),
            "cost_by_provider": self._calculate_cost_by_provider()
        }
    
    def _calculate_cost_by_provider(self) -> Dict[str, float]:
        """Calculate costs by provider"""
        costs = {}
        for history in self.request_history:
            provider = history.get("provider", "unknown")
            cost = history.get("cost", 0)
            costs[provider] = costs.get(provider, 0) + cost
        return costs