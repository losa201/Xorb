#!/usr/bin/env python3
"""
Intelligent LLM Client for XORB Supreme
Multi-provider, cost-aware, task-optimized LLM integration
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TaskType(Enum):
    PAYLOAD_GENERATION = "payload_generation"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    EXPLOITATION_STRATEGY = "exploitation_strategy"
    REPORT_ENHANCEMENT = "report_enhancement"
    TACTIC_SUGGESTION = "tactic_suggestion"
    CODE_REVIEW = "code_review"

class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    CLAUDE = "claude"
    LOCAL = "local"

@dataclass
class ModelConfig:
    name: str
    provider: LLMProvider
    cost_per_1k_tokens: float
    max_tokens: int
    best_for: List[TaskType]
    rate_limit_rpm: int = 60
    context_window: int = 4096
    supports_structured: bool = False

class LLMRequest(BaseModel):
    task_type: TaskType
    prompt: str
    target_info: Optional[Dict[str, Any]] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    structured_output: bool = False
    fallback_allowed: bool = True
    priority: int = 1  # 1-5, higher = more important

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

class IntelligentLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting tracking
        self.request_counts: Dict[str, List[datetime]] = {}
        self.total_cost = 0.0
        self.request_history: List[Dict[str, Any]] = []
        
        # Model configurations
        self.models = self._initialize_models()
        
        # Request queue for batching
        self.request_queue: List[LLMRequest] = []
        self.batch_size = 5
        self.batch_timeout = 30  # seconds
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize available models with their configurations"""
        return {
            # OpenRouter models - Qwen 2.5 235B (Primary)
            "qwen/qwen3-235b-a22b-07-25:free": ModelConfig(
                name="qwen/qwen3-235b-a22b-07-25:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free tier
                max_tokens=4096,
                best_for=[TaskType.PAYLOAD_GENERATION, TaskType.VULNERABILITY_ANALYSIS, TaskType.EXPLOITATION_STRATEGY],
                rate_limit_rpm=20,
                context_window=32768,
                supports_structured=True
            ),
            # Backup free models
            "moonshotai/kimi-k2:free": ModelConfig(
                name="moonshotai/kimi-k2:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free tier
                max_tokens=2048,
                best_for=[TaskType.PAYLOAD_GENERATION, TaskType.TACTIC_SUGGESTION],
                rate_limit_rpm=10,
                context_window=8192
            ),
            "google/gemini-flash-1.5": ModelConfig(
                name="google/gemini-flash-1.5", 
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0005,
                max_tokens=4096,
                best_for=[TaskType.VULNERABILITY_ANALYSIS, TaskType.CODE_REVIEW],
                rate_limit_rpm=30,
                context_window=32000,
                supports_structured=True
            ),
            "anthropic/claude-3-haiku": ModelConfig(
                name="anthropic/claude-3-haiku",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.002,
                max_tokens=4096,
                best_for=[TaskType.REPORT_ENHANCEMENT, TaskType.EXPLOITATION_STRATEGY],
                rate_limit_rpm=50,
                context_window=200000,
                supports_structured=True
            ),
            # Gemini direct API
            "gemini-1.5-flash": ModelConfig(
                name="gemini-1.5-flash",
                provider=LLMProvider.GEMINI,
                cost_per_1k_tokens=0.0005,
                max_tokens=8192,
                best_for=[TaskType.PAYLOAD_GENERATION, TaskType.VULNERABILITY_ANALYSIS],
                rate_limit_rpm=60,
                context_window=128000
            )
        }
    
    async def start(self):
        """Initialize the HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={"User-Agent": "XORB-Supreme/2.0"}
            )
            logger.info("LLM client session started")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def select_optimal_model(self, request: LLMRequest) -> str:
        """Select the best model for a given request"""
        # Filter models suitable for the task
        suitable_models = [
            (name, config) for name, config in self.models.items()
            if request.task_type in config.best_for
        ]
        
        if not suitable_models:
            # Fallback to general-purpose model
            suitable_models = list(self.models.items())
        
        # Score models based on cost, availability, and capability
        best_model = None
        best_score = -1
        
        for name, config in suitable_models:
            if not self._check_rate_limit(name):
                continue
                
            # Calculate score (lower cost = higher score)
            cost_score = 1.0 / (config.cost_per_1k_tokens + 0.001)
            
            # Prioritize structured output if needed
            structured_bonus = 0.5 if request.structured_output and config.supports_structured else 0
            
            # Context window bonus
            context_bonus = min(config.context_window / 10000, 1.0)
            
            total_score = cost_score + structured_bonus + context_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_model = name
        
        return best_model or "qwen/qwen3-235b-a22b-07-25:free"  # Ultimate fallback to Qwen 2.5
    
    async def generate_payload(self, request: LLMRequest) -> LLMResponse:
        """Generate security payload using optimal LLM"""
        if not self.session:
            await self.start()
        
        model_name = self.select_optimal_model(request)
        model_config = self.models[model_name]
        
        try:
            # Build prompt with security context
            enhanced_prompt = self._build_security_prompt(request)
            
            # Make API call based on provider
            if model_config.provider == LLMProvider.OPENROUTER:
                response = await self._call_openrouter(model_name, enhanced_prompt, request)
            elif model_config.provider == LLMProvider.GEMINI:
                response = await self._call_gemini(model_name, enhanced_prompt, request)
            else:
                raise Exception(f"Provider {model_config.provider} not implemented")
            
            # Track usage
            self._track_usage(model_name, response)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if request.fallback_allowed:
                return await self._fallback_generation(request)
            raise
    
    def _build_security_prompt(self, request: LLMRequest) -> str:
        """Build enhanced prompt with security context"""
        base_prompt = request.prompt
        
        # Add ethical constraints
        ethical_prefix = """
IMPORTANT: Generate payloads for AUTHORIZED SECURITY TESTING ONLY.
- Only suggest techniques for systems you own or have explicit permission to test
- Include defensive recommendations
- Focus on educational and defensive value
- Respect responsible disclosure practices

"""
        
        # Add target context if available
        context = ""
        if request.target_info:
            context = f"\nTarget Context: {json.dumps(request.target_info, indent=2)}\n"
        
        # Add task-specific guidance
        task_guidance = {
            TaskType.PAYLOAD_GENERATION: "Generate practical, well-commented payloads with explanation of how they work.",
            TaskType.VULNERABILITY_ANALYSIS: "Provide detailed technical analysis with CVSS scoring and remediation steps.",
            TaskType.EXPLOITATION_STRATEGY: "Outline step-by-step approach with detection evasion and persistence techniques.",
            TaskType.REPORT_ENHANCEMENT: "Enhance technical details while maintaining professional, clear communication.",
            TaskType.TACTIC_SUGGESTION: "Suggest creative but ethical testing approaches with success probability.",
            TaskType.CODE_REVIEW: "Identify security vulnerabilities with severity assessment and fix recommendations."
        }
        
        guidance = task_guidance.get(request.task_type, "Provide comprehensive security analysis.")
        
        return f"{ethical_prefix}\nTask: {guidance}\n{context}\nRequest: {base_prompt}"
    
    async def _call_openrouter(self, model: str, prompt: str, request: LLMRequest) -> LLMResponse:
        """Call OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.config.get('openrouter_api_key')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a cybersecurity expert specializing in ethical security testing and vulnerability research."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        if request.structured_output and self.models[model].supports_structured:
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
                confidence_score=0.8,  # Default confidence
                generated_at=datetime.now(timezone.utc),
                request_id=f"req_{int(time.time())}"
            )
    
    async def _call_gemini(self, model: str, prompt: str, request: LLMRequest) -> LLMResponse:
        """Call Gemini API directly"""
        # Implement Gemini API call
        # This would use the Google AI SDK
        raise NotImplementedError("Direct Gemini API not yet implemented")
    
    async def _fallback_generation(self, request: LLMRequest) -> LLMResponse:
        """Fallback to static payloads when APIs fail"""
        logger.warning("Using fallback payload generation")
        
        fallback_payloads = {
            TaskType.PAYLOAD_GENERATION: {
                "xss": ["<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>"],
                "sqli": ["' OR 1=1--", "' UNION SELECT 1,2,3--"],
                "ssrf": ["http://127.0.0.1:80", "http://metadata.google.internal/"],
                "rce": ["; whoami", "$(whoami)", "`id`"]
            }
        }
        
        # Return basic static payload
        content = "Fallback payload generation - API unavailable"
        if request.task_type in fallback_payloads:
            content = json.dumps(fallback_payloads[request.task_type], indent=2)
        
        return LLMResponse(
            content=content,
            model_used="fallback_static",
            provider=LLMProvider.LOCAL,
            tokens_used=0,
            cost_usd=0.0,
            confidence_score=0.3,  # Low confidence for fallback
            generated_at=datetime.now(timezone.utc),
            request_id=f"fallback_{int(time.time())}"
        )
    
    def _check_rate_limit(self, model_name: str) -> bool:
        """Check if model is within rate limits"""
        now = datetime.now(timezone.utc)
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
    
    def _calculate_cost(self, model_name: str, tokens_used: int) -> float:
        """Calculate cost for API call"""
        model_config = self.models[model_name]
        return (tokens_used / 1000) * model_config.cost_per_1k_tokens
    
    def _track_usage(self, model_name: str, response: LLMResponse):
        """Track API usage and costs"""
        now = datetime.now(timezone.utc)
        
        # Update request counts
        if model_name not in self.request_counts:
            self.request_counts[model_name] = []
        self.request_counts[model_name].append(now)
        
        # Update total cost
        self.total_cost += response.cost_usd
        
        # Add to history
        self.request_history.append({
            "timestamp": now.isoformat(),
            "model": model_name,
            "tokens": response.tokens_used,
            "cost": response.cost_usd,
            "task_type": "unknown"  # Would be passed from request
        })
        
        # Keep only last 1000 requests in memory
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate multiple payloads efficiently with batching"""
        responses = []
        
        # Group requests by optimal model
        model_groups = {}
        for req in requests:
            model = self.select_optimal_model(req)
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(req)
        
        # Process each group
        for model, model_requests in model_groups.items():
            # Respect rate limits
            batch_size = min(len(model_requests), self.models[model].rate_limit_rpm // 6)
            
            for i in range(0, len(model_requests), batch_size):
                batch = model_requests[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self.generate_payload(req) for req in batch]
                batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for resp in batch_responses:
                    if isinstance(resp, Exception):
                        logger.error(f"Batch request failed: {resp}")
                    else:
                        responses.append(resp)
                
                # Rate limiting delay between batches
                if i + batch_size < len(model_requests):
                    await asyncio.sleep(1)
        
        return responses
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": len(self.request_history),
            "total_cost_usd": self.total_cost,
            "requests_by_model": {
                model: len([h for h in self.request_history if h["model"] == model])
                for model in self.models.keys()
            },
            "avg_cost_per_request": self.total_cost / max(len(self.request_history), 1),
            "last_24h_requests": len([
                h for h in self.request_history
                if datetime.fromisoformat(h["timestamp"]) > datetime.now(timezone.utc) - timedelta(days=1)
            ])
        }