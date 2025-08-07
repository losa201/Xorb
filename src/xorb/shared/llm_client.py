#!/usr/bin/env python3
"""
Strategic LLM Integration Client for XORB Enhanced Platform
Integrates OpenRouter and NVIDIA APIs with supreme cognitive precision
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL_FREE = "local_free"

class SecurityTaskType(Enum):
    PAYLOAD_GENERATION = "payload_generation"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    THREAT_ASSESSMENT = "threat_assessment"
    REPORT_GENERATION = "report_generation"
    ANOMALY_DETECTION = "anomaly_detection"
    ATTACK_SIMULATION = "attack_simulation"
    RISK_SCORING = "risk_scoring"
    INTELLIGENCE_FUSION = "intelligence_fusion"

class TaskComplexity(Enum):
    BASIC = "basic"           # Simple analysis, basic payloads
    ENHANCED = "enhanced"     # Context-aware analysis with intelligence
    ADVANCED = "advanced"     # Multi-vector analysis with learning
    COGNITIVE = "cognitive"   # Supreme intelligence with reasoning chains

@dataclass
class ModelCapabilities:
    name: str
    provider: LLMProvider
    cost_per_1k_tokens: float
    max_tokens: int
    context_window: int
    security_tasks: List[SecurityTaskType]
    supports_structured_output: bool = False
    supports_function_calling: bool = False
    creativity_score: float = 0.7
    reliability_score: float = 0.8
    security_score: float = 0.9
    rate_limit_rpm: int = 60
    nvidia_optimized: bool = False

class StrategicLLMRequest(BaseModel):
    task_type: SecurityTaskType
    prompt: str
    target_context: Optional[Dict[str, Any]] = None
    complexity: TaskComplexity = TaskComplexity.ENHANCED
    max_tokens: int = 4000
    temperature: float = 0.7
    structured_output: bool = False
    use_nvidia_api: bool = True
    use_openrouter: bool = True
    budget_limit_usd: float = 1.0
    priority: int = 3  # 1-5, higher = more critical
    requires_creativity: bool = False
    chain_reasoning: bool = True
    epyc_optimized: bool = True

class LLMResponse(BaseModel):
    content: str
    model_used: str
    provider: LLMProvider
    tokens_used: int
    cost_usd: float
    confidence_score: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    request_id: str
    structured_data: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    was_fallback: bool = False
    security_analysis: Optional[Dict[str, Any]] = None
    intelligence_level: str = "enhanced"

class StrategicLLMClient:
    """Supreme cognitive LLM client with OpenRouter and NVIDIA integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Strategic cost management with fallback
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.daily_requests = 0
        self.monthly_requests = 0
        
        # Paid tier budget limits
        self.budget_limits = {
            "daily": config.get("daily_budget_limit", 25.0),
            "monthly": config.get("monthly_budget_limit", 500.0),
            "per_request": config.get("per_request_limit", 5.0),
            "emergency": config.get("emergency_budget", 50.0)
        }
        
        # Free tier fallback limits
        self.request_limits = {
            "daily": config.get("daily_request_limit", 100),
            "monthly": config.get("monthly_request_limit", 2000),
            "per_hour": config.get("hourly_request_limit", 20),
            "emergency": config.get("emergency_request_limit", 150)
        }
        
        self.prefer_paid = config.get("prefer_paid", True)
        self.enable_fallback = config.get("enable_fallback", True)
        
        # Intelligence optimization
        self.request_cache: Dict[str, LLMResponse] = {}
        self.cognitive_memory: Dict[str, Any] = {}
        self.request_history: List[Dict[str, Any]] = []
        
        # Enhanced model configurations
        self.models = self._initialize_strategic_models()
        self.prompt_templates = self._load_security_templates()
        
    def _initialize_strategic_models(self) -> Dict[str, ModelCapabilities]:
        """Initialize strategic security-optimized models with actual free models."""
        return {
            # NVIDIA FREE MODELS (via OpenAI client)
            "qwen/qwen3-235b-a22b": ModelCapabilities(
                name="qwen/qwen3-235b-a22b",
                provider=LLMProvider.NVIDIA,
                cost_per_1k_tokens=0.0,  # Free
                max_tokens=16384,
                context_window=131072,
                security_tasks=[SecurityTaskType.PAYLOAD_GENERATION, SecurityTaskType.VULNERABILITY_ANALYSIS, SecurityTaskType.THREAT_ASSESSMENT],
                supports_structured_output=True,
                creativity_score=0.98,
                reliability_score=0.99,
                security_score=0.995,
                nvidia_optimized=True,
                rate_limit_rpm=60
            ),
            
            # OPENROUTER FREE MODELS
            "qwen/qwen-2.5-coder-32b-instruct:free": ModelCapabilities(
                name="qwen/qwen-2.5-coder-32b-instruct:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free
                max_tokens=32768,
                context_window=131072,
                security_tasks=[SecurityTaskType.PAYLOAD_GENERATION, SecurityTaskType.VULNERABILITY_ANALYSIS],
                supports_structured_output=True,
                creativity_score=0.96,
                reliability_score=0.98,
                security_score=0.99,
                rate_limit_rpm=30
            ),
            "01-ai/yi-1.5-34b-chat:free": ModelCapabilities(
                name="01-ai/yi-1.5-34b-chat:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free
                max_tokens=32768,
                context_window=200000,
                security_tasks=[SecurityTaskType.THREAT_ASSESSMENT, SecurityTaskType.INTELLIGENCE_FUSION, SecurityTaskType.REPORT_GENERATION],
                supports_structured_output=True,
                creativity_score=0.92,
                reliability_score=0.95,
                security_score=0.96,
                rate_limit_rpm=25
            ),
            "zhipuai/glm-4-9b-chat:free": ModelCapabilities(
                name="zhipuai/glm-4-9b-chat:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free
                max_tokens=8192,
                context_window=128000,
                security_tasks=[SecurityTaskType.ANOMALY_DETECTION, SecurityTaskType.ATTACK_SIMULATION, SecurityTaskType.RISK_SCORING],
                supports_structured_output=True,
                creativity_score=0.88,
                reliability_score=0.9,
                security_score=0.92,
                rate_limit_rpm=40
            ),
            "openrouter/flavor": ModelCapabilities(
                name="openrouter/flavor",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free
                max_tokens=8192,
                context_window=32768,
                security_tasks=[SecurityTaskType.VULNERABILITY_ANALYSIS, SecurityTaskType.THREAT_ASSESSMENT],
                supports_structured_output=False,
                creativity_score=0.85,
                reliability_score=0.87,
                security_score=0.89,
                rate_limit_rpm=50
            ),
            "microsoft/phi-3-medium-128k-instruct:free": ModelCapabilities(
                name="microsoft/phi-3-medium-128k-instruct:free",
                provider=LLMProvider.OPENROUTER,
                cost_per_1k_tokens=0.0,  # Free
                max_tokens=4096,
                context_window=128000,
                security_tasks=[SecurityTaskType.PAYLOAD_GENERATION, SecurityTaskType.VULNERABILITY_ANALYSIS],
                supports_structured_output=True,
                creativity_score=0.82,
                reliability_score=0.85,
                security_score=0.87,
                rate_limit_rpm=35
            )
        }
    
    def _load_security_templates(self) -> Dict[str, str]:
        """Load security-optimized prompt templates."""
        return {
            SecurityTaskType.PAYLOAD_GENERATION.value: """
You are an elite cybersecurity specialist conducting authorized penetration testing.

TARGET CONTEXT:
{target_context}

TASK: Generate a sophisticated {complexity} payload for {task_type}

REQUIREMENTS:
- Ensure payload is for authorized testing only
- Include evasion techniques appropriate for {complexity} level
- Provide technical explanation of payload mechanics
- Include detection signatures for defensive purposes

RESPOND WITH:
1. Payload code/command
2. Technical analysis
3. Evasion techniques used
4. Detection methods
5. Risk assessment
""",
            
            SecurityTaskType.VULNERABILITY_ANALYSIS.value: """
You are a senior vulnerability researcher analyzing security findings.

SCAN RESULTS:
{scan_results}

TARGET CONTEXT:
{target_context}

TASK: Perform {complexity} vulnerability analysis

ANALYZE:
1. Criticality assessment (CVSS scoring)
2. Exploitation feasibility
3. Business impact analysis
4. Remediation recommendations
5. Attack chain possibilities

Provide structured analysis with risk scores and actionable recommendations.
""",
            
            SecurityTaskType.THREAT_ASSESSMENT.value: """
You are a threat intelligence analyst conducting strategic assessment.

INTELLIGENCE DATA:
{intelligence_data}

ENVIRONMENT:
{target_context}

TASK: Perform {complexity} threat landscape analysis

ASSESS:
1. Threat actor profiling
2. Attack vector probability
3. Asset vulnerability mapping
4. Risk quantification
5. Strategic recommendations

Provide executive-level threat assessment with actionable intelligence.
""",
            
            SecurityTaskType.INTELLIGENCE_FUSION.value: """
You are an intelligence fusion specialist with supreme analytical capabilities.

MULTI-SOURCE DATA:
{fusion_data}

CONTEXT:
{target_context}

TASK: Perform {complexity} intelligence fusion and correlation

SYNTHESIZE:
1. Cross-source validation
2. Pattern recognition
3. Anomaly identification
4. Predictive indicators
5. Strategic insights

Provide comprehensive intelligence picture with confidence levels.
"""
        }
    
    async def initialize(self):
        """Initialize the strategic LLM client."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            headers={'User-Agent': 'XORB-Strategic-Intelligence/2.0'}
        )
        logger.info("Strategic LLM Client initialized with supreme cognitive capabilities")
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
    
    def select_optimal_model(self, request: StrategicLLMRequest) -> str:
        """Select optimal model with intelligent availability checking."""
        # Filter models by task type
        suitable_models = [
            model for model in self.models.values()
            if request.task_type in model.security_tasks
        ]
        
        if not suitable_models:
            # Fallback to most capable model
            suitable_models = list(self.models.values())
        
        # Separate by cost (premium models with credits vs free models)
        premium_models = [m for m in suitable_models if m.cost_per_1k_tokens > 0]
        free_models = [m for m in suitable_models if m.cost_per_1k_tokens == 0]
        
        # Try premium models first if credits available and budget allows
        if self.prefer_paid and premium_models:
            for model in self._score_models(premium_models, request, prefer_premium=True):
                if self._check_budget_constraints(request, model):
                    logger.info(f"Selected PREMIUM model: {model.name} for task: {request.task_type}")
                    return model.name
            
            logger.info("Premium models not available or budget exceeded, using free tier models")
        
        # Use free tier models (NVIDIA and OpenRouter free options)
        if free_models:
            for model in self._score_models(free_models, request, prefer_premium=False):
                if self._check_model_specific_limits(model):
                    logger.info(f"Selected model: {model.name} for task: {request.task_type}")
                    return model.name
        
        # Emergency fallback - best available model regardless of constraints
        all_scored = self._score_models(suitable_models, request, prefer_premium=False)
        emergency_model = all_scored[0] if all_scored else self.models["nvidia/qwen2.5-coder-32b-instruct"]
        
        logger.warning(f"Emergency fallback to model: {emergency_model.name}")
        return emergency_model.name
    
    def _score_models(self, models: List[ModelCapabilities], request: StrategicLLMRequest, prefer_premium: bool = True) -> List[ModelCapabilities]:
        """Score and sort models based on request requirements."""
        scored_models = []
        
        for model in models:
            score = 0.0
            
            # Base scoring by complexity requirement
            if request.complexity == TaskComplexity.COGNITIVE:
                score += model.creativity_score * 0.4
                score += model.security_score * 0.4
                score += model.reliability_score * 0.2
            elif request.complexity == TaskComplexity.ADVANCED:
                score += model.security_score * 0.5
                score += model.reliability_score * 0.3
                score += model.creativity_score * 0.2
            else:
                score += model.reliability_score * 0.5
                score += model.security_score * 0.3
                score += model.creativity_score * 0.2
            
            # Provider preferences
            if request.use_nvidia_api and model.provider == LLMProvider.NVIDIA:
                score += 0.2
            if request.use_openrouter and model.provider == LLMProvider.OPENROUTER:
                score += 0.15
            
            # EPYC optimization bonus
            if request.epyc_optimized and model.nvidia_optimized:
                score += 0.15
            
            # Qwen3 bonus for code/security tasks
            if "qwen" in model.name.lower():
                if request.task_type in [SecurityTaskType.PAYLOAD_GENERATION, SecurityTaskType.VULNERABILITY_ANALYSIS]:
                    score += 0.25  # Qwen3 excels at code generation
            
            # Model availability and access
            if prefer_premium and model.cost_per_1k_tokens > 0:
                # Premium models get bonus if credits available
                score += 0.1 - (model.cost_per_1k_tokens * 0.01)
            elif not prefer_premium and model.cost_per_1k_tokens == 0:
                # Free models get bonus when no credits
                score += 0.2
            
            scored_models.append((model, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return [model for model, score in scored_models]
    
    async def execute_strategic_request(self, request: StrategicLLMRequest) -> LLMResponse:
        """Execute strategic LLM request with supreme intelligence."""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{time.time()}".encode()).hexdigest()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.request_cache and not request.requires_creativity:
            cached_response = self.request_cache[cache_key]
            logger.info(f"Returning cached response for request {request_id}")
            return cached_response
        
        # Select optimal model
        model_name = self.select_optimal_model(request)
        model = self.models[model_name]
        
        # Check constraints (budget for paid, rate limits for free)
        if model.cost_per_1k_tokens > 0:
            if not self._check_budget_constraints(request, model):
                raise ValueError("Budget constraints exceeded for strategic operation")
        else:
            if not self._check_request_limits(request, model):
                raise ValueError("Request rate limits exceeded for strategic operation")
        
        # Enhance prompt with template
        enhanced_prompt = self._enhance_prompt_with_template(request)
        
        # Execute request
        try:
            response = await self._execute_model_request(
                model, enhanced_prompt, request, request_id
            )
            
            # Add security analysis
            response.security_analysis = self._analyze_response_security(response)
            response.intelligence_level = request.complexity.value
            
            # Cache if appropriate
            if not request.requires_creativity:
                self.request_cache[cache_key] = response
            
            # Update metrics
            self._update_usage_metrics(response)
            
            execution_time = int((time.time() - start_time) * 1000)
            response.execution_time_ms = execution_time
            
            logger.info(
                f"Strategic request completed: {request_id}, "
                f"model: {model_name}, cost: ${response.cost_usd:.4f}, "
                f"time: {execution_time}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Strategic LLM request failed: {e}")
            # Attempt fallback
            return await self._execute_fallback_request(request, request_id, str(e))
    
    async def _execute_model_request(
        self, 
        model: ModelCapabilities, 
        prompt: str, 
        request: StrategicLLMRequest,
        request_id: str
    ) -> LLMResponse:
        """Execute request against specific model."""
        
        if model.provider == LLMProvider.NVIDIA:
            return await self._execute_nvidia_request(model, prompt, request, request_id)
        elif model.provider == LLMProvider.OPENROUTER:
            return await self._execute_openrouter_request(model, prompt, request, request_id)
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")
    
    async def _execute_nvidia_request(
        self,
        model: ModelCapabilities,
        prompt: str,
        request: StrategicLLMRequest,
        request_id: str
    ) -> LLMResponse:
        """Execute request using NVIDIA API via OpenAI client."""
        try:
            # Import OpenAI client dynamically
            try:
                from openai import OpenAI
            except ImportError:
                raise Exception("OpenAI package required for NVIDIA integration. Install with: pip install openai")
            
            # Create OpenAI client for NVIDIA
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.config.get('nvidia_api_key')
            )
            
            # Prepare completion parameters
            completion_params = {
                "model": model.name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": request.temperature,
                "top_p": 1,
                "max_tokens": min(request.max_tokens, model.max_tokens),
                "stream": False
            }
            
            # Add special parameters for Qwen3 thinking mode
            if "qwen3" in model.name.lower():
                completion_params["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
            
            # Execute completion
            completion = client.chat.completions.create(**completion_params)
            
            # Extract response
            choice = completion.choices[0]
            usage = completion.usage
            
            # Handle thinking content for Qwen3
            content = choice.message.content
            reasoning_content = getattr(choice.message, "reasoning_content", None)
            
            # Combine reasoning and content if available
            full_content = content
            if reasoning_content:
                full_content = f"Reasoning: {reasoning_content}\n\nResponse: {content}"
            
            return LLMResponse(
                content=full_content,
                model_used=model.name,
                provider=LLMProvider.NVIDIA,
                tokens_used=usage.total_tokens if usage else len(full_content) // 4,  # Estimate if no usage
                cost_usd=0.0,  # Free tier
                confidence_score=0.95,
                request_id=request_id
            )
            
        except Exception as e:
            logger.error(f"NVIDIA API error: {e}")
            raise Exception(f"NVIDIA API request failed: {str(e)}")
    
    async def _execute_openrouter_request(
        self,
        model: ModelCapabilities,
        prompt: str,
        request: StrategicLLMRequest,
        request_id: str
    ) -> LLMResponse:
        """Execute request using OpenRouter API."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.get('openrouter_api_key')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://xorb.platform",
            "X-Title": "XORB Strategic Intelligence Platform"
        }
        
        payload = {
            "model": model.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(request.max_tokens, model.max_tokens),
            "temperature": request.temperature
        }
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
            
            data = await response.json()
            choice = data["choices"][0]
            usage = data["usage"]
            
            return LLMResponse(
                content=choice["message"]["content"],
                model_used=model.name,
                provider=LLMProvider.OPENROUTER,
                tokens_used=usage["total_tokens"],
                cost_usd=usage["total_tokens"] * model.cost_per_1k_tokens / 1000,
                confidence_score=0.9,
                request_id=request_id
            )
    
    def _enhance_prompt_with_template(self, request: StrategicLLMRequest) -> str:
        """Enhance prompt with strategic template."""
        template = self.prompt_templates.get(request.task_type.value, "{prompt}")
        
        context_data = {
            "prompt": request.prompt,
            "complexity": request.complexity.value,
            "task_type": request.task_type.value,
            "target_context": json.dumps(request.target_context or {}, indent=2),
            "timestamp": datetime.utcnow().isoformat(),
            "epyc_optimized": request.epyc_optimized
        }
        
        try:
            return template.format(**context_data)
        except KeyError:
            return request.prompt
    
    def _check_request_limits(self, request: StrategicLLMRequest, model: ModelCapabilities) -> bool:
        """Check if request is within free tier rate limits."""
        if self.free_tier:
            # Free tier - check request limits instead of cost
            checks = [
                self.daily_requests < self.request_limits["daily"],
                self.monthly_requests < self.request_limits["monthly"],
                self._check_hourly_rate_limit(model),
                self._check_model_specific_limits(model)
            ]
        else:
            # Paid tier - original budget logic (if needed later)
            estimated_cost = (request.max_tokens * model.cost_per_1k_tokens) / 1000
            checks = [
                estimated_cost <= request.budget_limit_usd,
                self.daily_requests < self.request_limits["daily"]
            ]
        
        return all(checks)
    
    def _check_budget_constraints(self, request: StrategicLLMRequest, model: ModelCapabilities) -> bool:
        """Check if request is within budget constraints for paid models."""
        estimated_cost = (request.max_tokens * model.cost_per_1k_tokens) / 1000
        
        checks = [
            estimated_cost <= request.budget_limit_usd,
            estimated_cost <= self.budget_limits["per_request"],
            self.daily_spend + estimated_cost <= self.budget_limits["daily"],
            self.monthly_spend + estimated_cost <= self.budget_limits["monthly"]
        ]
        
        return all(checks)
    
    def _check_hourly_rate_limit(self, model: ModelCapabilities) -> bool:
        """Check hourly rate limits for the model."""
        import time
        current_hour = int(time.time()) // 3600
        
        # Simple hourly tracking (would be more sophisticated in production)
        hourly_key = f"hourly_{current_hour}"
        if not hasattr(self, 'hourly_requests'):
            self.hourly_requests = {}
        
        current_hourly = self.hourly_requests.get(hourly_key, 0)
        return current_hourly < min(self.request_limits["per_hour"], model.rate_limit_rpm)
    
    def _check_model_specific_limits(self, model: ModelCapabilities) -> bool:
        """Check model-specific rate limits."""
        # For free tier, respect model RPM limits
        if not hasattr(self, 'model_request_times'):
            self.model_request_times = {}
        
        import time
        current_time = time.time()
        model_key = model.name
        
        # Clean old timestamps (older than 1 minute)
        if model_key in self.model_request_times:
            self.model_request_times[model_key] = [
                t for t in self.model_request_times[model_key] 
                if current_time - t < 60
            ]
        else:
            self.model_request_times[model_key] = []
        
        # Check if we're under the RPM limit
        return len(self.model_request_times[model_key]) < model.rate_limit_rpm
    
    def _generate_cache_key(self, request: StrategicLLMRequest) -> str:
        """Generate cache key for request."""
        cache_data = {
            "task_type": request.task_type.value,
            "prompt_hash": hashlib.md5(request.prompt.encode()).hexdigest(),
            "complexity": request.complexity.value,
            "target_context": request.target_context
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _analyze_response_security(self, response: LLMResponse) -> Dict[str, Any]:
        """Analyze response for security implications."""
        return {
            "contains_sensitive_data": False,
            "security_classification": "authorized_testing",
            "confidence_level": response.confidence_score,
            "validation_required": response.confidence_score < 0.8,
            "model_reliability": self.models[response.model_used].reliability_score
        }
    
    def _update_usage_metrics(self, response: LLMResponse):
        """Update usage metrics for both paid and free models."""
        import time
        
        # Update request counters
        self.daily_requests += 1
        self.monthly_requests += 1
        
        # Update cost tracking (for paid models)
        if response.cost_usd > 0:
            self.daily_spend += response.cost_usd
            self.monthly_spend += response.cost_usd
        
        # Update hourly tracking
        current_hour = int(time.time()) // 3600
        hourly_key = f"hourly_{current_hour}"
        if not hasattr(self, 'hourly_requests'):
            self.hourly_requests = {}
        self.hourly_requests[hourly_key] = self.hourly_requests.get(hourly_key, 0) + 1
        
        # Update model-specific tracking
        if not hasattr(self, 'model_request_times'):
            self.model_request_times = {}
        model_key = response.model_used
        if model_key not in self.model_request_times:
            self.model_request_times[model_key] = []
        self.model_request_times[model_key].append(time.time())
        
        # Store request history
        self.request_history.append({
            "timestamp": response.generated_at.isoformat(),
            "model": response.model_used,
            "provider": response.provider.value,
            "tokens": response.tokens_used,
            "cost": response.cost_usd,
            "was_paid": response.cost_usd > 0,
            "was_fallback": response.was_fallback
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def _execute_fallback_request(
        self, 
        request: StrategicLLMRequest, 
        request_id: str, 
        error: str
    ) -> LLMResponse:
        """Execute fallback request with basic model."""
        logger.warning(f"Executing fallback for request {request_id}: {error}")
        
        # Use most reliable, cheapest model as fallback
        fallback_models = sorted(
            self.models.values(),
            key=lambda m: (m.reliability_score, -m.cost_per_1k_tokens),
            reverse=True
        )
        
        if not fallback_models:
            raise Exception("No fallback models available")
        
        fallback_model = fallback_models[0]
        
        try:
            response = await self._execute_model_request(
                fallback_model, request.prompt, request, request_id
            )
            response.was_fallback = True
            response.confidence_score *= 0.8  # Reduce confidence for fallback
            return response
        except Exception as fallback_error:
            logger.error(f"Fallback request also failed: {fallback_error}")
            raise Exception(f"All strategic LLM options exhausted: {error}")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics for hybrid paid/free system."""
        import time
        current_hour = int(time.time()) // 3600
        hourly_key = f"hourly_{current_hour}"
        
        # Calculate paid vs free usage
        paid_requests = len([r for r in self.request_history if r.get("was_paid", False)])
        free_requests = len([r for r in self.request_history if not r.get("was_paid", False)])
        fallback_requests = len([r for r in self.request_history if r.get("was_fallback", False)])
        
        return {
            "deployment_strategy": "paid_first_with_free_fallback",
            "prefer_paid": self.prefer_paid,
            "enable_fallback": self.enable_fallback,
            
            # Cost tracking
            "daily_spend": self.daily_spend,
            "monthly_spend": self.monthly_spend,
            "budget_limits": self.budget_limits,
            
            # Request tracking
            "daily_requests": self.daily_requests,
            "monthly_requests": self.monthly_requests,
            "hourly_requests": getattr(self, 'hourly_requests', {}).get(hourly_key, 0),
            "request_limits": self.request_limits,
            
            # Usage breakdown
            "paid_requests": paid_requests,
            "free_requests": free_requests,
            "fallback_requests": fallback_requests,
            "total_requests": len(self.request_history),
            
            # Efficiency metrics
            "cache_hit_rate": len(self.request_cache) / max(len(self.request_history), 1),
            "cost_per_request": self.daily_spend / max(paid_requests, 1) if paid_requests > 0 else 0,
            "fallback_rate": fallback_requests / max(len(self.request_history), 1),
            
            # System info
            "provider_distribution": self._calculate_provider_distribution(),
            "models_available": len(self.models),
            "qwen3_models": len([m for m in self.models.keys() if "qwen" in m.lower()]),
            "strategic_capabilities": [task.value for task in SecurityTaskType],
            
            # Status
            "budget_status": self._get_budget_status(),
            "rate_limit_status": self._get_rate_limit_status()
        }
    
    def _get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        import time
        current_hour = int(time.time()) // 3600
        hourly_key = f"hourly_{current_hour}"
        
        daily_usage = self.daily_requests / self.request_limits["daily"]
        monthly_usage = self.monthly_requests / self.request_limits["monthly"]
        hourly_usage = getattr(self, 'hourly_requests', {}).get(hourly_key, 0) / self.request_limits["per_hour"]
        
        return {
            "daily_usage_percentage": min(100, daily_usage * 100),
            "monthly_usage_percentage": min(100, monthly_usage * 100),
            "hourly_usage_percentage": min(100, hourly_usage * 100),
            "status": "healthy" if max(daily_usage, monthly_usage, hourly_usage) < 0.8 else "warning"
        }
    
    def _get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status for paid models."""
        daily_usage = self.daily_spend / self.budget_limits["daily"]
        monthly_usage = self.monthly_spend / self.budget_limits["monthly"]
        
        return {
            "daily_spend_percentage": min(100, daily_usage * 100),
            "monthly_spend_percentage": min(100, monthly_usage * 100),
            "remaining_daily_budget": max(0, self.budget_limits["daily"] - self.daily_spend),
            "remaining_monthly_budget": max(0, self.budget_limits["monthly"] - self.monthly_spend),
            "status": "healthy" if max(daily_usage, monthly_usage) < 0.8 else "warning"
        }
    
    def _calculate_provider_distribution(self) -> Dict[str, int]:
        """Calculate distribution of requests across providers."""
        distribution = {}
        for request in self.request_history:
            provider = request.get("provider", "unknown")
            distribution[provider] = distribution.get(provider, 0) + 1
        return distribution