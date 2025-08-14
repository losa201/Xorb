#!/usr/bin/env python3
"""
XORB Advanced LLM Orchestrator - PRODUCTION READY
Strategic AI-powered decision making with enterprise-grade reliability
"""

import asyncio
import logging
import json
import os
import time
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import hashlib
import uuid
import aiohttp

# Enhanced imports with fallbacks
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    """Enhanced AI provider enumeration with fallbacks"""
    OPENROUTER_QWEN = "openrouter_qwen"
    OPENROUTER_DEEPSEEK = "openrouter_deepseek"
    OPENROUTER_ANTHROPIC = "openrouter_anthropic"
    NVIDIA_QWEN = "nvidia_qwen"
    NVIDIA_LLAMA = "nvidia_llama"
    LOCAL_FALLBACK = "local_fallback"

class DecisionComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"

class DecisionDomain(Enum):
    SECURITY_ANALYSIS = "security_analysis"
    THREAT_ASSESSMENT = "threat_assessment"
    INCIDENT_RESPONSE = "incident_response"
    VULNERABILITY_PRIORITIZATION = "vulnerability_prioritization"
    ATTACK_SIMULATION = "attack_simulation"
    COMPLIANCE_VALIDATION = "compliance_validation"
    ARCHITECTURE_OPTIMIZATION = "architecture_optimization"

@dataclass
class AIDecisionRequest:
    """Enhanced decision request with validation"""
    decision_id: str
    domain: DecisionDomain
    complexity: DecisionComplexity
    context: Dict[str, Any]
    constraints: List[str]
    require_consensus: bool = False
    timeout_seconds: int = 30
    priority: str = "medium"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

        # Validate inputs
        if not self.decision_id:
            self.decision_id = f"decision_{uuid.uuid4().hex[:8]}"

        if self.timeout_seconds > 120:
            self.timeout_seconds = 120  # Max 2 minutes

        if not self.constraints:
            self.constraints = []

@dataclass
class AIDecisionResponse:
    """Enhanced decision response with confidence tracking"""
    decision_id: str
    decision: str
    confidence: float
    reasoning: List[str]
    alternatives: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    provider_used: str
    execution_time_ms: float
    token_usage: Dict[str, int]
    metadata: Dict[str, Any]
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

        # Ensure confidence is valid
        self.confidence = max(0.0, min(1.0, self.confidence))

class ProviderConfig:
    """Enhanced provider configuration with circuit breaker"""
    def __init__(self, provider: AIProvider, base_url: str, api_key: str,
                 model_name: str, max_tokens: int = 2000, temperature: float = 0.3):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limit_rpm = 60
        self.rate_limit_tpm = 50000
        self.last_request = 0
        self.request_count = 0
        self.error_count = 0
        self.success_count = 0
        self.circuit_breaker_state = "closed"  # closed, open, half_open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0

class ProductionLLMOrchestrator:
    """Production-grade LLM orchestrator with enhanced reliability"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers: Dict[AIProvider, ProviderConfig] = {}
        self.clients: Dict[AIProvider, AsyncOpenAI] = {}
        self.decision_history: List[AIDecisionResponse] = []
        self.provider_performance: Dict[AIProvider, Dict[str, float]] = {}
        self.consensus_threshold = 0.7
        self.max_retries = 3
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes

        # Enhanced fallback chain with local options
        self.fallback_chain = [
            AIProvider.OPENROUTER_QWEN,
            AIProvider.NVIDIA_QWEN,
            AIProvider.OPENROUTER_DEEPSEEK,
            AIProvider.OPENROUTER_ANTHROPIC,
            AIProvider.NVIDIA_LLAMA,
            AIProvider.LOCAL_FALLBACK
        ]

        # Initialize token counting
        self.token_encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.token_encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    async def initialize(self) -> bool:
        """Initialize the orchestrator with enhanced error handling"""
        try:
            logger.info("Initializing Production LLM Orchestrator...")

            # Initialize provider configurations
            await self._setup_providers()

            # Initialize HTTP clients with proper configuration
            await self._setup_clients()

            # Test connectivity with circuit breaker logic
            await self._test_provider_connectivity()

            # Initialize local fallback capabilities
            await self._initialize_local_fallback()

            active_providers = sum(1 for p in self.providers.values()
                                 if p.circuit_breaker_state == "closed")

            logger.info(f"LLM Orchestrator initialized with {active_providers} active providers")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LLM Orchestrator: {e}")
            return False

    async def _setup_providers(self):
        """Setup provider configurations with production settings"""
        # OpenRouter providers with API key validation
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key and openrouter_key != "dummy_key":
            self.providers[AIProvider.OPENROUTER_QWEN] = ProviderConfig(
                provider=AIProvider.OPENROUTER_QWEN,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_name="qwen/qwen-2.5-coder-32b-instruct",
                max_tokens=2000,
                temperature=0.2
            )

            self.providers[AIProvider.OPENROUTER_DEEPSEEK] = ProviderConfig(
                provider=AIProvider.OPENROUTER_DEEPSEEK,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_name="deepseek/deepseek-chat",
                max_tokens=2000,
                temperature=0.3
            )

            self.providers[AIProvider.OPENROUTER_ANTHROPIC] = ProviderConfig(
                provider=AIProvider.OPENROUTER_ANTHROPIC,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_name="anthropic/claude-3.5-sonnet",
                max_tokens=2000,
                temperature=0.2
            )

        # NVIDIA providers with API key validation
        nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        if nvidia_key and nvidia_key != "nvapi-dummy":
            self.providers[AIProvider.NVIDIA_QWEN] = ProviderConfig(
                provider=AIProvider.NVIDIA_QWEN,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key,
                model_name="qwen/qwen2.5-coder-32b-instruct",
                max_tokens=2000,
                temperature=0.2
            )

            self.providers[AIProvider.NVIDIA_LLAMA] = ProviderConfig(
                provider=AIProvider.NVIDIA_LLAMA,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key,
                model_name="meta/llama-3.1-405b-instruct",
                max_tokens=2000,
                temperature=0.3
            )

        # Always add local fallback
        self.providers[AIProvider.LOCAL_FALLBACK] = ProviderConfig(
            provider=AIProvider.LOCAL_FALLBACK,
            base_url="local",
            api_key="local",
            model_name="local_decision_engine",
            max_tokens=2000,
            temperature=0.5
        )

    async def _setup_clients(self):
        """Setup API clients with production configurations"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available - using local fallback only")
            return

        for provider, config in self.providers.items():
            if provider == AIProvider.LOCAL_FALLBACK:
                continue

            try:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                client = AsyncOpenAI(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    timeout=30.0,
                    max_retries=2
                )
                self.clients[provider] = client
                logger.info(f"Initialized client for {provider.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize client for {provider.value}: {e}")
                config.circuit_breaker_state = "open"

    async def _test_provider_connectivity(self):
        """Test provider connectivity with circuit breaker logic"""
        for provider, client in self.clients.items():
            try:
                config = self.providers[provider]

                # Skip if circuit breaker is open
                if config.circuit_breaker_state == "open":
                    if time.time() - config.circuit_breaker_last_failure < self.circuit_breaker_timeout:
                        continue
                    else:
                        config.circuit_breaker_state = "half_open"

                # Simple connectivity test
                test_messages = [
                    {"role": "system", "content": "You are XORB AI. Respond with 'OK' only."},
                    {"role": "user", "content": "Test connectivity"}
                ]

                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=config.model_name,
                        messages=test_messages,
                        max_tokens=10,
                        temperature=0
                    ),
                    timeout=10.0
                )

                if response.choices and response.choices[0].message.content:
                    logger.info(f"Provider {provider.value} connectivity: OK")
                    config.circuit_breaker_state = "closed"
                    config.circuit_breaker_failures = 0
                    self._update_provider_performance(provider, True, 0.5)
                else:
                    self._handle_provider_failure(config)

            except Exception as e:
                logger.warning(f"Provider {provider.value} connectivity test failed: {e}")
                self._handle_provider_failure(self.providers[provider])

    def _handle_provider_failure(self, config: ProviderConfig):
        """Handle provider failure with circuit breaker logic"""
        config.error_count += 1
        config.circuit_breaker_failures += 1
        config.circuit_breaker_last_failure = time.time()

        if config.circuit_breaker_failures >= self.circuit_breaker_threshold:
            config.circuit_breaker_state = "open"
            logger.warning(f"Circuit breaker opened for {config.provider.value}")

        self._update_provider_performance(config.provider, False, 0)

    async def _initialize_local_fallback(self):
        """Initialize local decision-making capabilities"""
        try:
            # Initialize local decision trees and heuristics
            self.local_decision_rules = {
                DecisionDomain.SECURITY_ANALYSIS: self._local_security_analysis,
                DecisionDomain.THREAT_ASSESSMENT: self._local_threat_assessment,
                DecisionDomain.INCIDENT_RESPONSE: self._local_incident_response,
                DecisionDomain.VULNERABILITY_PRIORITIZATION: self._local_vuln_prioritization,
                DecisionDomain.ATTACK_SIMULATION: self._local_attack_simulation,
                DecisionDomain.COMPLIANCE_VALIDATION: self._local_compliance_validation,
                DecisionDomain.ARCHITECTURE_OPTIMIZATION: self._local_architecture_optimization
            }

            logger.info("Local fallback decision engine initialized")

        except Exception as e:
            logger.error(f"Failed to initialize local fallback: {e}")

    async def make_decision(self, request: AIDecisionRequest) -> AIDecisionResponse:
        """Make AI-powered decision with enhanced reliability"""
        start_time = time.time()

        try:
            logger.info(f"Processing decision request: {request.decision_id} ({request.domain.value})")

            # Validate request
            if not self._validate_request(request):
                return self._create_error_response(
                    request,
                    "Invalid request parameters",
                    time.time() - start_time
                )

            # Choose strategy based on requirements
            if request.require_consensus:
                return await self._make_consensus_decision(request)
            else:
                return await self._make_single_decision(request)

        except Exception as e:
            logger.error(f"Decision making failed for {request.decision_id}: {e}")
            return self._create_error_response(request, str(e), time.time() - start_time)

    def _validate_request(self, request: AIDecisionRequest) -> bool:
        """Validate decision request parameters"""
        try:
            # Check required fields
            if not request.decision_id or not request.context:
                return False

            # Validate domain and complexity
            if not isinstance(request.domain, DecisionDomain):
                return False

            if not isinstance(request.complexity, DecisionComplexity):
                return False

            # Check context size (prevent token overflow)
            context_str = json.dumps(request.context)
            if len(context_str) > 10000:  # Limit context size
                return False

            return True

        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            return False

    async def _make_single_decision(self, request: AIDecisionRequest) -> AIDecisionResponse:
        """Make decision using best available provider with fallback"""
        start_time = time.time()

        # Select best provider based on performance and availability
        provider = self._select_best_provider(request.complexity)

        if not provider:
            # Use local fallback
            return await self._make_local_decision(request, start_time)

        # Attempt decision with retries and fallback
        for attempt in range(self.max_retries):
            try:
                if provider == AIProvider.LOCAL_FALLBACK:
                    return await self._make_local_decision(request, start_time)

                response = await self._query_provider(provider, request)

                # Update performance metrics
                execution_time = time.time() - start_time
                self._update_provider_performance(provider, True, response.confidence)

                # Store in history
                self.decision_history.append(response)
                if len(self.decision_history) > 1000:
                    self.decision_history = self.decision_history[-500:]  # Keep last 500

                return response

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for provider {provider.value}: {e}")
                self._handle_provider_failure(self.providers[provider])

                if attempt < self.max_retries - 1:
                    # Try next provider in fallback chain
                    provider = self._get_next_fallback_provider(provider)
                    if not provider:
                        break

        # All providers failed - use local fallback
        return await self._make_local_decision(request, start_time)

    async def _make_local_decision(self, request: AIDecisionRequest, start_time: float) -> AIDecisionResponse:
        """Make decision using local decision engine"""
        try:
            logger.info(f"Using local decision engine for {request.decision_id}")

            # Get domain-specific decision handler
            decision_handler = self.local_decision_rules.get(
                request.domain,
                self._local_generic_decision
            )

            # Execute local decision logic
            decision_data = await decision_handler(request)

            execution_time = time.time() - start_time

            return AIDecisionResponse(
                decision_id=request.decision_id,
                decision=decision_data["decision"],
                confidence=decision_data["confidence"],
                reasoning=decision_data["reasoning"],
                alternatives=decision_data["alternatives"],
                risk_factors=decision_data["risk_factors"],
                recommendations=decision_data["recommendations"],
                provider_used="local_fallback",
                execution_time_ms=execution_time * 1000,
                token_usage={"total_tokens": 0},
                metadata={
                    "local_engine": True,
                    "domain": request.domain.value,
                    "complexity": request.complexity.value
                }
            )

        except Exception as e:
            logger.error(f"Local decision failed: {e}")
            return self._create_error_response(request, f"Local decision failed: {e}", time.time() - start_time)

    # Local decision handlers
    async def _local_security_analysis(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local security analysis decision logic"""
        context = request.context

        # Analyze threat indicators
        threat_score = 0
        risk_factors = []

        if "vulnerability" in context:
            vuln = context["vulnerability"]
            cvss_score = vuln.get("cvss_score", 0)
            threat_score += cvss_score / 10

            if cvss_score >= 7.0:
                risk_factors.append("High CVSS score vulnerability detected")

        if "network_activity" in context:
            activity = context["network_activity"]
            if activity.get("external_connections", 0) > 10:
                threat_score += 0.3
                risk_factors.append("High volume of external connections")

        # Make decision based on threat score
        if threat_score >= 0.8:
            decision = "block"
            confidence = 0.9
        elif threat_score >= 0.5:
            decision = "investigate"
            confidence = 0.7
        else:
            decision = "monitor"
            confidence = 0.6

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": [
                f"Threat score calculated: {threat_score:.2f}",
                f"Analysis based on {len(context)} security indicators"
            ],
            "alternatives": ["manual_review", "automated_response"],
            "risk_factors": risk_factors,
            "recommendations": [
                "Implement additional monitoring",
                "Update security controls",
                "Review access policies"
            ]
        }

    async def _local_threat_assessment(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local threat assessment logic"""
        context = request.context

        # Basic threat classification
        threat_indicators = context.get("indicators", [])
        threat_level = "low"
        confidence = 0.5

        high_risk_indicators = ["malware", "exploit", "c2", "exfiltration"]
        medium_risk_indicators = ["scanning", "reconnaissance", "bruteforce"]

        high_count = sum(1 for indicator in threat_indicators if any(risk in str(indicator).lower() for risk in high_risk_indicators))
        medium_count = sum(1 for indicator in threat_indicators if any(risk in str(indicator).lower() for risk in medium_risk_indicators))

        if high_count > 0:
            threat_level = "high"
            confidence = 0.8
        elif medium_count > 2:
            threat_level = "medium"
            confidence = 0.7

        return {
            "decision": f"threat_level_{threat_level}",
            "confidence": confidence,
            "reasoning": [
                f"Analyzed {len(threat_indicators)} threat indicators",
                f"High-risk indicators: {high_count}",
                f"Medium-risk indicators: {medium_count}"
            ],
            "alternatives": ["escalate_to_analyst", "automated_containment"],
            "risk_factors": [f"Threat level assessed as {threat_level}"],
            "recommendations": [
                "Enhance monitoring for similar patterns",
                "Update threat intelligence feeds",
                "Review detection rules"
            ]
        }

    async def _local_incident_response(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local incident response decision logic"""
        context = request.context

        incident_type = context.get("incident_type", "unknown")
        severity = context.get("severity", "medium")
        affected_systems = context.get("affected_systems", [])

        # Determine response action
        if severity == "critical" or len(affected_systems) > 10:
            decision = "immediate_containment"
            confidence = 0.9
        elif severity == "high":
            decision = "rapid_response"
            confidence = 0.8
        else:
            decision = "standard_investigation"
            confidence = 0.6

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": [
                f"Incident type: {incident_type}",
                f"Severity level: {severity}",
                f"Affected systems: {len(affected_systems)}"
            ],
            "alternatives": ["manual_investigation", "automated_remediation"],
            "risk_factors": [f"Incident severity: {severity}"],
            "recommendations": [
                "Activate incident response team",
                "Preserve evidence for analysis",
                "Implement containment measures",
                "Notify stakeholders as appropriate"
            ]
        }

    async def _local_vuln_prioritization(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local vulnerability prioritization logic"""
        context = request.context
        vulnerabilities = context.get("vulnerabilities", [])

        # Priority scoring
        priority_scores = []
        for vuln in vulnerabilities:
            score = 0
            cvss = vuln.get("cvss_score", 0)
            exploitability = vuln.get("exploitability", "unknown")
            asset_criticality = vuln.get("asset_criticality", "medium")

            score += cvss * 10  # CVSS base score

            if exploitability == "high":
                score += 30
            elif exploitability == "medium":
                score += 15

            if asset_criticality == "critical":
                score += 25
            elif asset_criticality == "high":
                score += 15

            priority_scores.append((vuln.get("id", "unknown"), score))

        # Sort by priority
        priority_scores.sort(key=lambda x: x[1], reverse=True)

        if priority_scores:
            top_vuln_score = priority_scores[0][1]
            if top_vuln_score > 80:
                decision = "immediate_patching"
                confidence = 0.9
            elif top_vuln_score > 50:
                decision = "scheduled_patching"
                confidence = 0.8
            else:
                decision = "monitor_and_assess"
                confidence = 0.6
        else:
            decision = "no_action_required"
            confidence = 0.5

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": [
                f"Analyzed {len(vulnerabilities)} vulnerabilities",
                f"Highest priority score: {priority_scores[0][1] if priority_scores else 0}",
                "Prioritization based on CVSS, exploitability, and asset criticality"
            ],
            "alternatives": ["risk_acceptance", "compensating_controls"],
            "risk_factors": [f"Top {min(3, len(priority_scores))} vulnerabilities require attention"],
            "recommendations": [
                "Implement patching schedule",
                "Deploy compensating controls",
                "Monitor for exploitation attempts",
                "Update vulnerability scanning frequency"
            ]
        }

    async def _local_attack_simulation(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local attack simulation decision logic"""
        context = request.context

        simulation_type = context.get("simulation_type", "basic")
        target_environment = context.get("target_environment", "development")
        objectives = context.get("objectives", [])

        # Determine simulation approach
        if target_environment == "production":
            decision = "limited_simulation"
            confidence = 0.7
            risk_factors = ["Production environment requires careful testing"]
        elif simulation_type == "advanced":
            decision = "comprehensive_simulation"
            confidence = 0.8
            risk_factors = ["Advanced simulation may trigger security alerts"]
        else:
            decision = "standard_simulation"
            confidence = 0.6
            risk_factors = []

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": [
                f"Simulation type: {simulation_type}",
                f"Target environment: {target_environment}",
                f"Objectives: {len(objectives)}"
            ],
            "alternatives": ["tabletop_exercise", "red_team_engagement"],
            "risk_factors": risk_factors,
            "recommendations": [
                "Define clear rules of engagement",
                "Implement safety controls",
                "Coordinate with operations team",
                "Document all activities"
            ]
        }

    async def _local_compliance_validation(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local compliance validation logic"""
        context = request.context

        framework = context.get("framework", "unknown")
        controls = context.get("controls", [])
        findings = context.get("findings", [])

        # Calculate compliance score
        total_controls = len(controls)
        compliant_controls = len([c for c in controls if c.get("status") == "compliant"])

        if total_controls > 0:
            compliance_score = compliant_controls / total_controls
        else:
            compliance_score = 0.0

        if compliance_score >= 0.95:
            decision = "compliant"
            confidence = 0.9
        elif compliance_score >= 0.80:
            decision = "mostly_compliant"
            confidence = 0.7
        else:
            decision = "non_compliant"
            confidence = 0.8

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": [
                f"Framework: {framework}",
                f"Compliance score: {compliance_score:.2%}",
                f"Compliant controls: {compliant_controls}/{total_controls}"
            ],
            "alternatives": ["risk_acceptance", "remediation_plan"],
            "risk_factors": [f"Non-compliant controls: {total_controls - compliant_controls}"],
            "recommendations": [
                "Address non-compliant controls",
                "Implement monitoring procedures",
                "Schedule regular assessments",
                "Update policies and procedures"
            ]
        }

    async def _local_architecture_optimization(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Local architecture optimization logic"""
        context = request.context

        current_architecture = context.get("current_architecture", {})
        performance_metrics = context.get("performance_metrics", {})
        security_requirements = context.get("security_requirements", [])

        # Basic optimization recommendations
        optimizations = []

        if performance_metrics.get("cpu_usage", 0) > 80:
            optimizations.append("Scale compute resources")

        if performance_metrics.get("memory_usage", 0) > 85:
            optimizations.append("Optimize memory allocation")

        if len(security_requirements) > len(current_architecture.get("security_controls", [])):
            optimizations.append("Enhance security controls")

        if optimizations:
            decision = "optimization_recommended"
            confidence = 0.8
        else:
            decision = "architecture_acceptable"
            confidence = 0.6

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": [
                f"Current architecture components: {len(current_architecture)}",
                f"Performance issues identified: {len(optimizations)}",
                "Analysis based on resource utilization and security requirements"
            ],
            "alternatives": ["maintain_current", "major_redesign"],
            "risk_factors": ["Performance bottlenecks may impact operations"],
            "recommendations": optimizations + [
                "Implement monitoring and alerting",
                "Plan for scalability",
                "Regular architecture reviews"
            ]
        }

    async def _local_generic_decision(self, request: AIDecisionRequest) -> Dict[str, Any]:
        """Generic local decision handler"""
        return {
            "decision": "manual_review_required",
            "confidence": 0.3,
            "reasoning": [
                "No specific decision logic available for this domain",
                "Defaulting to manual review process"
            ],
            "alternatives": ["escalate_to_expert", "defer_decision"],
            "risk_factors": ["Limited automated analysis capability"],
            "recommendations": [
                "Engage subject matter experts",
                "Develop domain-specific decision logic",
                "Implement additional data collection"
            ]
        }

    # Enhanced provider management and selection methods
    def _select_best_provider(self, complexity: DecisionComplexity) -> Optional[AIProvider]:
        """Select the best provider based on complexity and performance"""
        available_providers = [
            p for p in self.fallback_chain
            if p in self.providers and self.providers[p].circuit_breaker_state != "open"
        ]

        if not available_providers:
            return AIProvider.LOCAL_FALLBACK

        # Score providers based on performance and suitability
        provider_scores = {}

        for provider in available_providers:
            config = self.providers[provider]
            performance = self.provider_performance.get(provider, {})

            score = 0.0

            # Performance metrics (0-40 points)
            success_rate = performance.get("success_rate", 0.5)
            avg_response_time = performance.get("avg_response_time", 5.0)
            score += success_rate * 30
            score += max(0, 10 - avg_response_time) * 1

            # Complexity suitability (0-30 points)
            complexity_scores = {
                DecisionComplexity.SIMPLE: {
                    AIProvider.OPENROUTER_QWEN: 30,
                    AIProvider.OPENROUTER_DEEPSEEK: 25,
                    AIProvider.NVIDIA_QWEN: 20,
                    AIProvider.OPENROUTER_ANTHROPIC: 15,
                    AIProvider.NVIDIA_LLAMA: 10,
                    AIProvider.LOCAL_FALLBACK: 5
                },
                DecisionComplexity.MODERATE: {
                    AIProvider.OPENROUTER_ANTHROPIC: 30,
                    AIProvider.NVIDIA_LLAMA: 25,
                    AIProvider.OPENROUTER_QWEN: 20,
                    AIProvider.OPENROUTER_DEEPSEEK: 15,
                    AIProvider.NVIDIA_QWEN: 10,
                    AIProvider.LOCAL_FALLBACK: 5
                },
                DecisionComplexity.COMPLEX: {
                    AIProvider.NVIDIA_LLAMA: 30,
                    AIProvider.OPENROUTER_ANTHROPIC: 25,
                    AIProvider.OPENROUTER_QWEN: 15,
                    AIProvider.OPENROUTER_DEEPSEEK: 10,
                    AIProvider.NVIDIA_QWEN: 5,
                    AIProvider.LOCAL_FALLBACK: 5
                },
                DecisionComplexity.CRITICAL: {
                    AIProvider.OPENROUTER_ANTHROPIC: 30,
                    AIProvider.NVIDIA_LLAMA: 25,
                    AIProvider.OPENROUTER_QWEN: 10,
                    AIProvider.OPENROUTER_DEEPSEEK: 5,
                    AIProvider.NVIDIA_QWEN: 0,
                    AIProvider.LOCAL_FALLBACK: 10  # Local fallback is reliable for critical decisions
                }
            }

            score += complexity_scores.get(complexity, {}).get(provider, 0)

            # Availability bonus (0-20 points)
            if config.circuit_breaker_state == "closed":
                score += 20
            elif config.circuit_breaker_state == "half_open":
                score += 10

            provider_scores[provider] = score

        # Return provider with highest score
        if provider_scores:
            best_provider = max(provider_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Selected provider {best_provider.value} with score {provider_scores[best_provider]}")
            return best_provider

        return AIProvider.LOCAL_FALLBACK

    def _get_next_fallback_provider(self, failed_provider: AIProvider) -> Optional[AIProvider]:
        """Get next provider in fallback chain"""
        try:
            current_index = self.fallback_chain.index(failed_provider)
            for i in range(current_index + 1, len(self.fallback_chain)):
                next_provider = self.fallback_chain[i]
                if (next_provider in self.providers and
                    self.providers[next_provider].circuit_breaker_state != "open"):
                    return next_provider
        except (ValueError, IndexError):
            pass

        return AIProvider.LOCAL_FALLBACK

    async def _query_provider(self, provider: AIProvider, request: AIDecisionRequest) -> AIDecisionResponse:
        """Query a specific provider with enhanced error handling"""
        if provider not in self.clients:
            raise Exception(f"Provider {provider.value} not available")

        client = self.clients[provider]
        config = self.providers[provider]

        # Check rate limits and circuit breaker
        if not self._check_rate_limit(config):
            raise Exception(f"Rate limit exceeded for {provider.value}")

        if config.circuit_breaker_state == "open":
            raise Exception(f"Circuit breaker open for {provider.value}")

        # Construct enhanced prompt
        prompt = self._construct_enhanced_prompt(request)

        # Track token usage
        input_tokens = self._count_tokens(prompt)

        try:
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(request.domain)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=request.timeout_seconds
            )

            if not response.choices or not response.choices[0].message.content:
                raise Exception("Empty response from provider")

            content = response.choices[0].message.content
            output_tokens = self._count_tokens(content)

            # Parse response with enhanced validation
            parsed_response = self._parse_ai_response(content, request)

            # Create response object
            ai_response = AIDecisionResponse(
                decision_id=request.decision_id,
                decision=parsed_response.get("decision", "defer"),
                confidence=parsed_response.get("confidence", 0.5),
                reasoning=parsed_response.get("reasoning", []),
                alternatives=parsed_response.get("alternatives", []),
                risk_factors=parsed_response.get("risk_factors", []),
                recommendations=parsed_response.get("recommendations", []),
                provider_used=provider.value,
                execution_time_ms=time.time() * 1000,
                token_usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                metadata={
                    "model": config.model_name,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "circuit_breaker_state": config.circuit_breaker_state
                }
            )

            # Update provider stats
            config.success_count += 1
            config.last_request = time.time()

            return ai_response

        except Exception as e:
            config.error_count += 1
            self._handle_provider_failure(config)
            raise Exception(f"Provider {provider.value} failed: {e}")

    def _construct_enhanced_prompt(self, request: AIDecisionRequest) -> str:
        """Construct enhanced prompt with better structure"""
        prompt = f"""
XORB SECURITY INTELLIGENCE DECISION REQUEST

Decision ID: {request.decision_id}
Domain: {request.domain.value}
Complexity Level: {request.complexity.value}
Priority: {request.priority}
Timestamp: {request.created_at.isoformat()}

CONTEXT ANALYSIS:
{json.dumps(request.context, indent=2)}

CONSTRAINTS:
{chr(10).join(f"• {constraint}" for constraint in request.constraints)}

DECISION REQUIREMENTS:
You must provide a comprehensive security decision following this exact JSON structure:

{{
    "decision": "approve|reject|defer|investigate|block|monitor",
    "confidence": 0.85,
    "reasoning": [
        "Primary analysis point",
        "Supporting evidence",
        "Risk assessment"
    ],
    "alternatives": [
        "Alternative approach 1",
        "Alternative approach 2"
    ],
    "risk_factors": [
        "Identified risk 1",
        "Identified risk 2"
    ],
    "recommendations": [
        "Immediate action 1",
        "Immediate action 2",
        "Long-term strategy"
    ]
}}

ANALYSIS GUIDELINES:
- Consider MITRE ATT&CK framework implications
- Evaluate business impact vs security risk
- Account for regulatory compliance requirements
- Assess operational feasibility
- Provide quantitative risk metrics where possible

Respond ONLY with valid JSON in the specified format.
"""
        return prompt.strip()

    def _get_system_prompt(self, domain: DecisionDomain) -> str:
        """Get enhanced domain-specific system prompt"""
        base_system = """You are XORB-AI, an advanced cybersecurity decision engine with expertise in threat analysis, incident response, vulnerability management, and security operations. You make precise, actionable decisions based on security frameworks, threat intelligence, and risk assessment methodologies."""

        domain_prompts = {
            DecisionDomain.SECURITY_ANALYSIS: """Specialize in technical security analysis using CVSS scoring, vulnerability correlation, and threat landscape assessment. Consider attack vectors, exploitation probability, and defense effectiveness.""",

            DecisionDomain.THREAT_ASSESSMENT: """Expert in threat intelligence analysis using MITRE ATT&CK framework, threat actor profiling, and campaign attribution. Focus on IOC validation, TTP analysis, and threat hunting methodologies.""",

            DecisionDomain.INCIDENT_RESPONSE: """Master of incident response following NIST and SANS frameworks. Prioritize containment strategies, evidence preservation, stakeholder communication, and recovery planning with minimal business disruption.""",

            DecisionDomain.VULNERABILITY_PRIORITIZATION: """Authority on vulnerability management using CVSS v3.1, EPSS scoring, and business context. Consider exploit availability, asset criticality, compensating controls, and patch management windows.""",

            DecisionDomain.ATTACK_SIMULATION: """Specialist in red team exercises and attack simulation design. Focus on realistic attack scenarios, defense testing, purple team integration, and security control validation.""",

            DecisionDomain.COMPLIANCE_VALIDATION: """Expert in regulatory compliance including PCI-DSS, HIPAA, SOX, GDPR, and ISO-27001. Provide gap analysis, control effectiveness assessment, and remediation guidance.""",

            DecisionDomain.ARCHITECTURE_OPTIMIZATION: """Authority on security architecture design using zero-trust principles, defense-in-depth, and operational efficiency. Consider scalability, maintainability, and threat model alignment."""
        }

        return f"{base_system} {domain_prompts.get(domain, '')}"

    def _parse_ai_response(self, content: str, request: AIDecisionRequest) -> Dict[str, Any]:
        """Parse AI response with enhanced validation"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Validate and sanitize response
                validated_response = self._validate_and_sanitize_response(parsed)
                return validated_response

            # Fallback parsing for non-JSON responses
            return self._fallback_parse_response(content)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._fallback_parse_response(content)
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return self._fallback_parse_response(content)

    def _validate_and_sanitize_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize parsed response"""
        # Ensure required fields exist
        validated = {
            "decision": parsed.get("decision", "defer"),
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "reasoning": parsed.get("reasoning", []),
            "alternatives": parsed.get("alternatives", []),
            "risk_factors": parsed.get("risk_factors", []),
            "recommendations": parsed.get("recommendations", [])
        }

        # Validate decision value
        valid_decisions = ["approve", "reject", "defer", "investigate", "block", "monitor"]
        if validated["decision"] not in valid_decisions:
            validated["decision"] = "defer"
            validated["reasoning"].append(f"Invalid decision value corrected to 'defer'")

        # Ensure lists are actually lists
        for list_field in ["reasoning", "alternatives", "risk_factors", "recommendations"]:
            if not isinstance(validated[list_field], list):
                validated[list_field] = [str(validated[list_field])] if validated[list_field] else []

        # Limit list sizes to prevent abuse
        for list_field in ["reasoning", "alternatives", "risk_factors", "recommendations"]:
            validated[list_field] = validated[list_field][:10]  # Max 10 items per list

        return validated

    def _fallback_parse_response(self, content: str) -> Dict[str, Any]:
        """Fallback parsing when structured parsing fails"""
        # Simple heuristic parsing
        content_lower = content.lower()

        # Determine decision based on keywords
        if any(word in content_lower for word in ["approve", "accept", "yes", "proceed"]):
            decision = "approve"
            confidence = 0.6
        elif any(word in content_lower for word in ["reject", "deny", "no", "block"]):
            decision = "reject"
            confidence = 0.6
        elif any(word in content_lower for word in ["investigate", "analyze", "review"]):
            decision = "investigate"
            confidence = 0.5
        else:
            decision = "defer"
            confidence = 0.3

        # Extract sentences as reasoning
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
        reasoning = sentences[:3] if sentences else ["Heuristic parsing of unstructured response"]

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "alternatives": ["manual_review"],
            "risk_factors": ["Unstructured AI response"],
            "recommendations": ["Review original AI response", "Implement structured prompting"]
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.token_encoder:
            try:
                return len(self.token_encoder.encode(text))
            except Exception:
                pass

        # Fallback: rough estimation (4 chars per token)
        return max(1, len(text) // 4)

    def _check_rate_limit(self, config: ProviderConfig) -> bool:
        """Check if provider is within rate limits"""
        current_time = time.time()

        # Reset counters if more than a minute has passed
        if current_time - config.last_request > 60:
            config.request_count = 0

        # Check RPM limit
        if config.request_count >= config.rate_limit_rpm:
            return False

        config.request_count += 1
        return True

    def _update_provider_performance(self, provider: AIProvider, success: bool, confidence: float):
        """Update provider performance metrics"""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                "success_count": 0,
                "error_count": 0,
                "total_response_time": 0.0,
                "request_count": 0,
                "avg_confidence": 0.0
            }

        metrics = self.provider_performance[provider]
        metrics["request_count"] += 1

        if success:
            metrics["success_count"] += 1
            # Update average confidence with exponential moving average
            if metrics["avg_confidence"] == 0:
                metrics["avg_confidence"] = confidence
            else:
                metrics["avg_confidence"] = (metrics["avg_confidence"] * 0.8) + (confidence * 0.2)
        else:
            metrics["error_count"] += 1

        # Calculate derived metrics
        total_requests = metrics["success_count"] + metrics["error_count"]
        metrics["success_rate"] = metrics["success_count"] / total_requests if total_requests > 0 else 0

    def _create_error_response(self, request: AIDecisionRequest, error_message: str, execution_time: float) -> AIDecisionResponse:
        """Create error response with local fallback attempt"""
        try:
            # Attempt local fallback for error case
            local_response = self._create_local_error_response(request, error_message)
            return local_response
        except Exception:
            # Ultimate fallback
            return AIDecisionResponse(
                decision_id=request.decision_id,
                decision="defer",
                confidence=0.1,
                reasoning=[f"All decision engines failed: {error_message}"],
                alternatives=["manual_analysis"],
                risk_factors=["Complete system failure"],
                recommendations=["Escalate to human experts", "Check system health", "Review logs"],
                provider_used="error_fallback",
                execution_time_ms=execution_time * 1000,
                token_usage={"total_tokens": 0},
                metadata={"error": error_message, "fallback_used": True}
            )

    def _create_local_error_response(self, request: AIDecisionRequest, error_message: str) -> AIDecisionResponse:
        """Create local error response with domain-specific logic"""
        # Use domain-specific safe defaults
        domain_defaults = {
            DecisionDomain.SECURITY_ANALYSIS: ("investigate", 0.5, ["Security analysis requires investigation"]),
            DecisionDomain.THREAT_ASSESSMENT: ("monitor", 0.5, ["Threat assessment requires monitoring"]),
            DecisionDomain.INCIDENT_RESPONSE: ("investigate", 0.7, ["Default to investigation during incidents"]),
            DecisionDomain.VULNERABILITY_PRIORITIZATION: ("defer", 0.4, ["Vulnerability prioritization requires expert review"]),
            DecisionDomain.ATTACK_SIMULATION: ("defer", 0.3, ["Attack simulation requires careful planning"]),
            DecisionDomain.COMPLIANCE_VALIDATION: ("defer", 0.5, ["Compliance validation requires expert review"]),
            DecisionDomain.ARCHITECTURE_OPTIMIZATION: ("defer", 0.4, ["Architecture changes require thorough analysis"])
        }

        decision, confidence, reasoning = domain_defaults.get(
            request.domain,
            ("defer", 0.3, ["Unknown domain requires manual review"])
        )

        return AIDecisionResponse(
            decision_id=request.decision_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning + [f"Local fallback due to: {error_message}"],
            alternatives=["manual_expert_analysis", "escalate_to_security_team"],
            risk_factors=["AI assistance unavailable", "Limited automated analysis"],
            recommendations=["Engage subject matter experts", "Implement backup decision procedures"],
            provider_used="local_error_fallback",
            execution_time_ms=50,  # Fast local response
            token_usage={"total_tokens": 0},
            metadata={"error": error_message, "domain": request.domain.value, "local_fallback": True}
        )

    async def get_orchestrator_health(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator health metrics"""
        try:
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "providers": {},
                "performance": {
                    "total_decisions": len(self.decision_history),
                    "recent_decisions": len([d for d in self.decision_history
                                           if d.created_at > datetime.utcnow() - timedelta(hours=1)]),
                    "avg_confidence": sum(d.confidence for d in self.decision_history[-100:]) / min(100, len(self.decision_history)) if self.decision_history else 0,
                    "avg_response_time": sum(d.execution_time_ms for d in self.decision_history[-100:]) / min(100, len(self.decision_history)) if self.decision_history else 0
                },
                "circuit_breakers": {},
                "fallback_usage": {
                    "local_decisions": len([d for d in self.decision_history if "local" in d.provider_used]),
                    "error_fallbacks": len([d for d in self.decision_history if "error" in d.provider_used])
                }
            }

            # Provider health
            for provider, config in self.providers.items():
                health_data["providers"][provider.value] = {
                    "available": provider in self.clients,
                    "circuit_breaker_state": config.circuit_breaker_state,
                    "success_count": config.success_count,
                    "error_count": config.error_count,
                    "success_rate": config.success_count / max(1, config.success_count + config.error_count)
                }

                health_data["circuit_breakers"][provider.value] = {
                    "state": config.circuit_breaker_state,
                    "failure_count": config.circuit_breaker_failures,
                    "last_failure": config.circuit_breaker_last_failure
                }

            # Overall health assessment
            active_providers = sum(1 for p in self.providers.values() if p.circuit_breaker_state == "closed")
            if active_providers == 0:
                health_data["status"] = "degraded"
            elif active_providers < len(self.providers) // 2:
                health_data["status"] = "warning"

            return health_data

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def cleanup(self):
        """Cleanup orchestrator resources"""
        try:
            # Close all HTTP clients
            for client in self.clients.values():
                if hasattr(client, 'close'):
                    await client.close()

            # Save decision history if needed
            if self.decision_history:
                logger.info(f"Processed {len(self.decision_history)} decisions during session")

            logger.info("Production LLM Orchestrator cleanup completed")

        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}")

# Global orchestrator instance
_production_orchestrator: Optional[ProductionLLMOrchestrator] = None

async def get_production_llm_orchestrator() -> ProductionLLMOrchestrator:
    """Get global production LLM orchestrator instance"""
    global _production_orchestrator

    if _production_orchestrator is None:
        _production_orchestrator = ProductionLLMOrchestrator()
        await _production_orchestrator.initialize()

    return _production_orchestrator

# Enhanced convenience functions with better error handling
async def analyze_security_incident(incident_data: Dict[str, Any], priority: str = "high") -> AIDecisionResponse:
    """Analyze security incident with production reliability"""
    orchestrator = await get_production_llm_orchestrator()

    request = AIDecisionRequest(
        decision_id=f"incident_{uuid.uuid4().hex[:8]}",
        domain=DecisionDomain.INCIDENT_RESPONSE,
        complexity=DecisionComplexity.CRITICAL,
        context=incident_data,
        constraints=["Minimize business impact", "Preserve evidence", "Contain threat"],
        priority=priority,
        timeout_seconds=45
    )

    return await orchestrator.make_decision(request)

async def prioritize_vulnerabilities(vulnerability_data: List[Dict[str, Any]]) -> AIDecisionResponse:
    """Prioritize vulnerabilities with enhanced analysis"""
    orchestrator = await get_production_llm_orchestrator()

    request = AIDecisionRequest(
        decision_id=f"vuln_prioritization_{uuid.uuid4().hex[:8]}",
        domain=DecisionDomain.VULNERABILITY_PRIORITIZATION,
        complexity=DecisionComplexity.MODERATE,
        context={"vulnerabilities": vulnerability_data},
        constraints=["Limited maintenance windows", "Business critical systems", "Resource constraints"],
        priority="medium"
    )

    return await orchestrator.make_decision(request)

async def validate_compliance_controls(framework: str, controls_data: Dict[str, Any]) -> AIDecisionResponse:
    """Validate compliance with enhanced framework support"""
    orchestrator = await get_production_llm_orchestrator()

    request = AIDecisionRequest(
        decision_id=f"compliance_{framework}_{uuid.uuid4().hex[:8]}",
        domain=DecisionDomain.COMPLIANCE_VALIDATION,
        complexity=DecisionComplexity.COMPLEX,
        context={"framework": framework, "controls": controls_data},
        constraints=["Regulatory requirements", "Audit timeline", "Documentation needs"],
        priority="high"
    )

    return await orchestrator.make_decision(request)

# Example usage demonstrating production reliability
if __name__ == "__main__":
    async def production_demo():
        """Demonstrate production LLM orchestrator capabilities"""
        orchestrator = ProductionLLMOrchestrator()
        await orchestrator.initialize()

        # Test with various scenarios
        test_scenarios = [
            {
                "incident_type": "data_exfiltration",
                "affected_systems": ["web_server", "database", "backup_system"],
                "indicators": ["unusual_network_traffic", "privilege_escalation", "data_access_anomaly"],
                "timeline": "ongoing",
                "business_impact": "high",
                "detected_by": "SIEM_alert"
            },
            {
                "incident_type": "malware_detection",
                "affected_systems": ["workstation_fleet"],
                "indicators": ["malware_signature", "c2_communication"],
                "timeline": "contained",
                "business_impact": "medium",
                "detected_by": "endpoint_protection"
            }
        ]

        for i, scenario in enumerate(test_scenarios):
            try:
                response = await analyze_security_incident(scenario, priority="critical")

                print(f"\n=== SCENARIO {i+1} ANALYSIS ===")
                print(f"Decision: {response.decision}")
                print(f"Confidence: {response.confidence:.2f}")
                print(f"Provider: {response.provider_used}")
                print(f"Execution Time: {response.execution_time_ms:.2f}ms")
                print(f"Reasoning: {', '.join(response.reasoning[:2])}")
                print(f"Recommendations: {', '.join(response.recommendations[:3])}")

            except Exception as e:
                print(f"Scenario {i+1} failed: {e}")

        # Get health metrics
        health = await orchestrator.get_orchestrator_health()
        print(f"\n=== ORCHESTRATOR HEALTH ===")
        print(f"Status: {health['status']}")
        print(f"Total Decisions: {health['performance']['total_decisions']}")
        print(f"Active Providers: {sum(1 for p in health['providers'].values() if p['available'])}")
        print(f"Local Fallback Usage: {health['fallback_usage']['local_decisions']}")

        await orchestrator.cleanup()

    asyncio.run(production_demo())
