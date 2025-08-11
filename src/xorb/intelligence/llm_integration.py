#!/usr/bin/env python3
"""
XORB Strategic LLM Integration
Advanced AI-powered decision making with OpenRouter and NVIDIA APIs
"""

import asyncio
import logging
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Graceful fallback - will use local reasoning
    class MockOpenAI:
        def __init__(self, *args, **kwargs): 
            logger.warning("OpenAI not available - using local reasoning fallbacks")
    
    class MockAsyncOpenAI:
        def __init__(self, *args, **kwargs): 
            logger.warning("AsyncOpenAI not available - using local reasoning fallbacks")
    
    OpenAI = MockOpenAI
    AsyncOpenAI = MockAsyncOpenAI

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - LLM provider connectivity limited")

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENROUTER_QWEN = "openrouter_qwen"
    OPENROUTER_GLM = "openrouter_glm"
    OPENROUTER_HORIZON = "openrouter_horizon"
    OPENROUTER_KIMI = "openrouter_kimi"
    NVIDIA_QWEN = "nvidia_qwen"

class DecisionType(Enum):
    FUSION_VALIDATION = "fusion_validation"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_SELECTION = "strategy_selection"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    THREAT_ANALYSIS = "threat_analysis"

@dataclass
class LLMDecision:
    """Structured LLM decision response."""
    decision: str
    confidence: float
    reasoning: List[str]
    alternatives: List[str]
    risk_factors: List[str]
    recommendations: List[str]

class XORBLLMOrchestrator:
    """Strategic LLM integration for intelligent decision making."""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, Any] = {}
        self.fallback_chain: List[LLMProvider] = [
            LLMProvider.OPENROUTER_QWEN,
            LLMProvider.NVIDIA_QWEN,
            LLMProvider.OPENROUTER_GLM,
            LLMProvider.OPENROUTER_HORIZON,
            LLMProvider.OPENROUTER_KIMI
        ]
        
        # Usage tracking
        self.decision_history: List[Dict[str, Any]] = []
        self.provider_performance: Dict[LLMProvider, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize LLM providers with strategic configuration."""
        logger.info("Initializing XORB LLM Orchestrator with multiple providers")
        
        # OpenRouter providers (secure configuration)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not set - OpenRouter providers will be unavailable")
            return
        
        openrouter_config = {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": openrouter_api_key
        }
        
        self.providers[LLMProvider.OPENROUTER_QWEN] = AsyncOpenAI(
            **openrouter_config
        )
        
        self.providers[LLMProvider.OPENROUTER_GLM] = AsyncOpenAI(
            **openrouter_config
        )
        
        self.providers[LLMProvider.OPENROUTER_HORIZON] = AsyncOpenAI(
            **openrouter_config
        )
        
        self.providers[LLMProvider.OPENROUTER_KIMI] = AsyncOpenAI(
            **openrouter_config
        )
        
        # NVIDIA provider (secure configuration)
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_api_key:
            self.providers[LLMProvider.NVIDIA_QWEN] = AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_api_key
            )
        else:
            logger.warning("NVIDIA_API_KEY not set - NVIDIA provider will be unavailable")
        
        logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    async def make_strategic_decision(self, decision_type: DecisionType, 
                                    context: Dict[str, Any],
                                    require_consensus: bool = False) -> LLMDecision:
        """Make strategic decisions using LLM intelligence."""
        
        prompt = self._generate_decision_prompt(decision_type, context)
        
        if require_consensus:
            return await self._consensus_decision(prompt, decision_type)
        else:
            return await self._single_provider_decision(prompt, decision_type)
    
    async def _single_provider_decision(self, prompt: str, decision_type: DecisionType) -> LLMDecision:
        """Get decision from single best-performing provider with fallback."""
        
        if not OPENAI_AVAILABLE:
            logger.info("OpenAI SDK not available, using local reasoning")
            return self._local_fallback_decision(decision_type)
        
        for provider in self.fallback_chain:
            try:
                logger.info(f"Attempting decision with {provider.value}")
                
                client = self.providers[provider]
                model_name = self._get_model_name(provider)
                
                # Check if this is a mock client
                if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                    try:
                        response = await client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": self._get_system_prompt(decision_type)},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=1000
                        )
                        
                        content = response.choices[0].message.content
                        decision = self._parse_llm_response(content)
                        
                        # Track successful provider
                        self._track_provider_success(provider, decision.confidence)
                        
                        logger.info(f"Decision obtained from {provider.value} with confidence {decision.confidence}")
                        return decision
                    except AttributeError:
                        # This is likely a mock client
                        logger.debug(f"Provider {provider.value} appears to be a mock client")
                        raise Exception("Mock client detected")
                else:
                    raise Exception("Invalid client configuration")
                
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                self._track_provider_failure(provider)
                continue
        
        # Fallback to local reasoning if all providers fail
        logger.warning("All LLM providers failed, using local reasoning")
        return self._local_fallback_decision(decision_type)
    
    async def _consensus_decision(self, prompt: str, decision_type: DecisionType) -> LLMDecision:
        """Get consensus decision from multiple providers."""
        
        decisions = []
        
        for provider in self.fallback_chain[:3]:  # Use top 3 providers for consensus
            try:
                client = self.providers[provider]
                model_name = self._get_model_name(provider)
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(decision_type)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                decision = self._parse_llm_response(content)
                decisions.append((provider, decision))
                
            except Exception as e:
                logger.warning(f"Consensus provider {provider.value} failed: {e}")
        
        if decisions:
            return self._merge_consensus_decisions(decisions)
        else:
            return self._local_fallback_decision(decision_type)
    
    def _generate_decision_prompt(self, decision_type: DecisionType, context: Dict[str, Any]) -> str:
        """Generate contextual prompt for decision type."""
        
        base_context = f"""
XORB Cybersecurity Platform Decision Request
Decision Type: {decision_type.value}
Timestamp: {datetime.utcnow().isoformat()}

Context Data:
{json.dumps(context, indent=2)}
"""
        
        if decision_type == DecisionType.FUSION_VALIDATION:
            return base_context + """
You are evaluating a service fusion plan for the XORB cybersecurity platform.
Analyze the fusion strategy, risk level, and validation criteria.

Consider:
1. Technical feasibility of the fusion
2. Risk mitigation strategies
3. Business impact and value preservation
4. System stability and performance implications
5. Security implications of service consolidation

Provide your decision in JSON format:
{
    "decision": "approve|reject|defer",
    "confidence": 0.0-1.0,
    "reasoning": ["reason1", "reason2", ...],
    "alternatives": ["alternative1", "alternative2", ...],
    "risk_factors": ["risk1", "risk2", ...],
    "recommendations": ["rec1", "rec2", ...]
}
"""
        
        elif decision_type == DecisionType.ARCHITECTURE_ANALYSIS:
            return base_context + """
You are analyzing the architecture optimization opportunities for XORB.
Evaluate service relationships, redundancies, and optimization potential.

Consider:
1. Service interdependencies and coupling
2. Redundancy elimination opportunities
3. Performance optimization potential
4. Security architecture improvements
5. Maintainability and operational efficiency

Provide your analysis in JSON format with decision, confidence, and detailed reasoning.
"""
        
        else:
            return base_context + """
Analyze the provided context and make a strategic decision for the XORB platform.
Consider security, performance, reliability, and business value.
Provide your response in JSON format with decision, confidence, and reasoning.
"""
    
    def _get_system_prompt(self, decision_type: DecisionType) -> str:
        """Get system prompt for decision type."""
        
        base_prompt = """You are XORB-AI, an advanced cybersecurity architecture strategist.
You have deep expertise in:
- Cybersecurity platform architecture
- Service mesh and microservices design
- AMD EPYC processor optimization
- Fault-tolerant distributed systems
- Penetration testing and threat analysis

Always provide strategic, well-reasoned decisions focused on security, performance, and reliability."""
        
        if decision_type == DecisionType.FUSION_VALIDATION:
            return base_prompt + """
You specialize in service fusion and architecture optimization.
Your decisions directly impact system stability and security posture.
Be conservative with high-risk changes, aggressive with clear improvements."""
        
        return base_prompt
    
    def _get_model_name(self, provider: LLMProvider) -> str:
        """Get model name for provider."""
        
        model_map = {
            LLMProvider.OPENROUTER_QWEN: "qwen/qwen-2.5-coder-32b-instruct",
            LLMProvider.OPENROUTER_GLM: "zhipuai/glm-4-9b-chat",
            LLMProvider.OPENROUTER_HORIZON: "openrouter/horizon-beta",
            LLMProvider.OPENROUTER_KIMI: "moonshot/moonshot-v1-8k",
            LLMProvider.NVIDIA_QWEN: "qwen/qwen2.5-coder-32b-instruct"
        }
        
        return model_map.get(provider, "gpt-3.5-turbo")
    
    def _parse_llm_response(self, content: str) -> LLMDecision:
        """Parse LLM response into structured decision."""
        
        try:
            # Try to parse as JSON
            if content.strip().startswith('{'):
                data = json.loads(content)
            else:
                # Extract JSON from markdown or text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            
            return LLMDecision(
                decision=data.get("decision", "defer"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", []),
                alternatives=data.get("alternatives", []),
                risk_factors=data.get("risk_factors", []),
                recommendations=data.get("recommendations", [])
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback parsing
            decision = "approve" if "approve" in content.lower() else "defer"
            confidence = 0.6 if "approve" in content.lower() else 0.3
            
            return LLMDecision(
                decision=decision,
                confidence=confidence,
                reasoning=[f"Parsed from unstructured response: {content[:100]}..."],
                alternatives=[],
                risk_factors=[],
                recommendations=[]
            )
    
    def _merge_consensus_decisions(self, decisions: List[tuple]) -> LLMDecision:
        """Merge multiple decisions into consensus."""
        
        # Count decisions
        decision_counts = {}
        total_confidence = 0
        all_reasoning = []
        all_alternatives = []
        all_risks = []
        all_recommendations = []
        
        for provider, decision in decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            total_confidence += decision.confidence
            all_reasoning.extend(decision.reasoning)
            all_alternatives.extend(decision.alternatives)
            all_risks.extend(decision.risk_factors)
            all_recommendations.extend(decision.recommendations)
        
        # Get consensus decision
        consensus_decision = max(decision_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = total_confidence / len(decisions)
        
        return LLMDecision(
            decision=consensus_decision,
            confidence=avg_confidence,
            reasoning=list(set(all_reasoning)),
            alternatives=list(set(all_alternatives)),
            risk_factors=list(set(all_risks)),
            recommendations=list(set(all_recommendations))
        )
    
    def _local_fallback_decision(self, decision_type: DecisionType) -> LLMDecision:
        """Fallback to local decision logic when LLMs unavailable."""
        
        if decision_type == DecisionType.FUSION_VALIDATION:
            return LLMDecision(
                decision="approve",
                confidence=0.7,
                reasoning=["Local fallback: Conservative approval based on heuristics"],
                alternatives=["Defer to manual review"],
                risk_factors=["LLM providers unavailable"],
                recommendations=["Implement gradual rollout", "Enhanced monitoring"]
            )
        
        return LLMDecision(
            decision="defer",
            confidence=0.5,
            reasoning=["Local fallback: Insufficient context for decision"],
            alternatives=["Manual analysis required"],
            risk_factors=["No AI assistance available"],
            recommendations=["Establish LLM connectivity", "Manual expert review"]
        )
    
    def _track_provider_success(self, provider: LLMProvider, confidence: float):
        """Track successful provider usage."""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {"success": 0, "total": 0, "avg_confidence": 0}
        
        perf = self.provider_performance[provider]
        perf["success"] += 1
        perf["total"] += 1
        perf["avg_confidence"] = (perf["avg_confidence"] + confidence) / 2
    
    def _track_provider_failure(self, provider: LLMProvider):
        """Track provider failure."""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {"success": 0, "total": 0, "avg_confidence": 0}
        
        self.provider_performance[provider]["total"] += 1

# Global LLM orchestrator instance
llm_orchestrator: Optional[XORBLLMOrchestrator] = None

async def initialize_llm_orchestrator() -> XORBLLMOrchestrator:
    """Initialize the global LLM orchestrator."""
    global llm_orchestrator
    llm_orchestrator = XORBLLMOrchestrator()
    await llm_orchestrator.initialize()
    return llm_orchestrator

async def get_llm_orchestrator() -> Optional[XORBLLMOrchestrator]:
    """Get the global LLM orchestrator."""
    return llm_orchestrator