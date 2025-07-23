#!/usr/bin/env python3

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field, validator


class ModelProvider(str, Enum):
    ANTHROPIC_CLAUDE = "anthropic/claude-3-haiku-20240307"
    OPENAI_GPT4 = "openai/gpt-4-turbo-preview" 
    GOOGLE_GEMINI = "google/gemini-pro"
    KIMI_K2 = "moonshot/moonshot-v1-8k"
    QWEN = "qwen/qwen-2-7b-instruct"


@dataclass
class PromptTemplate:
    name: str
    template: str
    model: ModelProvider
    temperature: float = 0.7
    max_tokens: int = 4000
    expected_output_type: str = "json"


class LLMRequest(BaseModel):
    prompt: str
    model: ModelProvider = ModelProvider.ANTHROPIC_CLAUDE
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=1, le=8000)
    system_message: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    @validator('prompt')
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class LLMResponse(BaseModel):
    content: str
    model_used: str
    tokens_used: int
    cost: float = 0.0
    processing_time: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SecurityResearch(BaseModel):
    technique_name: str = Field(description="Name of the security technique")
    description: str = Field(description="Detailed description of the technique")
    attack_vectors: List[str] = Field(description="List of attack vectors")
    payloads: List[str] = Field(description="Example payloads or commands")
    mitigation: List[str] = Field(description="Mitigation strategies")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    references: List[str] = Field(default_factory=list, description="External references")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class VulnerabilityInfo(BaseModel):
    cve_id: Optional[str] = Field(description="CVE identifier if available")
    title: str = Field(description="Vulnerability title")
    description: str = Field(description="Technical description")
    affected_systems: List[str] = Field(description="Affected systems or applications")
    exploitation_complexity: str = Field(description="Complexity: low, medium, high")
    proof_of_concept: Optional[str] = Field(description="Proof of concept code")
    remediation: str = Field(description="Remediation steps")
    cvss_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)


class ThreatIntelligence(BaseModel):
    threat_type: str = Field(description="Type of threat")
    iocs: List[str] = Field(description="Indicators of compromise")
    ttps: List[str] = Field(description="Tactics, techniques, and procedures")
    attribution: Optional[str] = Field(description="Threat actor attribution")
    timeline: str = Field(description="Attack timeline or discovery date")
    impact: str = Field(description="Potential impact assessment")


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limit = 60  # requests per minute
        self.rate_window = 60  # seconds
        self.request_times: List[datetime] = []
        
        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0
        
        # Model pricing (per 1K tokens)
        self.model_pricing = {
            ModelProvider.ANTHROPIC_CLAUDE: {"input": 0.00025, "output": 0.00125},
            ModelProvider.OPENAI_GPT4: {"input": 0.01, "output": 0.03},
            ModelProvider.GOOGLE_GEMINI: {"input": 0.0005, "output": 0.0015},
            ModelProvider.KIMI_K2: {"input": 0.0002, "output": 0.0002},
            ModelProvider.QWEN: {"input": 0.0001, "output": 0.0001}
        }
        
        self._setup_pydantic_agents()

    def _setup_pydantic_agents(self):
        """Setup Pydantic AI agents for structured outputs"""
        try:
            # Security Research Agent
            self.security_agent = Agent(
                model=OpenAIModel('anthropic/claude-3-haiku-20240307', base_url=self.base_url, api_key=self.api_key),
                result_type=SecurityResearch,
                system_prompt="""You are a cybersecurity expert providing detailed technical analysis of security techniques, vulnerabilities, and defensive measures. Always provide accurate, ethical, and defensive-focused information."""
            )
            
            # Vulnerability Analysis Agent  
            self.vuln_agent = Agent(
                model=OpenAIModel('anthropic/claude-3-haiku-20240307', base_url=self.base_url, api_key=self.api_key),
                result_type=VulnerabilityInfo,
                system_prompt="""You are a vulnerability researcher providing detailed technical analysis of security vulnerabilities for defensive purposes. Focus on accurate technical details and effective remediation."""
            )
            
            # Threat Intelligence Agent
            self.intel_agent = Agent(
                model=OpenAIModel('anthropic/claude-3-haiku-20240307', base_url=self.base_url, api_key=self.api_key),
                result_type=ThreatIntelligence,
                system_prompt="""You are a threat intelligence analyst providing structured analysis of threat actors, attack campaigns, and indicators of compromise for defensive security purposes."""
            )
            
            self.logger.info("Pydantic AI agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Pydantic agents: {e}")

    async def start(self):
        """Initialize the HTTP session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300, ttl_dns_cache_size=100)
            timeout = aiohttp.ClientTimeout(total=120, connect=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://xorb.ai",
                    "X-Title": "XORB Security Platform"
                }
            )
            
            self.logger.info("OpenRouter client session started")

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate_security_research(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> SecurityResearch:
        """Generate structured security research using Pydantic AI"""
        try:
            # Add context to the prompt if provided
            enhanced_prompt = prompt
            if context:
                enhanced_prompt = f"Context: {json.dumps(context, indent=2)}\n\nQuery: {prompt}"
            
            result = await self.security_agent.run(enhanced_prompt)
            
            self.logger.info(f"Generated security research: {result.data.technique_name}")
            return result.data
            
        except Exception as e:
            self.logger.error(f"Failed to generate security research: {e}")
            raise

    async def analyze_vulnerability(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> VulnerabilityInfo:
        """Analyze vulnerability using structured Pydantic output"""
        try:
            enhanced_prompt = prompt
            if context:
                enhanced_prompt = f"Context: {json.dumps(context, indent=2)}\n\nAnalyze: {prompt}"
            
            result = await self.vuln_agent.run(enhanced_prompt)
            
            self.logger.info(f"Analyzed vulnerability: {result.data.title}")
            return result.data
            
        except Exception as e:
            self.logger.error(f"Failed to analyze vulnerability: {e}")
            raise

    async def gather_threat_intelligence(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> ThreatIntelligence:
        """Gather threat intelligence using structured output"""
        try:
            enhanced_prompt = prompt
            if context:
                enhanced_prompt = f"Context: {json.dumps(context, indent=2)}\n\nIntelligence Query: {prompt}"
            
            result = await self.intel_agent.run(enhanced_prompt)
            
            self.logger.info(f"Gathered threat intelligence: {result.data.threat_type}")
            return result.data
            
        except Exception as e:
            self.logger.error(f"Failed to gather threat intelligence: {e}")
            raise

    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Make a raw chat completion request"""
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if not self.session:
            await self.start()
        
        start_time = datetime.utcnow()
        
        try:
            payload = {
                "model": request.model.value,
                "messages": [
                    {"role": "system", "content": request.system_message or "You are a helpful AI assistant focused on cybersecurity."},
                    {"role": "user", "content": request.prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False
            }
            
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
                
                data = await response.json()
                
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                cost = self._calculate_cost(request.model, tokens_used)
                
                self.total_cost += cost
                self.request_count += 1
                
                return LLMResponse(
                    content=content,
                    model_used=request.model.value,
                    tokens_used=tokens_used,
                    cost=cost,
                    processing_time=processing_time,
                    confidence=self._estimate_confidence(content),
                    metadata={
                        "request_id": data.get("id"),
                        "created": data.get("created"),
                        "context": request.context
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise

    async def batch_requests(self, requests: List[LLMRequest], max_concurrent: int = 5) -> List[LLMResponse]:
        """Process multiple requests concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(req: LLMRequest) -> LLMResponse:
            async with semaphore:
                await asyncio.sleep(1)  # Rate limiting between requests
                return await self.chat_completion(req)
        
        tasks = [process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for response in responses:
            if isinstance(response, LLMResponse):
                valid_responses.append(response)
            else:
                self.logger.error(f"Request failed: {response}")
        
        return valid_responses

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        if not self.session:
            await self.start()
        
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    self.logger.error(f"Failed to get models: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error getting models: {e}")
            return []

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        
        # Remove old requests outside the window
        cutoff = now - timedelta(seconds=self.rate_window)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        if len(self.request_times) >= self.rate_limit:
            self.logger.warning("Rate limit reached, request denied")
            return False
        
        self.request_times.append(now)
        return True

    def _calculate_cost(self, model: ModelProvider, tokens: int) -> float:
        """Calculate cost for a request"""
        if model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[model]
        # Simplified cost calculation (assuming equal input/output ratio)
        avg_price_per_1k = (pricing["input"] + pricing["output"]) / 2
        return (tokens / 1000.0) * avg_price_per_1k

    def _estimate_confidence(self, content: str) -> float:
        """Estimate confidence based on content characteristics"""
        if not content:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Longer responses tend to be more detailed
        if len(content) > 500:
            confidence += 0.1
        
        # Presence of technical terms
        technical_terms = ['vulnerability', 'exploit', 'payload', 'attack', 'security', 'CVE', 'CVSS']
        term_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        confidence += min(0.3, term_count * 0.05)
        
        # Presence of structured information
        if any(marker in content for marker in ['```', 'http://', 'https://', 'CVE-']):
            confidence += 0.1
        
        return min(1.0, confidence)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.request_count,
            "total_cost": round(self.total_cost, 4),
            "avg_cost_per_request": round(self.total_cost / max(1, self.request_count), 4),
            "recent_requests": len(self.request_times),
            "rate_limit": self.rate_limit,
            "supported_models": [model.value for model in ModelProvider]
        }