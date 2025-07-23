#!/usr/bin/env python3

import asyncio
import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    logging.warning("Local LLM dependencies not available. Using remote-only mode.")

import httpx
from ..llm.client import OpenRouterClient, LLMError


class ModelType(str, Enum):
    REMOTE_PRIMARY = "remote_primary"
    REMOTE_FALLBACK = "remote_fallback"
    LOCAL_FALLBACK = "local_fallback"
    EDGE_OPTIMIZED = "edge_optimized"


class PromptContext(str, Enum):
    VULNERABILITY_ANALYSIS = "vuln_analysis"
    PAYLOAD_GENERATION = "payload_gen"
    RECONNAISSANCE = "recon"
    REPORTING = "reporting"
    GENERAL_SECURITY = "general_sec"


@dataclass
class LLMRequest:
    prompt: str
    context: PromptContext
    max_tokens: int = 512
    temperature: float = 0.7
    priority: int = 5  # 1=highest, 10=lowest
    requires_structured: bool = False
    fallback_allowed: bool = True
    cache_key: Optional[str] = None
    
    def generate_cache_key(self) -> str:
        """Generate cache key for this request"""
        if self.cache_key:
            return self.cache_key
        
        content = f"{self.prompt[:100]}:{self.context.value}:{self.max_tokens}:{self.temperature}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class LLMResponse:
    content: str
    model_used: str
    model_type: ModelType
    tokens_used: int
    response_time_ms: int
    cached: bool = False
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalLLMManager:
    """Manages local LLM instances for fallback scenarios"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", cache_dir: str = "./models"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Force CPU for compatibility
        
        self.logger = logging.getLogger(__name__)
        self.max_context_length = 1024  # Conservative limit for CPU inference
        
        if LOCAL_LLM_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("Local LLM not available - missing dependencies")

    def _initialize_model(self):
        """Initialize local model for CPU inference"""
        try:
            # Use quantization for CPU efficiency
            quantization_config = None
            if hasattr(transformers, 'BitsAndBytesConfig'):
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            
            self.logger.info(f"Loading local model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info("Local LLM loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local LLM: {e}")
            self.model = None
            self.tokenizer = None

    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text using local model"""
        if not self.model or not self.tokenizer:
            raise LLMError("Local model not available")
        
        try:
            # Truncate prompt if too long
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=self.max_context_length, truncation=True)
            
            # Generate with conservative parameters for CPU
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=min(max_tokens, 128),  # Conservative token limit
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Local generation failed: {e}")
            raise LLMError(f"Local model generation error: {e}")

    def is_available(self) -> bool:
        """Check if local model is available"""
        return LOCAL_LLM_AVAILABLE and self.model is not None


class AdaptivePromptOptimizer:
    """Optimizes prompts based on context and performance history"""
    
    def __init__(self):
        self.performance_history: Dict[str, List[float]] = {}
        self.context_templates: Dict[PromptContext, Dict[str, str]] = {}
        self.optimization_cache: Dict[str, str] = {}
        
        self.logger = logging.getLogger(__name__)
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize context-specific prompt templates"""
        self.context_templates = {
            PromptContext.VULNERABILITY_ANALYSIS: {
                "prefix": "As a cybersecurity expert, analyze the following for potential security vulnerabilities:",
                "suffix": "Provide a concise analysis focusing on exploitability and impact.",
                "structured_format": "Return findings in JSON format with: vulnerability_type, severity, description, exploitation_steps"
            },
            
            PromptContext.PAYLOAD_GENERATION: {
                "prefix": "Generate security testing payloads for the following scenario:",
                "suffix": "Focus on safe, ethical testing payloads for authorized security assessment.",
                "structured_format": "Return as JSON array with: payload, description, risk_level, target_component"
            },
            
            PromptContext.RECONNAISSANCE: {
                "prefix": "Perform reconnaissance analysis on the following information:",
                "suffix": "Identify attack surfaces and potential entry points for authorized testing.",
                "structured_format": "Return JSON with: attack_surfaces, entry_points, reconnaissance_techniques, priority_targets"
            },
            
            PromptContext.REPORTING: {
                "prefix": "Generate a professional security assessment report for:",
                "suffix": "Use clear, actionable language suitable for technical and executive audiences.",
                "structured_format": "Structure as: executive_summary, technical_findings, risk_assessment, recommendations"
            },
            
            PromptContext.GENERAL_SECURITY: {
                "prefix": "From a cybersecurity perspective, provide analysis of:",
                "suffix": "Focus on practical security implications and defensive measures.",
                "structured_format": "Return structured analysis covering threats, mitigations, and recommendations"
            }
        }

    async def optimize_prompt(self, request: LLMRequest) -> str:
        """Optimize prompt based on context and historical performance"""
        cache_key = f"{request.context.value}:{request.generate_cache_key()}"
        
        # Check optimization cache
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Get template for context
        template = self.context_templates.get(request.context)
        if not template:
            return request.prompt  # Return original if no template
        
        # Build optimized prompt
        optimized_parts = []
        
        # Add context-specific prefix
        optimized_parts.append(template["prefix"])
        optimized_parts.append("")  # Empty line
        
        # Add the original prompt
        optimized_parts.append(request.prompt)
        optimized_parts.append("")  # Empty line
        
        # Add structured format instruction if requested
        if request.requires_structured:
            optimized_parts.append(template["structured_format"])
            optimized_parts.append("")
        
        # Add context-specific suffix
        optimized_parts.append(template["suffix"])
        
        optimized_prompt = "\n".join(optimized_parts)
        
        # Cache the optimization
        self.optimization_cache[cache_key] = optimized_prompt
        
        self.logger.debug(f"Optimized prompt for {request.context.value}")
        return optimized_prompt

    async def record_performance(self, prompt_hash: str, response_quality: float):
        """Record performance metrics for prompt optimization"""
        if prompt_hash not in self.performance_history:
            self.performance_history[prompt_hash] = []
        
        self.performance_history[prompt_hash].append(response_quality)
        
        # Keep only recent performance data
        if len(self.performance_history[prompt_hash]) > 10:
            self.performance_history[prompt_hash] = self.performance_history[prompt_hash][-10:]

    def get_prompt_performance(self, prompt_hash: str) -> float:
        """Get average performance score for a prompt pattern"""
        history = self.performance_history.get(prompt_hash, [])
        return sum(history) / len(history) if history else 0.5


class HybridLLMClient:
    """Hybrid LLM client with intelligent fallback capabilities"""
    
    def __init__(self, 
                 openrouter_api_key: str,
                 primary_model: str = "anthropic/claude-3-haiku",
                 fallback_model: str = "openai/gpt-3.5-turbo",
                 enable_local_fallback: bool = True):
        
        self.primary_client = OpenRouterClient(openrouter_api_key)
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        
        # Initialize local LLM manager
        self.local_manager = None
        if enable_local_fallback and LOCAL_LLM_AVAILABLE:
            self.local_manager = LocalLLMManager()
        
        # Initialize prompt optimizer
        self.prompt_optimizer = AdaptivePromptOptimizer()
        
        # Response cache
        self.response_cache: Dict[str, Tuple[LLMResponse, datetime]] = {}
        self.cache_ttl_minutes = 60
        
        # Performance tracking
        self.model_performance: Dict[str, List[float]] = {}
        self.model_availability: Dict[str, bool] = {
            self.primary_model: True,
            self.fallback_model: True,
            "local_fallback": self.local_manager.is_available() if self.local_manager else False
        }
        
        self.logger = logging.getLogger(__name__)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response with intelligent model selection and fallback"""
        start_time = time.time()
        
        # Check cache first
        cache_key = request.generate_cache_key()
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            cached_response.cached = True
            return cached_response
        
        # Optimize prompt
        optimized_prompt = await self.prompt_optimizer.optimize_prompt(request)
        
        # Try primary model first
        response = await self._try_generate_with_model(
            optimized_prompt, request, self.primary_model, ModelType.REMOTE_PRIMARY
        )
        
        if response:
            response.response_time_ms = int((time.time() - start_time) * 1000)
            self._cache_response(cache_key, response)
            return response
        
        # Try remote fallback
        if request.fallback_allowed:
            self.logger.warning(f"Primary model failed, trying fallback: {self.fallback_model}")
            response = await self._try_generate_with_model(
                optimized_prompt, request, self.fallback_model, ModelType.REMOTE_FALLBACK
            )
            
            if response:
                response.response_time_ms = int((time.time() - start_time) * 1000)
                self._cache_response(cache_key, response)
                return response
        
        # Try local fallback as last resort
        if request.fallback_allowed and self.local_manager and self.local_manager.is_available():
            self.logger.warning("Remote models failed, trying local fallback")
            try:
                local_content = await self.local_manager.generate(
                    optimized_prompt, request.max_tokens, request.temperature
                )
                
                response = LLMResponse(
                    content=local_content,
                    model_used="local_fallback",
                    model_type=ModelType.LOCAL_FALLBACK,
                    tokens_used=len(local_content.split()),  # Rough estimate
                    response_time_ms=int((time.time() - start_time) * 1000),
                    confidence_score=0.7,  # Lower confidence for local model
                    metadata={"fallback_reason": "remote_unavailable"}
                )
                
                self._cache_response(cache_key, response)
                return response
                
            except Exception as e:
                self.logger.error(f"Local fallback failed: {e}")
        
        # All models failed
        raise LLMError("All LLM models unavailable")

    async def _try_generate_with_model(self, 
                                     prompt: str, 
                                     request: LLMRequest, 
                                     model: str, 
                                     model_type: ModelType) -> Optional[LLMResponse]:
        """Try to generate with a specific model"""
        if not self.model_availability.get(model, False):
            return None
        
        try:
            # Use the existing OpenRouter client
            response_text = await self.primary_client.generate_text(
                prompt, 
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            # Estimate token usage (rough)
            estimated_tokens = len(response_text.split()) + len(prompt.split())
            
            response = LLMResponse(
                content=response_text,
                model_used=model,
                model_type=model_type,
                tokens_used=estimated_tokens,
                response_time_ms=0,  # Will be set by caller
                confidence_score=self._calculate_confidence_score(response_text, model_type)
            )
            
            # Mark model as available
            self.model_availability[model] = True
            
            return response
            
        except Exception as e:
            self.logger.error(f"Model {model} failed: {e}")
            
            # Mark model as unavailable temporarily
            self.model_availability[model] = False
            
            # Schedule availability check
            asyncio.create_task(self._check_model_availability_later(model))
            
            return None

    def _calculate_confidence_score(self, content: str, model_type: ModelType) -> float:
        """Calculate confidence score based on content and model type"""
        base_score = {
            ModelType.REMOTE_PRIMARY: 1.0,
            ModelType.REMOTE_FALLBACK: 0.9,
            ModelType.LOCAL_FALLBACK: 0.7,
            ModelType.EDGE_OPTIMIZED: 0.8
        }.get(model_type, 0.5)
        
        # Adjust based on response characteristics
        if len(content) < 10:  # Very short response
            base_score *= 0.5
        elif "error" in content.lower() or "sorry" in content.lower():
            base_score *= 0.6
        elif len(content) > 100:  # Substantial response
            base_score *= 1.1
        
        return min(1.0, max(0.0, base_score))

    async def _check_model_availability_later(self, model: str, delay_minutes: int = 5):
        """Check model availability after a delay"""
        await asyncio.sleep(delay_minutes * 60)
        
        try:
            # Test with a simple prompt
            test_response = await self.primary_client.generate_text(
                "Test connectivity", model=model, max_tokens=10
            )
            
            if test_response:
                self.model_availability[model] = True
                self.logger.info(f"Model {model} is available again")
            
        except Exception:
            self.logger.debug(f"Model {model} still unavailable")

    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if still valid"""
        if cache_key not in self.response_cache:
            return None
        
        response, cached_at = self.response_cache[cache_key]
        
        # Check if cache is still valid
        if datetime.now() - cached_at > timedelta(minutes=self.cache_ttl_minutes):
            del self.response_cache[cache_key]
            return None
        
        return response

    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache response with timestamp"""
        self.response_cache[cache_key] = (response, datetime.now())
        
        # Clean old cache entries
        if len(self.response_cache) > 1000:  # Limit cache size
            self._cleanup_cache()

    def _cleanup_cache(self):
        """Remove expired cache entries"""
        cutoff_time = datetime.now() - timedelta(minutes=self.cache_ttl_minutes)
        
        expired_keys = [
            key for key, (_, cached_at) in self.response_cache.items()
            if cached_at < cutoff_time
        ]
        
        for key in expired_keys:
            del self.response_cache[key]

    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all available models"""
        return {
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "model_availability": self.model_availability.copy(),
            "local_llm_available": self.local_manager.is_available() if self.local_manager else False,
            "cached_responses": len(self.response_cache),
            "optimization_cache_size": len(self.prompt_optimizer.optimization_cache)
        }

    async def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process multiple requests efficiently"""
        # Sort by priority (higher priority = lower number)
        sorted_requests = sorted(requests, key=lambda r: r.priority)
        
        # Process in batches to avoid overwhelming the service
        batch_size = 5
        responses = []
        
        for i in range(0, len(sorted_requests), batch_size):
            batch = sorted_requests[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [self.generate(request) for request in batch]
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Batch request {i+j} failed: {response}")
                    # Create error response
                    error_response = LLMResponse(
                        content=f"Error: {response}",
                        model_used="error",
                        model_type=ModelType.REMOTE_PRIMARY,
                        tokens_used=0,
                        response_time_ms=0,
                        confidence_score=0.0
                    )
                    responses.append(error_response)
                else:
                    responses.append(response)
            
            # Small delay between batches
            if i + batch_size < len(sorted_requests):
                await asyncio.sleep(0.5)
        
        return responses

    async def shutdown(self):
        """Clean shutdown of the hybrid client"""
        self.logger.info("Shutting down hybrid LLM client")
        
        # Clear caches
        self.response_cache.clear()
        self.prompt_optimizer.optimization_cache.clear()
        
        # Clean up local model if loaded
        if self.local_manager and self.local_manager.model:
            del self.local_manager.model
            del self.local_manager.tokenizer


# Convenience functions for different security contexts

async def analyze_vulnerability(client: HybridLLMClient, 
                               vulnerability_data: str, 
                               structured: bool = True) -> LLMResponse:
    """Analyze vulnerability data with security-optimized prompt"""
    request = LLMRequest(
        prompt=vulnerability_data,
        context=PromptContext.VULNERABILITY_ANALYSIS,
        max_tokens=512,
        temperature=0.3,  # Lower temperature for analytical tasks
        requires_structured=structured,
        priority=2  # High priority for security analysis
    )
    return await client.generate(request)


async def generate_payloads(client: HybridLLMClient, 
                           target_description: str,
                           payload_count: int = 5) -> LLMResponse:
    """Generate security testing payloads"""
    request = LLMRequest(
        prompt=f"Generate {payload_count} security testing payloads for: {target_description}",
        context=PromptContext.PAYLOAD_GENERATION,
        max_tokens=1024,
        temperature=0.8,  # Higher creativity for payload generation
        requires_structured=True,
        priority=3
    )
    return await client.generate(request)


async def perform_reconnaissance(client: HybridLLMClient, 
                                target_info: str) -> LLMResponse:
    """Perform reconnaissance analysis"""
    request = LLMRequest(
        prompt=target_info,
        context=PromptContext.RECONNAISSANCE,
        max_tokens=768,
        temperature=0.4,
        requires_structured=True,
        priority=4
    )
    return await client.generate(request)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_hybrid_llm():
        """Demonstrate hybrid LLM capabilities"""
        
        # Mock API key for demo
        api_key = "demo_key"
        
        client = HybridLLMClient(
            openrouter_api_key=api_key,
            enable_local_fallback=True
        )
        
        print("=== Hybrid LLM Client Demo ===")
        print(f"Local LLM available: {LOCAL_LLM_AVAILABLE}")
        
        # Show model status
        status = await client.get_model_status()
        print(f"Model status: {json.dumps(status, indent=2)}")
        
        # Test different contexts
        test_cases = [
            ("SQL injection in login form with parameter 'username'", PromptContext.VULNERABILITY_ANALYSIS),
            ("WordPress site with admin panel exposed", PromptContext.RECONNAISSANCE),
            ("Generate XSS payloads for input validation bypass", PromptContext.PAYLOAD_GENERATION),
        ]
        
        for prompt_text, context in test_cases:
            print(f"\n=== Testing {context.value} ===")
            
            request = LLMRequest(
                prompt=prompt_text,
                context=context,
                max_tokens=200,
                requires_structured=True
            )
            
            try:
                # This would normally work with real API keys
                print(f"Request: {prompt_text}")
                print("(Demo mode - would generate with hybrid model selection)")
                
            except Exception as e:
                print(f"Error (expected in demo): {e}")
        
        await client.shutdown()
    
    if "--demo" in sys.argv:
        asyncio.run(demo_hybrid_llm())
    elif "--install-deps" in sys.argv:
        print("To install local LLM dependencies:")
        print("pip install torch transformers bitsandbytes accelerate")
    else:
        print("XORB Hybrid LLM Client")
        print("Usage:")
        print("  python hybrid_client.py --demo        # Run demo")
        print("  python hybrid_client.py --install-deps # Show installation instructions")