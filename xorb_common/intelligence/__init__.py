"""
Xorb Intelligence Layer
LLM integration and reasoning capabilities for agents
"""

from .llm_gateway import LLMGateway, LLMRequest, LLMResponse, LLMProvider
from .reasoning_engine import ReasoningEngine, ReasoningContext
from .semantic_cache import SemanticCache
from .prompt_templates import PromptTemplateManager

__all__ = [
    'LLMGateway',
    'LLMRequest', 
    'LLMResponse',
    'LLMProvider',
    'ReasoningEngine',
    'ReasoningContext',
    'SemanticCache',
    'PromptTemplateManager'
]

__version__ = "2.0.0"