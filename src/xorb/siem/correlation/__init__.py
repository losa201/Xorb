"""
SIEM correlation module
Event correlation, pattern detection, and threat analysis
"""

from .correlation_engine import (
    CorrelationEngine, 
    CorrelationRule, 
    CorrelationAlert,
    CorrelationRuleType,
    CorrelationSeverity
)

__all__ = [
    'CorrelationEngine',
    'CorrelationRule', 
    'CorrelationAlert',
    'CorrelationRuleType',
    'CorrelationSeverity'
]