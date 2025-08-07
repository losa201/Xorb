"""
SIEM (Security Information and Event Management) module
Provides log ingestion, normalization, correlation, and alerting capabilities
"""

from .ingestion.log_parser import LogParser, LogParserFactory
from .ingestion.event_normalizer import EventNormalizer, NormalizedEvent
from .ingestion.stream_processor import StreamProcessor
from .correlation.correlation_engine import CorrelationEngine
from .correlation.rule_manager import RuleManager
from .correlation.threat_detector import ThreatDetector
from .alerting.alert_manager import AlertManager
from .alerting.notification_service import NotificationService

__all__ = [
    'LogParser',
    'LogParserFactory', 
    'EventNormalizer',
    'NormalizedEvent',
    'StreamProcessor',
    'CorrelationEngine',
    'RuleManager',
    'ThreatDetector',
    'AlertManager',
    'NotificationService'
]