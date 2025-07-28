"""
XORB AI Threat Hunting Module

This module provides comprehensive AI-driven threat hunting capabilities including
behavioral analysis, anomaly detection, pattern recognition, hypothesis generation,
and automated threat hunting workflows.
"""

from .ai_threat_hunter import (
    AIHypothesisGenerator,
    AIThreatHunter,
    Anomaly,
    AnomalyType,
    BehavioralAnomalyDetector,
    ConfidenceLevel,
    DataPoint,
    HuntSession,
    HuntTrigger,
    IAnomalyDetector,
    StatisticalAnomalyDetector,
    ThreatHypothesis,
    ThreatType,
    ai_threat_hunter,
    get_ai_threat_hunter,
    initialize_ai_threat_hunting,
    shutdown_ai_threat_hunting,
)

__all__ = [
    "DataPoint",
    "Anomaly",
    "ThreatHypothesis",
    "HuntSession",
    "HuntTrigger",
    "ThreatType",
    "ConfidenceLevel",
    "AnomalyType",
    "IAnomalyDetector",
    "StatisticalAnomalyDetector",
    "BehavioralAnomalyDetector",
    "AIHypothesisGenerator",
    "AIThreatHunter",
    "ai_threat_hunter",
    "initialize_ai_threat_hunting",
    "shutdown_ai_threat_hunting",
    "get_ai_threat_hunter"
]

__version__ = "2.0.0"
