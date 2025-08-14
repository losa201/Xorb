"""
XORB Penetration Testing as a Service (PTaaS) Modules
"""

__version__ = "1.0.0"

# Import only existing modules
try:
    from .attack_orchestrator import AttackOrchestrator, AttackTarget, AttackComplexity, AttackPhase, AttackVector
    __all__ = ['AttackOrchestrator', 'AttackTarget', 'AttackComplexity', 'AttackPhase', 'AttackVector']
except ImportError:
    __all__ = []

# Future modules can be imported as they are implemented
# from .scanner import NetworkScanner, VulnerabilityScanner
# from .exploit import ExploitEngine, MetasploitIntegration
# from .reporting import ReportGenerator, ComplianceChecker
# from .api import PTaaSAPI
