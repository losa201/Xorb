"""
XORB Core Exceptions

Custom exception hierarchy for XORB platform.
"""


class XORBError(Exception):
    """Base exception for all XORB-related errors."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "XORB_ERROR"
        self.details = details or {}


class ConfigurationError(XORBError):
    """Error in configuration setup."""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class ValidationError(XORBError):
    """Error in data validation."""

    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})


class AgentError(XORBError):
    """Error in agent execution."""

    def __init__(self, message: str, agent_id: str = None, agent_type: str = None):
        super().__init__(message, "AGENT_ERROR", {"agent_id": agent_id, "agent_type": agent_type})


class OrchestrationError(XORBError):
    """Error in orchestration logic."""

    def __init__(self, message: str, campaign_id: str = None):
        super().__init__(message, "ORCHESTRATION_ERROR", {"campaign_id": campaign_id})


class SecurityError(XORBError):
    """Security-related error."""

    def __init__(self, message: str, security_context: str = None):
        super().__init__(message, "SECURITY_ERROR", {"context": security_context})
