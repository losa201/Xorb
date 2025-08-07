import os
import requests
from dotenv import load_dotenv
from prometheus_client import Counter

# Load environment variables
load_dotenv()

# Prometheus metrics
THREAT_CONTEXT_CALLS = Counter('xorb_threat_context_calls_total', 'Threat context API calls', ['status'])

class ThreatIntelService:
    """Service for handling threat intelligence integration"""
    
    def __init__(self):
        """Initialize threat intel service with config from environment variables"""
        self.threat_intel_url = os.getenv('THREAT_INTEL_URL')
        if not self.threat_intel_url:
            raise ValueError("THREAT_INTEL_URL must be set in environment variables")
        self.timeout = float(os.getenv('THREAT_INTEL_TIMEOUT', '5.0'))
    
    def get_threat_context(self, state):
        """
        Get threat context for a given state
        
        Args:
            state: numpy array representing the current state
            
        Returns:
            dict: Threat context including risk factor and attack pattern
        """
        try:
            response = requests.post(
                self.threat_intel_url,
                json={'state': state.tolist()},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                THREAT_CONTEXT_CALLS.labels(status='success').inc()
                return response.json()
            else:
                THREAT_CONTEXT_CALLS.labels(status='error').inc()
                return self._get_default_context()
                
        except requests.exceptions.RequestException as e:
            THREAT_CONTEXT_CALLS.labels(status='exception').inc()
            return self._get_default_context()
    
    def _get_default_context(self):
        """
        Get default threat context when API is unavailable
        
        Returns:
            dict: Default threat context
        """
        return {
            'risk_factor': 0.1,
            'attack_pattern': 'unknown',
            'confidence': 0.5
        }