import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EnvironmentConfig:
    """Configuration class to manage environment variables"""
    
    @staticmethod
    def get_threat_intel_url() -> str:
        """
        Get the threat intelligence API URL from environment variables
        
        Returns:
            str: The threat intel API URL
            
        Raises:
            ValueError: If the URL is not set in environment variables
        """
        url = os.getenv('THREAT_INTEL_URL')
        if not url:
            raise ValueError("THREAT_INTEL_URL must be set in environment variables")
        return url

    @staticmethod
    def get_api_timeout() -> float:
        """
        Get the API timeout value from environment variables
        
        Returns:
            float: The API timeout in seconds
        """
        return float(os.getenv('API_TIMEOUT', '5.0'))

    @staticmethod
    def get_max_retries() -> int:
        """
        Get the maximum number of retries for API calls
        
        Returns:
            int: Maximum number of retries
        """
        return int(os.getenv('MAX_RETRIES', '3'))

    @staticmethod
    def get_log_level() -> str:
        """
        Get the logging level from environment variables
        
        Returns:
            str: Logging level (INFO, DEBUG, etc.)
        """
        return os.getenv('LOG_LEVEL', 'INFO')

    @staticmethod
    def get_prometheus_enabled() -> bool:
        """
        Get whether Prometheus metrics are enabled
        
        Returns:
            bool: True if Prometheus is enabled, False otherwise
        """
        return os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'