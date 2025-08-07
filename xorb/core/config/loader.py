import os
import json
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Load and merge configuration from multiple sources"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.default_config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        default_path = Path(self.base_path) / "config" / "default.json"
        try:
            if default_path.exists():
                with open(default_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading default config: {e}")
        return {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration with environment-specific overrides"""
        config_path = Path(self.base_path) / "config" / f"{config_name}.json"
        config = self.default_config.copy()
        
        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    env_config = json.load(f)
                    config.update(env_config)
        except Exception as e:
            print(f"Error loading config {config_name}: {e}")
        
        # Override with environment variables
        return self._apply_env_overrides(config)
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        for key in config:
            env_value = os.getenv(f"XORB_{key.upper()}")
            if env_value is not None:
                config[key] = self._convert_env_value(env_value)
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type"""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass
        return value