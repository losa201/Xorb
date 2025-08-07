from abc import ABC, abstractmethod
from typing import Dict, Any

class AttackPromptStrategy(ABC):
    @abstractmethod
    def generate_cuda_attack(self, context: Dict[str, Any]) -> str:
        """Generate CUDA attack pattern"""
        pass
    
    @abstractmethod
    def generate_speculative_attack(self, context: Dict[str, Any]) -> str:
        """Generate speculative execution attack pattern"""
        pass
    
    @abstractmethod
    def generate_sidechannel_attack(self, context: Dict[str, Any]) -> str:
        """Generate side-channel attack pattern"""
        pass

    @abstractmethod
    def get_attack_types(self) -> list:
        """Get list of supported attack types"""
        pass