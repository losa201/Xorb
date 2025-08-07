from .agent_core import AgentCore
from .threat_intel_service import ThreatIntelService
from .metrics_collector import MetricsCollector
from .environment_config import EnvironmentConfig
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextAwareAgent:
    """Main agent class that coordinates core functionality, threat intelligence, and metrics."""
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize the ContextAwareAgent.
        
        Args:
            state_size (int): Size of the state vector
            action_size (int): Number of possible actions
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = EnvironmentConfig(config)
        self.metrics = MetricsCollector()
        self.threat_intel = ThreatIntelService(self.config.threat_intel_url)
        self.core = AgentCore(state_size, action_size, self.config)
    
    def act(self, state):
        """
        Choose an action based on the current state.
        
        Args:
            state (np.array): Current state vector
        
        Returns:
            int: Selected action
        """
        try:
            threat_context = self.threat_intel.get_threat_context(state)
            q_values = self.core.predict(state)
            
            # Apply threat context weighting
            weighted_q = q_values * (1 + threat_context['risk_factor'])
            action = np.argmax(weighted_q)
            
            # Record the action type for metrics
            self.metrics.record_action(action)
            return action
            
        except Exception as e:
            logger.error(f"Error in act method: {e}", exc_info=True)
            # Return a safe default action in case of error
            return 0
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether the episode is complete
        """
        self.core.remember(state, action, reward, next_state, done)
    
    def replay(self, batch_size):
        """
        Train the agent using experience replay.
        
        Args:
            batch_size (int): Number of experiences to sample
        """
        return self.core.replay(batch_size)
    
    def get_config(self):
        """Get the current configuration."""
        return self.config.get_config()