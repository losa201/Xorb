import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PyTorchNetwork(nn.Module):
    """Neural network for agent decision making using PyTorch."""

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the neural network architecture.

        Args:
            input_size: Size of input state vector
            output_size: Size of output action vector
        """
        super(PyTorchNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.Tanh(),
            nn.Linear(24, output_size)
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

    def save(self, path: str) -> None:
        """Save model state to disk."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model state from disk."""
        try:
            self.model.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise
