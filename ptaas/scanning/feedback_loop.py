import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
import json
from collections import deque

@dataclass
class FeedbackSignal:
    """Represents a feedback signal from the system under test"""
    signal_type: str  # Type of signal (e.g., "block", "detect", "mitigate")
    timestamp: float
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return {
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }

class AdaptiveSimulationEngine:
    """Implements real-time feedback loop for adaptive attack simulations"""

    def __init__(self, update_interval: float = 1.0):
        self.logger = logging.getLogger(__name__)
        self.update_interval = update_interval
        self.signal_history = deque(maxlen=100)  # Store last 100 signals
        self.attack_patterns = {}
        self.current_strategy = None
        self.signal_handlers = {}
        self.adaptation_rules = []
        self._running = False
        self._adaptation_task = None

        # Initialize with default attack patterns
        self._load_default_attack_patterns()

        # Register default signal handlers
        self.register_signal_handler("block", self._handle_block_signal)
        self.register_signal_handler("detect", self._handle_detect_signal)
        self.register_signal_handler("mitigate", self._handle_mitigate_signal)

        # Register default adaptation rules
        self._setup_default_adaptation_rules()

    def _load_default_attack_patterns(self):
        """Load default attack patterns for simulation"""
        # In a real implementation, this would load from a database or API
        # For now, we'll use mock data
        self.attack_patterns = {
            "initial-access": [{
                "id": "AP-001",
                "name": "Phishing Email",
                "description": "Simulate phishing email attack",
                "tactics": ["initial-access"],
                "complexity": 2,
                "success_rate": 0.7
            }],
            "execution": [{
                "id": "AP-002",
                "name": "Malicious Script",
                "description": "Simulate script-based execution",
                "tactics": ["execution"],
                "complexity": 3,
                "success_rate": 0.6
            }],
            "persistence": [{
                "id": "AP-003",
                "name": "Scheduled Task",
                "description": "Simulate persistence via scheduled task",
                "tactics": ["persistence"],
                "complexity": 4,
                "success_rate": 0.5
            }],
            "privilege-escalation": [{
                "id": "AP-004",
                "name": "Exploit Vulnerability",
                "description": "Simulate privilege escalation via vulnerability",
                "tactics": ["privilege-escalation"],
                "complexity": 5,
                "success_rate": 0.4
            }],
            "defense-evasion": [{
                "id": "AP-005",
                "name": "Obfuscated Payload",
                "description": "Simulate obfuscated payload delivery",
                "tactics": ["defense-evasion"],
                "complexity": 4,
                "success_rate": 0.5
            }],
            "credential-access": [{
                "id": "AP-006",
                "name": "Credential Dumping",
                "description": "Simulate credential dumping attack",
                "tactics": ["credential-access"],
                "complexity": 5,
                "success_rate": 0.3
            }]
        }

    def register_signal_handler(self, signal_type: str, handler: Callable[[FeedbackSignal], None]):
        """Register a handler for a specific signal type"""
        self.signal_handlers[signal_type] = handler

    def _setup_default_adaptation_rules(self):
        """Set up default adaptation rules based on feedback signals"""
        # Rule format: (signal_type, confidence_threshold, adaptation_strategy)
        self.adaptation_rules = [
            ("block", 0.7, self._adapt_to_block),
            ("detect", 0.6, self._adapt_to_detection),
            ("mitigate", 0.8, self._adapt_to_mitigation)
        ]

    def start(self):
        """Start the feedback loop"""
        if self._running:
            self.logger.warning("Feedback loop already running")
            return

        self._running = True
        self._adaptation_task = asyncio.create_task(self._adaptation_loop())
        self.logger.info("Feedback loop started")

    def stop(self):
        """Stop the feedback loop"""
        if not self._running:
            self.logger.warning("Feedback loop not running")
            return

        self._running = False
        if self._adaptation_task:
            self._adaptation_task.cancel()
        self.logger.info("Feedback loop stopped")

    async def _adaptation_loop(self):
        """Main adaptation loop"""
        while self._running:
            try:
                # Process feedback signals and adapt attack strategy
                await self._process_feedback()

                # Wait for next iteration
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                self.logger.info("Adaptation loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _process_feedback(self):
        """Process feedback signals and adapt attack strategy"""
        if not self.signal_history:
            return  # No signals to process

        # Get recent signals (last 5 seconds)
        current_time = time.time()
        recent_signals = [s for s in self.signal_history
                         if current_time - s.timestamp <= 5]

        if not recent_signals:
            return

        # Process each signal type
        for signal in recent_signals:
            handler = self.signal_handlers.get(signal.signal_type)
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(signal)
                    else:
                        handler(signal)
                except Exception as e:
                    self.logger.error(f"Error handling signal {signal.signal_type}: {str(e)}")

        # Apply adaptation rules
        for signal_type, threshold, strategy in self.adaptation_rules:
            # Get relevant signals
            relevant_signals = [s for s in recent_signals
                               if s.signal_type == signal_type and
                               s.confidence >= threshold]

            if relevant_signals:
                try:
                    if asyncio.iscoroutinefunction(strategy):
                        await strategy(relevant_signals)
                    else:
                        strategy(relevant_signals)
                except Exception as e:
                    self.logger.error(f"Error applying adaptation strategy {strategy.__name__}: {str(e)}")

    def send_feedback(self, signal: FeedbackSignal):
        """Send feedback signal to the adaptation engine"""
        self.signal_history.append(signal)
        self.logger.debug(f"Received feedback signal: {signal.signal_type} (confidence: {signal.confidence})")

    def set_attack_strategy(self, strategy: str):
        """Set the current attack strategy"""
        if strategy not in self.attack_patterns:
            raise ValueError(f"Unknown attack strategy: {strategy}")

        self.current_strategy = strategy
        self.logger.info(f"Attack strategy set to: {strategy}")

    def get_available_strategies(self) -> List[str]:
        """Get list of available attack strategies"""
        return list(self.attack_patterns.keys())

    def get_attack_patterns(self, strategy: str = None) -> List[Dict]:
        """Get attack patterns for a specific strategy or all"""
        if strategy:
            return self.attack_patterns.get(strategy, [])
        else:
            all_patterns = []
            for patterns in self.attack_patterns.values():
                all_patterns.extend(patterns)
            return all_patterns

    def _handle_block_signal(self, signal: FeedbackSignal):
        """Handle block detection signals"""
        self.logger.debug(f"Handling block signal: {signal.metadata}")

    def _handle_detect_signal(self, signal: FeedbackSignal):
        """Handle detection signals"""
        self.logger.debug(f"Handling detection signal: {signal.metadata}")

    def _handle_mitigate_signal(self, signal: FeedbackSignal):
        """Handle mitigation signals"""
        self.logger.debug(f"Handling mitigation signal: {signal.metadata}")

    async def _adapt_to_block(self, signals: List[FeedbackSignal]):
        """Adapt to blocked attack signals"""
        self.logger.info(f"Adapting to {len(signals)} block signals")

        # Calculate success rate for current strategy
        blocked_signals = len([s for s in signals if s.confidence > 0.8])
        total_signals = len(signals)

        if total_signals == 0:
            return

        success_rate = 1 - (blocked_signals / total_signals)

        # If success rate is too low, switch strategy
        if success_rate < 0.3 and self.current_strategy:
            self.logger.info(f"Current strategy {self.current_strategy} has low success rate ({success_rate:.2f})")
            await self._find_better_strategy()

    async def _adapt_to_detection(self, signals: List[FeedbackSignal]):
        """Adapt to detection signals"""
        self.logger.info(f"Adapting to {len(signals)} detection signals")

        # If we're being detected too often, increase obfuscation
        high_confidence_detections = [s for s in signals if s.confidence > 0.7]

        if len(high_confidence_detections) > 3:
            self.logger.info("High confidence detections detected, increasing obfuscation")
            # In a real implementation, this would modify attack patterns
            # For now, just log the adaptation

    async def _adapt_to_mitigation(self, signals: List[FeedbackSignal]):
        """Adapt to mitigation signals"""
        self.logger.info(f"Adapting to {len(signals)} mitigation signals")

        # If we're being mitigated, try different attack vectors
        if len(signals) > 2:
            self.logger.info("Multiple mitigations detected, looking for alternative attack vectors")
            await self._find_alternative_attack_vectors()

    async def _find_better_strategy(self):
        """Find a better attack strategy based on success rates"""
        if not self.current_strategy:
            return

        # In a real implementation, this would analyze success rates across all strategies
        # For now, just pick the first available strategy that's different
        available_strategies = self.get_available_strategies()

        for strategy in available_strategies:
            if strategy != self.current_strategy:
                self.set_attack_strategy(strategy)
                return

    async def _find_alternative_attack_vectors(self):
        """Find alternative attack vectors when current ones are mitigated"""
        # In a real implementation, this would analyze available attack patterns
        # For now, just log the action
        self.logger.info("Looking for alternative attack vectors")

    def get_current_strategy(self) -> Optional[str]:
        """Get the current attack strategy"""
        return self.current_strategy

    def get_signal_history(self) -> List[FeedbackSignal]:
        """Get the history of feedback signals"""
        return list(self.signal_history)

    def get_attack_success_rate(self, strategy: str = None) -> float:
        """Get the estimated success rate for a strategy"""
        # In a real implementation, this would calculate based on feedback
        # For now, return a mock value
        return 0.7 if not strategy or strategy == self.current_strategy else 0.5

    def reset(self):
        """Reset the feedback loop"""
        self.signal_history.clear()
        self.current_strategy = None
        self.logger.info("Feedback loop reset")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the engine state to a dictionary"""
        return {
            "current_strategy": self.current_strategy,
            "signal_count": len(self.signal_history),
            "available_strategies": self.get_available_strategies(),
            "attack_patterns_count": {s: len(patterns) for s, patterns in self.attack_patterns.items()},
            "running": self._running
        }
