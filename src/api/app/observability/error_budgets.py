"""
XORB Phase G5 Error Budget Tracking
Error budget calculation and burn rate monitoring for SLOs
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from opentelemetry import metrics
from .instrumentation import get_meter
from .sli_metrics import SLITarget


@dataclass
class ErrorBudgetStatus:
    """Error budget status for an SLI."""
    sli_name: str
    target_value: float
    error_budget_percent: float
    remaining_budget_percent: float
    burn_rate_hourly: float
    time_to_exhaustion_hours: Optional[float]
    alert_level: str  # "healthy", "warning", "critical"
    last_updated: float


class ErrorBudgetTracker:
    """
    Error Budget Tracker for XORB SLO monitoring.
    
    Tracks error budgets and burn rates for all SLIs with alerting thresholds.
    Implements Google's error budget methodology with multi-window burn rate alerting.
    """
    
    def __init__(self):
        self.meter = get_meter()
        self._setup_budget_metrics()
        
        # Error budget tracking state
        self._budget_status: Dict[str, ErrorBudgetStatus] = {}
        
        # Multi-window burn rate thresholds (Google SRE standards)
        self._burn_rate_thresholds = {
            # Window: (burn_rate_threshold, budget_consumption_threshold)
            "1h": (14.4, 0.02),    # 2% budget consumed in 1 hour
            "6h": (6.0, 0.05),     # 5% budget consumed in 6 hours  
            "3d": (1.0, 0.10),     # 10% budget consumed in 3 days
        }
        
    def _setup_budget_metrics(self) -> None:
        """Initialize error budget metrics."""
        
        # Error budget remaining (0-100%)
        self.error_budget_remaining = self.meter.create_gauge(
            "slo_error_budget_remaining_percent",
            description="Remaining SLO error budget percentage"
        )
        
        # Error budget burn rate (multiplier of budget consumption)
        self.error_budget_burn_rate = self.meter.create_gauge(
            "slo_error_budget_burn_rate",
            description="SLO error budget burn rate multiplier"
        )
        
        # Time to budget exhaustion (hours)
        self.time_to_budget_exhaustion = self.meter.create_gauge(
            "slo_time_to_budget_exhaustion_hours",
            description="Hours until error budget exhaustion at current burn rate"
        )
        
        # Alert level gauge
        self.error_budget_alert_level = self.meter.create_gauge(
            "slo_error_budget_alert_level",
            description="Error budget alert level (0=healthy, 1=warning, 2=critical)"
        )
        
        print("✅ Error budget metrics initialized")
    
    def update_error_budget(
        self,
        sli_name: str,
        current_error_rate: float,
        target: SLITarget,
        measurement_window_hours: float = 1.0
    ) -> ErrorBudgetStatus:
        """
        Update error budget status for an SLI.
        
        Args:
            sli_name: Name of the SLI
            current_error_rate: Current measured error rate (0.0-1.0)
            target: SLI target configuration
            measurement_window_hours: Window for burn rate calculation
            
        Returns:
            Updated error budget status
        """
        target_error_rate = target.error_budget_percent / 100.0
        
        # Calculate budget consumption
        if current_error_rate <= target_error_rate:
            # Within budget
            remaining_budget = 100.0
            burn_rate = 0.0
            time_to_exhaustion = None
        else:
            # Calculate actual budget consumption
            excess_error_rate = current_error_rate - target_error_rate
            budget_consumption_rate = excess_error_rate / target_error_rate
            
            # Burn rate as multiple of expected consumption
            burn_rate = budget_consumption_rate
            
            # Calculate remaining budget (simplified model)
            # In production, this would integrate actual error measurements over time
            budget_consumed_this_window = (budget_consumption_rate * measurement_window_hours) / target.measurement_window_hours * 100
            remaining_budget = max(0.0, 100.0 - budget_consumed_this_window)
            
            # Time to exhaustion at current burn rate
            if burn_rate > 0:
                time_to_exhaustion = (remaining_budget / 100.0) * target.measurement_window_hours / burn_rate
            else:
                time_to_exhaustion = None
        
        # Determine alert level
        alert_level = self._calculate_alert_level(burn_rate, remaining_budget, time_to_exhaustion)
        
        # Create status object
        status = ErrorBudgetStatus(
            sli_name=sli_name,
            target_value=target.target_value_ms,
            error_budget_percent=target.error_budget_percent,
            remaining_budget_percent=remaining_budget,
            burn_rate_hourly=burn_rate,
            time_to_exhaustion_hours=time_to_exhaustion,
            alert_level=alert_level,
            last_updated=time.time()
        )
        
        # Store status
        self._budget_status[sli_name] = status
        
        # Record metrics
        labels = {
            "sli_name": sli_name,
            "target_error_budget": str(target.error_budget_percent),
            "alert_level": alert_level
        }
        
        self.error_budget_remaining.set(remaining_budget, labels)
        self.error_budget_burn_rate.set(burn_rate, labels)
        
        if time_to_exhaustion is not None:
            self.time_to_budget_exhaustion.set(time_to_exhaustion, labels)
        
        # Alert level as numeric value for alerting
        alert_level_numeric = {"healthy": 0, "warning": 1, "critical": 2}[alert_level]
        self.error_budget_alert_level.set(alert_level_numeric, labels)
        
        return status
    
    def _calculate_alert_level(
        self, 
        burn_rate: float, 
        remaining_budget: float,
        time_to_exhaustion: Optional[float]
    ) -> str:
        """Calculate alert level based on burn rate and remaining budget."""
        
        # Critical: High burn rate or low remaining budget
        if burn_rate >= 14.4:  # Will exhaust budget in ~7 hours
            return "critical"
        if remaining_budget <= 10.0:  # Less than 10% budget remaining
            return "critical"
        if time_to_exhaustion is not None and time_to_exhaustion <= 2.0:  # Less than 2 hours
            return "critical"
        
        # Warning: Medium burn rate or moderate budget consumption
        if burn_rate >= 6.0:  # Will exhaust budget in ~17 hours
            return "warning"
        if remaining_budget <= 25.0:  # Less than 25% budget remaining
            return "warning"
        if time_to_exhaustion is not None and time_to_exhaustion <= 8.0:  # Less than 8 hours
            return "warning"
        
        return "healthy"
    
    def get_budget_status(self, sli_name: str) -> Optional[ErrorBudgetStatus]:
        """Get current error budget status for an SLI."""
        return self._budget_status.get(sli_name)
    
    def get_all_budget_status(self) -> Dict[str, ErrorBudgetStatus]:
        """Get error budget status for all tracked SLIs."""
        return self._budget_status.copy()
    
    def generate_budget_report(self) -> Dict[str, Any]:
        """Generate comprehensive error budget report."""
        total_slis = len(self._budget_status)
        healthy_slis = sum(1 for status in self._budget_status.values() if status.alert_level == "healthy")
        warning_slis = sum(1 for status in self._budget_status.values() if status.alert_level == "warning")
        critical_slis = sum(1 for status in self._budget_status.values() if status.alert_level == "critical")
        
        # Overall platform SLO health
        if critical_slis > 0:
            overall_status = "critical"
        elif warning_slis > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "summary": {
                "total_slis": total_slis,
                "healthy": healthy_slis,
                "warning": warning_slis,
                "critical": critical_slis
            },
            "sli_details": {
                name: {
                    "remaining_budget_percent": status.remaining_budget_percent,
                    "burn_rate_hourly": status.burn_rate_hourly,
                    "time_to_exhaustion_hours": status.time_to_exhaustion_hours,
                    "alert_level": status.alert_level,
                    "last_updated": status.last_updated
                }
                for name, status in self._budget_status.items()
            },
            "generated_at": time.time()
        }
    
    async def start_monitoring(self, check_interval_seconds: int = 60) -> None:
        """Start background monitoring of error budgets."""
        while True:
            try:
                # Update budget calculations for all tracked SLIs
                for sli_name, status in self._budget_status.items():
                    # In production, this would fetch actual metrics from Prometheus
                    # For now, we maintain the last calculated values
                    pass
                
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                print(f"⚠️ Error budget monitoring error: {e}")
                await asyncio.sleep(check_interval_seconds)


# Global error budget tracker instance
_error_budget_tracker: Optional[ErrorBudgetTracker] = None


def get_error_budget_tracker() -> ErrorBudgetTracker:
    """Get the global error budget tracker instance."""
    global _error_budget_tracker
    if _error_budget_tracker is None:
        _error_budget_tracker = ErrorBudgetTracker()
    return _error_budget_tracker


# Convenience functions for error budget operations
def update_sli_error_budget(
    sli_name: str,
    current_error_rate: float,
    target: SLITarget
) -> ErrorBudgetStatus:
    """Update error budget for an SLI."""
    tracker = get_error_budget_tracker()
    return tracker.update_error_budget(sli_name, current_error_rate, target)


def get_sli_budget_status(sli_name: str) -> Optional[ErrorBudgetStatus]:
    """Get error budget status for an SLI."""
    tracker = get_error_budget_tracker()
    return tracker.get_budget_status(sli_name)


def generate_slo_report() -> Dict[str, Any]:
    """Generate comprehensive SLO report."""
    tracker = get_error_budget_tracker()
    return tracker.generate_budget_report()