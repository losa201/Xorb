#!/usr/bin/env python3
"""
Xorb Performance Budget Validator
Validates SLO compliance and performance budget adherence
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import prometheus_client.parser
from prometheus_client import CollectorRegistry, Gauge, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceBudget:
    """Performance budget definition"""
    name: str
    metric_query: str
    threshold: float
    unit: str
    severity: str
    description: str
    error_budget_monthly: float = 0.1  # 10% monthly error budget

@dataclass
class BudgetViolation:
    """Performance budget violation"""
    budget_name: str
    current_value: float
    threshold: float
    severity: str
    timestamp: datetime
    duration_minutes: int

class PerformanceBudgetValidator:
    """Validates performance budgets against Prometheus metrics"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.budgets = self._define_budgets()
        self.violations: List[BudgetViolation] = []
        
        # Prometheus client for custom metrics
        self.registry = CollectorRegistry()
        self.budget_violation_gauge = Gauge(
            'xorb_budget_violation_current',
            'Current budget violation status',
            ['budget_name', 'severity'],
            registry=self.registry
        )
        self.budget_compliance_histogram = Histogram(
            'xorb_budget_compliance_ratio',
            'Budget compliance ratio over time',
            ['budget_name'],
            registry=self.registry
        )
    
    def _define_budgets(self) -> List[PerformanceBudget]:
        """Define all performance budgets"""
        return [
            # API Latency Budgets
            PerformanceBudget(
                name="api_latency_p95",
                metric_query="xorb:api_latency_p95_5m",
                threshold=0.200,  # 200ms
                unit="seconds",
                severity="critical",
                description="API P95 latency budget",
                error_budget_monthly=0.05  # 5% monthly budget
            ),
            PerformanceBudget(
                name="api_latency_p99",
                metric_query="xorb:api_latency_p99_5m",
                threshold=0.500,  # 500ms
                unit="seconds",
                severity="critical",
                description="API P99 latency hard limit"
            ),
            
            # Scan Performance Budgets
            PerformanceBudget(
                name="scan_queue_lag",
                metric_query="xorb:scan_queue_age_p95_5m",
                threshold=300,  # 5 minutes
                unit="seconds",
                severity="critical",
                description="Scan queue lag budget"
            ),
            PerformanceBudget(
                name="scan_processing_time",
                metric_query="xorb:scan_processing_time_p95_5m",
                threshold=1800,  # 30 minutes
                unit="seconds",
                severity="warning",
                description="Scan processing time budget"
            ),
            PerformanceBudget(
                name="scan_error_rate",
                metric_query="xorb:scan_error_rate_5m",
                threshold=5.0,  # 5%
                unit="percent",
                severity="critical",
                description="Scan error rate budget"
            ),
            
            # Payout Budgets
            PerformanceBudget(
                name="payout_processing_time",
                metric_query="xorb:payout_processing_time_p95_5m",
                threshold=3600,  # 1 hour
                unit="seconds",
                severity="critical",
                description="Payout processing time budget"
            ),
            PerformanceBudget(
                name="payout_queue_age",
                metric_query="xorb:payout_queue_age_p95_5m",
                threshold=1800,  # 30 minutes
                unit="seconds",
                severity="critical",
                description="Payout queue age budget"
            ),
            PerformanceBudget(
                name="payout_success_rate",
                metric_query="xorb:payout_success_rate_5m",
                threshold=98.0,  # 98%
                unit="percent",
                severity="critical",
                description="Payout success rate budget",
                error_budget_monthly=0.02  # 2% monthly budget
            ),
            
            # Resource Budgets
            PerformanceBudget(
                name="cpu_utilization",
                metric_query="xorb:cpu_utilization_5m",
                threshold=75.0,  # 75%
                unit="percent",
                severity="warning",
                description="CPU utilization budget"
            ),
            PerformanceBudget(
                name="memory_utilization",
                metric_query="xorb:memory_utilization_5m",
                threshold=90.0,  # 90%
                unit="percent",
                severity="critical",
                description="Memory utilization budget"
            ),
            
            # Availability Budget
            PerformanceBudget(
                name="service_availability",
                metric_query="xorb:service_availability_5m",
                threshold=0.999,  # 99.9%
                unit="ratio",
                severity="critical",
                description="Service availability budget",
                error_budget_monthly=0.001  # 0.1% monthly budget
            )
        ]
    
    async def query_prometheus(self, query: str) -> Optional[float]:
        """Query Prometheus for metric value"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"query": query}
                async with session.get(f"{self.prometheus_url}/api/v1/query", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("data", {}).get("result", [])
                        if result:
                            return float(result[0]["value"][1])
                    return None
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return None
    
    async def validate_budget(self, budget: PerformanceBudget) -> Optional[BudgetViolation]:
        """Validate a single performance budget"""
        current_value = await self.query_prometheus(budget.metric_query)
        
        if current_value is None:
            logger.warning(f"Could not retrieve metric for budget: {budget.name}")
            return None
        
        # Check if budget is violated
        is_violated = False
        if budget.name == "payout_success_rate" or budget.name == "service_availability":
            # For success rates and availability, violation is below threshold
            is_violated = current_value < budget.threshold
        else:
            # For other metrics, violation is above threshold
            is_violated = current_value > budget.threshold
        
        # Update Prometheus metrics
        violation_value = 1.0 if is_violated else 0.0
        self.budget_violation_gauge.labels(
            budget_name=budget.name,
            severity=budget.severity
        ).set(violation_value)
        
        # Calculate compliance ratio
        if budget.name in ["payout_success_rate", "service_availability"]:
            compliance_ratio = min(current_value / budget.threshold, 1.0)
        else:
            compliance_ratio = min(budget.threshold / max(current_value, 0.001), 1.0)
        
        self.budget_compliance_histogram.labels(budget_name=budget.name).observe(compliance_ratio)
        
        if is_violated:
            violation = BudgetViolation(
                budget_name=budget.name,
                current_value=current_value,
                threshold=budget.threshold,
                severity=budget.severity,
                timestamp=datetime.now(),
                duration_minutes=5  # Assuming 5-minute evaluation window
            )
            logger.warning(f"Budget violation: {budget.name} = {current_value} {budget.unit} "
                         f"(threshold: {budget.threshold} {budget.unit})")
            return violation
        
        logger.info(f"Budget OK: {budget.name} = {current_value} {budget.unit} "
                   f"(threshold: {budget.threshold} {budget.unit})")
        return None
    
    async def validate_all_budgets(self) -> List[BudgetViolation]:
        """Validate all performance budgets"""
        logger.info("Starting performance budget validation...")
        violations = []
        
        for budget in self.budgets:
            violation = await self.validate_budget(budget)
            if violation:
                violations.append(violation)
        
        self.violations.extend(violations)
        return violations
    
    def calculate_error_budget_burn_rate(self) -> Dict[str, float]:
        """Calculate error budget burn rate for critical budgets"""
        burn_rates = {}
        
        # Group violations by budget name
        violation_counts = {}
        for violation in self.violations:
            if violation.budget_name not in violation_counts:
                violation_counts[violation.budget_name] = 0
            violation_counts[violation.budget_name] += violation.duration_minutes
        
        # Calculate burn rate as percentage of monthly budget used per hour
        for budget in self.budgets:
            if budget.name in violation_counts:
                minutes_violated = violation_counts[budget.name]
                # Monthly budget (43200 minutes) * error budget percentage
                monthly_allowance_minutes = 43200 * budget.error_budget_monthly
                burn_rate = (minutes_violated / monthly_allowance_minutes) * 100
                burn_rates[budget.name] = burn_rate
            else:
                burn_rates[budget.name] = 0.0
        
        return burn_rates
    
    def generate_budget_report(self) -> Dict:
        """Generate comprehensive budget report"""
        burn_rates = self.calculate_error_budget_burn_rate()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_budgets": len(self.budgets),
            "active_violations": len([v for v in self.violations if 
                                    datetime.now() - v.timestamp < timedelta(minutes=10)]),
            "budgets": [],
            "error_budget_burn_rates": burn_rates,
            "overall_health": "healthy"
        }
        
        # Determine overall health
        critical_violations = [v for v in self.violations if v.severity == "critical" and
                             datetime.now() - v.timestamp < timedelta(minutes=10)]
        if critical_violations:
            report["overall_health"] = "critical"
        elif len(report["active_violations"]) > 0:
            report["overall_health"] = "warning"
        
        # Add budget details
        for budget in self.budgets:
            budget_info = {
                "name": budget.name,
                "description": budget.description,
                "threshold": budget.threshold,
                "unit": budget.unit,
                "severity": budget.severity,
                "error_budget_monthly": budget.error_budget_monthly,
                "burn_rate_percentage": burn_rates.get(budget.name, 0.0),
                "status": "ok"
            }
            
            # Check for recent violations
            recent_violations = [v for v in self.violations if 
                               v.budget_name == budget.name and
                               datetime.now() - v.timestamp < timedelta(minutes=10)]
            if recent_violations:
                budget_info["status"] = "violated"
                budget_info["current_violation"] = {
                    "value": recent_violations[-1].current_value,
                    "threshold": recent_violations[-1].threshold,
                    "severity": recent_violations[-1].severity
                }
            
            report["budgets"].append(budget_info)
        
        return report
    
    async def continuous_monitoring(self, interval_seconds: int = 60):
        """Run continuous budget monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                violations = await self.validate_all_budgets()
                
                if violations:
                    logger.warning(f"Found {len(violations)} active budget violations")
                    for violation in violations:
                        logger.warning(f"  - {violation.budget_name}: {violation.current_value} "
                                     f"vs threshold {violation.threshold}")
                
                # Generate and save report every 5 minutes
                if int(time.time()) % 300 == 0:
                    report = self.generate_budget_report()
                    report_file = f"/tmp/xorb_performance_budget_report_{int(time.time())}.json"
                    with open(report_file, 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Performance budget report saved to {report_file}")
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)

async def main():
    """Main function for script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xorb Performance Budget Validator")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                       help="Prometheus server URL")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=60,
                       help="Monitoring interval in seconds")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report only, don't validate")
    
    args = parser.parse_args()
    
    validator = PerformanceBudgetValidator(prometheus_url=args.prometheus_url)
    
    if args.continuous:
        await validator.continuous_monitoring(interval_seconds=args.interval)
    else:
        violations = await validator.validate_all_budgets()
        report = validator.generate_budget_report()
        
        print(json.dumps(report, indent=2))
        
        if violations:
            print(f"\n⚠️  Found {len(violations)} budget violations!")
            exit(1)
        else:
            print("\n✅ All performance budgets are within limits")
            exit(0)

if __name__ == "__main__":
    asyncio.run(main())