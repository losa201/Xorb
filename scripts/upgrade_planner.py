#!/usr/bin/env python3
"""
Xorb Upgrade Planner
Analyzes system metrics and provides upgrade recommendations
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UpgradeRecommendation:
    """Upgrade recommendation data"""
    resource: str
    current_value: float
    threshold: float
    severity: str
    recommended_action: str
    cost_estimate: float | None = None
    timeline: str | None = None

@dataclass
class SystemSpecs:
    """Current system specifications"""
    cpu_cores: int = 16
    memory_gb: int = 32
    storage_gb: int = 600  # 400GB NVMe + 200GB SSD
    network_gbps: int = 1

class UpgradePlanner:
    """Analyzes metrics and provides upgrade recommendations"""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.current_specs = SystemSpecs()
        self.upgrade_costs = {
            "cpu": {
                24: 150,  # $150/month for 24 vCPU
                32: 250,  # $250/month for 32 vCPU
                48: 400   # $400/month for 48 vCPU
            },
            "memory": {
                64: 100,  # $100/month for 64GB
                128: 200, # $200/month for 128GB
                256: 400  # $400/month for 256GB
            },
            "storage": {
                1000: 50,  # $50/month for 1TB
                2000: 100, # $100/month for 2TB
                4000: 200  # $200/month for 4TB
            }
        }

    async def query_prometheus(self, query: str, time_range: str = "1h") -> float | None:
        """Query Prometheus for metric value"""
        try:
            async with aiohttp.ClientSession() as session:
                if time_range:
                    # For range queries, get the latest value
                    params = {"query": f"{query}[{time_range}]"}
                    endpoint = "query"
                else:
                    params = {"query": query}
                    endpoint = "query"

                async with session.get(f"{self.prometheus_url}/api/v1/{endpoint}",
                                     params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("data", {}).get("result", [])
                        if result:
                            # Get the latest value from the result
                            if "values" in result[0]:
                                return float(result[0]["values"][-1][1])
                            elif "value" in result[0]:
                                return float(result[0]["value"][1])
                    return None
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return None

    async def get_cpu_metrics(self) -> dict[str, float]:
        """Get CPU-related metrics"""
        metrics = {}

        # Current CPU utilization
        cpu_current = await self.query_prometheus("xorb:cpu_utilization_5m", "")
        metrics["cpu_current"] = cpu_current or 0

        # 1-hour average
        cpu_1h = await self.query_prometheus("xorb:cpu_utilization_1h", "")
        metrics["cpu_1h_avg"] = cpu_1h or 0

        # 6-hour trend prediction
        cpu_trend = await self.query_prometheus("xorb:cpu_utilization_trend_6h", "")
        metrics["cpu_6h_trend"] = cpu_trend or 0

        # Peak usage in last 24h
        cpu_peak = await self.query_prometheus("max_over_time(xorb:cpu_utilization_5m[24h])", "")
        metrics["cpu_24h_peak"] = cpu_peak or 0

        return metrics

    async def get_memory_metrics(self) -> dict[str, float]:
        """Get memory-related metrics"""
        metrics = {}

        # Current memory utilization
        mem_current = await self.query_prometheus("xorb:memory_utilization_5m", "")
        metrics["memory_current"] = mem_current or 0

        # 1-hour average
        mem_1h = await self.query_prometheus("xorb:memory_utilization_1h", "")
        metrics["memory_1h_avg"] = mem_1h or 0

        # 6-hour trend prediction
        mem_trend = await self.query_prometheus("xorb:memory_utilization_trend_6h", "")
        metrics["memory_6h_trend"] = mem_trend or 0

        # Peak usage in last 24h
        mem_peak = await self.query_prometheus("max_over_time(xorb:memory_utilization_5m[24h])", "")
        metrics["memory_24h_peak"] = mem_peak or 0

        return metrics

    async def get_storage_metrics(self) -> dict[str, float]:
        """Get storage-related metrics"""
        metrics = {}

        # Current storage utilization
        storage_current = await self.query_prometheus("max(xorb:storage_utilization_5m)", "")
        metrics["storage_current"] = storage_current or 0

        # Storage growth rate (MB/day)
        storage_growth = await self.query_prometheus(
            "predict_linear(node_filesystem_size_bytes{fstype!=\"tmpfs\"}[7d], 86400)"
        )
        metrics["storage_growth_daily"] = (storage_growth or 0) / (1024 * 1024 * 1024)  # Convert to GB

        return metrics

    async def get_performance_metrics(self) -> dict[str, float]:
        """Get performance-related metrics"""
        metrics = {}

        # API response time
        api_latency = await self.query_prometheus("xorb:api_response_time_p95_1h", "")
        metrics["api_p95_latency"] = api_latency or 0

        # Worker queue backlog
        worker_queue = await self.query_prometheus("xorb:worker_queue_avg_1h", "")
        metrics["worker_queue_avg"] = worker_queue or 0

        # Database connection utilization
        db_connections = await self.query_prometheus("xorb:db_connections_utilization", "")
        metrics["db_connections_pct"] = db_connections or 0

        # API request growth
        api_growth = await self.query_prometheus("xorb:api_requests_growth_24h", "")
        metrics["api_growth_24h"] = api_growth or 1.0

        return metrics

    def analyze_cpu_upgrade(self, cpu_metrics: dict[str, float]) -> list[UpgradeRecommendation]:
        """Analyze CPU upgrade needs"""
        recommendations = []

        current_avg = cpu_metrics["cpu_1h_avg"]
        peak_24h = cpu_metrics["cpu_24h_peak"]
        trend_6h = cpu_metrics["cpu_6h_trend"]

        # Immediate upgrade needed
        if current_avg > 85:
            recommendations.append(UpgradeRecommendation(
                resource="cpu",
                current_value=current_avg,
                threshold=85,
                severity="critical",
                recommended_action=f"Immediate upgrade from {self.current_specs.cpu_cores} to 32+ vCPU",
                cost_estimate=self.upgrade_costs["cpu"][32],
                timeline="Within 24 hours"
            ))
        # Warning threshold
        elif current_avg > 70:
            recommendations.append(UpgradeRecommendation(
                resource="cpu",
                current_value=current_avg,
                threshold=70,
                severity="warning",
                recommended_action=f"Plan upgrade from {self.current_specs.cpu_cores} to 24-32 vCPU",
                cost_estimate=self.upgrade_costs["cpu"][24],
                timeline="Within 1 week"
            ))
        # Predictive recommendation
        elif trend_6h > 80:
            recommendations.append(UpgradeRecommendation(
                resource="cpu",
                current_value=trend_6h,
                threshold=80,
                severity="info",
                recommended_action=f"Proactive upgrade from {self.current_specs.cpu_cores} to 24 vCPU recommended",
                cost_estimate=self.upgrade_costs["cpu"][24],
                timeline="Within 2 weeks"
            ))

        return recommendations

    def analyze_memory_upgrade(self, memory_metrics: dict[str, float]) -> list[UpgradeRecommendation]:
        """Analyze memory upgrade needs"""
        recommendations = []

        current_avg = memory_metrics["memory_1h_avg"]
        peak_24h = memory_metrics["memory_24h_peak"]
        trend_6h = memory_metrics["memory_6h_trend"]

        # Immediate upgrade needed
        if current_avg > 85:
            recommendations.append(UpgradeRecommendation(
                resource="memory",
                current_value=current_avg,
                threshold=85,
                severity="critical",
                recommended_action=f"Immediate upgrade from {self.current_specs.memory_gb}GB to 64GB+",
                cost_estimate=self.upgrade_costs["memory"][64],
                timeline="Within 24 hours"
            ))
        # Warning threshold
        elif current_avg > 70:
            recommendations.append(UpgradeRecommendation(
                resource="memory",
                current_value=current_avg,
                threshold=70,
                severity="warning",
                recommended_action=f"Plan upgrade from {self.current_specs.memory_gb}GB to 64GB",
                cost_estimate=self.upgrade_costs["memory"][64],
                timeline="Within 1 week"
            ))
        # Predictive recommendation
        elif trend_6h > 80:
            recommendations.append(UpgradeRecommendation(
                resource="memory",
                current_value=trend_6h,
                threshold=80,
                severity="info",
                recommended_action=f"Proactive upgrade from {self.current_specs.memory_gb}GB to 64GB recommended",
                cost_estimate=self.upgrade_costs["memory"][64],
                timeline="Within 2 weeks"
            ))

        return recommendations

    def analyze_storage_upgrade(self, storage_metrics: dict[str, float]) -> list[UpgradeRecommendation]:
        """Analyze storage upgrade needs"""
        recommendations = []

        current_usage = storage_metrics["storage_current"]
        daily_growth = storage_metrics["storage_growth_daily"]

        # Calculate days until full at current growth rate
        remaining_space = self.current_specs.storage_gb * (100 - current_usage) / 100
        if daily_growth > 0:
            days_until_full = remaining_space / daily_growth
        else:
            days_until_full = float('inf')

        # Immediate upgrade needed
        if current_usage > 85:
            recommendations.append(UpgradeRecommendation(
                resource="storage",
                current_value=current_usage,
                threshold=85,
                severity="critical",
                recommended_action=f"Immediate storage expansion from {self.current_specs.storage_gb}GB to 1TB+",
                cost_estimate=self.upgrade_costs["storage"][1000],
                timeline="Within 48 hours"
            ))
        # Warning threshold
        elif current_usage > 70:
            recommendations.append(UpgradeRecommendation(
                resource="storage",
                current_value=current_usage,
                threshold=70,
                severity="warning",
                recommended_action=f"Plan storage expansion from {self.current_specs.storage_gb}GB to 1TB",
                cost_estimate=self.upgrade_costs["storage"][1000],
                timeline="Within 1 week"
            ))
        # Predictive based on growth
        elif days_until_full < 30:
            recommendations.append(UpgradeRecommendation(
                resource="storage",
                current_value=current_usage,
                threshold=70,
                severity="info",
                recommended_action=f"Storage will be full in ~{int(days_until_full)} days. Plan expansion.",
                cost_estimate=self.upgrade_costs["storage"][1000],
                timeline="Within 2 weeks"
            ))

        return recommendations

    def analyze_performance_upgrade(self, perf_metrics: dict[str, float]) -> list[UpgradeRecommendation]:
        """Analyze performance-based upgrade needs"""
        recommendations = []

        api_latency = perf_metrics["api_p95_latency"]
        worker_queue = perf_metrics["worker_queue_avg"]
        db_connections = perf_metrics["db_connections_pct"]
        api_growth = perf_metrics["api_growth_24h"]

        # API latency degradation
        if api_latency > 0.5:
            recommendations.append(UpgradeRecommendation(
                resource="performance",
                current_value=api_latency,
                threshold=0.5,
                severity="warning",
                recommended_action="API performance degradation detected. Consider CPU/memory upgrade or horizontal scaling.",
                timeline="Within 1 week"
            ))

        # Worker queue backlog
        if worker_queue > 50:
            recommendations.append(UpgradeRecommendation(
                resource="workers",
                current_value=worker_queue,
                threshold=50,
                severity="warning",
                recommended_action="High worker queue backlog. Scale worker instances or increase CPU allocation.",
                timeline="Within 3 days"
            ))

        # Database connection pressure
        if db_connections > 70:
            recommendations.append(UpgradeRecommendation(
                resource="database",
                current_value=db_connections,
                threshold=70,
                severity="warning",
                recommended_action="High database connection utilization. Consider database scaling or connection pooling optimization.",
                timeline="Within 1 week"
            ))

        # Growth-based scaling
        if api_growth > 1.5:
            recommendations.append(UpgradeRecommendation(
                resource="capacity",
                current_value=api_growth,
                threshold=1.5,
                severity="info",
                recommended_action=f"API requests grew {api_growth:.1f}x in 24h. Plan capacity expansion.",
                timeline="Within 2 weeks"
            ))

        return recommendations

    def calculate_upgrade_priority(self, recommendations: list[UpgradeRecommendation]) -> list[UpgradeRecommendation]:
        """Sort recommendations by priority"""
        severity_order = {"critical": 0, "warning": 1, "info": 2}

        return sorted(recommendations, key=lambda x: (
            severity_order.get(x.severity, 3),
            -x.current_value if x.current_value else 0
        ))

    def generate_upgrade_plan(self, recommendations: list[UpgradeRecommendation]) -> dict:
        """Generate comprehensive upgrade plan"""
        critical_recs = [r for r in recommendations if r.severity == "critical"]
        warning_recs = [r for r in recommendations if r.severity == "warning"]
        info_recs = [r for r in recommendations if r.severity == "info"]

        total_cost = sum(r.cost_estimate for r in recommendations if r.cost_estimate)

        # Determine optimal upgrade path
        upgrade_plan = {
            "immediate_actions": critical_recs,
            "short_term_actions": warning_recs,
            "long_term_planning": info_recs,
            "total_estimated_cost": total_cost,
            "current_specs": {
                "cpu_cores": self.current_specs.cpu_cores,
                "memory_gb": self.current_specs.memory_gb,
                "storage_gb": self.current_specs.storage_gb
            }
        }

        # Suggest optimal new specs
        if critical_recs or warning_recs:
            cpu_upgrade = 32 if any("cpu" in r.resource for r in critical_recs + warning_recs) else self.current_specs.cpu_cores
            memory_upgrade = 64 if any("memory" in r.resource for r in critical_recs + warning_recs) else self.current_specs.memory_gb
            storage_upgrade = 1000 if any("storage" in r.resource for r in critical_recs + warning_recs) else self.current_specs.storage_gb

            upgrade_plan["recommended_specs"] = {
                "cpu_cores": cpu_upgrade,
                "memory_gb": memory_upgrade,
                "storage_gb": storage_upgrade
            }

        return upgrade_plan

    async def analyze_system(self) -> dict:
        """Perform comprehensive system analysis"""
        logger.info("Starting system upgrade analysis...")

        # Gather all metrics
        cpu_metrics = await self.get_cpu_metrics()
        memory_metrics = await self.get_memory_metrics()
        storage_metrics = await self.get_storage_metrics()
        performance_metrics = await self.get_performance_metrics()

        logger.info(f"CPU metrics: {cpu_metrics}")
        logger.info(f"Memory metrics: {memory_metrics}")
        logger.info(f"Storage metrics: {storage_metrics}")
        logger.info(f"Performance metrics: {performance_metrics}")

        # Generate recommendations
        all_recommendations = []
        all_recommendations.extend(self.analyze_cpu_upgrade(cpu_metrics))
        all_recommendations.extend(self.analyze_memory_upgrade(memory_metrics))
        all_recommendations.extend(self.analyze_storage_upgrade(storage_metrics))
        all_recommendations.extend(self.analyze_performance_upgrade(performance_metrics))

        # Sort by priority
        prioritized_recommendations = self.calculate_upgrade_priority(all_recommendations)

        # Generate upgrade plan
        upgrade_plan = self.generate_upgrade_plan(prioritized_recommendations)

        # Create comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "cpu": cpu_metrics,
                "memory": memory_metrics,
                "storage": storage_metrics,
                "performance": performance_metrics
            },
            "recommendations": [
                {
                    "resource": r.resource,
                    "current_value": r.current_value,
                    "threshold": r.threshold,
                    "severity": r.severity,
                    "action": r.recommended_action,
                    "cost_estimate": r.cost_estimate,
                    "timeline": r.timeline
                }
                for r in prioritized_recommendations
            ],
            "upgrade_plan": upgrade_plan,
            "summary": {
                "total_recommendations": len(prioritized_recommendations),
                "critical_actions": len([r for r in prioritized_recommendations if r.severity == "critical"]),
                "warnings": len([r for r in prioritized_recommendations if r.severity == "warning"]),
                "info": len([r for r in prioritized_recommendations if r.severity == "info"]),
                "estimated_monthly_cost": upgrade_plan["total_estimated_cost"]
            }
        }

        return report

async def main():
    """Main function for upgrade planning"""
    import argparse

    parser = argparse.ArgumentParser(description="Xorb Upgrade Planner")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                       help="Prometheus server URL")
    parser.add_argument("--output", choices=["json", "human"], default="human",
                       help="Output format")
    parser.add_argument("--save-report", help="Save report to file")

    args = parser.parse_args()

    planner = UpgradePlanner(prometheus_url=args.prometheus_url)

    try:
        report = await planner.analyze_system()

        if args.save_report:
            with open(args.save_report, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {args.save_report}")

        if args.output == "json":
            print(json.dumps(report, indent=2))
        else:
            # Human-readable output
            print("\n🔍 Xorb System Upgrade Analysis")
            print(f"📅 Generated: {report['analysis_timestamp']}")
            print(f"💰 Estimated monthly cost: ${report['summary']['estimated_monthly_cost']}")

            summary = report['summary']
            print("\n📊 Summary:")
            print(f"  • Total recommendations: {summary['total_recommendations']}")
            print(f"  • Critical actions: {summary['critical_actions']}")
            print(f"  • Warnings: {summary['warnings']}")
            print(f"  • Info: {summary['info']}")

            if report['recommendations']:
                print("\n⚠️  Recommendations (by priority):")
                for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
                    severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "💡"}.get(rec['severity'], "")
                    cost_text = f" (${rec['cost_estimate']}/mo)" if rec['cost_estimate'] else ""
                    timeline_text = f" - {rec['timeline']}" if rec['timeline'] else ""
                    print(f"  {i}. {severity_emoji} {rec['action']}{cost_text}{timeline_text}")

            upgrade_plan = report['upgrade_plan']
            if upgrade_plan.get('recommended_specs'):
                current = upgrade_plan['current_specs']
                recommended = upgrade_plan['recommended_specs']
                print("\n🚀 Recommended Upgrade:")
                print(f"  • CPU: {current['cpu_cores']} → {recommended['cpu_cores']} vCPU")
                print(f"  • Memory: {current['memory_gb']} → {recommended['memory_gb']} GB")
                print(f"  • Storage: {current['storage_gb']} → {recommended['storage_gb']} GB")

        # Exit with appropriate code
        if report['summary']['critical_actions'] > 0:
            exit(2)
        elif report['summary']['warnings'] > 0:
            exit(1)
        else:
            exit(0)

    except Exception as e:
        logger.error(f"Upgrade analysis failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
