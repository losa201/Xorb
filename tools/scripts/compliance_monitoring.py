#!/usr/bin/env python3
"""
XORB Compliance Monitoring Service
Real-time compliance validation with SIEM integration
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add XORB to Python path
XORB_ROOT = Path("/root/Xorb")
sys.path.insert(0, str(XORB_ROOT))
sys.path.insert(0, str(XORB_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("XORB-ComplianceMonitor")

class ComplianceMonitor:
    def __init__(self):
        self.compliance_check_interval = 300  # 5 minutes
        self.siem_query_interval = 60  # 1 minute
        self.siem_api_url = "http://localhost:8003/api/siem"
        self.compliance_api_url = "http://localhost:8000/api/compliance"
        self.active = True

    async def initialize(self):
        """Initialize compliance monitoring components"""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession()
            return True
        except ImportError as e:
            logger.error(f"❌ Failed to initialize compliance monitor: {e}")
            return False

    async def get_active_compliance_frameworks(self) -> List[str]:
        """Get list of active compliance frameworks"""
        try:
            async with self.session.get(f"{self.compliance_api_url}/frameworks") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('frameworks', [])
                return []
        except Exception as e:
            logger.error(f"❌ Failed to get compliance frameworks: {e}")
            return []

    async def validate_framework_compliance(self, framework: str) -> Dict:
        """Validate compliance for a specific framework"""
        logger.info(f"🔍 Validating {framework} compliance...")

        try:
            async with self.session.post(
                f"{self.compliance_api_url}/validate",
                json={"framework": framework}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ {framework} compliance validation completed")
                    return result
                return {"error": "Validation failed"}
        except Exception as e:
            logger.error(f"❌ {framework} validation error: {e}")
            return {"error": str(e)}

    async def query_siem_for_compliance_events(self, framework: str, timeframe: str = "24h") -> List[Dict]:
        """Query SIEM for compliance-related events"""
        logger.debug(f"🔍 Querying SIEM for {framework} events...")

        try:
            async with self.session.get(
                f"{self.siem_api_url}/correlation",
                params={
                    "compliance_framework": framework,
                    "timeframe": timeframe
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('events', [])
                return []
        except Exception as e:
            logger.error(f"❌ SIEM query error: {e}")
            return []

    async def generate_compliance_report(self, framework: str, validation_data: Dict, siem_events: List[Dict]) -> Dict:
        """Generate comprehensive compliance report"""
        report = {
            "framework": framework,
            "validation_timestamp": datetime.now().isoformat(),
            "validation_results": validation_data,
            "related_security_events": siem_events,
            "compliance_status": "COMPLIANT" if validation_data.get('compliant', False) else "NON-COMPLIANT",
            "risk_score": self.calculate_risk_score(validation_data, siem_events),
            "recommendations": self.generate_recommendations(validation_data, siem_events)
        }

        # Save report to file
        report_path = f"/root/Xorb/reports/compliance_{framework.lower()}_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Compliance report generated: {report_path}")
        return report

    def calculate_risk_score(self, validation_data: Dict, siem_events: List[Dict]) -> int:
        """Calculate overall compliance risk score"""
        base_score = 50 if validation_data.get('compliant', False) else 80

        # Increase score based on security events
        for event in siem_events:
            if event.get('severity') == "CRITICAL":
                base_score += 20
            elif event.get('severity') == "HIGH":
                base_score += 10
            elif event.get('severity') == "MEDIUM":
                base_score += 5

        return min(base_score, 100)

    def generate_recommendations(self, validation_data: Dict, siem_events: List[Dict]) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []

        # Add validation-based recommendations
        if not validation_data.get('compliant', False):
            recommendations.extend(validation_data.get('non_compliant_controls', []))

        # Add event-based recommendations
        for event in siem_events:
            if event.get('severity') in ["CRITICAL", "HIGH"]:
                recommendations.append(event.get('description', 'Investigate security event'))

        return list(set(recommendations))  # Remove duplicates

    async def send_alert(self, report: Dict):
        """Send compliance alert to security team"""
        logger.info(f"🔔 Sending compliance alert for {report['framework']}")
        # In production, this would integrate with notification systems
        # (email, Slack, Jira, etc.)

    async def monitor_compliance(self):
        """Main compliance monitoring loop"""
        logger.info("🛡️ Starting Compliance Monitoring Service...")

        if not await self.initialize():
            logger.error("❌ Compliance monitor initialization failed")
            return

        while self.active:
            try:
                # Get active compliance frameworks
                frameworks = await self.get_active_compliance_frameworks()

                for framework in frameworks:
                    # Validate compliance
                    validation_result = await self.validate_framework_compliance(framework)

                    # Get related security events from SIEM
                    siem_events = await self.query_siem_for_compliance_events(framework)

                    # Generate compliance report
                    report = await self.generate_compliance_report(framework, validation_result, siem_events)

                    # Check for non-compliance and send alerts
                    if report['compliance_status'] == "NON-COMPLIANT" or report['risk_score'] > 70:
                        await self.send_alert(report)

                    # Log summary
                    logger.info(f"📊 {framework} Compliance Status: {report['compliance_status']}")
                    logger.info(f"📈 Risk Score: {report['risk_score']}")
                    logger.info(f"🔍 Recommendations: {len(report['recommendations'])}")

                # Wait before next check
                logger.info(f"🔄 Next compliance check in {self.compliance_check_interval//60} minutes")
                await asyncio.sleep(self.compliance_check_interval)

            except Exception as e:
                logger.error(f"❌ Compliance monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def shutdown(self):
        """Gracefully shutdown the compliance monitor"""
        logger.info("🛑 Shutting down Compliance Monitoring Service...")
        self.active = False
        if hasattr(self, 'session'):
            await self.session.close()

async def main():
    """Main compliance monitoring entry point"""
    monitor = ComplianceMonitor()

    # Set up signal handlers
    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(
            sig, lambda: asyncio.create_task(monitor.shutdown())
        )

    try:
        await monitor.monitor_compliance()
    finally:
        await monitor.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Compliance monitor shutdown complete")
