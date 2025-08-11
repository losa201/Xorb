"""
XORB Security Compliance Integration
Handles integration between security events and compliance monitoring
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("XORB-Security-Compliance")

class ComplianceIntegrator:
    """Integrates security events with compliance monitoring framework"""
    
    def __init__(self, compliance_client, security_config: Dict[str, Any] = None):
        """Initialize compliance integrator"""
        self.compliance_client = compliance_client
        self.security_config = security_config or {
            "compliance_check_interval": 300,  # 5 minutes
            "security_event_thresholds": {
                "high_risk": 70,
                "medium_risk": 50
            }
        }
        self.last_compliance_check = 0
        self.active_frameworks = []
        
    async def initialize(self):
        """Initialize compliance integration"""
        try:
            self.active_frameworks = await self.compliance_client.get_active_compliance_frameworks()
            logger.info(f"Intialized compliance integration for frameworks: {self.active_frameworks}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize compliance integration: {e}")
            return False
    
    async def process_security_event(self, event: Dict[str, Any]) -> bool:
        """Process a security event for compliance impact"""
        try:
            logger.debug(f"Processing security event for compliance: {event.get('event_type')}")
            
            # Calculate risk score for the event
            risk_score = self._calculate_event_risk_score(event)
            
            # Determine compliance impact
            if risk_score > self.security_config["security_event_thresholds"]["high_risk"]:
                await self._handle_high_risk_event(event, risk_score)
            elif risk_score > self.security_config["security_event_thresholds"]["medium_risk"]:
                await self._handle_medium_risk_event(event, risk_score)
            
            return True
        except Exception as e:
            logger.error(f"Error processing security event: {e}")
            return False
    
    def _calculate_event_risk_score(self, event: Dict[str, Any]) -> int:
        """Calculate risk score for a security event"""
        base_score = 0
        
        # Base score by event type
        event_type_scores = {
            "blocked_request": 30,
            "invalid_headers": 20,
            "rate_limit_exceeded": 25,
            "suspicious_activity": 35,
            "potential_attack": 50
        }
        
        base_score += event_type_scores.get(event.get("event_type", "unknown"), 10)
        
        # Adjust by severity
        severity = event.get("severity", "medium")
        if severity == "high":
            base_score += 20
        elif severity == "critical":
            base_score += 30
        
        # Adjust by source
        if event.get("source") == "external":
            base_score += 10
        
        return min(base_score, 100)
    
    async def _handle_high_risk_event(self, event: Dict[str, Any], risk_score: int):
        """Handle high risk security events"""
        logger.warning(f"High risk event detected: {event.get('event_type')}")
        
        # Create compliance alert
        alert_data = {
            "alert_type": "high_risk_security_event",
            "risk_score": risk_score,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "severity": "high",
            "description": f"High risk security event detected: {event.get('event_type')}",
            "remediation": "Immediate investigation and mitigation required"
        }
        
        # Send alert through compliance system
        await self.compliance_client.send_alert(alert_data)
        
        # Trigger compliance check
        await self.trigger_compliance_check()
        
    async def _handle_medium_risk_event(self, event: Dict[str, Any], risk_score: int):
        """Handle medium risk security events"""
        logger.info(f"Medium risk event detected: {event.get('event_type')}")
        
        # Create compliance alert
        alert_data = {
            "alert_type": "medium_risk_security_event",
            "risk_score": risk_score,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "severity": "medium",
            "description": f"Medium risk security event detected: {event.get('event_type')}",
            "remediation": "Investigation and mitigation recommended"
        }
        
        # Send alert through compliance system
        await self.compliance_client.send_alert(alert_data)
        
    async def trigger_compliance_check(self, framework: str = None):
        """Trigger compliance check for all or specific framework"""
        logger.info("Triggering compliance check due to security event")
        
        if framework:
            frameworks = [framework]
        else:
            frameworks = self.active_frameworks
        
        for fw in frameworks:
            # Validate compliance
            validation_result = await self.compliance_client.validate_framework_compliance(fw)
            
            # Get related security events
            timeframe = "24h"
            siem_events = await self.compliance_client.query_siem_for_compliance_events(fw, timeframe)
            
            # Generate compliance report
            report = await self.compliance_client.generate_compliance_report(fw, validation_result, siem_events)
            
            # Log summary
            logger.info(f"{fw} Compliance Status: {report['compliance_status']}")
            logger.info(f"Risk Score: {report['risk_score']}")
            logger.info(f"Recommendations: {len(report['recommendations'])}")
    
    async def check_compliance_status(self, framework: str = None) -> Dict[str, Any]:
        """Check compliance status for all or specific framework"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "compliance_status": {}
        }
        
        if framework:
            frameworks = [framework]
        else:
            frameworks = self.active_frameworks
        
        for fw in frameworks:
            # Get validation status
            validation_status = await self.compliance_client.get_validation_status(fw)
            
            # Get risk score from recent events
            risk_score = await self._calculate_framework_risk_score(fw)
            
            result["compliance_status"][fw] = {
                "validation_status": validation_status,
                "risk_score": risk_score,
                "last_checked": datetime.now().isoformat()
            }
        
        return result
    
    async def _calculate_framework_risk_score(self, framework: str) -> int:
        """Calculate risk score for a compliance framework based on recent security events"""
        # Get recent security events
        timeframe = "7d"
        events = await self.compliance_client.query_siem_for_compliance_events(framework, timeframe)
        
        if not events:
            return 50  # Default score if no events
        
        # Calculate base score from events
        base_score = sum(event.get("risk_score", 50) for event in events) / len(events)
        
        # Adjust based on event recency
        now = datetime.now()
        recency_factor = 0
        
        for event in events:
            event_time = datetime.fromisoformat(event.get("timestamp", now.isoformat()))
            hours_old = (now - event_time).total_seconds() / 3600
            
            # More weight to recent events
            if hours_old < 24:  # Last day
                recency_factor += 0.3
            elif hours_old < 168:  # Last week
                recency_factor += 0.1
            
        # Calculate final score
        final_score = min(base_score * (1 + recency_factor), 100)
        return int(final_score)
    
    async def generate_compliance_recommendations(self, framework: str) -> List[str]:
        """Generate compliance recommendations based on security events and framework requirements"""
        # Get validation results
        validation_result = await self.compliance_client.validate_framework_compliance(framework)
        
        # Get related security events
        timeframe = "7d"
        siem_events = await self.compliance_client.query_siem_for_compliance_events(framework, timeframe)
        
        # Generate recommendations
        recommendations = []
        
        # Add validation-based recommendations
        if not validation_result.get('compliant', False):
            recommendations.extend(validation_result.get('non_compliant_controls', []))
        
        # Add event-based recommendations
        for event in siem_events:
            if event.get('severity') in ["CRITICAL", "HIGH"]:
                recommendations.append(event.get('description', 'Investigate security event'))
        
        return list(set(recommendations))  # Remove duplicates

# Factory function to create compliance integrator
async def create_compliance_integrator(compliance_client, security_config: Dict[str, Any] = None):
    """Create and initialize compliance integrator"""
    integrator = ComplianceIntegrator(compliance_client, security_config)
    if await integrator.initialize():
        return integrator
    return None