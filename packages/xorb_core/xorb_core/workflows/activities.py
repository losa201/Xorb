"""
Xorb 2.0 - Enhanced Temporal Activities with Metrics
Unified activity definitions for all Xorb workflows with comprehensive monitoring.
"""
from temporalio import activity
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from prometheus_client import Counter, Histogram, Gauge

from xorb_core.logging import get_logger
from xorb_core.models.agents import DiscoveryTarget, Finding

# Import activities with error handling
try:
    from xorb_core.agent_registry import agent_registry
except ImportError:
    agent_registry = None

try:
    from xorb_core.agents.discovery import enumerate_subdomains, resolve_dns
except ImportError:
    enumerate_subdomains = None
    resolve_dns = None

# Initialize logger
logger = get_logger(__name__)

# Prometheus metrics for activities
ACTIVITY_EXECUTIONS = Counter(
    'xorb_activity_executions_total',
    'Total number of activity executions',
    ['activity_name', 'status']
)

ACTIVITY_DURATION = Histogram(
    'xorb_activity_duration_seconds',
    'Duration of activity executions',
    ['activity_name']
)

ACTIVE_ACTIVITIES = Gauge(
    'xorb_active_activities',
    'Number of currently active activities'
)

def track_activity(activity_name: str):
    """Decorator to track activity metrics."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            ACTIVE_ACTIVITIES.inc()
            start_time = datetime.utcnow()
            
            try:
                with ACTIVITY_DURATION.labels(activity_name=activity_name).time():
                    result = await func(*args, **kwargs)
                ACTIVITY_EXECUTIONS.labels(activity_name=activity_name, status='success').inc()
                return result
            except Exception as e:
                ACTIVITY_EXECUTIONS.labels(activity_name=activity_name, status='failure').inc()
                logger.error(f"Activity {activity_name} failed", error=str(e), exc_info=True)
                raise
            finally:
                ACTIVE_ACTIVITIES.dec()
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Activity {activity_name} completed", duration_seconds=duration)
        
        return wrapper
    return decorator


@activity.defn
@track_activity("run_agent")
async def run_agent(agent_name: str, target: DiscoveryTarget) -> List[Finding]:
    """
    Generic activity to run any registered agent by name.
    Enhanced with comprehensive error handling and metrics.
    """
    activity.logger.info(f"Executing agent '{agent_name}' on target '{target.value}'")
    
    if not agent_registry:
        raise RuntimeError("Agent registry not available - check imports")
    
    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise ValueError(f"Agent '{agent_name}' not found in registry")
    
    if target.target_type not in agent.accepted_target_types:
        activity.logger.warning(
            f"Agent '{agent_name}' does not accept target type '{target.target_type}'. Skipping."
        )
        return []
    
    try:
        findings = await agent.run(target)
        activity.logger.info(f"Agent '{agent_name}' completed successfully", 
                           findings_count=len(findings))
        return findings
    except Exception as e:
        activity.logger.error(f"Agent '{agent_name}' execution failed", 
                            error=str(e), 
                            target=target.value,
                            exc_info=True)
        raise


@activity.defn 
@track_activity("enumerate_subdomains")
async def enumerate_subdomains_activity(domain: str, timeout_seconds: int = 300) -> List[Finding]:
    """
    Enumerate subdomains for a given domain with timeout and metrics.
    """
    activity.logger.info(f"Starting subdomain enumeration for domain: {domain}")
    
    if not enumerate_subdomains:
        raise RuntimeError("Subdomain enumeration function not available")
    
    try:
        # Set activity timeout
        findings = await asyncio.wait_for(
            enumerate_subdomains(domain, agent_id="discovery-agent-001"),
            timeout=timeout_seconds
        )
        
        activity.logger.info(f"Subdomain enumeration completed",
                           domain=domain,
                           subdomains_found=len(findings))
        return findings
        
    except asyncio.TimeoutError:
        activity.logger.error(f"Subdomain enumeration timed out", 
                            domain=domain, 
                            timeout_seconds=timeout_seconds)
        raise
    except Exception as e:
        activity.logger.error(f"Subdomain enumeration failed",
                            domain=domain,
                            error=str(e),
                            exc_info=True)
        raise


@activity.defn
@track_activity("resolve_dns")
async def resolve_dns_activity(hostname: str, timeout_seconds: int = 60) -> List[Finding]:
    """
    Resolve DNS for a given hostname with timeout and metrics.
    """
    activity.logger.info(f"Starting DNS resolution for hostname: {hostname}")
    
    if not resolve_dns:
        raise RuntimeError("DNS resolution function not available")
    
    try:
        findings = await asyncio.wait_for(
            resolve_dns(hostname, agent_id="discovery-agent-001"),
            timeout=timeout_seconds
        )
        
        activity.logger.info(f"DNS resolution completed",
                           hostname=hostname,
                           records_found=len(findings))
        return findings
        
    except asyncio.TimeoutError:
        activity.logger.error(f"DNS resolution timed out",
                            hostname=hostname,
                            timeout_seconds=timeout_seconds)
        raise
    except Exception as e:
        activity.logger.error(f"DNS resolution failed",
                            hostname=hostname,
                            error=str(e),
                            exc_info=True)
        raise


@activity.defn
@track_activity("perform_scan")
async def perform_scan(target: Dict[str, Any], scan_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generic scan activity that can be configured for different scan types.
    Used for target onboarding workflow.
    """
    scan_type = scan_config.get('type', 'basic') if scan_config else 'basic'
    activity.logger.info(f"Performing {scan_type} scan", target=target)
    
    try:
        # Simulate scan execution based on configuration
        if scan_type == 'basic':
            result = await _perform_basic_scan(target)
        elif scan_type == 'comprehensive':
            result = await _perform_comprehensive_scan(target)
        else:
            raise ValueError(f"Unknown scan type: {scan_type}")
        
        activity.logger.info(f"Scan completed successfully",
                           scan_type=scan_type,
                           target=target,
                           findings_count=result.get('findings_count', 0))
        return result
        
    except Exception as e:
        activity.logger.error(f"Scan failed",
                            scan_type=scan_type,
                            target=target,
                            error=str(e),
                            exc_info=True)
        raise


async def _perform_basic_scan(target: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a basic security scan."""
    # Simulate basic scan logic
    await asyncio.sleep(1)  # Simulate work
    return {
        'status': 'completed',
        'scan_type': 'basic',
        'target': target,
        'findings_count': 5,
        'findings': [
            {'type': 'info', 'description': 'Basic scan completed'},
            {'type': 'low', 'description': 'Minor configuration issue'},
        ]
    }


async def _perform_comprehensive_scan(target: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a comprehensive security scan."""
    # Simulate comprehensive scan logic
    await asyncio.sleep(3)  # Simulate more work
    return {
        'status': 'completed',
        'scan_type': 'comprehensive',
        'target': target,
        'findings_count': 12,
        'findings': [
            {'type': 'info', 'description': 'Comprehensive scan completed'},
            {'type': 'medium', 'description': 'Security header missing'},
            {'type': 'high', 'description': 'Potential vulnerability detected'},
        ]
    }


@activity.defn
@track_activity("health_check")
async def health_check_activity() -> Dict[str, Any]:
    """
    Simple health check activity for monitoring worker status.
    """
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'worker_id': activity.info().worker_identity,
        'task_queue': activity.info().task_queue
    }