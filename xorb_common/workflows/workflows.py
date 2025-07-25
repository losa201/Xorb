"""
Xorb 2.0 - Enhanced Temporal Workflows with Metrics
Comprehensive workflow definitions for security operations.
"""
from collections import deque
from datetime import timedelta
from typing import List, Set, Dict, Any, Optional
from temporalio import workflow
from prometheus_client import Counter, Histogram, Gauge

from xorb_core.logging import get_logger

# Import activities and models with error handling
with workflow.unsafe.imports_passed_through():
    from .activities import (
        run_agent, 
        enumerate_subdomains_activity, 
        resolve_dns_activity,
        perform_scan,
        health_check_activity
    )
    
    try:
        from xorb_core.agent_registry import agent_registry
        from xorb_core.models.agents import DiscoveryTarget, Finding
    except ImportError:
        agent_registry = None
        DiscoveryTarget = None
        Finding = None

# Initialize logger
logger = get_logger(__name__)

# Prometheus metrics for workflows
WORKFLOW_EXECUTIONS = Counter(
    'xorb_workflow_executions_total',
    'Total number of workflow executions',
    ['workflow_type', 'status']
)

WORKFLOW_DURATION = Histogram(
    'xorb_workflow_duration_seconds',
    'Duration of workflow executions',
    ['workflow_type']
)

ACTIVE_WORKFLOWS = Gauge(
    'xorb_active_workflows',
    'Number of currently active workflows'
)


def track_workflow(workflow_name: str):
    """Decorator to track workflow metrics."""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            ACTIVE_WORKFLOWS.inc()
            
            try:
                with WORKFLOW_DURATION.labels(workflow_type=workflow_name).time():
                    result = await func(self, *args, **kwargs)
                WORKFLOW_EXECUTIONS.labels(workflow_type=workflow_name, status='success').inc()
                return result
            except Exception as e:
                WORKFLOW_EXECUTIONS.labels(workflow_type=workflow_name, status='failure').inc()
                workflow.logger.error(f"Workflow {workflow_name} failed", error=str(e))
                raise
            finally:
                ACTIVE_WORKFLOWS.dec()
        
        return wrapper
    return decorator


@workflow.defn
class DynamicScanWorkflow:
    """
    Enhanced dynamic workflow that orchestrates security scans by intelligently
    selecting and running agents based on discovered findings.
    """

    @workflow.run
    @track_workflow("dynamic_scan")
    async def run(self, initial_target: DiscoveryTarget, max_depth: int = 3) -> List[Finding]:
        """
        Execute a dynamic scan workflow with configurable depth limiting.
        
        Args:
            initial_target: The initial target to scan
            max_depth: Maximum depth for recursive scanning (default: 3)
        """
        if not agent_registry:
            workflow.logger.error("Agent registry not available")
            return []
            
        all_findings: List[Finding] = []
        target_queue = deque([(initial_target, 0)])  # (target, depth)
        processed_targets: Set[str] = {f"{initial_target.target_type}::{initial_target.value}"}

        workflow.logger.info(f"Starting dynamic scan",
                           initial_target=initial_target.value,
                           max_depth=max_depth)

        while target_queue:
            current_target, depth = target_queue.popleft()
            
            if depth > max_depth:
                workflow.logger.info(f"Skipping target due to depth limit",
                                   target=current_target.value,
                                   depth=depth,
                                   max_depth=max_depth)
                continue

            workflow.logger.info(f"Processing target",
                               target=current_target.value,
                               depth=depth)

            # Get agents for this target type
            agents = agent_registry.get_agents_for_target_type(current_target.target_type)
            if not agents:
                workflow.logger.warning(f"No agents found for target type",
                                      target_type=current_target.target_type)
                continue

            # Schedule agent activities
            new_findings = await self._execute_agents(agents, current_target)
            all_findings.extend(new_findings)

            # Process findings to discover new targets
            for finding in new_findings:
                new_target = self._finding_to_target(finding)
                if new_target:
                    target_key = f"{new_target.target_type}::{new_target.value}"
                    if target_key not in processed_targets:
                        workflow.logger.info(f"Discovered new target",
                                           target=new_target.value,
                                           depth=depth + 1)
                        target_queue.append((new_target, depth + 1))
                        processed_targets.add(target_key)

        workflow.logger.info(f"Dynamic scan completed",
                           total_findings=len(all_findings),
                           targets_processed=len(processed_targets))
        return all_findings

    async def _execute_agents(self, agents: List, target: DiscoveryTarget) -> List[Finding]:
        """Execute multiple agents concurrently on a target."""
        activity_promises = []
        
        for agent in agents:
            promise = workflow.execute_activity(
                run_agent,
                args=[agent.name, target],
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=workflow.RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                    maximum_attempts=3
                )
            )
            activity_promises.append(promise)

        # Wait for all activities to complete
        results = await workflow.gather(*activity_promises, return_exceptions=True)
        
        all_findings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                workflow.logger.error(f"Agent execution failed",
                                    agent=agents[i].name,
                                    error=str(result))
            elif result:
                all_findings.extend(result)
        
        return all_findings

    def _finding_to_target(self, finding: Finding) -> Optional[DiscoveryTarget]:
        """Convert a finding into a new discovery target if applicable."""
        if finding.finding_type == "subdomain":
            return DiscoveryTarget(value=f"https://{finding.target}", target_type="url")
        elif finding.finding_type == "open_port":
            # Example: convert open port to HTTP endpoint
            return DiscoveryTarget(value=f"http://{finding.target}:80", target_type="url")
        elif finding.finding_type == "url":
            return DiscoveryTarget(value=finding.target, target_type="url")
        
        return None


@workflow.defn
class DiscoveryWorkflow:
    """
    Enhanced discovery workflow for DNS and subdomain enumeration.
    """

    @workflow.run
    @track_workflow("discovery")
    async def run(self, domain: str) -> Dict[str, Any]:
        """
        Execute DNS discovery and subdomain enumeration.
        
        Args:
            domain: The domain to discover
            
        Returns:
            Dictionary containing all discovery results
        """
        workflow.logger.info(f"Starting discovery workflow for domain: {domain}")
        
        results = {
            'domain': domain,
            'subdomains': [],
            'dns_records': [],
            'status': 'running'
        }
        
        try:
            # Execute subdomain enumeration and DNS resolution concurrently
            subdomain_promise = workflow.execute_activity(
                enumerate_subdomains_activity,
                args=[domain],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=workflow.RetryPolicy(maximum_attempts=2)
            )
            
            dns_promise = workflow.execute_activity(
                resolve_dns_activity,
                args=[domain],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=workflow.RetryPolicy(maximum_attempts=3)
            )
            
            # Wait for both activities to complete
            subdomain_findings, dns_findings = await workflow.gather(
                subdomain_promise, 
                dns_promise,
                return_exceptions=True
            )
            
            # Process results
            if not isinstance(subdomain_findings, Exception):
                results['subdomains'] = subdomain_findings
                workflow.logger.info(f"Subdomain enumeration completed",
                                   subdomains_found=len(subdomain_findings))
            else:
                workflow.logger.error(f"Subdomain enumeration failed",
                                    error=str(subdomain_findings))
            
            if not isinstance(dns_findings, Exception):
                results['dns_records'] = dns_findings
                workflow.logger.info(f"DNS resolution completed",
                                   records_found=len(dns_findings))
            else:
                workflow.logger.error(f"DNS resolution failed",
                                    error=str(dns_findings))
            
            results['status'] = 'completed'
            workflow.logger.info(f"Discovery workflow completed successfully for domain: {domain}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            workflow.logger.error(f"Discovery workflow failed",
                                domain=domain,
                                error=str(e))
        
        return results


@workflow.defn
class TargetOnboardingWorkflow:
    """
    Enhanced target onboarding workflow for new security targets.
    """

    @workflow.run
    @track_workflow("target_onboarding")
    async def run(self, target: Dict[str, Any], scan_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Onboard a new target with comprehensive scanning.
        
        Args:
            target: Target information (URL, IP, domain, etc.)
            scan_config: Optional scan configuration
            
        Returns:
            Onboarding results including scan findings
        """
        target_id = target.get('id', 'unknown')
        workflow.logger.info(f"Starting target onboarding",
                           target_id=target_id)
        
        results = {
            'target': target,
            'onboarding_status': 'in_progress',
            'scans': [],
            'findings': []
        }
        
        try:
            # Perform initial health check
            health_check = await workflow.execute_activity(
                health_check_activity,
                start_to_close_timeout=timedelta(seconds=30)
            )
            
            workflow.logger.info(f"Health check completed", health_status=health_check['status'])
            
            # Perform primary scan
            scan_result = await workflow.execute_activity(
                perform_scan,
                args=[target, scan_config],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=workflow.RetryPolicy(
                    initial_interval=timedelta(seconds=5),
                    maximum_attempts=2
                )
            )
            
            results['scans'].append(scan_result)
            results['findings'].extend(scan_result.get('findings', []))
            
            # If configured for comprehensive scanning, perform additional scans
            if scan_config and scan_config.get('comprehensive', False):
                additional_scan = await workflow.execute_activity(
                    perform_scan,
                    args=[target, {'type': 'comprehensive'}],
                    start_to_close_timeout=timedelta(minutes=60)
                )
                results['scans'].append(additional_scan)
                results['findings'].extend(additional_scan.get('findings', []))
            
            results['onboarding_status'] = 'completed'
            workflow.logger.info(f"Target onboarding completed successfully",
                               target_id=target_id,
                               total_findings=len(results['findings']))
            
        except Exception as e:
            results['onboarding_status'] = 'failed'
            results['error'] = str(e)
            workflow.logger.error(f"Target onboarding failed",
                                target_id=target_id,
                                error=str(e))
        
        return results


@workflow.defn
class HealthCheckWorkflow:
    """
    Simple health check workflow for monitoring worker status.
    """

    @workflow.run
    @track_workflow("health_check")
    async def run(self) -> Dict[str, Any]:
        """Execute a health check workflow."""
        return await workflow.execute_activity(
            health_check_activity,
            start_to_close_timeout=timedelta(seconds=30)
        )