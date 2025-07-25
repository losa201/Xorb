"""
Xorb 2.0 - Temporal Worker Entry Point with Metrics
Includes Prometheus metrics collection for worker operations.
"""
import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from xorb_core.logging import configure_logging, get_logger

# Configure logging
configure_logging(level="INFO", service_name="xorb-worker")
log = get_logger(__name__)

# Prometheus metrics for worker
WORKFLOW_EXECUTIONS = Counter(
    'xorb_workflow_executions_total',
    'Total number of workflow executions',
    ['workflow_type', 'status']
)

ACTIVITY_EXECUTIONS = Counter(
    'xorb_activity_executions_total', 
    'Total number of activity executions',
    ['activity_name', 'status']
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

WORKER_HEALTH = Gauge(
    'xorb_worker_health',
    'Worker health status (1=healthy, 0=unhealthy)'
)


async def main():
    """Main worker entry point with metrics and health monitoring."""
    try:
        # Start Prometheus metrics server on port 9000
        metrics_port = int(os.getenv('METRICS_PORT', '9000'))
        start_http_server(metrics_port)
        log.info("Metrics server started", port=metrics_port)
        
        # Set worker as healthy
        WORKER_HEALTH.set(1)
        
        # Connect to Temporal
        temporal_host = os.getenv('TEMPORAL_HOST', 'temporal:7233')
        client = await Client.connect(temporal_host, namespace="default")
        log.info("Connected to Temporal", host=temporal_host)
        
        # Import enhanced workflows and activities
        try:
            from xorb_core.workflows.workflows import (
                DynamicScanWorkflow, 
                DiscoveryWorkflow, 
                TargetOnboardingWorkflow,
                HealthCheckWorkflow
            )
            from xorb_core.workflows.activities import (
                run_agent, 
                enumerate_subdomains_activity, 
                resolve_dns_activity,
                perform_scan,
                health_check_activity
            )
            
            workflows = [
                DynamicScanWorkflow, 
                DiscoveryWorkflow, 
                TargetOnboardingWorkflow,
                HealthCheckWorkflow
            ]
            activities = [
                run_agent, 
                enumerate_subdomains_activity, 
                resolve_dns_activity,
                perform_scan,
                health_check_activity
            ]
            
            log.info("Successfully loaded enhanced workflows and activities",
                    workflows_count=len(workflows),
                    activities_count=len(activities))
            
        except ImportError as e:
            log.warning("Enhanced workflows not available, trying fallback", error=str(e))
            # Fallback to legacy activities if available
            try:
                from services.worker.activities import enumerate_subdomains_activity, resolve_dns_activity
                from services.worker.workflows import DiscoveryWorkflow
                from xorb_core.workflows.activities import health_check_activity
                
                workflows = [DiscoveryWorkflow]
                activities = [enumerate_subdomains_activity, resolve_dns_activity, health_check_activity]
                
                log.info("Loaded fallback workflows and activities")
                
            except ImportError:
                log.error("No workflows/activities available")
                WORKER_HEALTH.set(0)
                return
        
        # Create worker
        task_queue = os.getenv('TASK_QUEUE', 'xorb-task-queue')
        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=workflows,
            activities=activities,
        )
        
        log.info("Starting Temporal worker", 
                task_queue=task_queue,
                workflows=[w.__name__ for w in workflows],
                activities=[a.__name__ for a in activities])
        
        # Run worker
        await worker.run()
        
    except Exception as e:
        log.error("Worker failed", error=str(e))
        WORKER_HEALTH.set(0)
        raise
    finally:
        log.info("Worker shutting down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Worker interrupted by user")
    except Exception as e:
        log.error("Worker failed to start", error=str(e))
        exit(1)