from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from aiohttp import ClientSession
from src.orchestrator.core.workflow_engine import TaskExecutor, WorkflowTask, WorkflowExecution

@dataclass
class ComplianceCheckConfig:
    compliance_service_url: str
    timeout_minutes: int = 30
    retry_count: int = 1
    retry_delay_seconds: int = 60

class ComplianceCheckExecutor(TaskExecutor):
    """Executor for compliance checking tasks"""
    
    def __init__(self, config: ComplianceCheckConfig):
        self.config = config
        
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compliance check"""
        try:
            params = task.parameters
            framework = params.get('framework')  # GDPR, NIS2, SOC2, etc.
            scope = params.get('scope', 'full')
            
            async with ClientSession() as session:
                payload = {
                    'framework': framework,
                    'scope': scope,
                    'assets': params.get('assets', []),
                    'workflow_execution_id': context['execution_id']
                }
                
                async with session.post(f"{self.config.compliance_service_url}/api/v1/compliance/check", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'framework': framework,
                            'compliance_score': result.get('score', 0),
                            'passed_controls': result.get('passed_controls', []),
                            'failed_controls': result.get('failed_controls', []),
                            'recommendations': result.get('recommendations', []),
                            'risk_level': result.get('risk_level', 'unknown')
                        }
                    else:
                        raise Exception(f"Compliance check failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            raise
            
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate compliance check parameters"""
        required_fields = ['framework']
        return all(field in parameters for field in required_fields)