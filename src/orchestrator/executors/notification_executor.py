from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from aiohttp import ClientSession
from src.orchestrator.core.workflow_engine import TaskExecutor, WorkflowTask, WorkflowExecution

@dataclass
class NotificationConfig:
    """Configuration for notification executor"""
    notification_service_url: str
    default_channels: List[str] = None
    timeout_minutes: int = 5
    retry_count: int = 3
    retry_delay_seconds: int = 30

class NotificationExecutor(TaskExecutor):
    """Executor for notification tasks"""

    def __init__(self, config: NotificationConfig):
        self.config = config

    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification task"""
        try:
            params = task.parameters

            # Merge default channels with task-specific channels
            channels = params.get('channels', self.config.default_channels or ['email'])

            async with ClientSession() as session:
                payload = {
                    'recipients': params.get('recipients', []),
                    'subject': params.get('subject', ''),
                    'message': self._render_message(params.get('template', ''), context),
                    'channels': channels,
                    'priority': params.get('priority', 'normal'),
                    'attachments': params.get('attachments', [])
                }

                # Add execution context for tracing
                headers = {
                    'X-Workflow-Execution-ID': context['execution_id'],
                    'X-Workflow-ID': context['workflow_id'],
                    'X-Task-ID': task.id
                }

                async with session.post(
                    f"{self.config.notification_service_url}/api/v1/notifications/send",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'notification_id': result.get('id'),
                            'status': 'sent',
                            'recipients_count': len(payload['recipients']),
                            'channels_used': payload['channels'],
                            'delivery_time': datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Notification failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Notification failed: {e}")
            raise

    def _render_message(self, template: str, context: Dict[str, Any]) -> str:
        """Render message template with context variables"""
        try:
            # Simple template rendering - in production, use Jinja2 or similar
            message = template
            for key, value in context.get('variables', {}).items():
                message = message.replace(f"{{{key}}}", str(value))
            return message
        except Exception:
            return template

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate notification parameters"""
        required_fields = ['recipients', 'subject', 'template']
        return all(field in parameters for field in required_fields)
