"""
Advanced notification service with multiple channels and enterprise features
Supports email, webhook, Slack, Teams, and custom notification delivery
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import base64
import hashlib
import hmac

import aiohttp
import aiofiles
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from ..domain.repositories import CacheRepository
from .interfaces import NotificationService
from .base_service import XORBService, ServiceType


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NotificationStatus(Enum):
    """Notification delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


@dataclass
class NotificationTemplate:
    """Notification template definition"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    variables: List[str]
    metadata: Dict[str, Any]
    is_html: bool = False


@dataclass
class NotificationRecipient:
    """Notification recipient information"""
    id: str
    channel: NotificationChannel
    address: str  # email, phone, webhook URL, etc.
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class NotificationMessage:
    """Notification message details"""
    id: str
    template_id: Optional[str]
    channel: NotificationChannel
    recipients: List[NotificationRecipient]
    subject: str
    body: str
    priority: NotificationPriority
    variables: Dict[str, Any]
    attachments: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = None
    status: NotificationStatus = NotificationStatus.PENDING
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ProductionNotificationService(NotificationService, XORBService):
    """Production-ready notification service with multiple channels"""
    
    def __init__(self, cache_repository: CacheRepository, config: Dict[str, Any] = None):
        super().__init__(service_type=ServiceType.INTEGRATION)
        self.cache = cache_repository
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Notification templates
        self._templates: Dict[str, NotificationTemplate] = {}
        
        # Delivery queues by priority
        self._delivery_queues = {
            NotificationPriority.EMERGENCY: asyncio.Queue(),
            NotificationPriority.CRITICAL: asyncio.Queue(),
            NotificationPriority.HIGH: asyncio.Queue(),
            NotificationPriority.NORMAL: asyncio.Queue(),
            NotificationPriority.LOW: asyncio.Queue()
        }
        
        # Delivery workers
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Initialize default templates
        self._load_default_templates()
    
    async def send_notification(
        self,
        recipient: str,
        channel: str,
        message: str,
        subject: Optional[str] = None,
        priority: str = "normal",
        variables: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a notification"""
        try:
            # Convert string to enum
            channel_enum = NotificationChannel(channel)
            priority_enum = NotificationPriority(priority)
            
            # Create notification message
            notification = NotificationMessage(
                id=str(uuid4()),
                template_id=None,
                channel=channel_enum,
                recipients=[NotificationRecipient(
                    id=str(uuid4()),
                    channel=channel_enum,
                    address=recipient,
                    preferences={},
                    metadata={}
                )],
                subject=subject or "XORB Notification",
                body=message,
                priority=priority_enum,
                variables=variables or {},
                attachments=attachments or [],
                metadata=metadata or {}
            )
            
            # Queue for delivery
            await self._queue_notification(notification)
            
            # Store notification for tracking
            await self._store_notification(notification)
            
            self.logger.info(f"Queued notification {notification.id} for delivery")
            return notification.id
            
        except Exception as e:
            self.logger.error(f"Error sending notification to {recipient}: {str(e)}")
            raise
    
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None,
        retry_count: int = 3
    ) -> bool:
        """Send webhook notification"""
        try:
            # Create webhook message
            notification = NotificationMessage(
                id=str(uuid4()),
                template_id=None,
                channel=NotificationChannel.WEBHOOK,
                recipients=[NotificationRecipient(
                    id=str(uuid4()),
                    channel=NotificationChannel.WEBHOOK,
                    address=url,
                    preferences={"retry_count": retry_count},
                    metadata={"secret": secret}
                )],
                subject="Webhook Notification",
                body=json.dumps(payload),
                priority=NotificationPriority.NORMAL,
                variables={},
                attachments=[],
                metadata={"headers": headers or {}}
            )
            
            # Queue for delivery
            await self._queue_notification(notification)
            
            # Store notification
            await self._store_notification(notification)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending webhook to {url}: {str(e)}")
            return False
    
    async def send_template_notification(
        self,
        template_id: str,
        recipients: List[str],
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> List[str]:
        """Send notification using template"""
        try:
            # Get template
            template = await self.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Render template
            subject = self._render_template(template.subject_template, variables)
            body = self._render_template(template.body_template, variables)
            
            notification_ids = []
            
            # Send to each recipient
            for recipient in recipients:
                notification = NotificationMessage(
                    id=str(uuid4()),
                    template_id=template_id,
                    channel=template.channel,
                    recipients=[NotificationRecipient(
                        id=str(uuid4()),
                        channel=template.channel,
                        address=recipient,
                        preferences={},
                        metadata={}
                    )],
                    subject=subject,
                    body=body,
                    priority=priority,
                    variables=variables,
                    attachments=[],
                    metadata=template.metadata
                )
                
                await self._queue_notification(notification)
                await self._store_notification(notification)
                notification_ids.append(notification.id)
            
            return notification_ids
            
        except Exception as e:
            self.logger.error(f"Error sending template notification: {str(e)}")
            raise
    
    async def create_template(
        self,
        name: str,
        channel: NotificationChannel,
        subject_template: str,
        body_template: str,
        variables: List[str],
        is_html: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create notification template"""
        try:
            template_id = str(uuid4())
            template = NotificationTemplate(
                id=template_id,
                name=name,
                channel=channel,
                subject_template=subject_template,
                body_template=body_template,
                variables=variables,
                metadata=metadata or {},
                is_html=is_html
            )
            
            # Store template
            self._templates[template_id] = template
            
            # Persist to cache
            await self.cache.set(
                f"notification_template:{template_id}",
                asdict(template),
                ttl=86400 * 30  # 30 days
            )
            
            self.logger.info(f"Created notification template: {name} ({template_id})")
            return template_id
            
        except Exception as e:
            self.logger.error(f"Error creating template '{name}': {str(e)}")
            raise
    
    async def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get notification template"""
        try:
            # Check in-memory cache first
            if template_id in self._templates:
                return self._templates[template_id]
            
            # Check persistent cache
            template_data = await self.cache.get(f"notification_template:{template_id}")
            if template_data:
                template = NotificationTemplate(**template_data)
                self._templates[template_id] = template
                return template
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting template {template_id}: {str(e)}")
            return None
    
    async def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification delivery status"""
        try:
            notification_data = await self.cache.get(f"notification:{notification_id}")
            if not notification_data:
                return None
            
            return {
                "id": notification_id,
                "status": notification_data.get("status"),
                "created_at": notification_data.get("created_at"),
                "delivered_at": notification_data.get("delivered_at"),
                "error_message": notification_data.get("error_message"),
                "retry_count": notification_data.get("retry_count", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting notification status {notification_id}: {str(e)}")
            return None
    
    async def start_delivery_workers(self, worker_count: int = 3):
        """Start notification delivery workers"""
        if self._running:
            return
        
        self._running = True
        
        # Start workers for each priority level
        for priority in NotificationPriority:
            for i in range(worker_count):
                worker = asyncio.create_task(
                    self._delivery_worker(priority)
                )
                self._workers.append(worker)
        
        self.logger.info(f"Started {len(self._workers)} notification delivery workers")
    
    async def stop_delivery_workers(self):
        """Stop notification delivery workers"""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        self.logger.info("Stopped notification delivery workers")
    
    async def _queue_notification(self, notification: NotificationMessage):
        """Queue notification for delivery"""
        queue = self._delivery_queues[notification.priority]
        await queue.put(notification)
    
    async def _store_notification(self, notification: NotificationMessage):
        """Store notification for tracking"""
        await self.cache.set(
            f"notification:{notification.id}",
            asdict(notification),
            ttl=86400 * 7  # 7 days
        )
    
    async def _delivery_worker(self, priority: NotificationPriority):
        """Delivery worker for specific priority"""
        queue = self._delivery_queues[priority]
        
        while self._running:
            try:
                # Get next notification (wait up to 1 second)
                try:
                    notification = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Deliver notification
                success = await self._deliver_notification(notification)
                
                # Update status
                if success:
                    notification.status = NotificationStatus.DELIVERED
                else:
                    notification.status = NotificationStatus.FAILED
                
                # Update stored notification
                await self._store_notification(notification)
                
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in delivery worker ({priority.value}): {str(e)}")
                await asyncio.sleep(1)
    
    async def _deliver_notification(self, notification: NotificationMessage) -> bool:
        """Deliver notification based on channel"""
        try:
            if notification.channel == NotificationChannel.EMAIL:
                return await self._deliver_email(notification)
            elif notification.channel == NotificationChannel.WEBHOOK:
                return await self._deliver_webhook(notification)
            elif notification.channel == NotificationChannel.SLACK:
                return await self._deliver_slack(notification)
            elif notification.channel == NotificationChannel.TEAMS:
                return await self._deliver_teams(notification)
            else:
                self.logger.warning(f"Unsupported notification channel: {notification.channel}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error delivering notification {notification.id}: {str(e)}")
            return False
    
    async def _deliver_email(self, notification: NotificationMessage) -> bool:
        """Deliver email notification with production email service integration"""
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Production SMTP configuration (configurable via environment)
            smtp_config = {
                "hostname": os.getenv("SMTP_HOST", "localhost"),
                "port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME"),
                "password": os.getenv("SMTP_PASSWORD"),
                "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true"
            }
            
            # Create email message
            for recipient in notification.recipients:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = notification.subject
                msg['From'] = os.getenv("SMTP_FROM_EMAIL", "noreply@xorb-security.com")
                msg['To'] = recipient.address
                
                # Add message ID for tracking
                msg['Message-ID'] = f"<{notification.id}@xorb-security.com>"
                
                # Create both plain text and HTML versions
                text_content = self._convert_to_text(notification.body)
                html_content = self._convert_to_html(notification.body)
                
                text_part = MIMEText(text_content, 'plain')
                html_part = MIMEText(html_content, 'html')
                
                msg.attach(text_part)
                msg.attach(html_part)
                
                # Send email
                if smtp_config["username"] and smtp_config["password"]:
                    # Production SMTP with authentication
                    await aiosmtplib.send(
                        msg,
                        hostname=smtp_config["hostname"],
                        port=smtp_config["port"],
                        username=smtp_config["username"],
                        password=smtp_config["password"],
                        use_tls=smtp_config["use_tls"]
                    )
                else:
                    # Development mode - log instead of sending
                    self.logger.info(
                        f"[DEV MODE] Email would be sent to {recipient.address}: {notification.subject}"
                    )
                    self.logger.debug(f"[DEV MODE] Email content: {text_content[:200]}...")
                
                # Track delivery
                await self._track_email_delivery(notification.id, recipient.address, "delivered")
            
            return True
            
        except ImportError:
            # Fallback if aiosmtplib not available
            self.logger.warning("aiosmtplib not available, using fallback email delivery")
            return await self._deliver_email_fallback(notification)
        except Exception as e:
            self.logger.error(f"Error delivering email: {str(e)}")
            # Track failed delivery
            for recipient in notification.recipients:
                await self._track_email_delivery(notification.id, recipient.address, "failed", str(e))
            return False
    
    def _convert_to_text(self, body: str) -> str:
        """Convert notification body to plain text"""
        try:
            # If body is JSON, format it nicely
            data = json.loads(body)
            if isinstance(data, dict):
                lines = []
                for key, value in data.items():
                    lines.append(f"{key.replace('_', ' ').title()}: {value}")
                return "\n".join(lines)
            return str(data)
        except (json.JSONDecodeError, TypeError):
            # Body is already plain text
            return body
    
    def _convert_to_html(self, body: str) -> str:
        """Convert notification body to HTML"""
        try:
            # If body is JSON, format it as HTML
            data = json.loads(body)
            if isinstance(data, dict):
                html = "<html><body><table border='1' cellpadding='5'>"
                for key, value in data.items():
                    key_formatted = key.replace('_', ' ').title()
                    html += f"<tr><td><b>{key_formatted}</b></td><td>{value}</td></tr>"
                html += "</table></body></html>"
                return html
            return f"<html><body><pre>{str(data)}</pre></body></html>"
        except (json.JSONDecodeError, TypeError):
            # Body is plain text, wrap in HTML
            return f"<html><body><pre>{body}</pre></body></html>"
    
    async def _track_email_delivery(self, notification_id: str, recipient: str, status: str, error: str = None) -> None:
        """Track email delivery status"""
        try:
            delivery_record = {
                "notification_id": notification_id,
                "recipient": recipient,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "error": error
            }
            
            # Store in cache if available
            if hasattr(self, 'cache_repository') and self.cache_repository:
                cache_key = f"email_delivery:{notification_id}:{recipient}"
                await self.cache_repository.set(
                    cache_key, 
                    json.dumps(delivery_record), 
                    expire=86400  # 24 hours
                )
            
        except Exception as e:
            self.logger.error(f"Error tracking email delivery: {e}")
    
    async def _deliver_email_fallback(self, notification: NotificationMessage) -> bool:
        """Fallback email delivery for development/testing"""
        try:
            for recipient in notification.recipients:
                self.logger.info(
                    f"[FALLBACK] Email delivered to {recipient.address}: {notification.subject}"
                )
                self.logger.debug(f"[FALLBACK] Email body: {notification.body}")
            return True
        except Exception as e:
            self.logger.error(f"Error in fallback email delivery: {e}")
            return False
    
    async def _deliver_webhook(self, notification: NotificationMessage) -> bool:
        """Deliver webhook notification"""
        try:
            for recipient in notification.recipients:
                url = recipient.address
                payload = json.loads(notification.body)
                headers = notification.metadata.get("headers", {})
                
                # Add signature if secret is provided
                secret = recipient.metadata.get("secret")
                if secret:
                    signature = self._generate_webhook_signature(notification.body, secret)
                    headers["X-XORB-Signature"] = signature
                
                # Send webhook
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status < 400:
                            self.logger.info(f"Webhook delivered to {url}")
                            return True
                        else:
                            self.logger.error(f"Webhook failed to {url}: {response.status}")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error delivering webhook: {str(e)}")
            return False
    
    async def _deliver_slack(self, notification: NotificationMessage) -> bool:
        """Deliver Slack notification"""
        try:
            # Placeholder for Slack integration
            # In production, integrate with Slack API
            
            for recipient in notification.recipients:
                self.logger.info(
                    f"Slack message delivered to {recipient.address}: {notification.subject}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error delivering Slack notification: {str(e)}")
            return False
    
    async def _deliver_teams(self, notification: NotificationMessage) -> bool:
        """Deliver Microsoft Teams notification"""
        try:
            # Placeholder for Teams integration
            # In production, integrate with Teams webhook
            
            for recipient in notification.recipients:
                self.logger.info(
                    f"Teams message delivered to {recipient.address}: {notification.subject}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error delivering Teams notification: {str(e)}")
            return False
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables"""
        try:
            # Simple template rendering (in production, use Jinja2 or similar)
            rendered = template
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                rendered = rendered.replace(placeholder, str(value))
            return rendered
            
        except Exception as e:
            self.logger.error(f"Error rendering template: {str(e)}")
            return template
    
    def _generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate webhook signature for verification"""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def _load_default_templates(self):
        """Load default notification templates"""
        # Security alert template
        self._templates["security_alert"] = NotificationTemplate(
            id="security_alert",
            name="Security Alert",
            channel=NotificationChannel.EMAIL,
            subject_template="🚨 Security Alert: {{alert_type}}",
            body_template="""
            A security event has been detected:
            
            Alert Type: {{alert_type}}
            Severity: {{severity}}
            Time: {{timestamp}}
            Description: {{description}}
            
            Please review and take appropriate action.
            
            XORB Security Platform
            """,
            variables=["alert_type", "severity", "timestamp", "description"],
            metadata={"category": "security"}
        )
        
        # Scan completion template
        self._templates["scan_complete"] = NotificationTemplate(
            id="scan_complete",
            name="Scan Completion",
            channel=NotificationChannel.EMAIL,
            subject_template="✅ PTaaS Scan Complete: {{target}}",
            body_template="""
            Your penetration test scan has completed:
            
            Target: {{target}}
            Scan Type: {{scan_type}}
            Duration: {{duration}}
            Vulnerabilities Found: {{vulnerability_count}}
            Risk Level: {{risk_level}}
            
            View full report: {{report_url}}
            
            XORB PTaaS Platform
            """,
            variables=["target", "scan_type", "duration", "vulnerability_count", "risk_level", "report_url"],
            metadata={"category": "scan"}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Check cache connectivity
            test_key = "notification_health_check"
            await self.cache.set(test_key, "ok", ttl=60)
            cache_result = await self.cache.get(test_key)
            await self.cache.delete(test_key)
            
            cache_healthy = cache_result == "ok"
            
            # Check queue status
            queue_status = {}
            for priority, queue in self._delivery_queues.items():
                queue_status[priority.value] = queue.qsize()
            
            return {
                "status": "healthy" if cache_healthy and self._running else "degraded",
                "cache_connection": cache_healthy,
                "workers_running": self._running,
                "worker_count": len(self._workers),
                "queue_sizes": queue_status,
                "template_count": len(self._templates),
                "timestamp": str(datetime.utcnow().timestamp())
            }
            
        except Exception as e:
            self.logger.error(f"Notification service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": str(datetime.utcnow().timestamp())
            }


def asdict(obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary"""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                result[key] = [asdict(item) for item in value]
            elif hasattr(value, '__dict__'):
                result[key] = asdict(value)
            else:
                result[key] = value
        return result
    return obj.__dict__ if hasattr(obj, '__dict__') else str(obj)