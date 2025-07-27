#!/usr/bin/env python3

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as redis


class EventType(str, Enum):
    # Campaign events
    CAMPAIGN_CREATED = "campaign.created"
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_COMPLETED = "campaign.completed"
    CAMPAIGN_PAUSED = "campaign.paused"
    CAMPAIGN_FAILED = "campaign.failed"
    
    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_TASK_ASSIGNED = "agent.task.assigned"
    AGENT_TASK_COMPLETED = "agent.task.completed"
    AGENT_TASK_FAILED = "agent.task.failed"
    AGENT_ERROR = "agent.error"
    
    # Finding events
    FINDING_DISCOVERED = "finding.discovered"
    FINDING_VALIDATED = "finding.validated"
    FINDING_SUBMITTED = "finding.submitted"
    
    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_HEALTH_CHECK = "system.health.check"
    RESOURCE_THRESHOLD = "resource.threshold"
    
    # Knowledge events
    KNOWLEDGE_ATOM_CREATED = "knowledge.atom.created"
    KNOWLEDGE_ATOM_UPDATED = "knowledge.atom.updated"
    KNOWLEDGE_VALIDATION_COMPLETED = "knowledge.validation.completed"


@dataclass
class XORBEvent:
    event_id: str
    event_type: EventType
    source: str  # Component that generated the event
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None  # For tracing related events
    priority: int = 5  # 1=highest, 10=lowest
    ttl_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'ttl_seconds': self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XORBEvent':
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            correlation_id=data.get('correlation_id'),
            priority=data.get('priority', 5),
            ttl_seconds=data.get('ttl_seconds')
        )


class EventHandler:
    """Base class for event handlers"""
    
    def __init__(self, handler_id: str):
        self.handler_id = handler_id
        self.subscribed_events: Set[EventType] = set()
        self.logger = logging.getLogger(f"EventHandler[{handler_id}]")
    
    def subscribe(self, *event_types: EventType):
        """Subscribe to specific event types"""
        self.subscribed_events.update(event_types)
    
    async def handle_event(self, event: XORBEvent) -> bool:
        """Handle an event. Return True if handled successfully."""
        if event.event_type not in self.subscribed_events:
            return True  # Not interested in this event
        
        try:
            return await self._process_event(event)
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_id}: {e}")
            return False
    
    async def _process_event(self, event: XORBEvent) -> bool:
        """Override this method to implement event processing logic"""
        raise NotImplementedError()


class EventBus:
    """Redis Streams-based event bus for real-time communication"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.stream_key = "xorb:events"
        self.consumer_group = "xorb_system"
        
        self.handlers: Dict[str, EventHandler] = {}
        self.running = False
        self.consumer_tasks: List[asyncio.Task] = []
        
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.events_published = 0
        self.events_processed = 0
        self.events_failed = 0

    async def start(self):
        """Start the event bus"""
        if self.running:
            return
        
        self.running = True
        
        try:
            # Create consumer group if it doesn't exist
            await self.redis_client.xgroup_create(
                self.stream_key, 
                self.consumer_group, 
                id="0", 
                mkstream=True
            )
            self.logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            self.logger.debug("Consumer group already exists")
        
        # Start consumer task
        consumer_task = asyncio.create_task(self._consumer_loop())
        self.consumer_tasks.append(consumer_task)
        
        self.logger.info("Event bus started")

    async def stop(self):
        """Stop the event bus"""
        self.running = False
        
        # Cancel consumer tasks
        for task in self.consumer_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.consumer_tasks.clear()
        
        await self.redis_client.close()
        self.logger.info("Event bus stopped")

    async def publish_event(self, event: XORBEvent) -> str:
        """Publish an event to the stream"""
        try:
            event_data = event.to_dict()
            
            # Add to Redis stream
            message_id = await self.redis_client.xadd(
                self.stream_key,
                event_data,
                maxlen=10000  # Keep last 10k events
            )
            
            self.events_published += 1
            self.logger.debug(f"Published event {event.event_id} as {message_id}")
            
            return message_id.decode() if isinstance(message_id, bytes) else message_id
            
        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_id}: {e}")
            raise

    async def publish(self, 
                     event_type: EventType,
                     source: str,
                     data: Dict[str, Any],
                     correlation_id: Optional[str] = None,
                     priority: int = 5,
                     ttl_seconds: Optional[int] = None) -> str:
        """Convenience method to publish an event"""
        
        event = XORBEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            timestamp=datetime.utcnow(),
            data=data,
            correlation_id=correlation_id,
            priority=priority,
            ttl_seconds=ttl_seconds
        )
        
        return await self.publish_event(event)

    def register_handler(self, handler: EventHandler):
        """Register an event handler"""
        self.handlers[handler.handler_id] = handler
        self.logger.info(f"Registered handler: {handler.handler_id}")

    def unregister_handler(self, handler_id: str):
        """Unregister an event handler"""
        if handler_id in self.handlers:
            del self.handlers[handler_id]
            self.logger.info(f"Unregistered handler: {handler_id}")

    async def _consumer_loop(self):
        """Main consumer loop"""
        consumer_name = f"consumer_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Starting consumer: {consumer_name}")
        
        while self.running:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {self.stream_key: ">"},
                    count=10,  # Process up to 10 messages at once
                    block=1000  # Block for 1 second
                )
                
                # Process messages
                for stream, msgs in messages:
                    for message_id, fields in msgs:
                        await self._process_message(message_id, fields)
                        
                        # Acknowledge message
                        await self.redis_client.xack(
                            self.stream_key,
                            self.consumer_group,
                            message_id
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _process_message(self, message_id: bytes, fields: Dict[bytes, bytes]):
        """Process a single message"""
        try:
            # Convert bytes to strings
            str_fields = {k.decode(): v.decode() for k, v in fields.items()}
            
            # Reconstruct event
            event = XORBEvent.from_dict(str_fields)
            
            # Check TTL
            if event.ttl_seconds:
                age_seconds = (datetime.utcnow() - event.timestamp).total_seconds()
                if age_seconds > event.ttl_seconds:
                    self.logger.debug(f"Event {event.event_id} expired (age: {age_seconds}s)")
                    return
            
            # Process with all interested handlers
            handled_count = 0
            for handler in self.handlers.values():
                if event.event_type in handler.subscribed_events:
                    try:
                        if await handler.handle_event(event):
                            handled_count += 1
                        else:
                            self.events_failed += 1
                    except Exception as e:
                        self.logger.error(f"Handler {handler.handler_id} failed: {e}")
                        self.events_failed += 1
            
            self.events_processed += 1
            
            if handled_count == 0:
                self.logger.debug(f"No handlers for event type: {event.event_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to process message {message_id}: {e}")
            self.events_failed += 1

    async def get_stream_info(self) -> Dict[str, Any]:
        """Get information about the event stream"""
        try:
            stream_info = await self.redis_client.xinfo_stream(self.stream_key)
            groups_info = await self.redis_client.xinfo_groups(self.stream_key)
            
            return {
                'stream_length': stream_info.get('length', 0),
                'consumer_groups': len(groups_info),
                'first_entry_id': stream_info.get('first-entry', [None, None])[0],
                'last_entry_id': stream_info.get('last-entry', [None, None])[0],
                'events_published': self.events_published,
                'events_processed': self.events_processed,
                'events_failed': self.events_failed,
                'registered_handlers': len(self.handlers)
            }
        except Exception as e:
            self.logger.error(f"Failed to get stream info: {e}")
            return {}

    async def cleanup_old_events(self, max_age_hours: int = 24):
        """Remove old events from the stream"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            # XTRIM with MINID to remove old entries
            removed_count = await self.redis_client.xtrim(
                self.stream_key,
                minid=cutoff_timestamp,
                approximate=True
            )
            
            self.logger.info(f"Cleaned up {removed_count} old events")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old events: {e}")
            return 0


# Specialized event handlers for XORB components

class CampaignEventHandler(EventHandler):
    """Handler for campaign-related events"""
    
    def __init__(self, orchestrator):
        super().__init__("campaign_handler")
        self.orchestrator = orchestrator
        
        # Subscribe to campaign events
        self.subscribe(
            EventType.CAMPAIGN_CREATED,
            EventType.CAMPAIGN_STARTED,
            EventType.CAMPAIGN_COMPLETED,
            EventType.FINDING_DISCOVERED
        )
    
    async def _process_event(self, event: XORBEvent) -> bool:
        if event.event_type == EventType.CAMPAIGN_CREATED:
            return await self._handle_campaign_created(event)
        elif event.event_type == EventType.CAMPAIGN_STARTED:
            return await self._handle_campaign_started(event)
        elif event.event_type == EventType.CAMPAIGN_COMPLETED:
            return await self._handle_campaign_completed(event)
        elif event.event_type == EventType.FINDING_DISCOVERED:
            return await self._handle_finding_discovered(event)
        
        return True
    
    async def _handle_campaign_created(self, event: XORBEvent) -> bool:
        campaign_id = event.data.get('campaign_id')
        self.logger.info(f"Campaign created: {campaign_id}")
        
        # Could trigger additional actions like resource allocation
        return True
    
    async def _handle_campaign_started(self, event: XORBEvent) -> bool:
        campaign_id = event.data.get('campaign_id')
        self.logger.info(f"Campaign started: {campaign_id}")
        
        # Could start monitoring, logging, etc.
        return True
    
    async def _handle_campaign_completed(self, event: XORBEvent) -> bool:
        campaign_id = event.data.get('campaign_id')
        results = event.data.get('results', {})
        
        self.logger.info(f"Campaign completed: {campaign_id} - {results.get('findings_count', 0)} findings")
        
        # Could trigger report generation, cleanup, etc.
        return True
    
    async def _handle_finding_discovered(self, event: XORBEvent) -> bool:
        finding = event.data.get('finding', {})
        severity = finding.get('severity', 'unknown')
        
        # High-severity findings might trigger immediate alerts
        if severity.lower() in ['high', 'critical']:
            self.logger.warning(f"High-severity finding discovered: {finding.get('title', 'Unknown')}")
            
            # Could send alerts, notifications, etc.
            
        return True


class AgentEventHandler(EventHandler):
    """Handler for agent-related events"""
    
    def __init__(self, agent_manager):
        super().__init__("agent_handler")
        self.agent_manager = agent_manager
        
        self.subscribe(
            EventType.AGENT_TASK_ASSIGNED,
            EventType.AGENT_TASK_COMPLETED,
            EventType.AGENT_TASK_FAILED,
            EventType.AGENT_ERROR
        )
    
    async def _process_event(self, event: XORBEvent) -> bool:
        if event.event_type == EventType.AGENT_TASK_ASSIGNED:
            return await self._handle_task_assigned(event)
        elif event.event_type == EventType.AGENT_TASK_COMPLETED:
            return await self._handle_task_completed(event)
        elif event.event_type == EventType.AGENT_TASK_FAILED:
            return await self._handle_task_failed(event)
        elif event.event_type == EventType.AGENT_ERROR:
            return await self._handle_agent_error(event)
        
        return True
    
    async def _handle_task_assigned(self, event: XORBEvent) -> bool:
        agent_id = event.data.get('agent_id')
        task_id = event.data.get('task_id')
        
        self.logger.debug(f"Task {task_id} assigned to agent {agent_id}")
        return True
    
    async def _handle_task_completed(self, event: XORBEvent) -> bool:
        agent_id = event.data.get('agent_id')
        task_id = event.data.get('task_id')
        success = event.data.get('success', False)
        
        self.logger.info(f"Task {task_id} completed by {agent_id}: {'success' if success else 'failed'}")
        
        # Could update agent performance metrics
        return True
    
    async def _handle_task_failed(self, event: XORBEvent) -> bool:
        agent_id = event.data.get('agent_id')
        task_id = event.data.get('task_id')
        error = event.data.get('error', 'Unknown error')
        
        self.logger.warning(f"Task {task_id} failed on {agent_id}: {error}")
        
        # Could trigger task retry or agent health check
        return True
    
    async def _handle_agent_error(self, event: XORBEvent) -> bool:
        agent_id = event.data.get('agent_id')
        error = event.data.get('error', 'Unknown error')
        
        self.logger.error(f"Agent {agent_id} error: {error}")
        
        # Could trigger agent restart or failover
        return True


class SystemEventHandler(EventHandler):
    """Handler for system-level events"""
    
    def __init__(self):
        super().__init__("system_handler")
        
        self.subscribe(
            EventType.SYSTEM_ALERT,
            EventType.RESOURCE_THRESHOLD,
            EventType.SYSTEM_HEALTH_CHECK
        )
    
    async def _process_event(self, event: XORBEvent) -> bool:
        if event.event_type == EventType.SYSTEM_ALERT:
            return await self._handle_system_alert(event)
        elif event.event_type == EventType.RESOURCE_THRESHOLD:
            return await self._handle_resource_threshold(event)
        elif event.event_type == EventType.SYSTEM_HEALTH_CHECK:
            return await self._handle_health_check(event)
        
        return True
    
    async def _handle_system_alert(self, event: XORBEvent) -> bool:
        alert_type = event.data.get('alert_type')
        message = event.data.get('message')
        severity = event.data.get('severity', 'medium')
        
        self.logger.warning(f"System alert [{severity}]: {alert_type} - {message}")
        
        # Could send notifications, trigger automated responses
        return True
    
    async def _handle_resource_threshold(self, event: XORBEvent) -> bool:
        resource_type = event.data.get('resource_type')
        current_value = event.data.get('current_value')
        threshold = event.data.get('threshold')
        
        self.logger.warning(f"Resource threshold exceeded: {resource_type} = {current_value} (threshold: {threshold})")
        
        # Could trigger resource scaling, campaign throttling
        return True
    
    async def _handle_health_check(self, event: XORBEvent) -> bool:
        component = event.data.get('component')
        status = event.data.get('status')
        
        if status != 'healthy':
            self.logger.warning(f"Health check failed for {component}: {status}")
            
            # Could trigger component restart, alerts
        
        return True


# Event bus integration for existing components

class EventDrivenOrchestrator:
    """Mixin to add event publishing to orchestrator"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.source_id = "orchestrator"
    
    async def publish_campaign_created(self, campaign_id: str, campaign_data: Dict):
        """Publish campaign created event"""
        await self.event_bus.publish(
            EventType.CAMPAIGN_CREATED,
            self.source_id,
            {
                'campaign_id': campaign_id,
                'name': campaign_data.get('name'),
                'target_count': len(campaign_data.get('targets', [])),
                'priority': campaign_data.get('priority')
            },
            correlation_id=campaign_id
        )
    
    async def publish_campaign_started(self, campaign_id: str):
        """Publish campaign started event"""
        await self.event_bus.publish(
            EventType.CAMPAIGN_STARTED,
            self.source_id,
            {'campaign_id': campaign_id},
            correlation_id=campaign_id
        )
    
    async def publish_finding_discovered(self, campaign_id: str, finding: Dict):
        """Publish finding discovered event"""
        await self.event_bus.publish(
            EventType.FINDING_DISCOVERED,
            self.source_id,
            {
                'campaign_id': campaign_id,
                'finding': finding,
                'severity': finding.get('severity'),
                'target': finding.get('target')
            },
            correlation_id=campaign_id,
            priority=2 if finding.get('severity', '').lower() in ['high', 'critical'] else 5
        )


class EventDrivenAgent:
    """Mixin to add event publishing to agents"""
    
    def __init__(self, event_bus: EventBus, agent_id: str):
        self.event_bus = event_bus
        self.agent_id = agent_id
        self.source_id = f"agent_{agent_id}"
    
    async def publish_task_completed(self, task_id: str, success: bool, result_data: Dict = None):
        """Publish task completed event"""
        await self.event_bus.publish(
            EventType.AGENT_TASK_COMPLETED,
            self.source_id,
            {
                'agent_id': self.agent_id,
                'task_id': task_id,
                'success': success,
                'result': result_data or {}
            },
            correlation_id=task_id
        )
    
    async def publish_agent_error(self, error_message: str, context: Dict = None):
        """Publish agent error event"""
        await self.event_bus.publish(
            EventType.AGENT_ERROR,
            self.source_id,
            {
                'agent_id': self.agent_id,
                'error': error_message,
                'context': context or {}
            },
            priority=3  # High priority for errors
        )


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_event_system():
        """Demonstrate the event system"""
        
        # Create event bus
        event_bus = EventBus()
        await event_bus.start()
        
        # Create handlers
        campaign_handler = CampaignEventHandler(None)  # Mock orchestrator
        system_handler = SystemEventHandler()
        
        # Register handlers
        event_bus.register_handler(campaign_handler)
        event_bus.register_handler(system_handler)
        
        print("Event system demo started. Publishing test events...")
        
        # Publish some test events
        await event_bus.publish(
            EventType.CAMPAIGN_CREATED,
            "demo",
            {
                'campaign_id': 'demo-campaign-001',
                'name': 'Demo Campaign',
                'target_count': 3,
                'priority': 'high'
            }
        )
        
        await event_bus.publish(
            EventType.FINDING_DISCOVERED,
            "demo",
            {
                'campaign_id': 'demo-campaign-001',
                'finding': {
                    'title': 'SQL Injection',
                    'severity': 'high',
                    'target': 'example.com'
                }
            }
        )
        
        await event_bus.publish(
            EventType.SYSTEM_ALERT,
            "demo",
            {
                'alert_type': 'resource_usage',
                'message': 'CPU usage above 90%',
                'severity': 'warning'
            }
        )
        
        # Wait for events to be processed
        await asyncio.sleep(2)
        
        # Show stream info
        info = await event_bus.get_stream_info()
        print(f"Stream info: {info}")
        
        await event_bus.stop()
        print("Event system demo completed")
    
    if "--demo" in sys.argv:
        asyncio.run(demo_event_system())
    else:
        print("XORB Event System")
        print("Usage: python event_system.py --demo")