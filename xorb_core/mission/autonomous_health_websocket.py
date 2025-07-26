#!/usr/bin/env python3
"""
XORB Autonomous Health WebSocket Integration

Real-time health streaming integration with external intelligence API
for autonomous lifecycle management and mission planning.

Features:
- Real-time health data streaming via WebSocket
- Integration with external intelligence API
- Health event correlation with mission intelligence
- Autonomous remediation triggers based on intelligence input

Author: XORB Autonomous Systems
Version: 2.0.0
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import websockets
from websockets.exceptions import ConnectionClosed
import aiohttp
from dataclasses import dataclass, asdict

from xorb_core.autonomous.autonomous_health_manager import (
    AutonomousHealthManager, ServiceState, RemediationType, ErrorCategory
)
from xorb_core.mission.external_intelligence_api import ExternalIntelligenceAPI

logger = logging.getLogger(__name__)


@dataclass
class HealthStreamEvent:
    """Health streaming event structure"""
    timestamp: str
    event_type: str
    service_name: str
    health_state: str
    metrics: Dict[str, Any]
    remediation_action: Optional[str] = None
    intelligence_context: Optional[Dict[str, Any]] = None


class AutonomousHealthWebSocketStreamer:
    """
    WebSocket streamer for autonomous health management integration
    with external intelligence systems.
    """
    
    def __init__(self, health_manager: AutonomousHealthManager, 
                 external_intel: ExternalIntelligenceAPI,
                 config: Dict[str, Any]):
        self.health_manager = health_manager
        self.external_intel = external_intel
        self.config = config
        
        # WebSocket configuration
        self.websocket_host = config.get('websocket_host', 'localhost')
        self.websocket_port = config.get('websocket_port', 9092)
        self.external_api_endpoint = config.get('external_api_endpoint', 'ws://localhost:8005/intelligence')
        
        # Streaming state
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.health_event_queue = asyncio.Queue()
        self.intelligence_correlation_enabled = config.get('intelligence_correlation', True)
        
        logger.info("🔌 Autonomous Health WebSocket Streamer initialized")
    
    async def start_streaming_server(self):
        """Start the WebSocket streaming server"""
        try:
            # Start WebSocket server for health streaming
            server = await websockets.serve(
                self.handle_health_stream_connection,
                self.websocket_host,
                self.websocket_port
            )
            
            logger.info(f"🚀 Health streaming server started on ws://{self.websocket_host}:{self.websocket_port}")
            
            # Start background tasks
            await asyncio.gather(
                self._health_event_processor(),
                self._intelligence_correlation_engine(),
                self._external_api_connector()
            )
            
        except Exception as e:
            logger.error(f"Failed to start streaming server: {e}")
            raise
    
    async def handle_health_stream_connection(self, websocket, path):
        """Handle incoming WebSocket connections for health streaming"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        try:
            logger.info(f"🔗 New health stream connection: {client_id}")
            self.active_connections[client_id] = websocket
            
            # Send initial health summary
            await self._send_health_summary(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(client_id, data)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}: {message}")
                    await websocket.send(json.dumps({
                        'error': 'Invalid JSON format',
                        'timestamp': datetime.now().isoformat()
                    }))
                    
        except ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.active_connections.pop(client_id, None)
    
    async def _send_health_summary(self, websocket):
        """Send current health summary to a new connection"""
        try:
            health_summary = {
                'type': 'health_summary',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    name: {\n                        'state': state.value,\n                        'metrics': self.health_manager.health_history.get(name, [])[-1].__dict__ \n                                 if name in self.health_manager.health_history and self.health_manager.health_history[name] \n                                 else None\n                    }\n                    for name, state in self.health_manager.service_states.items()\n                },\n                'active_failures': len(self.health_manager.active_failures),\n                'remediation_queue_size': self.health_manager.remediation_queue.qsize(),\n                'autonomous_status': 'active'\n            }\n            \n            await websocket.send(json.dumps(health_summary, default=str))\n            \n        except Exception as e:\n            logger.error(f\"Failed to send health summary: {e}\")\n    \n    async def _handle_client_message(self, client_id: str, data: Dict[str, Any]):\n        \"\"\"Handle incoming messages from WebSocket clients\"\"\"\n        message_type = data.get('type')\n        \n        if message_type == 'get_health_status':\n            await self._send_health_summary(self.active_connections[client_id])\n            \n        elif message_type == 'trigger_remediation':\n            service_name = data.get('service')\n            remediation_type = data.get('remediation_type', 'restart')\n            \n            if service_name:\n                await self._trigger_manual_remediation(service_name, remediation_type)\n                \n        elif message_type == 'intelligence_input':\n            intelligence_data = data.get('intelligence')\n            await self._process_intelligence_input(intelligence_data)\n            \n        elif message_type == 'subscribe_service':\n            # Client wants to subscribe to specific service updates\n            service_name = data.get('service')\n            self._add_service_subscription(client_id, service_name)\n            \n        else:\n            logger.warning(f\"Unknown message type from {client_id}: {message_type}\")\n    \n    async def _health_event_processor(self):\n        \"\"\"Process health events and stream to connected clients\"\"\"\n        while True:\n            try:\n                # Monitor health manager state changes\n                await asyncio.sleep(5)  # Check every 5 seconds\n                \n                # Check for new health events\n                current_states = self.health_manager.service_states.copy()\n                \n                for service_name, current_state in current_states.items():\n                    # Create health event\n                    if service_name in self.health_manager.health_history:\n                        latest_metrics = self.health_manager.health_history[service_name][-1]\n                        \n                        health_event = HealthStreamEvent(\n                            timestamp=datetime.now().isoformat(),\n                            event_type='health_update',\n                            service_name=service_name,\n                            health_state=current_state.value,\n                            metrics={\n                                'cpu_usage': latest_metrics.cpu_usage,\n                                'memory_usage': latest_metrics.memory_usage,\n                                'response_time': latest_metrics.response_time,\n                                'error_rate': latest_metrics.error_rate,\n                                'uptime': latest_metrics.uptime\n                            }\n                        )\n                        \n                        await self.health_event_queue.put(health_event)\n                \n                # Process queued events\n                while not self.health_event_queue.empty():\n                    event = await self.health_event_queue.get()\n                    await self._broadcast_health_event(event)\n                    \n            except Exception as e:\n                logger.error(f\"Health event processor error: {e}\")\n                await asyncio.sleep(1)\n    \n    async def _broadcast_health_event(self, event: HealthStreamEvent):\n        \"\"\"Broadcast health event to all connected clients\"\"\"\n        if not self.active_connections:\n            return\n        \n        event_data = json.dumps(asdict(event), default=str)\n        \n        # Broadcast to all connections\n        disconnected_clients = []\n        \n        for client_id, websocket in self.active_connections.items():\n            try:\n                await websocket.send(event_data)\n                \n            except ConnectionClosed:\n                disconnected_clients.append(client_id)\n            except Exception as e:\n                logger.error(f\"Error sending to client {client_id}: {e}\")\n                disconnected_clients.append(client_id)\n        \n        # Clean up disconnected clients\n        for client_id in disconnected_clients:\n            self.active_connections.pop(client_id, None)\n    \n    async def _intelligence_correlation_engine(self):\n        \"\"\"Correlate health events with external intelligence\"\"\"\n        if not self.intelligence_correlation_enabled:\n            return\n        \n        while True:\n            try:\n                await asyncio.sleep(30)  # Process every 30 seconds\n                \n                # Get recent health events and correlate with intelligence\n                active_failures = list(self.health_manager.active_failures.values())\n                \n                for failure in active_failures:\n                    # Get intelligence context for this failure\n                    intelligence_context = await self._get_intelligence_context(failure)\n                    \n                    if intelligence_context:\n                        # Update remediation strategy based on intelligence\n                        await self._update_remediation_with_intelligence(\n                            failure, intelligence_context\n                        )\n                        \n            except Exception as e:\n                logger.error(f\"Intelligence correlation error: {e}\")\n                await asyncio.sleep(5)\n    \n    async def _get_intelligence_context(self, failure_event) -> Optional[Dict[str, Any]]:\n        \"\"\"Get intelligence context for a failure event\"\"\"\n        try:\n            # Query external intelligence API for context\n            context_request = {\n                'service_name': failure_event.service_name,\n                'error_category': failure_event.error_category.value,\n                'error_message': failure_event.error_message,\n                'timestamp': failure_event.timestamp.isoformat()\n            }\n            \n            intelligence_response = await self.external_intel.get_failure_intelligence(\n                context_request\n            )\n            \n            return intelligence_response\n            \n        except Exception as e:\n            logger.warning(f\"Failed to get intelligence context: {e}\")\n            return None\n    \n    async def _update_remediation_with_intelligence(self, failure_event, intelligence_context):\n        \"\"\"Update remediation strategy based on intelligence input\"\"\"\n        try:\n            # Analyze intelligence recommendations\n            recommended_action = intelligence_context.get('recommended_remediation')\n            confidence_score = intelligence_context.get('confidence', 0.0)\n            \n            if recommended_action and confidence_score > 0.8:\n                # High confidence intelligence - update remediation\n                new_remediation = RemediationType(recommended_action)\n                \n                logger.info(\n                    f\"🧠 Intelligence-guided remediation for {failure_event.service_name}: \"\n                    f\"{new_remediation.value} (confidence: {confidence_score})\"\n                )\n                \n                # Trigger intelligent remediation\n                await self.health_manager._execute_remediation(\n                    failure_event.service_name, new_remediation\n                )\n                \n                # Broadcast intelligence event\n                intelligence_event = HealthStreamEvent(\n                    timestamp=datetime.now().isoformat(),\n                    event_type='intelligence_remediation',\n                    service_name=failure_event.service_name,\n                    health_state=failure_event.metrics_snapshot.state.value,\n                    metrics={},\n                    remediation_action=new_remediation.value,\n                    intelligence_context=intelligence_context\n                )\n                \n                await self._broadcast_health_event(intelligence_event)\n                \n        except Exception as e:\n            logger.error(f\"Failed to update remediation with intelligence: {e}\")\n    \n    async def _external_api_connector(self):\n        \"\"\"Maintain connection to external intelligence API\"\"\"\n        while True:\n            try:\n                async with websockets.connect(self.external_api_endpoint) as websocket:\n                    logger.info(f\"🔗 Connected to external intelligence API: {self.external_api_endpoint}\")\n                    \n                    # Send health status updates to external API\n                    while True:\n                        await asyncio.sleep(60)  # Send updates every minute\n                        \n                        health_summary = {\n                            'type': 'xorb_health_update',\n                            'timestamp': datetime.now().isoformat(),\n                            'source': 'autonomous_health_manager',\n                            'services_count': len(self.health_manager.service_states),\n                            'healthy_services': len([\n                                s for s in self.health_manager.service_states.values() \n                                if s == ServiceState.HEALTHY\n                            ]),\n                            'failed_services': len([\n                                s for s in self.health_manager.service_states.values() \n                                if s == ServiceState.FAILED\n                            ]),\n                            'autonomous_actions': len(self.health_manager.active_failures)\n                        }\n                        \n                        await websocket.send(json.dumps(health_summary))\n                        \n            except ConnectionClosed:\n                logger.warning(\"Connection to external intelligence API closed\")\n                await asyncio.sleep(30)  # Retry after 30 seconds\n            except Exception as e:\n                logger.error(f\"External API connector error: {e}\")\n                await asyncio.sleep(60)  # Retry after 1 minute\n    \n    async def _trigger_manual_remediation(self, service_name: str, remediation_type: str):\n        \"\"\"Trigger manual remediation via WebSocket command\"\"\"\n        try:\n            remediation = RemediationType(remediation_type)\n            success = await self.health_manager._execute_remediation(service_name, remediation)\n            \n            response_event = HealthStreamEvent(\n                timestamp=datetime.now().isoformat(),\n                event_type='manual_remediation_result',\n                service_name=service_name,\n                health_state='unknown',\n                metrics={},\n                remediation_action=remediation_type,\n                intelligence_context={'success': success, 'trigger': 'manual'}\n            )\n            \n            await self._broadcast_health_event(response_event)\n            \n        except Exception as e:\n            logger.error(f\"Manual remediation failed: {e}\")\n    \n    async def _process_intelligence_input(self, intelligence_data: Dict[str, Any]):\n        \"\"\"Process external intelligence input for autonomous decision making\"\"\"\n        try:\n            intelligence_type = intelligence_data.get('type')\n            \n            if intelligence_type == 'threat_intel':\n                # Process threat intelligence\n                await self._process_threat_intelligence(intelligence_data)\n                \n            elif intelligence_type == 'performance_intel':\n                # Process performance intelligence\n                await self._process_performance_intelligence(intelligence_data)\n                \n            elif intelligence_type == 'remediation_guidance':\n                # Process remediation guidance\n                await self._process_remediation_guidance(intelligence_data)\n                \n        except Exception as e:\n            logger.error(f\"Intelligence input processing error: {e}\")\n    \n    async def _process_threat_intelligence(self, threat_data: Dict[str, Any]):\n        \"\"\"Process threat intelligence and update autonomous security posture\"\"\"\n        threat_level = threat_data.get('threat_level', 'low')\n        affected_services = threat_data.get('affected_services', [])\n        \n        if threat_level == 'high':\n            # Implement defensive measures\n            for service in affected_services:\n                if service in self.health_manager.service_states:\n                    logger.warning(f\"🛡️ Implementing defensive measures for {service} due to threat intelligence\")\n                    # Could trigger preventive scaling, security hardening, etc.\n    \n    async def _process_performance_intelligence(self, perf_data: Dict[str, Any]):\n        \"\"\"Process performance intelligence for proactive optimization\"\"\"\n        performance_predictions = perf_data.get('predictions', {})\n        \n        for service, prediction in performance_predictions.items():\n            if prediction.get('degradation_risk', 0) > 0.8:\n                logger.info(f\"📈 Proactive scaling recommended for {service} based on performance intelligence\")\n                # Could trigger proactive scaling\n    \n    async def _process_remediation_guidance(self, guidance_data: Dict[str, Any]):\n        \"\"\"Process external remediation guidance\"\"\"\n        service = guidance_data.get('service')\n        recommended_action = guidance_data.get('action')\n        \n        if service and recommended_action:\n            logger.info(f\"🎯 External remediation guidance for {service}: {recommended_action}\")\n            await self._trigger_manual_remediation(service, recommended_action)\n    \n    def _add_service_subscription(self, client_id: str, service_name: str):\n        \"\"\"Add service-specific subscription for a client\"\"\"\n        # Implementation for service-specific subscriptions\n        logger.info(f\"📡 Client {client_id} subscribed to {service_name} updates\")\n\n\n# Factory function for easy initialization\nasync def create_health_websocket_streamer(\n    health_manager: AutonomousHealthManager,\n    external_intel: ExternalIntelligenceAPI,\n    config: Dict[str, Any]\n) -> AutonomousHealthWebSocketStreamer:\n    \"\"\"Factory function to create WebSocket streamer\"\"\"\n    streamer = AutonomousHealthWebSocketStreamer(health_manager, external_intel, config)\n    return streamer\n\n\nif __name__ == \"__main__\":\n    # Example usage\n    async def main():\n        from xorb_core.autonomous.autonomous_health_manager import create_autonomous_health_manager\n        from xorb_core.mission.external_intelligence_api import ExternalIntelligenceAPI\n        \n        # Configuration\n        config = {\n            'websocket_host': 'localhost',\n            'websocket_port': 9092,\n            'external_api_endpoint': 'ws://localhost:8005/intelligence',\n            'intelligence_correlation': True\n        }\n        \n        # Initialize components\n        health_manager = await create_autonomous_health_manager(config)\n        external_intel = ExternalIntelligenceAPI()\n        \n        # Create and start streamer\n        streamer = await create_health_websocket_streamer(\n            health_manager, external_intel, config\n        )\n        \n        await streamer.start_streaming_server()\n    \n    asyncio.run(main())