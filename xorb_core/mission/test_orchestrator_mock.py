#!/usr/bin/env python3
"""
Mock Orchestrator for Mission Module Testing

Provides basic orchestrator interface without heavy dependencies
"""

import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock

class MockIntelligentOrchestrator:
    """Mock orchestrator for testing mission modules"""
    
    def __init__(self):
        self.episodic_memory = AsyncMock()
        self.autonomous_workers = {}
        self.execution_contexts = {}
        self._running = True
        
        # Mock episodic memory methods
        self.episodic_memory.store_memory = AsyncMock(return_value="mock_memory_id")
        self.episodic_memory.retrieve_similar_memories = AsyncMock(return_value=[])
        self.episodic_memory.store_mission_outcome = AsyncMock(return_value="mock_mission_memory")
        self.episodic_memory.store_bounty_interaction = AsyncMock(return_value="mock_bounty_memory")
        self.episodic_memory.store_compliance_event = AsyncMock(return_value="mock_compliance_memory")
        self.episodic_memory.store_remediation_outcome = AsyncMock(return_value="mock_remediation_memory")
        
    async def submit_task(self, task):
        """Mock task submission"""
        return "mock_task_id"
    
    async def get_status(self):
        """Mock status"""
        return {
            'running': self._running,
            'workers': len(self.autonomous_workers),
            'contexts': len(self.execution_contexts)
        }
    
    async def handle_emergency_shutdown(self):
        """Mock emergency shutdown"""
        self._running = False

    async def initialize_intelligence_coordination(self):
        """Mock intelligence coordination initialization"""
        pass