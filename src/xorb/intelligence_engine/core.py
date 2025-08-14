import asyncio
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum

import aioredis
import aiohttp
from aiohttp import web
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel, Field

from xorb.shared.epyc_config import EPYCConfig
from xorb.shared.enums import CampaignStatus, AgentType, ThreatSeverity
from xorb.shared.models import UnifiedTarget, UnifiedAgent, UnifiedCampaign, ThreatIntelligence
from xorb.shared.llm_client import StrategicLLMClient, StrategicLLMRequest, SecurityTaskType, TaskComplexity
from xorb.database.database import AsyncSessionLocal
from xorb.database.repositories import CampaignRepository, TargetRepository, AgentRepository, ThreatIntelligenceRepository
from .ml_planner import MLCampaignPlanner
from .agent_orchestrator import IntelligentAgentOrchestrator
from .llm_integration import IntelligenceLLMOrchestrator

# Unified Intelligence Engine
class UnifiedIntelligenceEngine:
    def __init__(self, db_session: AsyncSession, llm_config: Optional[Dict[str, Any]] = None):
        self.app = web.Application()
        self.redis = None
        self.db_session = db_session
        self.ml_planner = MLCampaignPlanner()
        self.agent_orchestrator = IntelligentAgentOrchestrator(self.db_session, self.redis)
        self.campaign_repo = CampaignRepository(db_session)
        self.target_repo = TargetRepository(db_session)
        self.agent_repo = AgentRepository(db_session)
        self.threat_intel_repo = ThreatIntelligenceRepository(db_session)
        self.active_campaigns = {}
        self.threat_intelligence = {}
        self.logger = logging.getLogger(__name__)

        # Strategic LLM Integration
        self.llm_config = llm_config or self._get_default_llm_config()
        self.llm_client = None
        self.llm_orchestrator = None

    async def initialize(self):
        """Initialize the intelligence engine with strategic LLM capabilities."""
        # Redis connection
        self.redis = aioredis.from_url("redis://redis:6379")

        # Initialize ML components
        await self.ml_planner.initialize()

        # Initialize Strategic LLM Integration
        self.llm_client = StrategicLLMClient(self.llm_config)
        await self.llm_client.initialize()
        self.llm_orchestrator = IntelligenceLLMOrchestrator(self.llm_client)

        # Initialize agent orchestrator
        self.agent_orchestrator = IntelligentAgentOrchestrator(self.db_session, self.redis)

        # Setup routes
        self.setup_routes()

        self.logger.info("Unified Intelligence Engine initialized")

    def setup_routes(self):
        """Setup API routes."""
        # Campaign management
        self.app.router.add_post('/campaigns', self.create_campaign)
        self.app.router.add_get('/campaigns', self.list_campaigns)
    async def list_campaigns(self, request: web.Request):
        campaigns = await self.campaign_repo.get_all_campaigns()
        return web.json_response([c.dict() for c in campaigns])
        self.app.router.add_get('/campaigns/{campaign_id}', self.get_campaign)
        self.app.router.add_post('/campaigns/{campaign_id}/start', self.start_campaign)
        self.app.router.add_post('/campaigns/{campaign_id}/pause', self.pause_campaign)
        self.app.router.add_post('/campaigns/{campaign_id}/stop', self.stop_campaign)

        # AI-Enhanced endpoints
        self.app.router.add_post('/campaigns/{campaign_id}/ai-analysis', self.analyze_with_ai)
        self.app.router.add_post('/intelligence/fusion', self.fuse_intelligence)
        self.app.router.add_post('/payloads/generate', self.generate_ai_payloads)
        self.app.router.add_get('/intelligence/statistics', self.get_intelligence_statistics)

        # Health and status
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.get_status)

    def _get_default_llm_config(self) -> Dict[str, Any]:
        """Get default LLM configuration for paid-first with free fallback strategy."""
        import os
        return {
            "nvidia_api_key": os.getenv("NVIDIA_API_KEY", ""),
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "prefer_paid": os.getenv("LLM_PREFER_PAID", "true").lower() == "true",
            "enable_fallback": os.getenv("LLM_ENABLE_FALLBACK", "true").lower() == "true",
            "daily_budget_limit": float(os.getenv("LLM_DAILY_BUDGET", "25.0")),
            "monthly_budget_limit": float(os.getenv("LLM_MONTHLY_BUDGET", "500.0")),
            "per_request_limit": float(os.getenv("LLM_REQUEST_LIMIT", "5.0")),
            "daily_request_limit": int(os.getenv("LLM_DAILY_REQUEST_LIMIT", "100")),
            "monthly_request_limit": int(os.getenv("LLM_MONTHLY_REQUEST_LIMIT", "2000")),
            "hourly_request_limit": int(os.getenv("LLM_HOURLY_REQUEST_LIMIT", "20")),
            "emergency_request_limit": int(os.getenv("LLM_EMERGENCY_REQUEST_LIMIT", "150")),
            "epyc_optimized": True,
            "prefer_nvidia": True,
            "enable_caching": True,
            "strategy": "paid_first_with_free_fallback"
        }

    async def analyze_with_ai(self, request: web.Request):
        """AI-enhanced campaign analysis endpoint."""
        try:
            campaign_id = request.match_info['campaign_id']
            data = await request.json()

            # Get campaign data
            campaign = await self.campaign_repo.get_campaign_by_id(campaign_id)
            if not campaign:
                return web.json_response({'error': 'Campaign not found'}, status=404)

            # Get related scan results (this would need to be implemented)
            scan_results = data.get('scan_results', [])
            target_context = {
                "campaign_name": campaign.name,
                "targets": campaign.targets,
                "rules_of_engagement": campaign.rules_of_engagement
            }

            # Perform AI analysis
            if self.llm_orchestrator:
                analysis = await self.llm_orchestrator.analyze_scan_results_with_ai(
                    scan_results, target_context
                )

                return web.json_response({
                    'campaign_id': campaign_id,
                    'ai_analysis': analysis,
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'ai_enhanced': True
                })
            else:
                return web.json_response({'error': 'AI analysis not available'}, status=503)

        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def fuse_intelligence(self, request: web.Request):
        """Intelligence fusion endpoint using AI."""
        try:
            data = await request.json()
            multi_source_data = data.get('sources', {})
            fusion_objectives = data.get('objectives', ['correlation', 'threat_assessment'])

            if self.llm_orchestrator:
                fusion_result = await self.llm_orchestrator.fuse_intelligence_sources(
                    multi_source_data, fusion_objectives
                )

                return web.json_response({
                    'fusion_result': fusion_result,
                    'fusion_timestamp': datetime.utcnow().isoformat(),
                    'sources_processed': len(multi_source_data),
                    'ai_enhanced': True
                })
            else:
                return web.json_response({'error': 'Intelligence fusion not available'}, status=503)

        except Exception as e:
            self.logger.error(f"Intelligence fusion error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def generate_ai_payloads(self, request: web.Request):
        """AI-powered payload generation endpoint."""
        try:
            data = await request.json()
            vulnerability_data = data.get('vulnerabilities', {})
            target_context = data.get('target_context', {})
            complexity_level = TaskComplexity(data.get('complexity', 'enhanced'))

            if self.llm_orchestrator:
                payloads = await self.llm_orchestrator.generate_strategic_payloads(
                    vulnerability_data, target_context, complexity_level
                )

                return web.json_response({
                    'payloads': payloads,
                    'generation_timestamp': datetime.utcnow().isoformat(),
                    'complexity_level': complexity_level.value,
                    'payload_count': len(payloads),
                    'ai_enhanced': True
                })
            else:
                return web.json_response({'error': 'AI payload generation not available'}, status=503)

        except Exception as e:
            self.logger.error(f"AI payload generation error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_intelligence_statistics(self, request: web.Request):
        """Get comprehensive intelligence statistics."""
        try:
            stats = {
                'campaigns': {
                    'total': len(self.active_campaigns),
                    'active': len([c for c in self.active_campaigns.values() if c.get('status') == 'running'])
                },
                'threat_intelligence': {
                    'total_indicators': len(self.threat_intelligence)
                }
            }

            if self.llm_orchestrator:
                ai_stats = await self.llm_orchestrator.get_intelligence_statistics()
                stats['ai_integration'] = ai_stats

            return web.json_response(stats)

        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def health_check(self, request: web.Request):
        """Enhanced health check with AI status."""
        try:
            components = {
                'database': True,  # Would check actual DB connection
                'redis': await self._check_redis_health(),
                'ml_planner': self.ml_planner is not None,
                'ai_integration': self.llm_client is not None
            }

            ai_rate_limit_status = "healthy"
            if self.llm_client:
                stats = self.llm_client.get_usage_statistics()
                rate_limit_status = stats.get('rate_limit_status', {})
                max_usage = max(
                    rate_limit_status.get('daily_usage_percentage', 0) / 100,
                    rate_limit_status.get('hourly_usage_percentage', 0) / 100
                )

                if max_usage > 0.8:
                    ai_rate_limit_status = "warning"
                elif max_usage > 0.95:
                    ai_rate_limit_status = "critical"

            return web.json_response({
                'status': 'healthy' if all(components.values()) else 'degraded',
                'service': 'intelligence-engine',
                'timestamp': datetime.utcnow().isoformat(),
                'components': components,
                'ai_rate_limit_status': ai_rate_limit_status,
                'free_tier_enabled': True,
                'active_campaigns': len(self.active_campaigns),
                'epyc_optimized': True
            })

        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return web.json_response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=500)

    async def get_status(self, request: web.Request):
        """Get detailed engine status."""
        try:
            status = {
                'engine_type': 'unified_intelligence',
                'capabilities': [
                    'campaign_management',
                    'ml_planning',
                    'agent_orchestration',
                    'ai_enhanced_analysis',
                    'payload_generation',
                    'intelligence_fusion',
                    'threat_assessment'
                ],
                'epyc_optimization': True,
                'ai_integration': {
                    'nvidia_api': bool(self.llm_config.get('nvidia_api_key')),
                    'openrouter_api': bool(self.llm_config.get('openrouter_api_key')),
                    'models_available': True,
                    'strategic_capabilities': True
                }
            }

            if self.llm_client:
                llm_stats = self.llm_client.get_usage_statistics()
                status['ai_usage'] = llm_stats

            return web.json_response(status)

        except Exception as e:
            self.logger.error(f"Status error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity."""
        try:
            if self.redis:
                await self.redis.ping()
                return True
            return False
        except Exception:
            return False

    async def close(self):
        """Clean shutdown of intelligence engine."""
        if self.llm_client:
            await self.llm_client.close()
        if self.redis:
            await self.redis.close()
        self.logger.info("Intelligence Engine shutdown complete")
