#!/usr/bin/env python3
"""
Xorb AI-Powered Vulnerability Prioritization Engine
Phase 6.1 - Intelligent Context-Aware Risk Scoring
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import asyncpg
import openai
from openai import AsyncOpenAI
import aioredis
import numpy as np
from sklearn.preprocessing import StandardScaler
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("xorb.ai_prioritization")

# Phase 6.1 Metrics
vulnerability_prioritization_total = Counter(
    'vulnerability_prioritization_total',
    'Total vulnerability prioritizations',
    ['priority_level', 'asset_type', 'ai_model']
)

prioritization_processing_duration = Histogram(
    'prioritization_processing_duration_seconds',
    'Time to process vulnerability prioritization',
    ['priority_algorithm', 'complexity_level']
)

ai_context_analysis_duration = Histogram(
    'ai_context_analysis_duration_seconds',
    'Duration of AI context analysis',
    ['model_type', 'context_type']
)

priority_score_distribution = Histogram(
    'priority_score_distribution',
    'Distribution of priority scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

threat_intelligence_matches = Counter(
    'threat_intelligence_matches_total',
    'Total threat intelligence matches',
    ['intel_source', 'match_type']
)

business_context_score = Gauge(
    'business_context_score',
    'Business context score for assets',
    ['asset_id', 'context_type']
)

class PriorityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AssetType(Enum):
    WEB_APP = "web_application"
    API = "api_endpoint"
    MOBILE_APP = "mobile_application"
    INFRASTRUCTURE = "infrastructure"
    DATABASE = "database"
    CLOUD_SERVICE = "cloud_service"

@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    cve_id: Optional[str]
    cvss_score: float
    exploit_available: bool
    exploit_maturity: str  # "functional", "poc", "high", "not_defined"
    temporal_score: float
    environmental_score: float
    attack_vector: str
    attack_complexity: str
    privileges_required: str
    user_interaction: str
    scope: str
    confidentiality_impact: str
    integrity_impact: str
    availability_impact: str
    mitre_tactics: List[str]
    mitre_techniques: List[str]
    known_exploited: bool
    exploit_prediction_scoring_system: float  # EPSS score

@dataclass
class BusinessContext:
    """Business context for asset"""
    asset_criticality: float  # 0-1 scale
    revenue_impact: float     # 0-1 scale
    compliance_importance: float  # 0-1 scale
    data_sensitivity: float   # 0-1 scale
    user_base_size: int
    regulatory_requirements: List[str]
    business_function: str
    asset_value: float        # Dollar value
    downtime_cost_per_hour: float

@dataclass
class EnvironmentalContext:
    """Environmental context for vulnerability"""
    network_exposure: str     # "internet", "internal", "isolated"
    access_controls: List[str]
    monitoring_coverage: float  # 0-1 scale
    patch_deployment_speed: str  # "immediate", "fast", "normal", "slow"
    backup_availability: bool
    incident_response_capability: float  # 0-1 scale
    security_controls: List[str]

@dataclass
class VulnerabilityContext:
    """Complete vulnerability context for prioritization"""
    vulnerability_id: str
    title: str
    description: str
    discovered_at: datetime
    asset_id: str
    asset_type: AssetType
    
    # Technical details
    severity: str
    cvss_base_score: float
    cwe_id: Optional[str]
    
    # Threat intelligence
    threat_intel: ThreatIntelligence
    
    # Business context
    business_context: BusinessContext
    
    # Environmental context
    environmental_context: EnvironmentalContext
    
    # Dynamic factors
    trending_attacks: List[str]
    recent_incidents: List[str]
    patch_availability: bool
    exploit_mentions: int

@dataclass
class PrioritizationResult:
    """Vulnerability prioritization result"""
    vulnerability_id: str
    priority_score: float  # 0-1 scale
    priority_level: PriorityLevel
    confidence: float      # 0-1 scale
    
    # Score breakdown
    threat_score: float
    business_impact_score: float
    exploitability_score: float
    environmental_risk_score: float
    temporal_urgency_score: float
    
    # AI analysis
    ai_insights: Dict[str, any]
    recommended_actions: List[str]
    risk_factors: List[str]
    mitigation_urgency: str
    
    # Metadata
    model_version: str
    analyzed_at: datetime
    expires_at: datetime

class ThreatIntelligenceProvider:
    """Integrates with threat intelligence feeds"""
    
    def __init__(self):
        self.redis = None
        self.mitre_cache_ttl = 3600  # 1 hour
        
    async def initialize(self, redis_url: str):
        """Initialize threat intelligence provider"""
        self.redis = await aioredis.from_url(redis_url)
        
    async def enrich_vulnerability(self, cve_id: str, cwe_id: str) -> ThreatIntelligence:
        """Enrich vulnerability with threat intelligence"""
        
        # Check cache first
        cache_key = f"threat_intel:{cve_id}:{cwe_id}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            return ThreatIntelligence(**json.loads(cached))
        
        # Simulate threat intelligence lookup
        # In production, this would integrate with:
        # - MITRE ATT&CK API
        # - NVD API
        # - EPSS API
        # - Commercial threat feeds
        
        intel = ThreatIntelligence(
            cve_id=cve_id,
            cvss_score=7.5,  # Would be fetched from NVD
            exploit_available=True,
            exploit_maturity="functional",
            temporal_score=8.2,
            environmental_score=6.8,
            attack_vector="network",
            attack_complexity="low",
            privileges_required="none",
            user_interaction="none",
            scope="unchanged",
            confidentiality_impact="high",
            integrity_impact="high", 
            availability_impact="high",
            mitre_tactics=["initial_access", "execution"],
            mitre_techniques=["T1190", "T1059"],
            known_exploited=True,
            exploit_prediction_scoring_system=0.85
        )
        
        # Cache for future use
        await self.redis.setex(
            cache_key, 
            self.mitre_cache_ttl, 
            json.dumps(asdict(intel))
        )
        
        threat_intelligence_matches.labels(
            intel_source="mitre_attack",
            match_type="technique"
        ).inc()
        
        return intel

class BusinessContextAnalyzer:
    """Analyzes business context for assets"""
    
    def __init__(self):
        self.db_pool = None
        
    async def initialize(self, database_url: str):
        """Initialize business context analyzer"""
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        
    async def get_asset_context(self, asset_id: str) -> BusinessContext:
        """Get business context for asset"""
        
        async with self.db_pool.acquire() as conn:
            asset_data = await conn.fetchrow("""
                SELECT 
                    a.criticality_score,
                    a.revenue_impact,
                    a.compliance_score,
                    a.data_sensitivity,
                    a.user_count,
                    a.business_function,
                    a.asset_value,
                    a.downtime_cost_hour,
                    array_agg(DISTINCT c.requirement_type) as compliance_reqs
                FROM assets a
                LEFT JOIN asset_compliance c ON a.id = c.asset_id
                WHERE a.id = $1 AND a.active = true
                GROUP BY a.id
            """, asset_id)
            
            if not asset_data:
                # Default context for unknown assets
                return BusinessContext(
                    asset_criticality=0.5,
                    revenue_impact=0.3,
                    compliance_importance=0.2,
                    data_sensitivity=0.4,
                    user_base_size=100,
                    regulatory_requirements=[],
                    business_function="unknown",
                    asset_value=10000,
                    downtime_cost_per_hour=500
                )
            
            context = BusinessContext(
                asset_criticality=float(asset_data['criticality_score']) / 10.0,
                revenue_impact=float(asset_data['revenue_impact']) / 10.0,
                compliance_importance=float(asset_data['compliance_score']) / 10.0,
                data_sensitivity=float(asset_data['data_sensitivity']) / 10.0,
                user_base_size=asset_data['user_count'] or 0,
                regulatory_requirements=asset_data['compliance_reqs'] or [],
                business_function=asset_data['business_function'] or "unknown",
                asset_value=float(asset_data['asset_value']) or 0,
                downtime_cost_per_hour=float(asset_data['downtime_cost_hour']) or 0
            )
            
            # Update metric
            business_context_score.labels(
                asset_id=asset_id,
                context_type="criticality"
            ).set(context.asset_criticality)
            
            return context

class AIContextAnalyzer:
    """Uses GPT-4 for context-aware vulnerability analysis"""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4"
        
    async def initialize(self, api_key: str):
        """Initialize AI context analyzer"""
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def analyze_vulnerability_context(
        self, 
        vuln_context: VulnerabilityContext
    ) -> Dict[str, any]:
        """Perform AI-powered context analysis"""
        
        start_time = datetime.now()
        
        # Build context prompt
        prompt = self._build_analysis_prompt(vuln_context)
        
        try:
            with ai_context_analysis_duration.labels(
                model_type="gpt4",
                context_type="vulnerability_analysis"
            ).time():
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cybersecurity expert specializing in vulnerability risk assessment. Analyze the provided vulnerability context and provide structured insights about business impact, exploitability, and prioritization recommendations."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Validate and structure the response
            structured_analysis = {
                "impact_assessment": analysis.get("impact_assessment", {}),
                "exploitability_analysis": analysis.get("exploitability_analysis", {}),
                "business_risk_factors": analysis.get("business_risk_factors", []),
                "technical_risk_factors": analysis.get("technical_risk_factors", []),
                "recommended_actions": analysis.get("recommended_actions", []),
                "mitigation_urgency": analysis.get("mitigation_urgency", "medium"),
                "confidence_factors": analysis.get("confidence_factors", []),
                "ai_reasoning": analysis.get("reasoning", ""),
                "analysis_duration": (datetime.now() - start_time).total_seconds()
            }
            
            logger.info("AI vulnerability analysis completed",
                       vulnerability_id=vuln_context.vulnerability_id,
                       model=self.model,
                       confidence=structured_analysis.get("confidence", 0.5))
            
            return structured_analysis
            
        except Exception as e:
            logger.error("AI context analysis failed", 
                        error=str(e),
                        vulnerability_id=vuln_context.vulnerability_id)
            
            # Return default analysis on failure
            return {
                "impact_assessment": {"score": 0.5, "reasoning": "AI analysis failed"},
                "exploitability_analysis": {"score": 0.5, "reasoning": "AI analysis failed"},
                "business_risk_factors": ["AI analysis unavailable"],
                "technical_risk_factors": ["AI analysis unavailable"],
                "recommended_actions": ["Manual review required"],
                "mitigation_urgency": "medium",
                "confidence_factors": ["AI analysis failed"],
                "ai_reasoning": f"Analysis failed: {str(e)}",
                "analysis_duration": (datetime.now() - start_time).total_seconds()
            }
    
    def _build_analysis_prompt(self, vuln_context: VulnerabilityContext) -> str:
        """Build analysis prompt for GPT-4"""
        
        return f"""
Analyze this vulnerability for intelligent prioritization:

VULNERABILITY DETAILS:
- ID: {vuln_context.vulnerability_id}
- Title: {vuln_context.title}
- Description: {vuln_context.description}
- CVSS Base Score: {vuln_context.cvss_base_score}
- Asset Type: {vuln_context.asset_type.value}

THREAT INTELLIGENCE:
- CVE: {vuln_context.threat_intel.cve_id}
- CVSS Score: {vuln_context.threat_intel.cvss_score}
- Exploit Available: {vuln_context.threat_intel.exploit_available}
- Exploit Maturity: {vuln_context.threat_intel.exploit_maturity}
- EPSS Score: {vuln_context.threat_intel.exploit_prediction_scoring_system}
- Known Exploited: {vuln_context.threat_intel.known_exploited}
- MITRE Tactics: {', '.join(vuln_context.threat_intel.mitre_tactics)}
- MITRE Techniques: {', '.join(vuln_context.threat_intel.mitre_techniques)}

BUSINESS CONTEXT:
- Asset Criticality: {vuln_context.business_context.asset_criticality}
- Revenue Impact: {vuln_context.business_context.revenue_impact}
- Compliance Importance: {vuln_context.business_context.compliance_importance}
- Data Sensitivity: {vuln_context.business_context.data_sensitivity}
- User Base Size: {vuln_context.business_context.user_base_size}
- Business Function: {vuln_context.business_context.business_function}
- Downtime Cost/Hour: ${vuln_context.business_context.downtime_cost_per_hour}

ENVIRONMENTAL CONTEXT:
- Network Exposure: {vuln_context.environmental_context.network_exposure}
- Monitoring Coverage: {vuln_context.environmental_context.monitoring_coverage}
- Patch Deployment Speed: {vuln_context.environmental_context.patch_deployment_speed}
- Security Controls: {', '.join(vuln_context.environmental_context.security_controls)}

DYNAMIC FACTORS:
- Trending Attacks: {', '.join(vuln_context.trending_attacks)}
- Recent Incidents: {len(vuln_context.recent_incidents)} related incidents
- Patch Available: {vuln_context.patch_availability}
- Exploit Mentions: {vuln_context.exploit_mentions}

Please provide your analysis in the following JSON format:
{{
  "impact_assessment": {{
    "score": 0-1,
    "reasoning": "detailed explanation of business and technical impact"
  }},
  "exploitability_analysis": {{
    "score": 0-1,
    "reasoning": "analysis of how easily this could be exploited"
  }},
  "business_risk_factors": ["list of key business risk factors"],
  "technical_risk_factors": ["list of key technical risk factors"],
  "recommended_actions": ["prioritized list of recommended actions"],
  "mitigation_urgency": "immediate|high|medium|low",
  "confidence_factors": ["factors affecting confidence in this analysis"],
  "reasoning": "comprehensive reasoning for prioritization decision"
}}
"""

class VulnerabilityPrioritizationEngine:
    """Main prioritization engine combining all analysis components"""
    
    def __init__(self):
        self.threat_intel = ThreatIntelligenceProvider()
        self.business_analyzer = BusinessContextAnalyzer()
        self.ai_analyzer = AIContextAnalyzer()
        self.db_pool = None
        self.redis = None
        
        # ML model for priority scoring (simplified)
        self.scaler = StandardScaler()
        
    async def initialize(self, config: Dict):
        """Initialize prioritization engine"""
        logger.info("Initializing AI Vulnerability Prioritization Engine...")
        
        # Initialize database
        database_url = config.get("database_url", 
                                 "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=3, max_size=8)
        
        # Initialize Redis
        redis_url = config.get("redis_url", "redis://redis:6379/0")
        self.redis = await aioredis.from_url(redis_url)
        
        # Initialize components
        await self.threat_intel.initialize(redis_url)
        await self.business_analyzer.initialize(database_url)
        await self.ai_analyzer.initialize(config.get("openai_api_key"))
        
        # Create database tables
        await self._create_prioritization_tables()
        
        logger.info("AI Vulnerability Prioritization Engine initialized successfully")
    
    async def _create_prioritization_tables(self):
        """Create database tables for prioritization"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vulnerability_priorities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    vulnerability_id VARCHAR(255) NOT NULL,
                    priority_score FLOAT NOT NULL,
                    priority_level VARCHAR(20) NOT NULL,
                    confidence FLOAT NOT NULL,
                    
                    -- Score breakdown
                    threat_score FLOAT NOT NULL,
                    business_impact_score FLOAT NOT NULL,
                    exploitability_score FLOAT NOT NULL,
                    environmental_risk_score FLOAT NOT NULL,
                    temporal_urgency_score FLOAT NOT NULL,
                    
                    -- AI analysis
                    ai_insights JSONB NOT NULL,
                    recommended_actions JSONB NOT NULL,
                    risk_factors JSONB NOT NULL,
                    mitigation_urgency VARCHAR(20) NOT NULL,
                    
                    -- Metadata
                    model_version VARCHAR(50) NOT NULL,
                    analyzed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_vuln_priorities_vuln_id 
                ON vulnerability_priorities(vulnerability_id);
                
                CREATE INDEX IF NOT EXISTS idx_vuln_priorities_score 
                ON vulnerability_priorities(priority_score DESC);
                
                CREATE INDEX IF NOT EXISTS idx_vuln_priorities_level 
                ON vulnerability_priorities(priority_level);
                
                CREATE INDEX IF NOT EXISTS idx_vuln_priorities_expires
                ON vulnerability_priorities(expires_at);
                
                -- Asset context table
                CREATE TABLE IF NOT EXISTS assets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    asset_type VARCHAR(50) NOT NULL,
                    criticality_score INTEGER DEFAULT 5,
                    revenue_impact INTEGER DEFAULT 5,
                    compliance_score INTEGER DEFAULT 5,
                    data_sensitivity INTEGER DEFAULT 5,
                    user_count INTEGER DEFAULT 0,
                    business_function VARCHAR(100),
                    asset_value DECIMAL(15,2) DEFAULT 0,
                    downtime_cost_hour DECIMAL(10,2) DEFAULT 0,
                    active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS asset_compliance (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    asset_id UUID REFERENCES assets(id),
                    requirement_type VARCHAR(50) NOT NULL,
                    active BOOLEAN DEFAULT true
                );
            """)
    
    async def prioritize_vulnerability(self, vulnerability_id: str) -> PrioritizationResult:
        """Main entry point for vulnerability prioritization"""
        
        start_time = datetime.now()
        
        try:
            # Check if we have a recent prioritization
            cached_result = await self._get_cached_prioritization(vulnerability_id)
            if cached_result:
                logger.info("Using cached prioritization", vulnerability_id=vulnerability_id)
                return cached_result
            
            # Gather vulnerability context
            vuln_context = await self._gather_vulnerability_context(vulnerability_id)
            
            with prioritization_processing_duration.labels(
                priority_algorithm="ai_enhanced",
                complexity_level="high"
            ).time():
                
                # Perform AI analysis
                ai_insights = await self.ai_analyzer.analyze_vulnerability_context(vuln_context)
                
                # Calculate component scores
                threat_score = self._calculate_threat_score(vuln_context)
                business_score = self._calculate_business_impact_score(vuln_context)
                exploitability_score = self._calculate_exploitability_score(vuln_context)
                environmental_score = self._calculate_environmental_risk_score(vuln_context)
                temporal_score = self._calculate_temporal_urgency_score(vuln_context)
                
                # Combine scores with AI insights
                priority_score = self._combine_scores(
                    threat_score, business_score, exploitability_score,
                    environmental_score, temporal_score, ai_insights
                )
                
                # Determine priority level
                priority_level = self._score_to_priority_level(priority_score)
                
                # Calculate confidence
                confidence = self._calculate_confidence(vuln_context, ai_insights)
                
                # Build result
                result = PrioritizationResult(
                    vulnerability_id=vulnerability_id,
                    priority_score=priority_score,
                    priority_level=priority_level,
                    confidence=confidence,
                    threat_score=threat_score,
                    business_impact_score=business_score,
                    exploitability_score=exploitability_score,
                    environmental_risk_score=environmental_score,
                    temporal_urgency_score=temporal_score,
                    ai_insights=ai_insights,
                    recommended_actions=ai_insights.get("recommended_actions", []),
                    risk_factors=ai_insights.get("business_risk_factors", []) + 
                               ai_insights.get("technical_risk_factors", []),
                    mitigation_urgency=ai_insights.get("mitigation_urgency", "medium"),
                    model_version="6.1.0",
                    analyzed_at=start_time,
                    expires_at=start_time + timedelta(hours=24)
                )
                
                # Store result
                await self._store_prioritization_result(result)
                
                # Update metrics
                vulnerability_prioritization_total.labels(
                    priority_level=priority_level.value,
                    asset_type=vuln_context.asset_type.value,
                    ai_model="gpt4_enhanced"
                ).inc()
                
                priority_score_distribution.observe(priority_score)
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info("Vulnerability prioritization completed",
                           vulnerability_id=vulnerability_id,
                           priority_score=priority_score,
                           priority_level=priority_level.value,
                           confidence=confidence,
                           duration=duration)
                
                return result
                
        except Exception as e:
            logger.error("Vulnerability prioritization failed",
                        vulnerability_id=vulnerability_id,
                        error=str(e))
            raise
    
    async def _gather_vulnerability_context(self, vulnerability_id: str) -> VulnerabilityContext:
        """Gather comprehensive context for vulnerability"""
        
        async with self.db_pool.acquire() as conn:
            # Get vulnerability details
            vuln_data = await conn.fetchrow("""
                SELECT v.*, a.asset_type, a.id as asset_id
                FROM vulnerabilities v
                JOIN assets a ON v.asset_id = a.id
                WHERE v.id = $1
            """, vulnerability_id)
            
            if not vuln_data:
                raise ValueError(f"Vulnerability {vulnerability_id} not found")
            
            # Get threat intelligence
            cve_id = vuln_data.get('cve_id')
            cwe_id = vuln_data.get('cwe_id')
            threat_intel = await self.threat_intel.enrich_vulnerability(cve_id, cwe_id)
            
            # Get business context
            business_context = await self.business_analyzer.get_asset_context(
                vuln_data['asset_id']
            )
            
            # Get environmental context (simplified)
            environmental_context = EnvironmentalContext(
                network_exposure="internet",  # Would be determined from asset data
                access_controls=["authentication", "authorization"],
                monitoring_coverage=0.8,
                patch_deployment_speed="normal",
                backup_availability=True,
                incident_response_capability=0.7,
                security_controls=["waf", "ids", "firewall"]
            )
            
            # Build context
            context = VulnerabilityContext(
                vulnerability_id=vulnerability_id,
                title=vuln_data['title'],
                description=vuln_data['description'],
                discovered_at=vuln_data['discovered_at'],
                asset_id=vuln_data['asset_id'],
                asset_type=AssetType(vuln_data['asset_type']),
                severity=vuln_data['severity'],
                cvss_base_score=float(vuln_data.get('cvss_score', 5.0)),
                cwe_id=cwe_id,
                threat_intel=threat_intel,
                business_context=business_context,
                environmental_context=environmental_context,
                trending_attacks=["web_exploitation", "privilege_escalation"],
                recent_incidents=["incident_1", "incident_2"],
                patch_availability=True,
                exploit_mentions=5
            )
            
            return context
    
    def _calculate_threat_score(self, context: VulnerabilityContext) -> float:
        """Calculate threat score based on threat intelligence"""
        
        intel = context.threat_intel
        
        # Base CVSS score (normalized to 0-1)
        base_score = intel.cvss_score / 10.0
        
        # EPSS score (probability of exploitation)
        epss_factor = intel.exploit_prediction_scoring_system
        
        # Exploit availability factor
        exploit_factor = 1.0 if intel.known_exploited else (
            0.8 if intel.exploit_available else 0.4
        )
        
        # Combine factors
        threat_score = (base_score * 0.4) + (epss_factor * 0.4) + (exploit_factor * 0.2)
        
        return min(1.0, threat_score)
    
    def _calculate_business_impact_score(self, context: VulnerabilityContext) -> float:
        """Calculate business impact score"""
        
        biz = context.business_context
        
        # Weighted combination of business factors
        impact_score = (
            biz.asset_criticality * 0.3 +
            biz.revenue_impact * 0.25 +
            biz.compliance_importance * 0.2 +
            biz.data_sensitivity * 0.25
        )
        
        # Adjust for user base size (logarithmic scaling)
        user_factor = min(1.0, np.log10(max(1, biz.user_base_size)) / 6.0)
        impact_score = impact_score * (0.8 + 0.2 * user_factor)
        
        return min(1.0, impact_score)
    
    def _calculate_exploitability_score(self, context: VulnerabilityContext) -> float:
        """Calculate exploitability score"""
        
        intel = context.threat_intel
        env = context.environmental_context
        
        # CVSS exploitability metrics
        attack_vector_score = {
            "network": 1.0, "adjacent": 0.7, "local": 0.4, "physical": 0.2
        }.get(intel.attack_vector.lower(), 0.5)
        
        attack_complexity_score = {
            "low": 1.0, "high": 0.4
        }.get(intel.attack_complexity.lower(), 0.7)
        
        privileges_score = {
            "none": 1.0, "low": 0.7, "high": 0.3
        }.get(intel.privileges_required.lower(), 0.5)
        
        user_interaction_score = {
            "none": 1.0, "required": 0.7
        }.get(intel.user_interaction.lower(), 0.8)
        
        # Environmental factors
        exposure_score = {
            "internet": 1.0, "internal": 0.6, "isolated": 0.2
        }.get(env.network_exposure, 0.5)
        
        # Combine scores
        exploitability = (
            attack_vector_score * 0.25 +
            attack_complexity_score * 0.2 +
            privileges_score * 0.2 +
            user_interaction_score * 0.15 +
            exposure_score * 0.2
        )
        
        return min(1.0, exploitability)
    
    def _calculate_environmental_risk_score(self, context: VulnerabilityContext) -> float:
        """Calculate environmental risk score"""
        
        env = context.environmental_context
        
        # Security controls effectiveness
        controls_score = 1.0 - (len(env.security_controls) * 0.1)
        controls_score = max(0.2, controls_score)
        
        # Monitoring coverage (inverse - less monitoring = higher risk)
        monitoring_risk = 1.0 - env.monitoring_coverage
        
        # Incident response capability (inverse)
        response_risk = 1.0 - env.incident_response_capability
        
        # Patch deployment speed
        patch_risk = {
            "immediate": 0.1, "fast": 0.3, "normal": 0.6, "slow": 1.0
        }.get(env.patch_deployment_speed, 0.8)
        
        # Combine factors
        env_risk = (
            controls_score * 0.3 +
            monitoring_risk * 0.25 +
            response_risk * 0.25 +
            patch_risk * 0.2
        )
        
        return min(1.0, env_risk)
    
    def _calculate_temporal_urgency_score(self, context: VulnerabilityContext) -> float:
        """Calculate temporal urgency based on dynamic factors"""
        
        # Age of vulnerability (older = higher urgency for patching)
        age_days = (datetime.now() - context.discovered_at).days
        age_factor = min(1.0, age_days / 30.0)  # Max urgency after 30 days
        
        # Trending attacks factor
        trending_factor = len(context.trending_attacks) * 0.2
        
        # Recent incidents factor
        incident_factor = len(context.recent_incidents) * 0.15
        
        # Exploit mentions in threat feeds
        mention_factor = min(1.0, context.exploit_mentions / 10.0)
        
        # Patch availability (urgent if patch is available)
        patch_factor = 0.8 if context.patch_availability else 0.3
        
        # Combine factors
        urgency = (
            age_factor * 0.2 +
            trending_factor * 0.25 +
            incident_factor * 0.2 +
            mention_factor * 0.15 +
            patch_factor * 0.2
        )
        
        return min(1.0, urgency)
    
    def _combine_scores(
        self, 
        threat: float, 
        business: float, 
        exploitability: float,
        environmental: float, 
        temporal: float, 
        ai_insights: Dict
    ) -> float:
        """Combine all scores with AI insights to get final priority score"""
        
        # Base combination (traditional approach)
        base_score = (
            threat * 0.25 +
            business * 0.25 +
            exploitability * 0.2 +
            environmental * 0.15 +
            temporal * 0.15
        )
        
        # AI enhancement factor
        ai_impact_score = ai_insights.get("impact_assessment", {}).get("score", 0.5)
        ai_exploit_score = ai_insights.get("exploitability_analysis", {}).get("score", 0.5)
        
        # Combine AI insights (weighted average with base score)
        ai_enhanced_score = (base_score * 0.7) + ((ai_impact_score + ai_exploit_score) / 2 * 0.3)
        
        return min(1.0, ai_enhanced_score)
    
    def _score_to_priority_level(self, score: float) -> PriorityLevel:
        """Convert numeric score to priority level"""
        
        if score >= 0.9:
            return PriorityLevel.CRITICAL
        elif score >= 0.7:
            return PriorityLevel.HIGH
        elif score >= 0.4:
            return PriorityLevel.MEDIUM
        elif score >= 0.2:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.INFO
    
    def _calculate_confidence(
        self, 
        context: VulnerabilityContext, 
        ai_insights: Dict
    ) -> float:
        """Calculate confidence in prioritization"""
        
        confidence_factors = []
        
        # Data completeness
        if context.threat_intel.cve_id:
            confidence_factors.append(0.2)
        if context.business_context.asset_criticality > 0:
            confidence_factors.append(0.2)
        if context.threat_intel.exploit_available:
            confidence_factors.append(0.15)
        if len(ai_insights.get("confidence_factors", [])) > 2:
            confidence_factors.append(0.15)
        
        # AI analysis quality
        if ai_insights.get("ai_reasoning") and len(ai_insights["ai_reasoning"]) > 100:
            confidence_factors.append(0.15)
        
        # Threat intelligence freshness
        confidence_factors.append(0.15)  # Assume fresh data
        
        return min(1.0, sum(confidence_factors))
    
    async def _get_cached_prioritization(self, vulnerability_id: str) -> Optional[PrioritizationResult]:
        """Get cached prioritization if still valid"""
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT * FROM vulnerability_priorities
                    WHERE vulnerability_id = $1 
                    AND expires_at > NOW()
                    ORDER BY analyzed_at DESC
                    LIMIT 1
                """, vulnerability_id)
                
                if result:
                    return PrioritizationResult(
                        vulnerability_id=result['vulnerability_id'],
                        priority_score=result['priority_score'],
                        priority_level=PriorityLevel(result['priority_level']),
                        confidence=result['confidence'],
                        threat_score=result['threat_score'],
                        business_impact_score=result['business_impact_score'],
                        exploitability_score=result['exploitability_score'],
                        environmental_risk_score=result['environmental_risk_score'],
                        temporal_urgency_score=result['temporal_urgency_score'],
                        ai_insights=result['ai_insights'],
                        recommended_actions=result['recommended_actions'],
                        risk_factors=result['risk_factors'],
                        mitigation_urgency=result['mitigation_urgency'],
                        model_version=result['model_version'],
                        analyzed_at=result['analyzed_at'],
                        expires_at=result['expires_at']
                    )
                    
        except Exception as e:
            logger.error("Failed to get cached prioritization", error=str(e))
            
        return None
    
    async def _store_prioritization_result(self, result: PrioritizationResult):
        """Store prioritization result in database"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO vulnerability_priorities
                    (vulnerability_id, priority_score, priority_level, confidence,
                     threat_score, business_impact_score, exploitability_score,
                     environmental_risk_score, temporal_urgency_score,
                     ai_insights, recommended_actions, risk_factors, mitigation_urgency,
                     model_version, analyzed_at, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, 
                result.vulnerability_id, result.priority_score, result.priority_level.value,
                result.confidence, result.threat_score, result.business_impact_score,
                result.exploitability_score, result.environmental_risk_score,
                result.temporal_urgency_score, json.dumps(result.ai_insights),
                json.dumps(result.recommended_actions), json.dumps(result.risk_factors),
                result.mitigation_urgency, result.model_version, result.analyzed_at,
                result.expires_at)
                
        except Exception as e:
            logger.error("Failed to store prioritization result", error=str(e))
    
    async def get_prioritization_statistics(self) -> Dict:
        """Get comprehensive prioritization statistics"""
        
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_prioritized,
                        AVG(priority_score) as avg_score,
                        COUNT(*) FILTER (WHERE priority_level = 'critical') as critical_count,
                        COUNT(*) FILTER (WHERE priority_level = 'high') as high_count,
                        COUNT(*) FILTER (WHERE priority_level = 'medium') as medium_count,
                        COUNT(*) FILTER (WHERE priority_level = 'low') as low_count,
                        AVG(confidence) as avg_confidence,
                        MAX(analyzed_at) as last_analysis
                    FROM vulnerability_priorities
                    WHERE analyzed_at >= NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    "total_prioritized": stats['total_prioritized'],
                    "average_priority_score": float(stats['avg_score'] or 0),
                    "priority_distribution": {
                        "critical": stats['critical_count'],
                        "high": stats['high_count'],
                        "medium": stats['medium_count'],
                        "low": stats['low_count']
                    },
                    "average_confidence": float(stats['avg_confidence'] or 0),
                    "last_analysis": stats['last_analysis'],
                    "analysis_model": "AI Enhanced v6.1",
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get prioritization statistics", error=str(e))
            return {"error": str(e)}

async def main():
    """Main prioritization engine service"""
    
    # Start Prometheus metrics server
    start_http_server(8010)
    
    # Initialize prioritization engine
    config = {
        "database_url": os.getenv("DATABASE_URL", 
                                 "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379/0"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    engine = VulnerabilityPrioritizationEngine()
    await engine.initialize(config)
    
    logger.info("🧠 Xorb AI Vulnerability Prioritization Engine started",
               service_version="6.1.0",
               features=["ai_context_analysis", "threat_intelligence", "business_context", "ml_scoring"])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down prioritization engine")

if __name__ == "__main__":
    asyncio.run(main())