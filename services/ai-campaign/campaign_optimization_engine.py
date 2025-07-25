#!/usr/bin/env python3
"""
Xorb AI-Driven Campaign Optimization Engine
Phase 6.5 - Autonomous Testing Strategy Evolution & Resource Optimization
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import asyncpg
import aioredis
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
logger = structlog.get_logger("xorb.ai_campaign")

# Phase 6.5 Metrics
campaign_optimizations_total = Counter(
    'campaign_optimizations_total',
    'Total campaign optimizations performed',
    ['optimization_type', 'strategy_type']
)

optimization_performance_gain = Histogram(
    'optimization_performance_gain',
    'Performance gain from optimizations',
    ['metric_type']
)

resource_efficiency_score = Gauge(
    'resource_efficiency_score',
    'Resource efficiency score',
    ['resource_type', 'campaign_type']
)

vulnerability_discovery_rate = Gauge(
    'vulnerability_discovery_rate',
    'Rate of vulnerability discovery',
    ['scanner_type', 'target_type']
)

campaign_success_rate = Gauge(
    'campaign_success_rate',
    'Campaign success rate by strategy',
    ['strategy_type', 'target_complexity']
)

ai_strategy_recommendations = Counter(
    'ai_strategy_recommendations_total',
    'AI strategy recommendations generated',
    ['strategy_category', 'confidence_level']
)

class OptimizationType(Enum):
    TARGET_SELECTION = "target_selection"
    SCANNER_CONFIGURATION = "scanner_configuration"
    RESOURCE_ALLOCATION = "resource_allocation"
    STRATEGY_EVOLUTION = "strategy_evolution"
    PARAMETER_TUNING = "parameter_tuning"

class CampaignStrategy(Enum):
    COMPREHENSIVE = "comprehensive"
    FOCUSED = "focused"
    RAPID = "rapid"
    DEEP_DIVE = "deep_dive"
    STEALTH = "stealth"
    BASELINE = "baseline"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    SCANNER_LICENSES = "scanner_licenses"

@dataclass
class AssetProfile:
    """Asset profiling for targeted testing"""
    asset_id: str
    asset_type: str
    technology_stack: List[str]
    complexity_score: float
    attack_surface_size: int
    historical_vulnerability_density: float
    business_criticality: float
    security_maturity: float
    discovery_probability: float
    testing_cost_estimate: float

@dataclass
class ScannerProfile:
    """Scanner performance characteristics"""
    scanner_name: str
    scanner_type: str
    strengths: List[str]
    weaknesses: List[str]
    optimal_target_types: List[str]
    
    # Performance metrics
    avg_scan_duration: float
    vulnerability_discovery_rate: float
    false_positive_rate: float
    resource_consumption: Dict[str, float]
    cost_per_scan: float
    
    # Effectiveness by target type
    effectiveness_scores: Dict[str, float]

@dataclass
class CampaignConfiguration:
    """Optimized campaign configuration"""
    campaign_id: str
    strategy: CampaignStrategy
    
    # Target selection
    selected_assets: List[str]
    asset_priorities: Dict[str, float]
    
    # Scanner configuration
    scanner_assignments: Dict[str, str]  # asset_id -> scanner_name
    scanner_parameters: Dict[str, Dict]  # scanner_name -> parameters
    
    # Resource allocation
    resource_allocation: Dict[ResourceType, float]
    execution_schedule: List[Dict]
    
    # Performance predictions
    predicted_vulnerabilities: int
    predicted_duration_hours: float
    predicted_cost: float
    confidence_score: float
    
    # Optimization metadata
    optimization_reasoning: str
    alternative_strategies: List[Dict]
    generated_at: datetime

@dataclass
class OptimizationResult:
    """Campaign optimization result"""
    optimization_id: str
    campaign_id: str
    optimization_type: OptimizationType
    
    # Performance improvements
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percentage: Dict[str, float]
    
    # Strategy changes
    strategy_changes: List[Dict]
    parameter_adjustments: Dict[str, Any]
    resource_reallocation: Dict[str, float]
    
    # AI insights
    ai_reasoning: str
    confidence_level: str
    success_probability: float
    
    # Validation
    a_b_test_results: Optional[Dict]
    performance_validation: Dict[str, bool]
    
    # Metadata
    optimized_at: datetime
    model_version: str

class AssetProfilingEngine:
    """Profiles assets for intelligent targeting"""
    
    def __init__(self):
        self.db_pool = None
        self.profiling_models = {}
        
    async def initialize(self, database_url: str):
        """Initialize asset profiling engine"""
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        
    async def profile_asset(self, asset_id: str) -> AssetProfile:
        """Create comprehensive asset profile"""
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get asset data
                asset_data = await conn.fetchrow("""
                    SELECT a.*, 
                           COUNT(v.id) as vuln_count,
                           AVG(v.cvss_score) as avg_cvss,
                           COUNT(DISTINCT s.id) as scan_count
                    FROM assets a
                    LEFT JOIN vulnerabilities v ON a.id = v.asset_id
                    LEFT JOIN scans s ON a.id = s.asset_id
                    WHERE a.id = $1
                    GROUP BY a.id
                """, asset_id)
                
                if not asset_data:
                    raise ValueError(f"Asset {asset_id} not found")
                
                # Analyze technology stack
                tech_stack = await self._analyze_technology_stack(asset_id)
                
                # Calculate complexity score
                complexity_score = await self._calculate_complexity_score(asset_data, tech_stack)
                
                # Estimate attack surface
                attack_surface_size = await self._estimate_attack_surface(asset_id, tech_stack)
                
                # Calculate historical vulnerability density
                vuln_density = (asset_data['vuln_count'] or 0) / max(1, asset_data['scan_count'] or 1)
                
                # Get business criticality
                business_criticality = float(asset_data.get('criticality_score', 5)) / 10.0
                
                # Assess security maturity
                security_maturity = await self._assess_security_maturity(asset_id)
                
                # Predict discovery probability
                discovery_probability = await self._predict_discovery_probability(
                    complexity_score, vuln_density, security_maturity
                )
                
                # Estimate testing cost
                testing_cost = await self._estimate_testing_cost(
                    attack_surface_size, complexity_score, tech_stack
                )
                
                profile = AssetProfile(
                    asset_id=asset_id,
                    asset_type=asset_data.get('asset_type', 'unknown'),
                    technology_stack=tech_stack,
                    complexity_score=complexity_score,
                    attack_surface_size=attack_surface_size,
                    historical_vulnerability_density=vuln_density,
                    business_criticality=business_criticality,
                    security_maturity=security_maturity,
                    discovery_probability=discovery_probability,
                    testing_cost_estimate=testing_cost
                )
                
                logger.info("Asset profile created",
                           asset_id=asset_id,
                           complexity_score=complexity_score,
                           discovery_probability=discovery_probability)
                
                return profile
                
        except Exception as e:
            logger.error("Asset profiling failed", asset_id=asset_id, error=str(e))
            raise
    
    async def _analyze_technology_stack(self, asset_id: str) -> List[str]:
        """Analyze technology stack for asset"""
        
        # This would integrate with asset discovery tools
        # For now, simulate based on asset type and known patterns
        
        try:
            async with self.db_pool.acquire() as conn:
                asset_info = await conn.fetchrow("""
                    SELECT asset_type, metadata FROM assets WHERE id = $1
                """, asset_id)
                
                if not asset_info:
                    return ["unknown"]
                
                asset_type = asset_info['asset_type']
                metadata = asset_info.get('metadata', {})
                
                # Technology stack inference
                tech_stack = []
                
                if asset_type == 'web_application':
                    # Common web technologies
                    web_techs = ["html", "css", "javascript"]
                    
                    # Backend inference based on patterns
                    if 'framework' in metadata:
                        framework = metadata['framework'].lower()
                        if 'django' in framework or 'flask' in framework:
                            tech_stack.extend(["python", framework])
                        elif 'express' in framework or 'node' in framework:
                            tech_stack.extend(["nodejs", "javascript"])
                        elif 'spring' in framework:
                            tech_stack.extend(["java", "spring"])
                    
                    # Database inference
                    if 'database' in metadata:
                        tech_stack.append(metadata['database'])
                    else:
                        tech_stack.append("database")  # Generic
                    
                    tech_stack.extend(web_techs)
                
                elif asset_type == 'api_endpoint':
                    tech_stack.extend(["rest_api", "json"])
                    
                    # API technology inference
                    if 'api_type' in metadata:
                        tech_stack.append(metadata['api_type'])
                
                elif asset_type == 'mobile_application':
                    tech_stack.extend(["mobile", "https"])
                    
                    # Platform inference
                    if 'platform' in metadata:
                        platform = metadata['platform'].lower()
                        if 'android' in platform:
                            tech_stack.extend(["android", "java", "kotlin"])
                        elif 'ios' in platform:
                            tech_stack.extend(["ios", "swift", "objective-c"])
                
                elif asset_type == 'infrastructure':
                    tech_stack.extend(["infrastructure", "network"])
                    
                    # Infrastructure components
                    if 'services' in metadata:
                        tech_stack.extend(metadata['services'])
                
                return list(set(tech_stack))  # Remove duplicates
                
        except Exception as e:
            logger.warning("Technology stack analysis failed", error=str(e))
            return ["unknown"]
    
    async def _calculate_complexity_score(self, asset_data: Dict, tech_stack: List[str]) -> float:
        """Calculate asset complexity score"""
        
        base_complexity = 0.3
        
        # Technology stack complexity
        tech_complexity = {
            "javascript": 0.1, "python": 0.1, "java": 0.15, "csharp": 0.15,
            "php": 0.1, "ruby": 0.1, "go": 0.05, "rust": 0.05,
            "react": 0.1, "angular": 0.15, "vue": 0.08,
            "spring": 0.15, "django": 0.1, "rails": 0.12,
            "database": 0.1, "redis": 0.05, "elasticsearch": 0.1,
            "microservices": 0.2, "kubernetes": 0.15, "docker": 0.1
        }
        
        for tech in tech_stack:
            base_complexity += tech_complexity.get(tech.lower(), 0.05)
        
        # Asset type complexity
        asset_type_complexity = {
            "web_application": 0.3,
            "api_endpoint": 0.2,
            "mobile_application": 0.25,
            "infrastructure": 0.4,
            "database": 0.35,
            "cloud_service": 0.3
        }
        
        asset_type = asset_data.get('asset_type', 'unknown')
        base_complexity += asset_type_complexity.get(asset_type, 0.2)
        
        # Historical vulnerability count influence
        vuln_count = asset_data.get('vuln_count', 0) or 0
        if vuln_count > 10:
            base_complexity += 0.2
        elif vuln_count > 5:
            base_complexity += 0.1
        
        return min(1.0, base_complexity)
    
    async def _estimate_attack_surface(self, asset_id: str, tech_stack: List[str]) -> int:
        """Estimate attack surface size"""
        
        base_surface = 10  # Minimum attack surface
        
        # Technology-based surface expansion
        surface_multipliers = {
            "web_application": 5,
            "api_endpoint": 3,
            "database": 2,
            "javascript": 2,
            "php": 3,  # Often more vulnerable patterns
            "java": 2,
            "microservices": 4,  # More endpoints
            "rest_api": 3,
            "graphql": 4,
            "websockets": 2
        }
        
        for tech in tech_stack:
            base_surface += surface_multipliers.get(tech.lower(), 1)
        
        # Add randomness for realism
        variation = int(base_surface * 0.3)
        surface_size = base_surface + random.randint(-variation, variation)
        
        return max(5, surface_size)
    
    async def _assess_security_maturity(self, asset_id: str) -> float:
        """Assess security maturity of asset"""
        
        try:
            async with self.db_pool.acquire() as conn:
                # Look at security measures and vulnerability history
                security_data = await conn.fetchrow("""
                    SELECT 
                        COUNT(v.id) FILTER (WHERE v.severity = 'critical') as critical_vulns,
                        COUNT(v.id) FILTER (WHERE v.severity = 'high') as high_vulns,
                        COUNT(v.id) FILTER (WHERE v.created_at >= NOW() - INTERVAL '30 days') as recent_vulns,
                        MAX(s.created_at) as last_scan
                    FROM assets a
                    LEFT JOIN vulnerabilities v ON a.id = v.asset_id
                    LEFT JOIN scans s ON a.id = s.asset_id
                    WHERE a.id = $1
                """, asset_id)
                
                if not security_data:
                    return 0.5  # Default medium maturity
                
                # Start with medium maturity
                maturity = 0.6
                
                # Penalize for critical/high vulnerabilities
                critical_vulns = security_data['critical_vulns'] or 0
                high_vulns = security_data['high_vulns'] or 0
                
                if critical_vulns > 0:
                    maturity -= 0.3
                elif high_vulns > 3:
                    maturity -= 0.2
                elif high_vulns > 0:
                    maturity -= 0.1
                
                # Penalize for recent vulnerabilities (indicates poor maintenance)
                recent_vulns = security_data['recent_vulns'] or 0
                if recent_vulns > 5:
                    maturity -= 0.2
                elif recent_vulns > 2:
                    maturity -= 0.1
                
                # Bonus for regular scanning (indicates good security practices)
                last_scan = security_data['last_scan']
                if last_scan:
                    days_since_scan = (datetime.now() - last_scan).days
                    if days_since_scan <= 7:
                        maturity += 0.1
                    elif days_since_scan <= 30:
                        maturity += 0.05
                
                return max(0.1, min(1.0, maturity))
                
        except Exception as e:
            logger.warning("Security maturity assessment failed", error=str(e))
            return 0.5
    
    async def _predict_discovery_probability(
        self, 
        complexity: float, 
        vuln_density: float, 
        maturity: float
    ) -> float:
        """Predict probability of discovering vulnerabilities"""
        
        # Higher complexity and lower maturity = higher discovery probability
        # Higher historical vulnerability density = higher probability
        
        base_probability = 0.3
        
        # Complexity factor (more complex = more likely to have issues)
        complexity_factor = complexity * 0.4
        
        # Vulnerability density factor (history predicts future)
        density_factor = min(vuln_density / 5.0, 0.3)  # Cap at 0.3
        
        # Maturity factor (lower maturity = higher probability)
        maturity_factor = (1.0 - maturity) * 0.3
        
        probability = base_probability + complexity_factor + density_factor + maturity_factor
        
        return min(1.0, probability)
    
    async def _estimate_testing_cost(
        self, 
        attack_surface: int, 
        complexity: float, 
        tech_stack: List[str]
    ) -> float:
        """Estimate cost of testing asset"""
        
        # Base cost per attack surface point
        base_cost_per_point = 5.0  # dollars
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + complexity
        
        # Technology-specific cost adjustments
        tech_cost_factors = {
            "java": 1.2,
            "csharp": 1.2,
            "php": 0.9,  # Often simpler to test
            "python": 1.0,
            "javascript": 1.1,
            "microservices": 1.5,  # More complex to test
            "kubernetes": 1.3,
            "database": 1.4,  # Requires specialized testing
            "mobile": 1.3
        }
        
        tech_multiplier = 1.0
        for tech in tech_stack:
            tech_multiplier *= tech_cost_factors.get(tech.lower(), 1.0)
        
        # Cap multiplier to prevent extreme costs
        tech_multiplier = min(tech_multiplier, 2.0)
        
        total_cost = attack_surface * base_cost_per_point * complexity_multiplier * tech_multiplier
        
        return round(total_cost, 2)

class ScannerOptimizationEngine:
    """Optimizes scanner selection and configuration"""
    
    def __init__(self):
        self.scanner_profiles = {}
        self.optimization_models = {}
        
    async def initialize(self):
        """Initialize scanner optimization engine"""
        
        # Initialize scanner profiles
        self.scanner_profiles = {
            "nuclei": ScannerProfile(
                scanner_name="nuclei",
                scanner_type="template_based",
                strengths=["speed", "coverage", "accuracy"],
                weaknesses=["limited_deep_analysis"],
                optimal_target_types=["web_application", "api_endpoint"],
                avg_scan_duration=300.0,  # 5 minutes
                vulnerability_discovery_rate=0.7,
                false_positive_rate=0.1,
                resource_consumption={"cpu": 2.0, "memory": 512.0, "network": 1.0},
                cost_per_scan=2.50,
                effectiveness_scores={
                    "web_application": 0.9,
                    "api_endpoint": 0.85,
                    "mobile_application": 0.3,
                    "infrastructure": 0.6
                }
            ),
            "zap": ScannerProfile(
                scanner_name="zap",
                scanner_type="dynamic_analysis",
                strengths=["interactive_testing", "comprehensive_web"],
                weaknesses=["slow", "resource_intensive"],
                optimal_target_types=["web_application"],
                avg_scan_duration=1800.0,  # 30 minutes
                vulnerability_discovery_rate=0.8,
                false_positive_rate=0.15,
                resource_consumption={"cpu": 4.0, "memory": 2048.0, "network": 2.0},
                cost_per_scan=8.00,
                effectiveness_scores={
                    "web_application": 0.95,
                    "api_endpoint": 0.7,
                    "mobile_application": 0.2,
                    "infrastructure": 0.3
                }
            ),
            "nmap": ScannerProfile(
                scanner_name="nmap",
                scanner_type="network_discovery",
                strengths=["network_mapping", "service_detection"],
                weaknesses=["limited_application_testing"],
                optimal_target_types=["infrastructure", "network"],
                avg_scan_duration=600.0,  # 10 minutes
                vulnerability_discovery_rate=0.5,
                false_positive_rate=0.05,
                resource_consumption={"cpu": 1.0, "memory": 256.0, "network": 3.0},
                cost_per_scan=1.50,
                effectiveness_scores={
                    "infrastructure": 0.9,
                    "network": 0.95,
                    "web_application": 0.3,
                    "api_endpoint": 0.4
                }
            ),
            "custom_mobile": ScannerProfile(
                scanner_name="custom_mobile",
                scanner_type="mobile_analysis",
                strengths=["mobile_specific", "binary_analysis"],
                weaknesses=["limited_scope"],
                optimal_target_types=["mobile_application"],
                avg_scan_duration=2400.0,  # 40 minutes
                vulnerability_discovery_rate=0.75,
                false_positive_rate=0.12,
                resource_consumption={"cpu": 3.0, "memory": 1536.0, "network": 1.0},
                cost_per_scan=12.00,
                effectiveness_scores={
                    "mobile_application": 0.9,
                    "web_application": 0.1,
                    "api_endpoint": 0.2,
                    "infrastructure": 0.1
                }
            )
        }
        
        logger.info("Scanner optimization engine initialized",
                   scanners_loaded=len(self.scanner_profiles))
    
    async def optimize_scanner_selection(
        self, 
        asset_profiles: List[AssetProfile],
        resource_constraints: Dict[ResourceType, float],
        objectives: Dict[str, float]
    ) -> Dict[str, str]:
        """Optimize scanner selection for assets"""
        
        scanner_assignments = {}
        
        # Sort assets by priority (discovery probability * business criticality)
        prioritized_assets = sorted(
            asset_profiles,
            key=lambda a: a.discovery_probability * a.business_criticality,
            reverse=True
        )
        
        # Track resource usage
        resource_usage = {rt: 0.0 for rt in ResourceType}
        total_cost = 0.0
        
        for asset in prioritized_assets:
            best_scanner = self._select_best_scanner(
                asset, resource_constraints, resource_usage, objectives
            )
            
            if best_scanner:
                scanner_assignments[asset.asset_id] = best_scanner.scanner_name
                
                # Update resource usage
                for resource_type, usage in best_scanner.resource_consumption.items():
                    if ResourceType(resource_type) in resource_usage:
                        resource_usage[ResourceType(resource_type)] += usage
                
                total_cost += best_scanner.cost_per_scan
                
                logger.debug("Scanner assigned",
                           asset_id=asset.asset_id,
                           scanner=best_scanner.scanner_name,
                           effectiveness=best_scanner.effectiveness_scores.get(asset.asset_type, 0.5))
        
        logger.info("Scanner selection optimized",
                   assignments=len(scanner_assignments),
                   total_estimated_cost=total_cost,
                   resource_utilization={rt.value: usage for rt, usage in resource_usage.items()})
        
        return scanner_assignments
    
    def _select_best_scanner(
        self,
        asset: AssetProfile,
        resource_constraints: Dict[ResourceType, float],
        current_usage: Dict[ResourceType, float],
        objectives: Dict[str, float]
    ) -> Optional[ScannerProfile]:
        """Select best scanner for specific asset"""
        
        best_scanner = None
        best_score = 0.0
        
        for scanner in self.scanner_profiles.values():
            # Check resource constraints
            can_run = True
            for resource_name, consumption in scanner.resource_consumption.items():
                resource_type = ResourceType(resource_name)
                if resource_type in resource_constraints:
                    total_usage = current_usage[resource_type] + consumption
                    if total_usage > resource_constraints[resource_type]:
                        can_run = False
                        break
            
            if not can_run:
                continue
            
            # Calculate effectiveness score
            effectiveness = scanner.effectiveness_scores.get(asset.asset_type, 0.1)
            
            # Multi-objective scoring
            discovery_score = effectiveness * asset.discovery_probability
            efficiency_score = effectiveness / max(scanner.avg_scan_duration / 600.0, 0.1)  # Normalize to 10 min
            cost_score = 1.0 / max(scanner.cost_per_scan / 5.0, 0.1)  # Normalize to $5
            accuracy_score = 1.0 - scanner.false_positive_rate
            
            # Weighted combination based on objectives
            total_score = (
                discovery_score * objectives.get("discovery", 0.4) +
                efficiency_score * objectives.get("speed", 0.2) +
                cost_score * objectives.get("cost", 0.2) +
                accuracy_score * objectives.get("accuracy", 0.2)
            )
            
            if total_score > best_score:
                best_score = total_score
                best_scanner = scanner
        
        return best_scanner
    
    async def optimize_scanner_parameters(
        self,
        scanner_name: str,
        asset_profile: AssetProfile,
        historical_performance: Dict
    ) -> Dict[str, Any]:
        """Optimize scanner parameters for specific asset"""
        
        scanner = self.scanner_profiles.get(scanner_name)
        if not scanner:
            return {}
        
        optimized_params = {}
        
        if scanner_name == "nuclei":
            # Nuclei-specific optimizations
            optimized_params.update(await self._optimize_nuclei_parameters(asset_profile, historical_performance))
        
        elif scanner_name == "zap":
            # ZAP-specific optimizations
            optimized_params.update(await self._optimize_zap_parameters(asset_profile, historical_performance))
        
        elif scanner_name == "nmap":
            # Nmap-specific optimizations
            optimized_params.update(await self._optimize_nmap_parameters(asset_profile, historical_performance))
        
        elif scanner_name == "custom_mobile":
            # Mobile scanner optimizations
            optimized_params.update(await self._optimize_mobile_parameters(asset_profile, historical_performance))
        
        return optimized_params
    
    async def _optimize_nuclei_parameters(
        self, 
        asset: AssetProfile, 
        performance: Dict
    ) -> Dict[str, Any]:
        """Optimize Nuclei scanner parameters"""
        
        params = {
            "rate_limit": 150,  # Default requests per second
            "timeout": 10,      # Default timeout in seconds
            "retries": 1,       # Default retry count
            "concurrency": 25   # Default concurrent templates
        }
        
        # Adjust based on asset complexity
        if asset.complexity_score > 0.7:
            params["timeout"] = 15
            params["retries"] = 2
            params["concurrency"] = 15  # Reduce for complex targets
        elif asset.complexity_score < 0.3:
            params["rate_limit"] = 300  # Increase for simple targets
            params["concurrency"] = 50
        
        # Adjust based on historical performance
        if performance.get("avg_response_time", 1000) > 2000:  # Slow target
            params["rate_limit"] = 50
            params["timeout"] = 20
        
        # Technology-specific adjustments
        if "php" in asset.technology_stack:
            params["rate_limit"] = 100  # PHP apps often slower
        
        if "microservices" in asset.technology_stack:
            params["concurrency"] = 10  # Be gentler with distributed systems
        
        return params
    
    async def _optimize_zap_parameters(
        self, 
        asset: AssetProfile, 
        performance: Dict
    ) -> Dict[str, Any]:
        """Optimize ZAP scanner parameters"""
        
        params = {
            "attack_strength": "medium",
            "alert_threshold": "medium",
            "max_scan_duration": 1800,  # 30 minutes
            "threads_per_host": 2
        }
        
        # Adjust based on asset characteristics
        if asset.business_criticality > 0.8:
            params["attack_strength"] = "high"
            params["max_scan_duration"] = 3600  # 1 hour for critical assets
        
        if asset.security_maturity < 0.3:
            params["alert_threshold"] = "low"  # Catch more issues
            params["attack_strength"] = "high"
        
        # Performance-based adjustments
        if performance.get("error_rate", 0) > 0.1:
            params["threads_per_host"] = 1  # Reduce load
            params["attack_strength"] = "low"
        
        return params
    
    async def _optimize_nmap_parameters(
        self, 
        asset: AssetProfile, 
        performance: Dict
    ) -> Dict[str, Any]:
        """Optimize Nmap scanner parameters"""
        
        params = {
            "timing_template": "T3",  # Normal timing
            "host_timeout": "30m",
            "scan_delay": "0",
            "max_retries": 1
        }
        
        # Infrastructure-specific optimizations
        if asset.asset_type == "infrastructure":
            params["timing_template"] = "T2"  # Slower for infrastructure
            params["host_timeout"] = "60m"
        
        # Stealth requirements
        if asset.security_maturity > 0.7:  # Well-monitored targets
            params["timing_template"] = "T1"  # Stealthier
            params["scan_delay"] = "1s"
        
        return params
    
    async def _optimize_mobile_parameters(
        self, 
        asset: AssetProfile, 
        performance: Dict
    ) -> Dict[str, Any]:
        """Optimize mobile scanner parameters"""
        
        params = {
            "analysis_depth": "medium",
            "dynamic_analysis": True,
            "static_analysis": True,
            "network_analysis": True
        }
        
        # Complexity-based adjustments
        if asset.complexity_score > 0.8:
            params["analysis_depth"] = "deep"
        elif asset.complexity_score < 0.3:
            params["analysis_depth"] = "fast"
        
        # Platform-specific optimizations
        if "android" in asset.technology_stack:
            params["android_specific"] = True
            params["apk_analysis"] = True
        
        if "ios" in asset.technology_stack:
            params["ios_specific"] = True
            params["ipa_analysis"] = True
        
        return params

class ResourceOptimizationEngine:
    """Optimizes resource allocation and scheduling"""
    
    def __init__(self):
        self.allocation_models = {}
        
    async def optimize_resource_allocation(
        self,
        campaign_requirements: Dict,
        available_resources: Dict[ResourceType, float],
        performance_targets: Dict[str, float]
    ) -> Dict[ResourceType, float]:
        """Optimize resource allocation for campaign"""
        
        # Start with baseline allocation
        allocation = {
            ResourceType.CPU: available_resources.get(ResourceType.CPU, 8.0) * 0.7,
            ResourceType.MEMORY: available_resources.get(ResourceType.MEMORY, 16384.0) * 0.7,
            ResourceType.NETWORK: available_resources.get(ResourceType.NETWORK, 1000.0) * 0.8,
            ResourceType.STORAGE: available_resources.get(ResourceType.STORAGE, 100.0) * 0.5,
            ResourceType.SCANNER_LICENSES: available_resources.get(ResourceType.SCANNER_LICENSES, 10.0)
        }
        
        # Adjust based on campaign characteristics
        asset_count = campaign_requirements.get("asset_count", 10)
        complexity_avg = campaign_requirements.get("avg_complexity", 0.5)
        
        # Scale CPU and memory with asset count and complexity
        cpu_scale = min(2.0, 1.0 + (asset_count / 20.0) + complexity_avg)
        memory_scale = min(2.0, 1.0 + (asset_count / 15.0) + (complexity_avg * 0.5))
        
        allocation[ResourceType.CPU] = min(
            allocation[ResourceType.CPU] * cpu_scale,
            available_resources.get(ResourceType.CPU, 8.0) * 0.9
        )
        
        allocation[ResourceType.MEMORY] = min(
            allocation[ResourceType.MEMORY] * memory_scale,
            available_resources.get(ResourceType.MEMORY, 16384.0) * 0.9
        )
        
        # Adjust for performance targets
        if performance_targets.get("speed", 0.5) > 0.8:
            # High speed requirement - allocate more resources
            allocation[ResourceType.CPU] *= 1.3
            allocation[ResourceType.MEMORY] *= 1.2
            allocation[ResourceType.NETWORK] *= 1.4
        
        # Constraint enforcement
        for resource_type in allocation:
            max_available = available_resources.get(resource_type, 0)
            allocation[resource_type] = min(allocation[resource_type], max_available * 0.95)
        
        return allocation
    
    async def optimize_execution_schedule(
        self,
        scanner_assignments: Dict[str, str],
        asset_profiles: List[AssetProfile],
        resource_allocation: Dict[ResourceType, float]
    ) -> List[Dict]:
        """Optimize execution schedule for maximum efficiency"""
        
        schedule = []
        
        # Group assets by scanner type
        scanner_groups = {}
        asset_map = {a.asset_id: a for a in asset_profiles}
        
        for asset_id, scanner_name in scanner_assignments.items():
            if scanner_name not in scanner_groups:
                scanner_groups[scanner_name] = []
            scanner_groups[scanner_name].append(asset_id)
        
        # Schedule each scanner group
        current_time = datetime.now()
        
        for scanner_name, asset_ids in scanner_groups.items():
            # Sort assets within group by priority
            sorted_assets = sorted(
                asset_ids,
                key=lambda aid: asset_map[aid].discovery_probability * asset_map[aid].business_criticality,
                reverse=True
            )
            
            # Determine parallelism based on resource allocation
            max_parallel = self._calculate_max_parallel_scans(scanner_name, resource_allocation)
            
            # Create batches for parallel execution
            for i in range(0, len(sorted_assets), max_parallel):
                batch = sorted_assets[i:i + max_parallel]
                
                # Estimate batch duration
                batch_duration = self._estimate_batch_duration(scanner_name, batch, asset_map)
                
                schedule.append({
                    "start_time": current_time,
                    "end_time": current_time + timedelta(seconds=batch_duration),
                    "scanner": scanner_name,
                    "assets": batch,
                    "parallel_count": len(batch),
                    "estimated_duration": batch_duration
                })
                
                current_time += timedelta(seconds=batch_duration + 60)  # 1 minute buffer
        
        # Optimize schedule order for resource efficiency
        optimized_schedule = self._optimize_schedule_order(schedule)
        
        return optimized_schedule
    
    def _calculate_max_parallel_scans(
        self, 
        scanner_name: str, 
        resource_allocation: Dict[ResourceType, float]
    ) -> int:
        """Calculate maximum parallel scans for scanner"""
        
        # Get scanner resource requirements (mock data)
        scanner_requirements = {
            "nuclei": {"cpu": 0.5, "memory": 128.0, "network": 10.0},
            "zap": {"cpu": 2.0, "memory": 1024.0, "network": 20.0},
            "nmap": {"cpu": 0.3, "memory": 64.0, "network": 50.0},
            "custom_mobile": {"cpu": 1.5, "memory": 512.0, "network": 5.0}
        }
        
        requirements = scanner_requirements.get(scanner_name, {"cpu": 1.0, "memory": 256.0, "network": 10.0})
        
        # Calculate maximum based on each resource constraint
        max_by_cpu = int(resource_allocation.get(ResourceType.CPU, 4.0) / requirements["cpu"])
        max_by_memory = int(resource_allocation.get(ResourceType.MEMORY, 8192.0) / requirements["memory"])
        max_by_network = int(resource_allocation.get(ResourceType.NETWORK, 1000.0) / requirements["network"])
        
        # Return the most constraining factor
        return max(1, min(max_by_cpu, max_by_memory, max_by_network, 10))  # Cap at 10
    
    def _estimate_batch_duration(
        self,
        scanner_name: str,
        asset_batch: List[str],
        asset_map: Dict[str, AssetProfile]
    ) -> float:
        """Estimate duration for batch of assets"""
        
        # Base durations by scanner (seconds)
        base_durations = {
            "nuclei": 300,
            "zap": 1800,
            "nmap": 600,
            "custom_mobile": 2400
        }
        
        base_duration = base_durations.get(scanner_name, 600)
        
        if not asset_batch:
            return base_duration
        
        # Calculate average complexity for batch
        avg_complexity = np.mean([
            asset_map[asset_id].complexity_score 
            for asset_id in asset_batch 
            if asset_id in asset_map
        ])
        
        # Duration scales with complexity
        complexity_multiplier = 0.5 + (avg_complexity * 1.5)
        
        # For parallel execution, duration is dominated by the slowest asset
        return base_duration * complexity_multiplier
    
    def _optimize_schedule_order(self, initial_schedule: List[Dict]) -> List[Dict]:
        """Optimize schedule order for resource efficiency"""
        
        # Simple optimization: sort by resource utilization efficiency
        def efficiency_score(schedule_item):
            # Favor shorter, more parallel tasks first
            duration = schedule_item["estimated_duration"]
            parallel_count = schedule_item["parallel_count"]
            return parallel_count / max(duration / 600.0, 0.1)  # Normalize to 10 minutes
        
        optimized = sorted(initial_schedule, key=efficiency_score, reverse=True)
        
        # Recalculate start/end times
        current_time = datetime.now()
        for item in optimized:
            item["start_time"] = current_time
            item["end_time"] = current_time + timedelta(seconds=item["estimated_duration"])
            current_time = item["end_time"] + timedelta(minutes=1)  # Buffer
        
        return optimized

class AIStrategyEvolutionEngine:
    """Evolves testing strategies using ML and AI"""
    
    def __init__(self):
        self.ai_client = None
        self.evolution_models = {}
        self.strategy_performance = {}
        
    async def initialize(self, openai_key: str):
        """Initialize strategy evolution engine"""
        self.ai_client = AsyncOpenAI(api_key=openai_key)
        
    async def evolve_campaign_strategy(
        self,
        historical_campaigns: List[Dict],
        current_context: Dict,
        performance_targets: Dict[str, float]
    ) -> CampaignStrategy:
        """Evolve campaign strategy based on historical performance"""
        
        # Analyze historical performance by strategy
        strategy_analysis = await self._analyze_strategy_performance(historical_campaigns)
        
        # Generate AI recommendations
        ai_recommendation = await self._generate_ai_strategy_recommendation(
            strategy_analysis, current_context, performance_targets
        )
        
        # Apply genetic algorithm for strategy evolution
        evolved_strategy = await self._apply_genetic_optimization(
            strategy_analysis, ai_recommendation, performance_targets
        )
        
        return evolved_strategy
    
    async def _analyze_strategy_performance(self, campaigns: List[Dict]) -> Dict:
        """Analyze performance of different strategies"""
        
        analysis = {}
        
        # Group campaigns by strategy
        strategy_groups = {}
        for campaign in campaigns:
            strategy = campaign.get("strategy", "baseline")
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(campaign)
        
        # Calculate performance metrics for each strategy
        for strategy, strategy_campaigns in strategy_groups.items():
            if not strategy_campaigns:
                continue
            
            # Calculate key metrics
            vuln_discovery_rates = [c.get("vulnerabilities_found", 0) / max(c.get("duration_hours", 1), 1) 
                                   for c in strategy_campaigns]
            cost_efficiency = [c.get("vulnerabilities_found", 0) / max(c.get("total_cost", 1), 1) 
                              for c in strategy_campaigns]
            success_rates = [1.0 if c.get("vulnerabilities_found", 0) > 0 else 0.0 
                            for c in strategy_campaigns]
            
            analysis[strategy] = {
                "campaign_count": len(strategy_campaigns),
                "avg_discovery_rate": np.mean(vuln_discovery_rates),
                "avg_cost_efficiency": np.mean(cost_efficiency),
                "success_rate": np.mean(success_rates),
                "total_vulnerabilities": sum(c.get("vulnerabilities_found", 0) for c in strategy_campaigns),
                "avg_duration": np.mean([c.get("duration_hours", 0) for c in strategy_campaigns]),
                "avg_cost": np.mean([c.get("total_cost", 0) for c in strategy_campaigns])
            }
        
        return analysis
    
    async def _generate_ai_strategy_recommendation(
        self,
        strategy_analysis: Dict,
        context: Dict,
        targets: Dict[str, float]
    ) -> Dict:
        """Generate AI-powered strategy recommendation"""
        
        prompt = self._build_strategy_recommendation_prompt(strategy_analysis, context, targets)
        
        try:
            response = await self.ai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cybersecurity strategist. Analyze campaign performance data and recommend optimal testing strategies."
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
            
            recommendation = json.loads(response.choices[0].message.content)
            
            # Update metrics
            ai_strategy_recommendations.labels(
                strategy_category=recommendation.get("recommended_strategy", "unknown"),
                confidence_level=recommendation.get("confidence", "medium")
            ).inc()
            
            return recommendation
            
        except Exception as e:
            logger.error("AI strategy recommendation failed", error=str(e))
            
            # Return default recommendation
            return {
                "recommended_strategy": "comprehensive",
                "confidence": "low",
                "reasoning": "AI analysis failed, using default strategy",
                "adjustments": []
            }
    
    def _build_strategy_recommendation_prompt(
        self, 
        analysis: Dict, 
        context: Dict, 
        targets: Dict
    ) -> str:
        """Build prompt for AI strategy recommendation"""
        
        return f"""
Analyze the following campaign performance data and recommend the optimal testing strategy:

HISTORICAL PERFORMANCE BY STRATEGY:
{json.dumps(analysis, indent=2)}

CURRENT CONTEXT:
- Asset count: {context.get('asset_count', 'unknown')}
- Asset types: {', '.join(context.get('asset_types', []))}
- Budget constraint: ${context.get('budget', 'unlimited')}
- Time constraint: {context.get('time_limit', 'flexible')}
- Business criticality: {context.get('avg_criticality', 'medium')}

PERFORMANCE TARGETS:
- Discovery rate target: {targets.get('discovery_rate', 0.5)} vulnerabilities/hour
- Cost efficiency target: {targets.get('cost_efficiency', 1.0)} vulnerabilities/$
- Success rate target: {targets.get('success_rate', 0.8)}

Available strategies: comprehensive, focused, rapid, deep_dive, stealth, baseline

Please provide recommendation in this JSON format:
{{
  "recommended_strategy": "strategy_name",
  "confidence": "high|medium|low",
  "reasoning": "detailed explanation of why this strategy is recommended",
  "expected_performance": {{
    "discovery_rate": estimated_rate,
    "cost_efficiency": estimated_efficiency,
    "success_probability": estimated_probability
  }},
  "adjustments": [
    "specific parameter or approach adjustments"
  ],
  "alternative_strategies": [
    {{"strategy": "name", "conditions": "when to use this instead"}}
  ]
}}

Focus on data-driven recommendations based on historical performance and current context.
"""
    
    async def _apply_genetic_optimization(
        self,
        strategy_analysis: Dict,
        ai_recommendation: Dict,
        targets: Dict[str, float]
    ) -> CampaignStrategy:
        """Apply genetic algorithm for strategy optimization"""
        
        # Simple genetic algorithm simulation
        # In production, would implement full GA with crossover and mutation
        
        # Define strategy "genes" and their performance
        strategies = [
            CampaignStrategy.COMPREHENSIVE,
            CampaignStrategy.FOCUSED,
            CampaignStrategy.RAPID,
            CampaignStrategy.DEEP_DIVE,
            CampaignStrategy.STEALTH,
            CampaignStrategy.BASELINE
        ]
        
        # Score strategies based on historical performance and AI recommendation
        strategy_scores = {}
        
        for strategy in strategies:
            strategy_name = strategy.value
            
            # Historical performance score
            historical_score = 0.0
            if strategy_name in strategy_analysis:
                perf_data = strategy_analysis[strategy_name]
                historical_score = (
                    perf_data.get("avg_discovery_rate", 0) * targets.get("discovery_rate", 1.0) +
                    perf_data.get("avg_cost_efficiency", 0) * targets.get("cost_efficiency", 1.0) +
                    perf_data.get("success_rate", 0) * targets.get("success_rate", 1.0)
                ) / 3.0
            
            # AI recommendation bonus
            ai_bonus = 0.0
            if ai_recommendation.get("recommended_strategy") == strategy_name:
                confidence = ai_recommendation.get("confidence", "medium")
                ai_bonus = {"high": 0.3, "medium": 0.2, "low": 0.1}.get(confidence, 0.1)
            
            strategy_scores[strategy] = historical_score + ai_bonus
        
        # Select best strategy
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        
        logger.info("Strategy evolution completed",
                   selected_strategy=best_strategy.value,
                   score=strategy_scores[best_strategy],
                   ai_recommendation=ai_recommendation.get("recommended_strategy"))
        
        return best_strategy

class CampaignOptimizationEngine:
    """Main campaign optimization orchestrator"""
    
    def __init__(self):
        self.asset_profiler = AssetProfilingEngine()
        self.scanner_optimizer = ScannerOptimizationEngine()
        self.resource_optimizer = ResourceOptimizationEngine()
        self.strategy_evolver = AIStrategyEvolutionEngine()
        self.db_pool = None
        
    async def initialize(self, config: Dict):
        """Initialize campaign optimization engine"""
        
        logger.info("Initializing Campaign Optimization Engine...")
        
        # Initialize database
        database_url = config.get("database_url")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        
        # Initialize components
        await self.asset_profiler.initialize(database_url)
        await self.scanner_optimizer.initialize()
        await self.strategy_evolver.initialize(config.get("openai_api_key"))
        
        # Create optimization tables
        await self._create_optimization_tables()
        
        logger.info("Campaign Optimization Engine initialized successfully")
    
    async def _create_optimization_tables(self):
        """Create database tables for optimization"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS campaign_configurations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    campaign_id VARCHAR(255) UNIQUE NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    selected_assets JSONB NOT NULL,
                    asset_priorities JSONB NOT NULL,
                    scanner_assignments JSONB NOT NULL,
                    scanner_parameters JSONB NOT NULL,
                    resource_allocation JSONB NOT NULL,
                    execution_schedule JSONB NOT NULL,
                    predicted_vulnerabilities INTEGER NOT NULL,
                    predicted_duration_hours FLOAT NOT NULL,
                    predicted_cost FLOAT NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    optimization_reasoning TEXT,
                    alternative_strategies JSONB,
                    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    optimization_id VARCHAR(255) NOT NULL,
                    campaign_id VARCHAR(255) NOT NULL,
                    optimization_type VARCHAR(50) NOT NULL,
                    before_metrics JSONB NOT NULL,
                    after_metrics JSONB NOT NULL,
                    improvement_percentage JSONB NOT NULL,
                    strategy_changes JSONB NOT NULL,
                    parameter_adjustments JSONB NOT NULL,
                    resource_reallocation JSONB NOT NULL,
                    ai_reasoning TEXT,
                    confidence_level VARCHAR(20) NOT NULL,
                    success_probability FLOAT NOT NULL,
                    a_b_test_results JSONB,
                    performance_validation JSONB NOT NULL,
                    optimized_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    model_version VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_campaign_configs_campaign 
                ON campaign_configurations(campaign_id);
                
                CREATE INDEX IF NOT EXISTS idx_optimization_results_campaign 
                ON optimization_results(campaign_id);
            """)
    
    async def optimize_campaign(
        self,
        campaign_id: str,
        asset_ids: List[str],
        objectives: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> CampaignConfiguration:
        """Perform comprehensive campaign optimization"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Profile all assets
            logger.info("Profiling assets for optimization", campaign_id=campaign_id, asset_count=len(asset_ids))
            asset_profiles = []
            for asset_id in asset_ids:
                try:
                    profile = await self.asset_profiler.profile_asset(asset_id)
                    asset_profiles.append(profile)
                except Exception as e:
                    logger.warning("Asset profiling failed", asset_id=asset_id, error=str(e))
            
            # Step 2: Evolve strategy based on historical data
            logger.info("Evolving campaign strategy", campaign_id=campaign_id)
            historical_campaigns = await self._get_historical_campaigns()
            context = self._build_campaign_context(asset_profiles, constraints)
            
            optimal_strategy = await self.strategy_evolver.evolve_campaign_strategy(
                historical_campaigns, context, objectives
            )
            
            # Step 3: Optimize scanner selection
            logger.info("Optimizing scanner selection", campaign_id=campaign_id)
            resource_constraints = constraints.get("resources", {})
            scanner_assignments = await self.scanner_optimizer.optimize_scanner_selection(
                asset_profiles, resource_constraints, objectives
            )
            
            # Step 4: Optimize scanner parameters
            logger.info("Optimizing scanner parameters", campaign_id=campaign_id)
            scanner_parameters = {}
            for asset_id, scanner_name in scanner_assignments.items():
                asset_profile = next(a for a in asset_profiles if a.asset_id == asset_id)
                historical_perf = await self._get_scanner_performance(scanner_name, asset_profile)
                
                params = await self.scanner_optimizer.optimize_scanner_parameters(
                    scanner_name, asset_profile, historical_perf
                )
                scanner_parameters[scanner_name] = params
            
            # Step 5: Optimize resource allocation
            logger.info("Optimizing resource allocation", campaign_id=campaign_id)
            campaign_requirements = {
                "asset_count": len(asset_profiles),
                "avg_complexity": np.mean([a.complexity_score for a in asset_profiles]),
                "total_attack_surface": sum(a.attack_surface_size for a in asset_profiles)
            }
            
            available_resources = constraints.get("available_resources", {
                ResourceType.CPU: 16.0,
                ResourceType.MEMORY: 32768.0,
                ResourceType.NETWORK: 1000.0,
                ResourceType.STORAGE: 500.0,
                ResourceType.SCANNER_LICENSES: 20.0
            })
            
            resource_allocation = await self.resource_optimizer.optimize_resource_allocation(
                campaign_requirements, available_resources, objectives
            )
            
            # Step 6: Optimize execution schedule
            logger.info("Optimizing execution schedule", campaign_id=campaign_id)
            execution_schedule = await self.resource_optimizer.optimize_execution_schedule(
                scanner_assignments, asset_profiles, resource_allocation
            )
            
            # Step 7: Generate predictions
            predictions = self._generate_performance_predictions(
                asset_profiles, scanner_assignments, optimal_strategy
            )
            
            # Step 8: Calculate asset priorities
            asset_priorities = {
                a.asset_id: a.discovery_probability * a.business_criticality 
                for a in asset_profiles
            }
            
            # Step 9: Build configuration
            configuration = CampaignConfiguration(
                campaign_id=campaign_id,
                strategy=optimal_strategy,
                selected_assets=[a.asset_id for a in asset_profiles],
                asset_priorities=asset_priorities,
                scanner_assignments=scanner_assignments,
                scanner_parameters=scanner_parameters,
                resource_allocation={rt.value: allocation for rt, allocation in resource_allocation.items()},
                execution_schedule=execution_schedule,
                predicted_vulnerabilities=predictions["vulnerabilities"],
                predicted_duration_hours=predictions["duration_hours"],
                predicted_cost=predictions["cost"],
                confidence_score=predictions["confidence"],
                optimization_reasoning=predictions["reasoning"],
                alternative_strategies=predictions["alternatives"],
                generated_at=start_time
            )
            
            # Store configuration
            await self._store_campaign_configuration(configuration)
            
            # Update metrics
            campaign_optimizations_total.labels(
                optimization_type="comprehensive",
                strategy_type=optimal_strategy.value
            ).inc()
            
            for resource_type, allocation in resource_allocation.items():
                resource_efficiency_score.labels(
                    resource_type=resource_type.value,
                    campaign_type=optimal_strategy.value
                ).set(allocation / available_resources.get(resource_type, 1.0))
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("Campaign optimization completed",
                       campaign_id=campaign_id,
                       strategy=optimal_strategy.value,
                       assets_optimized=len(asset_profiles),
                       predicted_vulnerabilities=predictions["vulnerabilities"],
                       duration=duration)
            
            return configuration
            
        except Exception as e:
            logger.error("Campaign optimization failed", campaign_id=campaign_id, error=str(e))
            raise
    
    async def _get_historical_campaigns(self) -> List[Dict]:
        """Get historical campaign data for learning"""
        
        try:
            async with self.db_pool.acquire() as conn:
                campaigns = await conn.fetch("""
                    SELECT 
                        c.id as campaign_id,
                        c.strategy,
                        c.created_at,
                        c.completed_at,
                        COUNT(v.id) as vulnerabilities_found,
                        SUM(s.duration_seconds) / 3600.0 as duration_hours,
                        SUM(s.cost) as total_cost,
                        AVG(a.complexity_score) as avg_complexity,
                        COUNT(DISTINCT a.id) as asset_count
                    FROM campaigns c
                    LEFT JOIN scans s ON c.id = s.campaign_id
                    LEFT JOIN vulnerabilities v ON s.id = v.scan_id
                    LEFT JOIN assets a ON s.asset_id = a.id
                    WHERE c.completed_at >= NOW() - INTERVAL '90 days'
                    GROUP BY c.id, c.strategy, c.created_at, c.completed_at
                    ORDER BY c.created_at DESC
                    LIMIT 50
                """)
                
                return [dict(campaign) for campaign in campaigns]
                
        except Exception as e:
            logger.warning("Failed to get historical campaigns", error=str(e))
            return []
    
    def _build_campaign_context(self, asset_profiles: List[AssetProfile], constraints: Dict) -> Dict:
        """Build campaign context for strategy evolution"""
        
        asset_types = list(set(a.asset_type for a in asset_profiles))
        
        return {
            "asset_count": len(asset_profiles),
            "asset_types": asset_types,
            "avg_complexity": np.mean([a.complexity_score for a in asset_profiles]),
            "avg_criticality": np.mean([a.business_criticality for a in asset_profiles]),
            "total_attack_surface": sum(a.attack_surface_size for a in asset_profiles),
            "budget": constraints.get("budget", float('inf')),
            "time_limit": constraints.get("time_limit_hours", 24),
            "stealth_required": constraints.get("stealth_mode", False)
        }
    
    async def _get_scanner_performance(self, scanner_name: str, asset_profile: AssetProfile) -> Dict:
        """Get historical performance data for scanner on similar assets"""
        
        try:
            async with self.db_pool.acquire() as conn:
                perf_data = await conn.fetchrow("""
                    SELECT 
                        AVG(s.duration_seconds) as avg_duration,
                        AVG(s.response_time_ms) as avg_response_time,
                        COUNT(v.id) / COUNT(DISTINCT s.id) as avg_vulns_per_scan,
                        COUNT(v.id) FILTER (WHERE v.false_positive = true) / NULLIF(COUNT(v.id), 0) as fp_rate
                    FROM scans s
                    LEFT JOIN vulnerabilities v ON s.id = v.scan_id
                    JOIN assets a ON s.asset_id = a.id
                    WHERE s.scanner_name = $1
                    AND a.asset_type = $2
                    AND s.completed_at >= NOW() - INTERVAL '60 days'
                """, scanner_name, asset_profile.asset_type)
                
                if perf_data:
                    return {
                        "avg_duration": float(perf_data['avg_duration'] or 600),
                        "avg_response_time": float(perf_data['avg_response_time'] or 1000),
                        "avg_vulns_per_scan": float(perf_data['avg_vulns_per_scan'] or 0.5),
                        "false_positive_rate": float(perf_data['fp_rate'] or 0.1)
                    }
                    
        except Exception as e:
            logger.warning("Failed to get scanner performance", error=str(e))
        
        # Return defaults
        return {
            "avg_duration": 600,
            "avg_response_time": 1000,
            "avg_vulns_per_scan": 0.5,
            "false_positive_rate": 0.1
        }
    
    def _generate_performance_predictions(
        self,
        asset_profiles: List[AssetProfile],
        scanner_assignments: Dict[str, str],
        strategy: CampaignStrategy
    ) -> Dict:
        """Generate performance predictions for campaign"""
        
        # Calculate predictions based on asset profiles and scanner effectiveness
        total_vulnerabilities = 0
        total_duration = 0.0
        total_cost = 0.0
        
        for asset in asset_profiles:
            scanner_name = scanner_assignments.get(asset.asset_id, "nuclei")
            
            # Predict vulnerabilities for this asset
            base_discovery_rate = asset.discovery_probability
            
            # Strategy adjustments
            strategy_multipliers = {
                CampaignStrategy.COMPREHENSIVE: 1.2,
                CampaignStrategy.FOCUSED: 1.0,
                CampaignStrategy.RAPID: 0.7,
                CampaignStrategy.DEEP_DIVE: 1.5,
                CampaignStrategy.STEALTH: 0.8,
                CampaignStrategy.BASELINE: 1.0
            }
            
            adjusted_rate = base_discovery_rate * strategy_multipliers.get(strategy, 1.0)
            predicted_vulns = int(adjusted_rate * asset.attack_surface_size / 10.0)
            total_vulnerabilities += predicted_vulns
            
            # Predict duration
            base_duration = 600  # 10 minutes base
            complexity_factor = 1.0 + asset.complexity_score
            duration = base_duration * complexity_factor
            total_duration += duration
            
            # Predict cost
            total_cost += asset.testing_cost_estimate
        
        # Convert duration to hours
        total_duration_hours = total_duration / 3600.0
        
        # Calculate confidence based on data quality
        confidence = min(1.0, len(asset_profiles) / 20.0 + 0.5)  # More assets = higher confidence
        
        # Generate reasoning
        reasoning = f"Predictions based on {len(asset_profiles)} asset profiles using {strategy.value} strategy. " \
                   f"Average asset complexity: {np.mean([a.complexity_score for a in asset_profiles]):.2f}. " \
                   f"Average discovery probability: {np.mean([a.discovery_probability for a in asset_profiles]):.2f}."
        
        # Alternative strategies
        alternatives = [
            {"strategy": "rapid", "tradeoff": "Faster execution, 30% fewer vulnerabilities"},
            {"strategy": "deep_dive", "tradeoff": "50% more vulnerabilities, 2x longer duration"},
            {"strategy": "stealth", "tradeoff": "Lower detection risk, 20% fewer vulnerabilities"}
        ]
        
        return {
            "vulnerabilities": max(1, total_vulnerabilities),
            "duration_hours": max(0.5, total_duration_hours),
            "cost": max(10.0, total_cost),
            "confidence": confidence,
            "reasoning": reasoning,
            "alternatives": alternatives
        }
    
    async def _store_campaign_configuration(self, config: CampaignConfiguration):
        """Store campaign configuration in database"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO campaign_configurations
                    (campaign_id, strategy, selected_assets, asset_priorities,
                     scanner_assignments, scanner_parameters, resource_allocation,
                     execution_schedule, predicted_vulnerabilities, predicted_duration_hours,
                     predicted_cost, confidence_score, optimization_reasoning,
                     alternative_strategies, generated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                config.campaign_id, config.strategy.value,
                json.dumps(config.selected_assets), json.dumps(config.asset_priorities),
                json.dumps(config.scanner_assignments), json.dumps(config.scanner_parameters),
                json.dumps(config.resource_allocation), json.dumps(config.execution_schedule),
                config.predicted_vulnerabilities, config.predicted_duration_hours,
                config.predicted_cost, config.confidence_score,
                config.optimization_reasoning, json.dumps(config.alternative_strategies),
                config.generated_at)
                
        except Exception as e:
            logger.error("Failed to store campaign configuration", error=str(e))
    
    async def get_optimization_statistics(self) -> Dict:
        """Get comprehensive optimization statistics"""
        
        try:
            async with self.db_pool.acquire() as conn:
                # Campaign configuration statistics
                config_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_campaigns_optimized,
                        AVG(predicted_vulnerabilities) as avg_predicted_vulns,
                        AVG(predicted_duration_hours) as avg_predicted_duration,
                        AVG(predicted_cost) as avg_predicted_cost,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(*) FILTER (WHERE strategy = 'comprehensive') as comprehensive_count,
                        COUNT(*) FILTER (WHERE strategy = 'focused') as focused_count,
                        COUNT(*) FILTER (WHERE strategy = 'rapid') as rapid_count
                    FROM campaign_configurations
                    WHERE generated_at >= NOW() - INTERVAL '30 days'
                """)
                
                # Optimization results statistics
                opt_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_optimizations,
                        AVG(success_probability) as avg_success_probability,
                        COUNT(*) FILTER (WHERE confidence_level = 'high') as high_confidence_count
                    FROM optimization_results
                    WHERE optimized_at >= NOW() - INTERVAL '30 days'
                """)
                
                return {
                    "campaign_optimization": {
                        "total_campaigns_optimized": config_stats['total_campaigns_optimized'] if config_stats else 0,
                        "avg_predicted_vulnerabilities": float(config_stats['avg_predicted_vulns'] or 0) if config_stats else 0,
                        "avg_predicted_duration_hours": float(config_stats['avg_predicted_duration'] or 0) if config_stats else 0,
                        "avg_predicted_cost": float(config_stats['avg_predicted_cost'] or 0) if config_stats else 0,
                        "avg_confidence": float(config_stats['avg_confidence'] or 0) if config_stats else 0
                    },
                    "strategy_distribution": {
                        "comprehensive": config_stats['comprehensive_count'] if config_stats else 0,
                        "focused": config_stats['focused_count'] if config_stats else 0,
                        "rapid": config_stats['rapid_count'] if config_stats else 0
                    },
                    "optimization_performance": {
                        "total_optimizations": opt_stats['total_optimizations'] if opt_stats else 0,
                        "avg_success_probability": float(opt_stats['avg_success_probability'] or 0) if opt_stats else 0,
                        "high_confidence_optimizations": opt_stats['high_confidence_count'] if opt_stats else 0
                    },
                    "ai_capabilities": [
                        "asset_profiling",
                        "scanner_optimization",
                        "resource_allocation",
                        "strategy_evolution",
                        "performance_prediction"
                    ],
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get optimization statistics", error=str(e))
            return {"error": str(e)}

async def main():
    """Main campaign optimization service"""
    
    # Start Prometheus metrics server
    start_http_server(8014)
    
    # Initialize optimization engine
    config = {
        "database_url": os.getenv("DATABASE_URL", 
                                 "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    engine = CampaignOptimizationEngine()
    await engine.initialize(config)
    
    logger.info("🚀 Xorb AI Campaign Optimization Engine started",
               service_version="6.5.0",
               features=["asset_profiling", "scanner_optimization", "resource_allocation", 
                        "strategy_evolution", "ai_predictions"])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down campaign optimization engine")

if __name__ == "__main__":
    asyncio.run(main())