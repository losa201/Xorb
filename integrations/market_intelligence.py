#!/usr/bin/env python3

import asyncio
import json
import logging
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.warning("Analytics libraries not available. Install with: pip install pandas numpy scikit-learn")

try:
    import aiohttp
    import asyncio
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    logging.warning("HTTP libraries not available. Install with: pip install aiohttp")

from .hackerone_client import HackerOneClient
from .bounty_intelligence import BountyIntelligenceEngine


@dataclass
class MarketTrend:
    """Represents a market trend in bug bounty programs."""
    trend_id: str
    trend_type: str  # 'program_growth', 'payout_increase', 'scope_expansion'
    description: str
    confidence: float
    timeframe: str  # 'daily', 'weekly', 'monthly'
    impact_score: float
    affected_programs: List[str]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class CompetitiveAnalysis:
    """Competitive analysis for a bug bounty program."""
    program_handle: str
    researcher_count: int
    top_researchers: List[Dict[str, Any]]
    submission_frequency: float
    avg_response_time: float
    competition_score: float  # 0-1 scale
    market_saturation: float  # 0-1 scale
    opportunity_score: float  # 0-1 scale
    recommendations: List[str]
    analysis_date: datetime


@dataclass
class ProgramValuation:
    """Comprehensive program valuation metrics."""
    program_handle: str
    estimated_monthly_revenue: float
    roi_score: float
    time_investment_hours: float
    success_probability: float
    market_position: str  # 'premium', 'growth', 'mature', 'declining'
    valuation_confidence: float
    key_factors: List[str]
    risk_factors: List[str]
    updated_at: datetime


@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence report."""
    report_id: str
    market_overview: Dict[str, Any]
    trending_programs: List[Dict[str, Any]]
    emerging_opportunities: List[Dict[str, Any]]
    market_predictions: List[Dict[str, Any]]
    competitive_landscape: Dict[str, Any]
    generated_at: datetime


class ProgramAnalytics:
    """Advanced analytics for bug bounty programs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        
    async def analyze_program_performance(self, program_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze program performance metrics."""
        try:
            performance_metrics = {
                "response_time_score": self._calculate_response_time_score(program_data),
                "payout_competitiveness": self._calculate_payout_competitiveness(program_data),
                "scope_attractiveness": self._calculate_scope_attractiveness(program_data),
                "program_activity": self._calculate_program_activity(program_data),
                "researcher_satisfaction": self._calculate_researcher_satisfaction(program_data)
            }
            
            # Calculate overall performance score
            weights = {
                "response_time_score": 0.25,
                "payout_competitiveness": 0.30,
                "scope_attractiveness": 0.20,
                "program_activity": 0.15,
                "researcher_satisfaction": 0.10
            }
            
            overall_score = sum(
                performance_metrics[metric] * weight 
                for metric, weight in weights.items()
            )
            
            performance_metrics["overall_score"] = overall_score
            performance_metrics["performance_tier"] = self._determine_performance_tier(overall_score)
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Program performance analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_response_time_score(self, program_data: Dict[str, Any]) -> float:
        """Calculate response time performance score."""
        avg_response_time = program_data.get("average_response_time_hours", 168)  # Default 1 week
        
        # Score based on response time (faster is better)
        if avg_response_time <= 24:
            return 1.0  # Excellent
        elif avg_response_time <= 72:
            return 0.8  # Good
        elif avg_response_time <= 168:
            return 0.6  # Average
        elif avg_response_time <= 336:
            return 0.4  # Below average
        else:
            return 0.2  # Poor
    
    def _calculate_payout_competitiveness(self, program_data: Dict[str, Any]) -> float:
        """Calculate payout competitiveness score."""
        bounty_stats = program_data.get("bounty_statistics", {})
        
        avg_bounty = bounty_stats.get("average_bounty", 0)
        max_bounty = bounty_stats.get("maximum_bounty", 0)
        
        # Industry benchmarks (approximate)
        industry_avg = 1500
        industry_max = 25000
        
        avg_score = min(1.0, avg_bounty / industry_avg)
        max_score = min(1.0, max_bounty / industry_max)
        
        return (avg_score * 0.7) + (max_score * 0.3)
    
    def _calculate_scope_attractiveness(self, program_data: Dict[str, Any]) -> float:
        """Calculate scope attractiveness score."""
        scope = program_data.get("scope", {})
        
        score_factors = []
        
        # Number of in-scope assets
        in_scope_count = len(scope.get("in_scope", []))
        score_factors.append(min(1.0, in_scope_count / 10))
        
        # Asset types diversity
        asset_types = set()
        for asset in scope.get("in_scope", []):
            asset_types.add(asset.get("asset_type", "other"))
        
        score_factors.append(min(1.0, len(asset_types) / 5))
        
        # Technologies mentioned
        technologies = scope.get("technologies", [])
        score_factors.append(min(1.0, len(technologies) / 10))
        
        return sum(score_factors) / len(score_factors) if score_factors else 0.0
    
    def _calculate_program_activity(self, program_data: Dict[str, Any]) -> float:
        """Calculate program activity score."""
        activity_stats = program_data.get("activity_statistics", {})
        
        reports_last_month = activity_stats.get("reports_last_30_days", 0)
        resolved_reports = activity_stats.get("resolved_reports_last_30_days", 0)
        
        # Activity score based on report volume and resolution rate
        volume_score = min(1.0, reports_last_month / 20)  # 20 reports/month = max score
        
        resolution_rate = resolved_reports / max(1, reports_last_month)
        resolution_score = min(1.0, resolution_rate)
        
        return (volume_score * 0.6) + (resolution_score * 0.4)
    
    def _calculate_researcher_satisfaction(self, program_data: Dict[str, Any]) -> float:
        """Calculate researcher satisfaction score."""
        satisfaction_data = program_data.get("researcher_feedback", {})
        
        avg_rating = satisfaction_data.get("average_rating", 3.0)  # Out of 5
        response_rate = satisfaction_data.get("feedback_response_rate", 0.5)
        
        rating_score = avg_rating / 5.0
        engagement_score = response_rate
        
        return (rating_score * 0.8) + (engagement_score * 0.2)
    
    def _determine_performance_tier(self, score: float) -> str:
        """Determine performance tier based on overall score."""
        if score >= 0.8:
            return "premium"
        elif score >= 0.6:
            return "high_performing"
        elif score >= 0.4:
            return "average"
        elif score >= 0.2:
            return "below_average"
        else:
            return "underperforming"


class MarketTrendAnalyzer:
    """Analyzes trends in the bug bounty market."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trend_data = []
    
    async def detect_market_trends(self, programs_data: List[Dict[str, Any]], 
                                 historical_data: Optional[List[Dict[str, Any]]] = None) -> List[MarketTrend]:
        """Detect trends in the bug bounty market."""
        trends = []
        
        try:
            # Analyze payout trends
            payout_trend = await self._analyze_payout_trends(programs_data, historical_data)
            if payout_trend:
                trends.append(payout_trend)
            
            # Analyze program growth trends
            growth_trend = await self._analyze_program_growth_trends(programs_data, historical_data)
            if growth_trend:
                trends.append(growth_trend)
            
            # Analyze technology trends
            tech_trends = await self._analyze_technology_trends(programs_data)
            trends.extend(tech_trends)
            
            # Analyze vulnerability type trends
            vuln_trends = await self._analyze_vulnerability_trends(programs_data)
            trends.extend(vuln_trends)
            
        except Exception as e:
            self.logger.error(f"Market trend analysis failed: {e}")
        
        return trends
    
    async def _analyze_payout_trends(self, programs_data: List[Dict[str, Any]], 
                                   historical_data: Optional[List[Dict[str, Any]]]) -> Optional[MarketTrend]:
        """Analyze payout trends across programs."""
        try:
            current_payouts = []
            for program in programs_data:
                bounty_stats = program.get("bounty_statistics", {})
                avg_bounty = bounty_stats.get("average_bounty", 0)
                if avg_bounty > 0:
                    current_payouts.append(avg_bounty)
            
            if not current_payouts:
                return None
            
            current_avg = statistics.mean(current_payouts)
            
            # Compare with historical data if available
            trend_direction = "stable"
            confidence = 0.5
            
            if historical_data:
                historical_payouts = []
                for program in historical_data:
                    bounty_stats = program.get("bounty_statistics", {})
                    avg_bounty = bounty_stats.get("average_bounty", 0)
                    if avg_bounty > 0:
                        historical_payouts.append(avg_bounty)
                
                if historical_payouts:
                    historical_avg = statistics.mean(historical_payouts)
                    change_percent = ((current_avg - historical_avg) / historical_avg) * 100
                    
                    if change_percent > 10:
                        trend_direction = "increasing"
                        confidence = min(0.9, 0.5 + abs(change_percent) / 100)
                    elif change_percent < -10:
                        trend_direction = "decreasing"
                        confidence = min(0.9, 0.5 + abs(change_percent) / 100)
            
            return MarketTrend(
                trend_id=f"payout_trend_{datetime.utcnow().strftime('%Y%m%d')}",
                trend_type="payout_trend",
                description=f"Average bounty payouts are {trend_direction} (current avg: ${current_avg:.0f})",
                confidence=confidence,
                timeframe="monthly",
                impact_score=0.8 if trend_direction != "stable" else 0.3,
                affected_programs=[p.get("handle", "unknown") for p in programs_data[:10]],
                metadata={
                    "current_average": current_avg,
                    "trend_direction": trend_direction,
                    "sample_size": len(current_payouts)
                },
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Payout trend analysis failed: {e}")
            return None
    
    async def _analyze_program_growth_trends(self, programs_data: List[Dict[str, Any]], 
                                           historical_data: Optional[List[Dict[str, Any]]]) -> Optional[MarketTrend]:
        """Analyze program growth trends."""
        try:
            # Analyze by program launch dates
            new_programs_30d = 0
            new_programs_90d = 0
            
            cutoff_30d = datetime.utcnow() - timedelta(days=30)
            cutoff_90d = datetime.utcnow() - timedelta(days=90)
            
            for program in programs_data:
                launch_date_str = program.get("launched_at")
                if launch_date_str:
                    try:
                        launch_date = datetime.fromisoformat(launch_date_str.replace('Z', '+00:00'))
                        if launch_date >= cutoff_30d:
                            new_programs_30d += 1
                        if launch_date >= cutoff_90d:
                            new_programs_90d += 1
                    except:
                        continue
            
            growth_rate_30d = new_programs_30d
            growth_rate_90d = new_programs_90d / 3  # Average per 30 days
            
            trend_direction = "stable"
            if growth_rate_30d > growth_rate_90d * 1.2:
                trend_direction = "accelerating"
            elif growth_rate_30d < growth_rate_90d * 0.8:
                trend_direction = "slowing"
            
            return MarketTrend(
                trend_id=f"growth_trend_{datetime.utcnow().strftime('%Y%m%d')}",
                trend_type="program_growth",
                description=f"Program growth is {trend_direction} ({new_programs_30d} new programs in 30 days)",
                confidence=0.7,
                timeframe="monthly",
                impact_score=0.6,
                affected_programs=[],
                metadata={
                    "new_programs_30d": new_programs_30d,
                    "new_programs_90d": new_programs_90d,
                    "growth_direction": trend_direction
                },
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Program growth trend analysis failed: {e}")
            return None
    
    async def _analyze_technology_trends(self, programs_data: List[Dict[str, Any]]) -> List[MarketTrend]:
        """Analyze trending technologies in bug bounty programs."""
        tech_counts = {}
        
        for program in programs_data:
            scope = program.get("scope", {})
            technologies = scope.get("technologies", [])
            
            for tech in technologies:
                tech_lower = tech.lower()
                tech_counts[tech_lower] = tech_counts.get(tech_lower, 0) + 1
        
        # Sort by frequency
        sorted_techs = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)
        
        trends = []
        total_programs = len(programs_data)
        
        for tech, count in sorted_techs[:5]:  # Top 5 technologies
            prevalence = count / total_programs
            
            if prevalence > 0.1:  # Only if present in >10% of programs
                trends.append(MarketTrend(
                    trend_id=f"tech_trend_{tech}_{datetime.utcnow().strftime('%Y%m%d')}",
                    trend_type="technology_trend",
                    description=f"{tech.title()} is trending ({prevalence:.1%} of programs)",
                    confidence=min(0.9, prevalence * 2),
                    timeframe="quarterly",
                    impact_score=prevalence,
                    affected_programs=[p.get("handle", "unknown") for p in programs_data 
                                     if tech in [t.lower() for t in p.get("scope", {}).get("technologies", [])]][:10],
                    metadata={
                        "technology": tech,
                        "program_count": count,
                        "prevalence": prevalence
                    },
                    created_at=datetime.utcnow()
                ))
        
        return trends
    
    async def _analyze_vulnerability_trends(self, programs_data: List[Dict[str, Any]]) -> List[MarketTrend]:
        """Analyze trending vulnerability types."""
        # This would typically analyze recent submissions/findings
        # For now, provide common vulnerability trends
        
        common_vulns = [
            {"type": "xss", "trend": "stable", "impact": 0.4},
            {"type": "sqli", "trend": "decreasing", "impact": 0.6},
            {"type": "csrf", "trend": "stable", "impact": 0.3},
            {"type": "idor", "trend": "increasing", "impact": 0.5},
            {"type": "rce", "trend": "stable", "impact": 0.9},
            {"type": "ssrf", "trend": "increasing", "impact": 0.7}
        ]
        
        trends = []
        for vuln in common_vulns:
            if vuln["trend"] != "stable":
                trends.append(MarketTrend(
                    trend_id=f"vuln_trend_{vuln['type']}_{datetime.utcnow().strftime('%Y%m%d')}",
                    trend_type="vulnerability_trend",
                    description=f"{vuln['type'].upper()} vulnerabilities are {vuln['trend']}",
                    confidence=0.6,
                    timeframe="quarterly",
                    impact_score=vuln["impact"],
                    affected_programs=[],
                    metadata={
                        "vulnerability_type": vuln["type"],
                        "trend_direction": vuln["trend"]
                    },
                    created_at=datetime.utcnow()
                ))
        
        return trends


class MarketIntelligenceEngine:
    """Main market intelligence engine for XORB."""
    
    def __init__(self, hackerone_client: HackerOneClient):
        self.hackerone_client = hackerone_client
        self.bounty_intelligence = BountyIntelligenceEngine()
        self.program_analytics = ProgramAnalytics()
        self.trend_analyzer = MarketTrendAnalyzer()
        
        self.logger = logging.getLogger(__name__)
        
        # Cache for market data
        self.market_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize the market intelligence engine."""
        await self.bounty_intelligence.initialize()
        self.logger.info("Market Intelligence Engine initialized")
    
    async def generate_market_intelligence_report(self) -> MarketIntelligence:
        """Generate comprehensive market intelligence report."""
        report_id = f"market_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Gather market data
            programs_data = await self._gather_market_data()
            
            # Market overview
            market_overview = await self._generate_market_overview(programs_data)
            
            # Trending programs
            trending_programs = await self._identify_trending_programs(programs_data)
            
            # Emerging opportunities
            opportunities = await self._identify_emerging_opportunities(programs_data)
            
            # Market predictions
            predictions = await self._generate_market_predictions(programs_data)
            
            # Competitive landscape
            competitive_landscape = await self._analyze_competitive_landscape(programs_data)
            
            return MarketIntelligence(
                report_id=report_id,
                market_overview=market_overview,
                trending_programs=trending_programs,
                emerging_opportunities=opportunities,
                market_predictions=predictions,
                competitive_landscape=competitive_landscape,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Market intelligence report generation failed: {e}")
            raise
    
    async def _gather_market_data(self) -> List[Dict[str, Any]]:
        """Gather market data from various sources."""
        # Check cache first
        cache_key = "market_data"
        cached_data = self.market_cache.get(cache_key)
        
        if cached_data and (datetime.utcnow() - cached_data["timestamp"]).seconds < self.cache_ttl:
            return cached_data["data"]
        
        try:
            # Fetch from HackerOne
            h1_programs = await self.hackerone_client.get_programs()
            
            # Add analytics for each program
            enhanced_programs = []
            for program in h1_programs[:50]:  # Limit for performance
                try:
                    # Get additional program details
                    program_details = await self.hackerone_client.get_program_details(program.get("handle", ""))
                    
                    # Add performance metrics
                    performance_metrics = await self.program_analytics.analyze_program_performance(program_details)
                    
                    enhanced_program = {
                        **program_details,
                        "performance_metrics": performance_metrics
                    }
                    
                    enhanced_programs.append(enhanced_program)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to enhance program {program.get('handle', 'unknown')}: {e}")
                    enhanced_programs.append(program)
            
            # Cache the data
            self.market_cache[cache_key] = {
                "data": enhanced_programs,
                "timestamp": datetime.utcnow()
            }
            
            return enhanced_programs
            
        except Exception as e:
            self.logger.error(f"Market data gathering failed: {e}")
            return []
    
    async def _generate_market_overview(self, programs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate market overview statistics."""
        if not programs_data:
            return {"error": "No market data available"}
        
        # Calculate market metrics
        total_programs = len(programs_data)
        active_programs = len([p for p in programs_data if p.get("state") == "public_mode"])
        
        # Payout statistics
        all_payouts = []
        for program in programs_data:
            bounty_stats = program.get("bounty_statistics", {})
            avg_bounty = bounty_stats.get("average_bounty", 0)
            if avg_bounty > 0:
                all_payouts.append(avg_bounty)
        
        payout_stats = {}
        if all_payouts:
            payout_stats = {
                "median_payout": statistics.median(all_payouts),
                "mean_payout": statistics.mean(all_payouts),
                "max_payout": max(all_payouts),
                "min_payout": min(all_payouts),
                "total_programs_with_payouts": len(all_payouts)
            }
        
        # Performance distribution
        performance_distribution = {
            "premium": 0,
            "high_performing": 0,
            "average": 0,
            "below_average": 0,
            "underperforming": 0
        }
        
        for program in programs_data:
            performance_metrics = program.get("performance_metrics", {})
            tier = performance_metrics.get("performance_tier", "unknown")
            if tier in performance_distribution:
                performance_distribution[tier] += 1
        
        return {
            "total_programs": total_programs,
            "active_programs": active_programs,
            "payout_statistics": payout_stats,
            "performance_distribution": performance_distribution,
            "market_health_score": self._calculate_market_health_score(programs_data),
            "data_freshness": datetime.utcnow().isoformat()
        }
    
    def _calculate_market_health_score(self, programs_data: List[Dict[str, Any]]) -> float:
        """Calculate overall market health score."""
        if not programs_data:
            return 0.0
        
        health_factors = []
        
        # Active program ratio
        active_ratio = len([p for p in programs_data if p.get("state") == "public_mode"]) / len(programs_data)
        health_factors.append(active_ratio)
        
        # Average performance score
        performance_scores = []
        for program in programs_data:
            performance_metrics = program.get("performance_metrics", {})
            overall_score = performance_metrics.get("overall_score", 0.5)
            performance_scores.append(overall_score)
        
        if performance_scores:
            avg_performance = sum(performance_scores) / len(performance_scores)
            health_factors.append(avg_performance)
        
        # Payout competitiveness
        competitive_programs = 0
        for program in programs_data:
            performance_metrics = program.get("performance_metrics", {})
            payout_score = performance_metrics.get("payout_competitiveness", 0)
            if payout_score > 0.6:
                competitive_programs += 1
        
        competitiveness_ratio = competitive_programs / len(programs_data)
        health_factors.append(competitiveness_ratio)
        
        return sum(health_factors) / len(health_factors) if health_factors else 0.0
    
    async def _identify_trending_programs(self, programs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify trending bug bounty programs."""
        trending = []
        
        # Sort by performance score and recent activity
        scored_programs = []
        for program in programs_data:
            performance_metrics = program.get("performance_metrics", {})
            overall_score = performance_metrics.get("overall_score", 0)
            
            activity_stats = program.get("activity_statistics", {})
            recent_activity = activity_stats.get("reports_last_30_days", 0)
            
            # Combine performance and activity for trending score
            trending_score = (overall_score * 0.7) + (min(1.0, recent_activity / 20) * 0.3)
            
            scored_programs.append({
                "program": program,
                "trending_score": trending_score
            })
        
        # Sort and take top programs
        scored_programs.sort(key=lambda x: x["trending_score"], reverse=True)
        
        for item in scored_programs[:10]:
            program = item["program"]
            trending.append({
                "handle": program.get("handle", "unknown"),
                "name": program.get("name", "Unknown Program"),
                "trending_score": item["trending_score"],
                "performance_tier": program.get("performance_metrics", {}).get("performance_tier", "unknown"),
                "recent_activity": program.get("activity_statistics", {}).get("reports_last_30_days", 0),
                "reason": self._generate_trending_reason(program, item["trending_score"])
            })
        
        return trending
    
    def _generate_trending_reason(self, program: Dict[str, Any], score: float) -> str:
        """Generate reason why program is trending."""
        performance_metrics = program.get("performance_metrics", {})
        
        reasons = []
        
        if performance_metrics.get("payout_competitiveness", 0) > 0.7:
            reasons.append("competitive payouts")
        
        if performance_metrics.get("response_time_score", 0) > 0.8:
            reasons.append("fast response times")
        
        activity = program.get("activity_statistics", {}).get("reports_last_30_days", 0)
        if activity > 15:
            reasons.append("high activity")
        
        if not reasons:
            reasons.append("strong overall performance")
        
        return f"Trending due to {', '.join(reasons)}"
    
    async def _identify_emerging_opportunities(self, programs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify emerging opportunities in the market."""
        opportunities = []
        
        # Look for programs with good potential but low competition
        for program in programs_data:
            performance_metrics = program.get("performance_metrics", {})
            overall_score = performance_metrics.get("overall_score", 0)
            
            # Good performance but potentially underutilized
            if 0.6 <= overall_score <= 0.8:
                activity_stats = program.get("activity_statistics", {})
                reports_last_month = activity_stats.get("reports_last_30_days", 0)
                
                # Low competition indicator
                if reports_last_month < 10:
                    opportunity_score = overall_score / max(1, reports_last_month / 5)
                    
                    opportunities.append({
                        "handle": program.get("handle", "unknown"),
                        "name": program.get("name", "Unknown Program"),
                        "opportunity_score": min(1.0, opportunity_score),
                        "performance_score": overall_score,
                        "competition_level": "low" if reports_last_month < 5 else "medium",
                        "key_advantages": self._identify_program_advantages(program),
                        "estimated_roi": self._estimate_program_roi(program)
                    })
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        return opportunities[:15]  # Top 15 opportunities
    
    def _identify_program_advantages(self, program: Dict[str, Any]) -> List[str]:
        """Identify key advantages of a program."""
        advantages = []
        performance_metrics = program.get("performance_metrics", {})
        
        if performance_metrics.get("payout_competitiveness", 0) > 0.6:
            advantages.append("competitive payouts")
        
        if performance_metrics.get("response_time_score", 0) > 0.7:
            advantages.append("responsive team")
        
        if performance_metrics.get("scope_attractiveness", 0) > 0.6:
            advantages.append("broad scope")
        
        scope = program.get("scope", {})
        if len(scope.get("technologies", [])) > 5:
            advantages.append("diverse technology stack")
        
        return advantages
    
    def _estimate_program_roi(self, program: Dict[str, Any]) -> float:
        """Estimate ROI for a program."""
        bounty_stats = program.get("bounty_statistics", {})
        avg_bounty = bounty_stats.get("average_bounty", 0)
        
        performance_metrics = program.get("performance_metrics", {})
        overall_score = performance_metrics.get("overall_score", 0)
        
        activity_stats = program.get("activity_statistics", {})
        reports_last_month = activity_stats.get("reports_last_30_days", 1)
        
        # Simple ROI estimation
        success_probability = overall_score
        expected_time_hours = max(10, 40 - (reports_last_month * 2))  # Less competition = less time needed
        expected_revenue = avg_bounty * success_probability
        
        roi = expected_revenue / expected_time_hours if expected_time_hours > 0 else 0
        
        return min(100.0, roi)  # Cap at $100/hour
    
    async def _generate_market_predictions(self, programs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate market predictions."""
        predictions = []
        
        # Analyze trends
        trends = await self.trend_analyzer.detect_market_trends(programs_data)
        
        for trend in trends:
            if trend.confidence > 0.6:
                prediction = {
                    "prediction_type": trend.trend_type,
                    "description": trend.description,
                    "confidence": trend.confidence,
                    "timeframe": trend.timeframe,
                    "impact_level": "high" if trend.impact_score > 0.7 else "medium" if trend.impact_score > 0.4 else "low",
                    "implications": self._generate_prediction_implications(trend)
                }
                predictions.append(prediction)
        
        # Add general market predictions
        market_health = self._calculate_market_health_score(programs_data)
        if market_health > 0.7:
            predictions.append({
                "prediction_type": "market_growth",
                "description": "Market shows strong health indicators, expect continued growth",
                "confidence": 0.7,
                "timeframe": "quarterly",
                "impact_level": "medium",
                "implications": ["More programs likely to launch", "Increased competition for researchers", "Higher average payouts"]
            })
        
        return predictions
    
    def _generate_prediction_implications(self, trend: MarketTrend) -> List[str]:
        """Generate implications from a market trend."""
        implications = []
        
        if trend.trend_type == "payout_trend":
            if "increasing" in trend.description:
                implications.extend([
                    "Higher earnings potential for researchers",
                    "Increased competition among programs",
                    "Need for more sophisticated research techniques"
                ])
            elif "decreasing" in trend.description:
                implications.extend([
                    "Focus on efficiency and automation",
                    "Target programs with stable payouts",
                    "Consider diverse income streams"
                ])
        
        elif trend.trend_type == "technology_trend":
            implications.extend([
                f"Specialize in {trend.metadata.get('technology', 'trending')} security",
                "Develop specific tools and techniques",
                "Build expertise in emerging technologies"
            ])
        
        return implications
    
    async def _analyze_competitive_landscape(self, programs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the competitive landscape."""
        landscape = {
            "market_segments": {},
            "competition_levels": {},
            "market_concentration": {},
            "entry_barriers": {}
        }
        
        # Segment by performance tiers
        segments = {
            "premium": [],
            "high_performing": [],
            "average": [],
            "below_average": [],
            "underperforming": []
        }
        
        for program in programs_data:
            performance_metrics = program.get("performance_metrics", {})
            tier = performance_metrics.get("performance_tier", "unknown")
            
            if tier in segments:
                segments[tier].append(program)
        
        landscape["market_segments"] = {
            tier: {
                "count": len(programs),
                "avg_payout": self._calculate_segment_avg_payout(programs),
                "competition_level": self._calculate_segment_competition(programs)
            }
            for tier, programs in segments.items()
        }
        
        return landscape
    
    def _calculate_segment_avg_payout(self, programs: List[Dict[str, Any]]) -> float:
        """Calculate average payout for a market segment."""
        payouts = []
        for program in programs:
            bounty_stats = program.get("bounty_statistics", {})
            avg_bounty = bounty_stats.get("average_bounty", 0)
            if avg_bounty > 0:
                payouts.append(avg_bounty)
        
        return statistics.mean(payouts) if payouts else 0.0
    
    def _calculate_segment_competition(self, programs: List[Dict[str, Any]]) -> str:
        """Calculate competition level for a market segment."""
        total_activity = 0
        for program in programs:
            activity_stats = program.get("activity_statistics", {})
            reports = activity_stats.get("reports_last_30_days", 0)
            total_activity += reports
        
        avg_activity = total_activity / len(programs) if programs else 0
        
        if avg_activity > 20:
            return "high"
        elif avg_activity > 10:
            return "medium"
        else:
            return "low"
    
    async def get_program_competitive_analysis(self, program_handle: str) -> CompetitiveAnalysis:
        """Get detailed competitive analysis for a specific program."""
        try:
            # Get program data
            program_data = await self.hackerone_client.get_program_details(program_handle)
            
            # Analyze competition metrics
            activity_stats = program_data.get("activity_statistics", {})
            researcher_count = activity_stats.get("unique_researchers_last_90_days", 0)
            submission_frequency = activity_stats.get("reports_last_30_days", 0) / 30.0
            avg_response_time = activity_stats.get("average_response_time_hours", 168)
            
            # Calculate competition scores
            competition_score = min(1.0, researcher_count / 100)
            market_saturation = min(1.0, submission_frequency / 2.0)  # 2 reports/day = saturated
            opportunity_score = 1.0 - ((competition_score + market_saturation) / 2.0)
            
            # Generate recommendations
            recommendations = self._generate_competitive_recommendations(
                competition_score, market_saturation, opportunity_score
            )
            
            return CompetitiveAnalysis(
                program_handle=program_handle,
                researcher_count=researcher_count,
                top_researchers=[],  # Would fetch from API if available
                submission_frequency=submission_frequency,
                avg_response_time=avg_response_time,
                competition_score=competition_score,
                market_saturation=market_saturation,
                opportunity_score=opportunity_score,
                recommendations=recommendations,
                analysis_date=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Competitive analysis failed for {program_handle}: {e}")
            raise
    
    def _generate_competitive_recommendations(self, competition_score: float, 
                                           market_saturation: float, 
                                           opportunity_score: float) -> List[str]:
        """Generate competitive recommendations."""
        recommendations = []
        
        if competition_score > 0.7:
            recommendations.append("High competition - focus on unique vulnerabilities and advanced techniques")
        elif competition_score < 0.3:
            recommendations.append("Low competition - good opportunity for systematic testing")
        
        if market_saturation > 0.6:
            recommendations.append("Market is saturated - consider targeting specific components or advanced attack vectors")
        
        if opportunity_score > 0.6:
            recommendations.append("Good opportunity - favorable risk/reward ratio")
        elif opportunity_score < 0.3:
            recommendations.append("Limited opportunity - consider alternative programs")
        
        return recommendations