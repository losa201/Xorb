#!/usr/bin/env python3

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .hackerone_client import HackerOneClient, VulnerabilitySubmission
from ..reports.report_generator import Finding, CVSSMetrics


class BountyProgramTier(str, Enum):
    PREMIUM = "premium"    # Top-tier programs (Google, Facebook, etc.)
    ESTABLISHED = "established"  # Well-established programs
    GROWING = "growing"    # Newer but active programs
    BASIC = "basic"        # Basic programs with lower payouts


@dataclass
class ProgramMetrics:
    program_handle: str
    tier: BountyProgramTier
    avg_bounty_amount: float
    bounty_frequency: float  # Bounties per month
    response_time_hours: float
    resolution_time_hours: float
    scope_size: int
    last_activity_days: int
    reputation_score: float
    accepts_duplicates: bool
    minimum_severity: str
    maximum_bounty: float
    total_paid: float
    researcher_count: int
    submission_volume: int


@dataclass
class ROIAnalysis:
    program_handle: str
    expected_bounty: float
    effort_hours: float
    roi_score: float
    confidence: float
    risk_factors: List[str]
    success_probability: float
    time_to_payout: float
    competitive_level: float


class BountyIntelligenceEngine:
    """Advanced bounty program analysis and ROI optimization"""
    
    def __init__(self, hackerone_client: HackerOneClient):
        self.h1_client = hackerone_client
        self.logger = logging.getLogger(__name__)
        
        # ML Models
        self.bounty_predictor = None
        self.effort_predictor = None
        self.success_predictor = None
        
        # Data storage
        self.program_cache: Dict[str, ProgramMetrics] = {}
        self.historical_data: List[Dict] = []
        self.roi_cache: Dict[str, ROIAnalysis] = {}
        
        # Configuration
        self.min_roi_threshold = 50.0  # Minimum $/hour ROI
        self.max_effort_hours = 40.0   # Maximum effort to invest
        self.confidence_threshold = 0.7
        
        if ML_AVAILABLE:
            self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for predictions"""
        # Bounty amount prediction
        self.bounty_predictor = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        # Effort estimation
        self.effort_predictor = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )
        
        # Success probability
        self.success_predictor = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )

    async def analyze_program_value(self, program_handle: str, force_refresh: bool = False) -> ROIAnalysis:
        """Comprehensive ROI analysis of a bounty program"""
        
        # Check cache first
        if not force_refresh and program_handle in self.roi_cache:
            cached_analysis = self.roi_cache[program_handle]
            # Return cached if less than 24 hours old
            if (datetime.utcnow() - datetime.fromisoformat(cached_analysis.time_to_payout)).hours < 24:
                return cached_analysis
        
        try:
            # Get program metrics
            metrics = await self._get_program_metrics(program_handle)
            
            # Predict expected bounty
            expected_bounty = await self._predict_bounty_amount(metrics)
            
            # Estimate effort required
            effort_hours = await self._estimate_effort_hours(metrics)
            
            # Calculate success probability
            success_prob = await self._predict_success_probability(metrics)
            
            # Calculate ROI
            roi_score = (expected_bounty * success_prob) / max(effort_hours, 1.0)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(metrics)
            
            # Calculate competitive level
            competitive_level = self._calculate_competitive_level(metrics)
            
            # Generate analysis
            analysis = ROIAnalysis(
                program_handle=program_handle,
                expected_bounty=expected_bounty,
                effort_hours=effort_hours,
                roi_score=roi_score,
                confidence=self._calculate_confidence(metrics),
                risk_factors=risk_factors,
                success_probability=success_prob,
                time_to_payout=metrics.resolution_time_hours,
                competitive_level=competitive_level
            )
            
            # Cache result
            self.roi_cache[program_handle] = analysis
            
            self.logger.info(f"Analyzed program {program_handle}: ROI=${roi_score:.2f}/hr, Success={success_prob:.2f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze program {program_handle}: {e}")
            # Return default low-value analysis
            return ROIAnalysis(
                program_handle=program_handle,
                expected_bounty=100.0,
                effort_hours=20.0,
                roi_score=5.0,
                confidence=0.1,
                risk_factors=["analysis_failed"],
                success_probability=0.1,
                time_to_payout=720.0,  # 30 days
                competitive_level=0.5
            )

    async def _get_program_metrics(self, program_handle: str) -> ProgramMetrics:
        """Get comprehensive program metrics"""
        
        if program_handle in self.program_cache:
            return self.program_cache[program_handle]
        
        try:
            # Get program details from HackerOne
            programs = await self.h1_client.get_programs(eligible_only=True)
            program_data = None
            
            for program in programs:
                if program.handle == program_handle:
                    program_data = program
                    break
            
            if not program_data:
                raise ValueError(f"Program {program_handle} not found")
            
            # Calculate metrics
            metrics = ProgramMetrics(
                program_handle=program_handle,
                tier=self._classify_program_tier(program_data),
                avg_bounty_amount=self._calculate_avg_bounty(program_data),
                bounty_frequency=self._calculate_bounty_frequency(program_data),
                response_time_hours=self._get_response_time(program_data),
                resolution_time_hours=self._get_resolution_time(program_data),
                scope_size=len(program_data.scopes),
                last_activity_days=self._calculate_days_since_activity(program_data),
                reputation_score=self._calculate_reputation_score(program_data),
                accepts_duplicates=self._check_duplicate_policy(program_data),
                minimum_severity="medium",  # Default assumption
                maximum_bounty=program_data.average_bounty_upper_amount or 5000,
                total_paid=self._estimate_total_paid(program_data),
                researcher_count=self._estimate_researcher_count(program_data),
                submission_volume=self._estimate_submission_volume(program_data)
            )
            
            # Cache metrics
            self.program_cache[program_handle] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics for {program_handle}: {e}")
            raise

    def _classify_program_tier(self, program_data) -> BountyProgramTier:
        """Classify program into tier based on characteristics"""
        
        # Premium programs (known high-value companies)
        premium_programs = {
            'google', 'facebook', 'microsoft', 'apple', 'amazon',
            'netflix', 'uber', 'airbnb', 'twitter', 'dropbox',
            'github', 'shopify', 'yahoo', 'linkedin', 'adobe'
        }
        
        handle_lower = program_data.handle.lower()
        
        if any(premium in handle_lower for premium in premium_programs):
            return BountyProgramTier.PREMIUM
        
        # Check bounty amounts for tier classification
        avg_bounty = program_data.average_bounty_lower_amount or 0
        max_bounty = program_data.average_bounty_upper_amount or 0
        
        if max_bounty > 10000:
            return BountyProgramTier.PREMIUM
        elif max_bounty > 2500:
            return BountyProgramTier.ESTABLISHED
        elif max_bounty > 500:
            return BountyProgramTier.GROWING
        else:
            return BountyProgramTier.BASIC

    def _calculate_avg_bounty(self, program_data) -> float:
        """Calculate average bounty amount"""
        lower = program_data.average_bounty_lower_amount or 0
        upper = program_data.average_bounty_upper_amount or 0
        
        if upper > 0:
            return (lower + upper) / 2
        elif lower > 0:
            return lower * 1.5  # Estimate upper bound
        else:
            # Use tier-based defaults
            tier_defaults = {
                BountyProgramTier.PREMIUM: 2500,
                BountyProgramTier.ESTABLISHED: 1000,
                BountyProgramTier.GROWING: 400,
                BountyProgramTier.BASIC: 150
            }
            tier = self._classify_program_tier(program_data)
            return tier_defaults.get(tier, 200)

    def _calculate_bounty_frequency(self, program_data) -> float:
        """Estimate bounties per month for this program"""
        # This would ideally use historical data
        # For now, estimate based on program characteristics
        
        base_frequency = {
            BountyProgramTier.PREMIUM: 15.0,     # 15 bounties/month
            BountyProgramTier.ESTABLISHED: 8.0,   # 8 bounties/month
            BountyProgramTier.GROWING: 4.0,      # 4 bounties/month
            BountyProgramTier.BASIC: 1.5         # 1.5 bounties/month
        }
        
        tier = self._classify_program_tier(program_data)
        return base_frequency.get(tier, 2.0)

    def _get_response_time(self, program_data) -> float:
        """Get average first response time in hours"""
        # This would come from HackerOne API if available
        # Using tier-based estimates
        tier_response_times = {
            BountyProgramTier.PREMIUM: 24.0,      # 1 day
            BountyProgramTier.ESTABLISHED: 72.0,  # 3 days
            BountyProgramTier.GROWING: 168.0,     # 7 days
            BountyProgramTier.BASIC: 336.0        # 14 days
        }
        
        tier = self._classify_program_tier(program_data)
        return tier_response_times.get(tier, 168.0)

    def _get_resolution_time(self, program_data) -> float:
        """Get average resolution time in hours"""
        tier_resolution_times = {
            BountyProgramTier.PREMIUM: 720.0,     # 30 days
            BountyProgramTier.ESTABLISHED: 1440.0, # 60 days
            BountyProgramTier.GROWING: 2160.0,     # 90 days
            BountyProgramTier.BASIC: 2880.0        # 120 days
        }
        
        tier = self._classify_program_tier(program_data)
        return tier_resolution_times.get(tier, 1440.0)

    async def _predict_bounty_amount(self, metrics: ProgramMetrics) -> float:
        """Predict expected bounty amount"""
        
        if ML_AVAILABLE and self.bounty_predictor and len(self.historical_data) > 10:
            # Use ML prediction if trained
            features = self._extract_bounty_features(metrics)
            try:
                prediction = self.bounty_predictor.predict([features])[0]
                return max(50.0, prediction)  # Minimum $50 bounty
            except:
                pass
        
        # Fallback to heuristic prediction
        base_bounty = metrics.avg_bounty_amount
        
        # Adjust based on various factors
        multiplier = 1.0
        
        # Tier adjustment
        tier_multipliers = {
            BountyProgramTier.PREMIUM: 1.2,
            BountyProgramTier.ESTABLISHED: 1.0,
            BountyProgramTier.GROWING: 0.8,
            BountyProgramTier.BASIC: 0.6
        }
        multiplier *= tier_multipliers.get(metrics.tier, 1.0)
        
        # Activity adjustment
        if metrics.last_activity_days < 30:
            multiplier *= 1.1
        elif metrics.last_activity_days > 90:
            multiplier *= 0.9
        
        # Scope size adjustment
        if metrics.scope_size > 20:
            multiplier *= 1.05
        elif metrics.scope_size < 5:
            multiplier *= 0.95
        
        # Reputation adjustment
        if metrics.reputation_score > 0.8:
            multiplier *= 1.1
        elif metrics.reputation_score < 0.5:
            multiplier *= 0.9
        
        return base_bounty * multiplier

    async def _estimate_effort_hours(self, metrics: ProgramMetrics) -> float:
        """Estimate effort hours required for this program"""
        
        # Base effort by tier
        base_effort = {
            BountyProgramTier.PREMIUM: 25.0,      # Premium = harder targets
            BountyProgramTier.ESTABLISHED: 20.0,
            BountyProgramTier.GROWING: 15.0,
            BountyProgramTier.BASIC: 12.0
        }
        
        effort = base_effort.get(metrics.tier, 20.0)
        
        # Adjust based on scope size
        scope_factor = min(2.0, 1.0 + (metrics.scope_size / 50.0))  # More scope = more effort
        effort *= scope_factor
        
        # Adjust based on competitive level
        if metrics.researcher_count > 1000:
            effort *= 1.3  # More competition = more effort needed
        elif metrics.researcher_count < 100:
            effort *= 0.8  # Less competition = easier findings
        
        # Adjust based on recent activity
        if metrics.last_activity_days > 60:
            effort *= 0.9  # Less active programs might be easier
        
        return min(self.max_effort_hours, effort)

    async def _predict_success_probability(self, metrics: ProgramMetrics) -> float:
        """Predict probability of successful submission"""
        
        # Base success rate by tier
        base_success = {
            BountyProgramTier.PREMIUM: 0.15,      # 15% - harder but pays well
            BountyProgramTier.ESTABLISHED: 0.25,  # 25% - good balance
            BountyProgramTier.GROWING: 0.35,      # 35% - easier targets
            BountyProgramTier.BASIC: 0.40         # 40% - easiest but lower pay
        }
        
        success_prob = base_success.get(metrics.tier, 0.25)
        
        # Adjust based on various factors
        
        # Program activity
        if metrics.last_activity_days < 30:
            success_prob *= 1.1  # Active programs more likely to pay
        elif metrics.last_activity_days > 90:
            success_prob *= 0.8  # Inactive programs riskier
        
        # Reputation
        if metrics.reputation_score > 0.8:
            success_prob *= 1.2
        elif metrics.reputation_score < 0.5:
            success_prob *= 0.7
        
        # Response time (faster = better)
        if metrics.response_time_hours < 48:
            success_prob *= 1.1
        elif metrics.response_time_hours > 168:
            success_prob *= 0.9
        
        # Competitive level
        if metrics.researcher_count > 2000:
            success_prob *= 0.8  # Too much competition
        elif metrics.researcher_count < 50:
            success_prob *= 0.9  # Might be inactive
        
        return min(1.0, max(0.05, success_prob))

    async def _identify_risk_factors(self, metrics: ProgramMetrics) -> List[str]:
        """Identify risk factors for this program"""
        risks = []
        
        if metrics.last_activity_days > 90:
            risks.append("program_inactive")
        
        if metrics.reputation_score < 0.5:
            risks.append("low_reputation")
        
        if metrics.response_time_hours > 336:  # 2 weeks
            risks.append("slow_response_time")
        
        if metrics.resolution_time_hours > 2160:  # 3 months
            risks.append("slow_resolution")
        
        if metrics.researcher_count > 3000:
            risks.append("high_competition")
        
        if metrics.scope_size < 3:
            risks.append("limited_scope")
        
        if not metrics.accepts_duplicates:
            risks.append("no_duplicates")
        
        if metrics.avg_bounty_amount < 200:
            risks.append("low_payout")
        
        return risks

    def _calculate_competitive_level(self, metrics: ProgramMetrics) -> float:
        """Calculate competitive level (0-1 scale)"""
        # Based on researcher count and submission volume
        
        base_competitive = min(1.0, metrics.researcher_count / 5000.0)
        
        # Adjust based on bounty amounts (higher bounties = more competition)
        if metrics.avg_bounty_amount > 1000:
            base_competitive *= 1.2
        elif metrics.avg_bounty_amount < 300:
            base_competitive *= 0.8
        
        # Adjust based on tier
        tier_competition = {
            BountyProgramTier.PREMIUM: 1.3,
            BountyProgramTier.ESTABLISHED: 1.0,
            BountyProgramTier.GROWING: 0.8,
            BountyProgramTier.BASIC: 0.6
        }
        
        base_competitive *= tier_competition.get(metrics.tier, 1.0)
        
        return min(1.0, base_competitive)

    def _calculate_confidence(self, metrics: ProgramMetrics) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for well-known programs
        if metrics.tier == BountyProgramTier.PREMIUM:
            confidence += 0.3
        elif metrics.tier == BountyProgramTier.ESTABLISHED:
            confidence += 0.2
        
        # Higher confidence for active programs
        if metrics.last_activity_days < 30:
            confidence += 0.1
        elif metrics.last_activity_days > 180:
            confidence -= 0.2
        
        # Higher confidence for programs with good reputation
        if metrics.reputation_score > 0.8:
            confidence += 0.1
        elif metrics.reputation_score < 0.4:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))

    async def prioritize_programs(self, program_handles: List[str], max_programs: int = 10) -> List[Tuple[str, ROIAnalysis]]:
        """Prioritize programs by ROI potential"""
        
        analyses = []
        
        for program_handle in program_handles:
            try:
                analysis = await self.analyze_program_value(program_handle)
                
                # Filter by minimum thresholds
                if (analysis.roi_score >= self.min_roi_threshold and 
                    analysis.confidence >= self.confidence_threshold and
                    analysis.success_probability > 0.1):
                    
                    analyses.append((program_handle, analysis))
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze program {program_handle}: {e}")
                continue
        
        # Sort by ROI score (weighted by confidence and success probability)
        analyses.sort(key=lambda x: x[1].roi_score * x[1].confidence * x[1].success_probability, reverse=True)
        
        return analyses[:max_programs]

    async def find_optimal_targets(self, 
                                 vulnerability_types: List[str],
                                 available_hours: float = 40.0,
                                 min_roi: float = 30.0) -> List[Dict[str, Any]]:
        """Find optimal targets based on available time and ROI requirements"""
        
        # Get all available programs
        all_programs = await self.h1_client.get_programs(eligible_only=True)
        program_handles = [p.handle for p in all_programs]
        
        # Analyze programs
        prioritized = await self.prioritize_programs(program_handles, max_programs=50)
        
        # Select optimal combination within time budget
        optimal_targets = []
        total_effort = 0.0
        
        for program_handle, analysis in prioritized:
            if total_effort + analysis.effort_hours <= available_hours and analysis.roi_score >= min_roi:
                
                # Check if program scope matches vulnerability types
                scope_match = await self._check_scope_compatibility(program_handle, vulnerability_types)
                
                if scope_match:
                    optimal_targets.append({
                        'program_handle': program_handle,
                        'expected_roi': analysis.roi_score,
                        'expected_bounty': analysis.expected_bounty,
                        'effort_hours': analysis.effort_hours,
                        'success_probability': analysis.success_probability,
                        'confidence': analysis.confidence,
                        'risk_factors': analysis.risk_factors,
                        'competitive_level': analysis.competitive_level
                    })
                    
                    total_effort += analysis.effort_hours
        
        self.logger.info(f"Selected {len(optimal_targets)} optimal targets requiring {total_effort:.1f} hours")
        
        return optimal_targets

    async def _check_scope_compatibility(self, program_handle: str, vulnerability_types: List[str]) -> bool:
        """Check if program scope is compatible with vulnerability types"""
        
        try:
            scopes = await self.h1_client.get_program_scopes(program_handle)
            
            # Map vulnerability types to scope types
            vuln_scope_mapping = {
                'sql_injection': ['web', 'api'],
                'xss': ['web', 'api'],
                'ssrf': ['web', 'api'],
                'rce': ['web', 'api', 'mobile'],
                'privilege_escalation': ['web', 'api', 'mobile', 'desktop'],
                'authentication_bypass': ['web', 'api', 'mobile'],
                'file_upload': ['web', 'api'],
                'directory_traversal': ['web', 'api']
            }
            
            # Check if any vulnerability type matches scope
            for vuln_type in vulnerability_types:
                required_scopes = vuln_scope_mapping.get(vuln_type, ['web'])
                
                for scope in scopes:
                    scope_type = scope.get('asset_type', '').lower()
                    
                    if any(req_scope in scope_type for req_scope in required_scopes):
                        return True
            
            return len(scopes) > 0  # If we can't match specifically, assume compatible if has scope
            
        except Exception as e:
            self.logger.warning(f"Failed to check scope compatibility for {program_handle}: {e}")
            return True  # Assume compatible on error

    def _extract_bounty_features(self, metrics: ProgramMetrics) -> List[float]:
        """Extract features for ML bounty prediction"""
        return [
            float(metrics.tier.value == 'premium'),
            float(metrics.tier.value == 'established'),
            float(metrics.tier.value == 'growing'),
            metrics.avg_bounty_amount,
            metrics.bounty_frequency,
            metrics.response_time_hours,
            metrics.resolution_time_hours,
            metrics.scope_size,
            metrics.last_activity_days,
            metrics.reputation_score,
            float(metrics.accepts_duplicates),
            metrics.maximum_bounty,
            metrics.total_paid,
            metrics.researcher_count,
            metrics.submission_volume
        ]

    # Helper methods for gathering historical data
    def _calculate_days_since_activity(self, program_data) -> int:
        """Calculate days since last activity"""
        # This would use actual API data
        return 15  # Mock value

    def _calculate_reputation_score(self, program_data) -> float:
        """Calculate program reputation score"""
        # This would be based on various factors
        return 0.8  # Mock value

    def _check_duplicate_policy(self, program_data) -> bool:
        """Check if program accepts duplicates"""
        # This would come from program policy
        return False  # Most don't accept duplicates

    def _estimate_total_paid(self, program_data) -> float:
        """Estimate total amount paid by program"""
        # This would come from historical data
        return 100000.0  # Mock value

    def _estimate_researcher_count(self, program_data) -> int:
        """Estimate number of active researchers"""
        # This would come from program statistics
        return 500  # Mock value

    def _estimate_submission_volume(self, program_data) -> int:
        """Estimate monthly submission volume"""
        # This would come from program statistics  
        return 50  # Mock value


class AutoSubmissionEngine:
    """Automated vulnerability submission with ROI optimization"""
    
    def __init__(self, 
                 bounty_intel: BountyIntelligenceEngine,
                 hackerone_client: HackerOneClient):
        
        self.bounty_intel = bounty_intel
        self.h1_client = hackerone_client
        self.logger = logging.getLogger(__name__)
        
        # Submission strategy
        self.min_cvss_score = 4.0
        self.min_confidence = 0.7
        self.max_daily_submissions = 5

    async def process_findings_for_submission(self, 
                                            findings: List[Finding],
                                            campaign_targets: List[str]) -> Dict[str, Any]:
        """Process campaign findings for optimal submission"""
        
        submission_results = {
            'submitted': [],
            'queued': [],
            'rejected': [],
            'errors': []
        }
        
        # Filter high-quality findings
        quality_findings = self._filter_quality_findings(findings)
        
        # Find matching programs for each target
        target_programs = {}
        for target in campaign_targets:
            programs = await self.h1_client.find_matching_programs(target)
            if programs:
                # Get ROI analysis for each program
                program_roi = []
                for program in programs:
                    analysis = await self.bounty_intel.analyze_program_value(program)
                    program_roi.append((program, analysis))
                
                # Sort by ROI
                program_roi.sort(key=lambda x: x[1].roi_score, reverse=True)
                target_programs[target] = program_roi
        
        # Process each quality finding
        for finding in quality_findings:
            try:
                submission_result = await self._process_single_finding(
                    finding, target_programs
                )
                
                if submission_result['submitted']:
                    submission_results['submitted'].append(submission_result)
                elif submission_result['queued']:
                    submission_results['queued'].append(submission_result)
                else:
                    submission_results['rejected'].append(submission_result)
                    
            except Exception as e:
                self.logger.error(f"Failed to process finding {finding.id}: {e}")
                submission_results['errors'].append({
                    'finding_id': finding.id,
                    'error': str(e)
                })
        
        return submission_results

    def _filter_quality_findings(self, findings: List[Finding]) -> List[Finding]:
        """Filter findings for submission quality"""
        quality_findings = []
        
        for finding in findings:
            # CVSS score check
            if finding.cvss_score and finding.cvss_score < self.min_cvss_score:
                continue
            
            # Confidence check
            if hasattr(finding, 'confidence') and finding.confidence < self.min_confidence:
                continue
            
            # Must have proof of concept
            if not finding.proof_of_concept:
                continue
            
            # Must have clear remediation
            if not finding.remediation:
                continue
            
            # Severity check
            if finding.severity.value.lower() not in ['medium', 'high', 'critical']:
                continue
            
            quality_findings.append(finding)
        
        return quality_findings

    async def _process_single_finding(self, 
                                    finding: Finding, 
                                    target_programs: Dict[str, List]) -> Dict[str, Any]:
        """Process a single finding for submission"""
        
        # Find best program for this finding
        best_program = None
        best_roi = 0.0
        
        for target in finding.affected_targets:
            if target in target_programs:
                for program_handle, analysis in target_programs[target]:
                    if analysis.roi_score > best_roi:
                        best_program = program_handle
                        best_roi = analysis.roi_score
        
        if not best_program:
            return {
                'finding_id': finding.id,
                'submitted': False,
                'queued': False,
                'reason': 'no_matching_programs'
            }
        
        # Check if we should submit now or queue
        if await self._should_submit_immediately(finding, best_roi):
            try:
                # Create enhanced submission
                submission = await self._create_enhanced_submission(finding, best_program)
                
                # Submit to HackerOne
                result = await self.h1_client.submit_report(submission)
                
                return {
                    'finding_id': finding.id,
                    'program': best_program,
                    'report_id': result['report_id'],
                    'submitted': True,
                    'queued': False,
                    'expected_roi': best_roi,
                    'submission_url': result['url']
                }
                
            except Exception as e:
                self.logger.error(f"Submission failed for finding {finding.id}: {e}")
                return {
                    'finding_id': finding.id,
                    'submitted': False,
                    'queued': True,
                    'reason': f'submission_failed: {str(e)}'
                }
        else:
            # Queue for later submission
            return {
                'finding_id': finding.id,
                'program': best_program,
                'submitted': False,
                'queued': True,
                'expected_roi': best_roi,
                'reason': 'queued_for_optimal_timing'
            }

    async def _should_submit_immediately(self, finding: Finding, roi_score: float) -> bool:
        """Determine if finding should be submitted immediately"""
        
        # Always submit critical findings immediately
        if finding.severity.value.lower() == 'critical':
            return True
        
        # Submit high ROI findings immediately
        if roi_score > 100.0:
            return True
        
        # Check daily submission limit
        today_submissions = await self._count_todays_submissions()
        if today_submissions >= self.max_daily_submissions:
            return False
        
        # Submit high-confidence findings
        if hasattr(finding, 'confidence') and finding.confidence > 0.9:
            return True
        
        # Default to queueing
        return False

    async def _create_enhanced_submission(self, finding: Finding, program_handle: str) -> VulnerabilitySubmission:
        """Create enhanced submission with AI-powered descriptions"""
        
        # Create basic submission
        submission = VulnerabilitySubmission(
            title=finding.title,
            description=finding.description,
            impact=self._generate_business_impact(finding),
            severity_rating=finding.severity.value.lower(),
            program_handle=program_handle,
            proof_of_concept=finding.proof_of_concept,
            cvss_vector=finding.cvss_vector,
            cvss_score=finding.cvss_score,
            asset_identifier=finding.affected_targets[0] if finding.affected_targets else None
        )
        
        # Enhance description with professional formatting
        submission.description = self._format_professional_description(finding)
        
        return submission

    def _generate_business_impact(self, finding: Finding) -> str:
        """Generate business impact description"""
        
        impact_templates = {
            'sql_injection': 'This vulnerability could allow an attacker to access, modify, or delete sensitive database information, potentially leading to data breaches and compliance violations.',
            'xss': 'This cross-site scripting vulnerability could enable attackers to steal user credentials, hijack user sessions, or distribute malware to site visitors.',
            'ssrf': 'This server-side request forgery vulnerability could allow attackers to access internal systems, scan internal networks, or potentially access cloud metadata services.',
            'rce': 'This remote code execution vulnerability could allow attackers to gain complete control over the affected system, potentially leading to full compromise of the server and associated data.',
            'authentication_bypass': 'This authentication bypass could allow unauthorized access to user accounts or administrative functions, potentially compromising user data and system integrity.'
        }
        
        # Determine finding type from title/description
        finding_text = (finding.title + ' ' + finding.description).lower()
        
        for vuln_type, template in impact_templates.items():
            if vuln_type.replace('_', ' ') in finding_text or vuln_type in finding.tags:
                return template
        
        # Generic impact
        return 'This security vulnerability could potentially compromise the confidentiality, integrity, or availability of the affected system and its data.'

    def _format_professional_description(self, finding: Finding) -> str:
        """Format finding as professional vulnerability report"""
        
        sections = []
        
        # Summary
        sections.append("## Summary")
        sections.append(finding.description)
        sections.append("")
        
        # Technical Details
        sections.append("## Technical Details")
        sections.append(f"**Affected URL(s):** {', '.join(finding.affected_targets)}")
        
        if finding.cvss_vector:
            sections.append(f"**CVSS Vector:** {finding.cvss_vector}")
        if finding.cvss_score:
            sections.append(f"**CVSS Score:** {finding.cvss_score} ({finding.severity.value.title()})")
        
        sections.append("")
        
        # Proof of Concept
        if finding.proof_of_concept:
            sections.append("## Proof of Concept")
            sections.append("```")
            sections.append(finding.proof_of_concept)
            sections.append("```")
            sections.append("")
        
        # Impact
        sections.append("## Impact")
        sections.append(self._generate_business_impact(finding))
        sections.append("")
        
        # Remediation
        sections.append("## Remediation")
        sections.append(finding.remediation)
        sections.append("")
        
        # References
        if finding.references:
            sections.append("## References")
            for ref in finding.references:
                sections.append(f"- {ref}")
            sections.append("")
        
        return '\n'.join(sections)

    async def _count_todays_submissions(self) -> int:
        """Count submissions made today"""
        # This would track actual submissions
        return 0  # Mock implementation

    async def get_submission_stats(self) -> Dict[str, Any]:
        """Get submission engine statistics"""
        return {
            'total_submissions': len(self.h1_client.submitted_reports),
            'success_rate': 0.75,  # Mock
            'average_roi': 85.0,   # Mock
            'total_earned': sum([500, 1200, 300]),  # Mock
            'pending_submissions': 3,  # Mock
            'programs_targeted': 5   # Mock
        }


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_bounty_intelligence():
        """Demo bounty intelligence system"""
        
        # Mock HackerOne client
        h1_client = HackerOneClient("mock_api_key")
        
        # Create intelligence engine
        intel_engine = BountyIntelligenceEngine(h1_client)
        
        # Demo program analysis
        test_programs = ['google', 'github', 'shopify']
        
        print("=== Bounty Intelligence Demo ===\n")
        
        for program in test_programs:
            try:
                analysis = await intel_engine.analyze_program_value(program)
                
                print(f"Program: {program}")
                print(f"  Expected Bounty: ${analysis.expected_bounty:.2f}")
                print(f"  Effort Hours: {analysis.effort_hours:.1f}")
                print(f"  ROI Score: ${analysis.roi_score:.2f}/hour")
                print(f"  Success Probability: {analysis.success_probability:.1%}")
                print(f"  Confidence: {analysis.confidence:.1%}")
                print(f"  Risk Factors: {', '.join(analysis.risk_factors)}")
                print()
                
            except Exception as e:
                print(f"Failed to analyze {program}: {e}")
        
        # Demo program prioritization
        print("=== Program Prioritization ===")
        prioritized = await intel_engine.prioritize_programs(test_programs, max_programs=5)
        
        for i, (program, analysis) in enumerate(prioritized, 1):
            print(f"{i}. {program} - ROI: ${analysis.roi_score:.2f}/hr (Confidence: {analysis.confidence:.1%})")
        
        print("\nBounty intelligence demo completed")
    
    if "--demo" in sys.argv:
        asyncio.run(demo_bounty_intelligence())
    else:
        print("XORB Bounty Intelligence Engine")
        print("Usage: python bounty_intelligence.py --demo")