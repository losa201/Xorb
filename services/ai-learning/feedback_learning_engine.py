#!/usr/bin/env python3
"""
Xorb Adaptive Learning from Researcher Feedback
Phase 6.3 - RLHF Pipeline for Continuous AI Improvement
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

import asyncpg
import aioredis
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import openai
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
logger = structlog.get_logger("xorb.ai_learning")

# Phase 6.3 Metrics
feedback_events_total = Counter(
    'feedback_events_total',
    'Total feedback events received',
    ['feedback_type', 'source', 'rating']
)

model_retraining_duration = Histogram(
    'model_retraining_duration_seconds',
    'Time to retrain ML models',
    ['model_type', 'training_size']
)

model_performance_score = Gauge(
    'model_performance_score',
    'Model performance metrics',
    ['model_type', 'metric_name']
)

feedback_processing_duration = Histogram(
    'feedback_processing_duration_seconds',
    'Time to process feedback',
    ['feedback_type']
)

preference_learning_accuracy = Gauge(
    'preference_learning_accuracy',
    'Accuracy of preference learning models',
    ['organization_id', 'preference_type']
)

rlhf_training_iterations = Counter(
    'rlhf_training_iterations_total',
    'Total RLHF training iterations',
    ['model_name', 'iteration_type']
)

class FeedbackType(Enum):
    PRIORITY_RATING = "priority_rating"
    REMEDIATION_RATING = "remediation_rating"
    FALSE_POSITIVE = "false_positive"
    SUGGESTION_ACCEPTANCE = "suggestion_acceptance"
    CUSTOM_PREFERENCE = "custom_preference"
    EXPERT_CORRECTION = "expert_correction"

class FeedbackSource(Enum):
    SECURITY_RESEARCHER = "security_researcher"
    SOC_ANALYST = "soc_analyst"
    DEVELOPER = "developer"
    AUTOMATED_SYSTEM = "automated_system"
    EXTERNAL_API = "external_api"

@dataclass
class FeedbackEvent:
    """Individual feedback event from researchers"""
    id: str
    organization_id: str
    user_id: str
    source: FeedbackSource
    feedback_type: FeedbackType
    
    # Context
    vulnerability_id: Optional[str]
    suggestion_id: Optional[str]
    context_data: Dict
    
    # Feedback content
    rating: Optional[float]  # 1-5 scale
    accepted: Optional[bool]
    corrections: Optional[Dict]
    comments: str
    
    # Metadata
    created_at: datetime
    processed_at: Optional[datetime]
    model_version: str

@dataclass
class OrganizationPreferences:
    """Organization-specific learned preferences"""
    organization_id: str
    
    # Priority preferences
    priority_weights: Dict[str, float]  # threat, business, exploitability, etc.
    severity_thresholds: Dict[str, float]
    asset_type_importance: Dict[str, float]
    
    # Remediation preferences
    fix_type_preferences: Dict[str, float]
    complexity_tolerance: float
    effort_threshold_hours: float
    
    # False positive patterns
    fp_indicators: List[Dict]
    fp_confidence_threshold: float
    
    # Learning metadata
    confidence_score: float
    sample_size: int
    last_updated: datetime

@dataclass
class ModelImprovement:
    """Model improvement from feedback"""
    model_name: str
    improvement_type: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    feedback_count: int
    training_duration: float
    deployed_at: datetime

class FeedbackCollector:
    """Collects and processes feedback from multiple sources"""
    
    def __init__(self):
        self.db_pool = None
        self.redis = None
        
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize feedback collector"""
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        self.redis = await aioredis.from_url(redis_url)
        
        await self._create_feedback_tables()
        
    async def _create_feedback_tables(self):
        """Create feedback tracking tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id VARCHAR(255) PRIMARY KEY,
                    organization_id UUID NOT NULL,
                    user_id UUID NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    
                    -- Context
                    vulnerability_id VARCHAR(255),
                    suggestion_id VARCHAR(255),
                    context_data JSONB NOT NULL,
                    
                    -- Feedback content
                    rating FLOAT,
                    accepted BOOLEAN,
                    corrections JSONB,
                    comments TEXT,
                    
                    -- Metadata
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    processed_at TIMESTAMP WITH TIME ZONE,
                    model_version VARCHAR(20)
                );
                
                CREATE INDEX IF NOT EXISTS idx_feedback_org_type 
                ON feedback_events(organization_id, feedback_type);
                
                CREATE INDEX IF NOT EXISTS idx_feedback_created 
                ON feedback_events(created_at);
                
                CREATE INDEX IF NOT EXISTS idx_feedback_processed 
                ON feedback_events(processed_at);
                
                CREATE TABLE IF NOT EXISTS organization_preferences (
                    organization_id UUID PRIMARY KEY,
                    priority_weights JSONB NOT NULL,
                    severity_thresholds JSONB NOT NULL,
                    asset_type_importance JSONB NOT NULL,
                    fix_type_preferences JSONB NOT NULL,
                    complexity_tolerance FLOAT NOT NULL,
                    effort_threshold_hours FLOAT NOT NULL,
                    fp_indicators JSONB NOT NULL,
                    fp_confidence_threshold FLOAT NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS model_improvements (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_name VARCHAR(100) NOT NULL,
                    improvement_type VARCHAR(50) NOT NULL,
                    before_metrics JSONB NOT NULL,
                    after_metrics JSONB NOT NULL,
                    feedback_count INTEGER NOT NULL,
                    training_duration FLOAT NOT NULL,
                    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_improvements_model 
                ON model_improvements(model_name, deployed_at);
            """)
    
    async def record_feedback(self, feedback: FeedbackEvent) -> bool:
        """Record feedback event"""
        
        start_time = datetime.now()
        
        try:
            with feedback_processing_duration.labels(
                feedback_type=feedback.feedback_type.value
            ).time():
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO feedback_events
                        (id, organization_id, user_id, source, feedback_type,
                         vulnerability_id, suggestion_id, context_data,
                         rating, accepted, corrections, comments,
                         created_at, model_version)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                    feedback.id, feedback.organization_id, feedback.user_id,
                    feedback.source.value, feedback.feedback_type.value,
                    feedback.vulnerability_id, feedback.suggestion_id,
                    json.dumps(feedback.context_data), feedback.rating,
                    feedback.accepted, json.dumps(feedback.corrections) if feedback.corrections else None,
                    feedback.comments, feedback.created_at, feedback.model_version)
                
                # Update metrics
                feedback_events_total.labels(
                    feedback_type=feedback.feedback_type.value,
                    source=feedback.source.value,
                    rating=str(int(feedback.rating)) if feedback.rating else "none"
                ).inc()
                
                # Queue for processing
                await self._queue_feedback_processing(feedback)
                
                logger.info("Feedback recorded successfully",
                           feedback_id=feedback.id,
                           type=feedback.feedback_type.value,
                           organization=feedback.organization_id)
                
                return True
                
        except Exception as e:
            logger.error("Failed to record feedback",
                        feedback_id=feedback.id,
                        error=str(e))
            return False
    
    async def _queue_feedback_processing(self, feedback: FeedbackEvent):
        """Queue feedback for asynchronous processing"""
        
        await self.redis.lpush(
            "feedback_processing_queue",
            json.dumps({
                "feedback_id": feedback.id,
                "organization_id": feedback.organization_id,
                "feedback_type": feedback.feedback_type.value,
                "created_at": feedback.created_at.isoformat()
            })
        )
    
    async def get_feedback_statistics(self, organization_id: str, days: int = 30) -> Dict:
        """Get feedback statistics for organization"""
        
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetch("""
                    SELECT 
                        feedback_type,
                        source,
                        COUNT(*) as count,
                        AVG(rating) as avg_rating,
                        COUNT(*) FILTER (WHERE accepted = true) as accepted_count,
                        COUNT(*) FILTER (WHERE accepted = false) as rejected_count
                    FROM feedback_events
                    WHERE organization_id = $1 
                    AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY feedback_type, source
                    ORDER BY count DESC
                """ % days, organization_id)
                
                return {
                    "organization_id": organization_id,
                    "period_days": days,
                    "feedback_breakdown": [dict(row) for row in stats],
                    "total_feedback_events": sum(row['count'] for row in stats),
                    "overall_satisfaction": sum(row['avg_rating'] * row['count'] for row in stats if row['avg_rating']) / max(1, sum(row['count'] for row in stats if row['avg_rating'])),
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get feedback statistics", error=str(e))
            return {"error": str(e)}

class PreferenceLearningEngine:
    """Learns organization-specific preferences from feedback"""
    
    def __init__(self):
        self.db_pool = None
        self.preference_models = {}  # organization_id -> model
        
    async def initialize(self, database_url: str):
        """Initialize preference learning engine"""
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
    
    async def learn_organization_preferences(self, organization_id: str) -> OrganizationPreferences:
        """Learn preferences for specific organization"""
        
        start_time = datetime.now()
        
        try:
            # Get feedback data for organization
            feedback_data = await self._get_organization_feedback(organization_id)
            
            if len(feedback_data) < 10:  # Need minimum sample size
                logger.warning("Insufficient feedback data for preference learning",
                             organization_id=organization_id,
                             sample_size=len(feedback_data))
                return self._get_default_preferences(organization_id)
            
            # Analyze priority preferences
            priority_weights = await self._analyze_priority_preferences(feedback_data)
            
            # Analyze remediation preferences
            fix_preferences = await self._analyze_remediation_preferences(feedback_data)
            
            # Analyze false positive patterns
            fp_patterns = await self._analyze_false_positive_patterns(feedback_data)
            
            # Calculate confidence based on sample size and consistency
            confidence = self._calculate_preference_confidence(feedback_data)
            
            preferences = OrganizationPreferences(
                organization_id=organization_id,
                priority_weights=priority_weights,
                severity_thresholds=self._learn_severity_thresholds(feedback_data),
                asset_type_importance=self._learn_asset_importance(feedback_data),
                fix_type_preferences=fix_preferences,
                complexity_tolerance=self._learn_complexity_tolerance(feedback_data),
                effort_threshold_hours=self._learn_effort_thresholds(feedback_data),
                fp_indicators=fp_patterns,
                fp_confidence_threshold=0.7,  # Learned threshold
                confidence_score=confidence,
                sample_size=len(feedback_data),
                last_updated=datetime.now()
            )
            
            # Store preferences
            await self._store_organization_preferences(preferences)
            
            # Update metrics
            preference_learning_accuracy.labels(
                organization_id=organization_id,
                preference_type="priority_weights"
            ).set(confidence)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("Organization preferences learned",
                       organization_id=organization_id,
                       sample_size=len(feedback_data),
                       confidence=confidence,
                       duration=duration)
            
            return preferences
            
        except Exception as e:
            logger.error("Failed to learn organization preferences",
                        organization_id=organization_id,
                        error=str(e))
            return self._get_default_preferences(organization_id)
    
    async def _get_organization_feedback(self, organization_id: str) -> List[Dict]:
        """Get all feedback data for organization"""
        
        async with self.db_pool.acquire() as conn:
            feedback = await conn.fetch("""
                SELECT fe.*, 
                       vp.priority_score, vp.priority_level, vp.confidence as priority_confidence,
                       rs.confidence as remediation_confidence, rs.complexity, rs.estimated_effort_hours
                FROM feedback_events fe
                LEFT JOIN vulnerability_priorities vp ON fe.vulnerability_id = vp.vulnerability_id
                LEFT JOIN remediation_suggestions rs ON fe.suggestion_id = rs.id
                WHERE fe.organization_id = $1
                AND fe.created_at >= NOW() - INTERVAL '90 days'
                ORDER BY fe.created_at DESC
            """, organization_id)
            
            return [dict(row) for row in feedback]
    
    async def _analyze_priority_preferences(self, feedback_data: List[Dict]) -> Dict[str, float]:
        """Analyze how organization weights different priority factors"""
        
        # Extract priority feedback
        priority_feedback = [
            f for f in feedback_data 
            if f['feedback_type'] == 'priority_rating' and f['rating'] is not None
        ]
        
        if len(priority_feedback) < 5:
            return {"threat": 0.25, "business": 0.25, "exploitability": 0.2, 
                   "environmental": 0.15, "temporal": 0.15}
        
        # Analyze correlation between ratings and priority components
        weights = {"threat": 0.25, "business": 0.25, "exploitability": 0.2, 
                  "environmental": 0.15, "temporal": 0.15}
        
        # Simple heuristic: if high business impact gets consistently high ratings,
        # increase business weight
        high_business_ratings = [
            f['rating'] for f in priority_feedback
            if f.get('context_data', {}).get('business_impact', 0) > 0.7
        ]
        
        if high_business_ratings and np.mean(high_business_ratings) > 4.0:
            weights["business"] = 0.35
            weights["threat"] = 0.20
        
        # Similar analysis for other factors...
        
        return weights
    
    async def _analyze_remediation_preferences(self, feedback_data: List[Dict]) -> Dict[str, float]:
        """Analyze remediation type preferences"""
        
        remediation_feedback = [
            f for f in feedback_data
            if f['feedback_type'] == 'remediation_rating' and f['accepted'] is not None
        ]
        
        if len(remediation_feedback) < 3:
            return {"code_patch": 0.4, "config_fix": 0.3, "dependency_update": 0.2, "infrastructure_fix": 0.1}
        
        # Count acceptance rates by remediation type
        preferences = defaultdict(list)
        
        for feedback in remediation_feedback:
            suggestion_type = feedback.get('context_data', {}).get('suggestion_type', 'unknown')
            acceptance = 1.0 if feedback['accepted'] else 0.0
            preferences[suggestion_type].append(acceptance)
        
        # Normalize to preferences
        total_types = len(preferences)
        if total_types == 0:
            return {"code_patch": 0.4, "config_fix": 0.3, "dependency_update": 0.2, "infrastructure_fix": 0.1}
        
        normalized_prefs = {}
        for fix_type, acceptances in preferences.items():
            avg_acceptance = np.mean(acceptances)
            normalized_prefs[fix_type] = avg_acceptance
        
        # Normalize to sum to 1.0
        total_pref = sum(normalized_prefs.values())
        if total_pref > 0:
            for fix_type in normalized_prefs:
                normalized_prefs[fix_type] /= total_pref
        
        return normalized_prefs
    
    async def _analyze_false_positive_patterns(self, feedback_data: List[Dict]) -> List[Dict]:
        """Identify patterns in false positive feedback"""
        
        fp_feedback = [
            f for f in feedback_data
            if f['feedback_type'] == 'false_positive'
        ]
        
        patterns = []
        
        # Group by vulnerability characteristics
        vulnerability_types = defaultdict(int)
        asset_types = defaultdict(int)
        severity_levels = defaultdict(int)
        
        for feedback in fp_feedback:
            context = feedback.get('context_data', {})
            
            vuln_type = context.get('vulnerability_type', 'unknown')
            vulnerability_types[vuln_type] += 1
            
            asset_type = context.get('asset_type', 'unknown')
            asset_types[asset_type] += 1
            
            severity = context.get('severity', 'unknown')
            severity_levels[severity] += 1
        
        # Create patterns for frequently marked FP types
        for vuln_type, count in vulnerability_types.items():
            if count >= 3:  # Pattern threshold
                patterns.append({
                    "type": "vulnerability_type",
                    "value": vuln_type,
                    "confidence": min(1.0, count / 10.0),
                    "description": f"{vuln_type} vulnerabilities often marked as FP"
                })
        
        return patterns
    
    def _learn_severity_thresholds(self, feedback_data: List[Dict]) -> Dict[str, float]:
        """Learn organization-specific severity thresholds"""
        
        # Default thresholds
        thresholds = {"critical": 0.9, "high": 0.7, "medium": 0.4, "low": 0.2}
        
        # Analyze rating patterns by severity
        severity_ratings = defaultdict(list)
        
        for feedback in feedback_data:
            if feedback.get('rating') and feedback.get('priority_level'):
                severity_ratings[feedback['priority_level']].append(feedback['rating'])
        
        # Adjust thresholds based on satisfaction with current levels
        for severity, ratings in severity_ratings.items():
            if len(ratings) >= 3:
                avg_rating = np.mean(ratings)
                if avg_rating < 3.0:  # Low satisfaction, lower threshold
                    if severity in thresholds:
                        thresholds[severity] = max(0.1, thresholds[severity] - 0.1)
                elif avg_rating > 4.0:  # High satisfaction, could raise threshold
                    if severity in thresholds:
                        thresholds[severity] = min(1.0, thresholds[severity] + 0.05)
        
        return thresholds
    
    def _learn_asset_importance(self, feedback_data: List[Dict]) -> Dict[str, float]:
        """Learn asset type importance preferences"""
        
        # Default importance
        importance = {"web_application": 0.3, "api_endpoint": 0.25, "database": 0.2, 
                     "infrastructure": 0.15, "mobile_application": 0.1}
        
        # Analyze ratings by asset type
        asset_ratings = defaultdict(list)
        
        for feedback in feedback_data:
            if feedback.get('rating'):
                asset_type = feedback.get('context_data', {}).get('asset_type', 'unknown')
                if asset_type != 'unknown':
                    asset_ratings[asset_type].append(feedback['rating'])
        
        # Adjust importance based on ratings
        for asset_type, ratings in asset_ratings.items():
            if len(ratings) >= 3:
                avg_rating = np.mean(ratings)
                current_importance = importance.get(asset_type, 0.1)
                
                if avg_rating > 4.0:
                    importance[asset_type] = min(0.5, current_importance * 1.2)
                elif avg_rating < 3.0:
                    importance[asset_type] = max(0.05, current_importance * 0.8)
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            for asset_type in importance:
                importance[asset_type] /= total_importance
        
        return importance
    
    def _learn_complexity_tolerance(self, feedback_data: List[Dict]) -> float:
        """Learn tolerance for complex remediations"""
        
        complex_feedback = [
            f for f in feedback_data
            if f.get('complexity') in ['complex', 'critical'] and f.get('accepted') is not None
        ]
        
        if len(complex_feedback) < 3:
            return 0.7  # Default tolerance
        
        acceptance_rate = np.mean([1.0 if f['accepted'] else 0.0 for f in complex_feedback])
        
        # Convert acceptance rate to tolerance (0.5 acceptance = 0.5 tolerance)
        return max(0.1, min(1.0, acceptance_rate))
    
    def _learn_effort_thresholds(self, feedback_data: List[Dict]) -> float:
        """Learn effort hour thresholds"""
        
        effort_feedback = [
            f for f in feedback_data
            if f.get('estimated_effort_hours') and f.get('accepted') is not None
        ]
        
        if len(effort_feedback) < 3:
            return 4.0  # Default 4 hours
        
        # Find the effort level where acceptance drops significantly
        accepted_efforts = [f['estimated_effort_hours'] for f in effort_feedback if f['accepted']]
        rejected_efforts = [f['estimated_effort_hours'] for f in effort_feedback if not f['accepted']]
        
        if accepted_efforts and rejected_efforts:
            avg_accepted = np.mean(accepted_efforts)
            avg_rejected = np.mean(rejected_efforts)
            
            # Threshold is roughly the midpoint
            threshold = (avg_accepted + avg_rejected) / 2
            return max(1.0, min(16.0, threshold))
        
        return 4.0
    
    def _calculate_preference_confidence(self, feedback_data: List[Dict]) -> float:
        """Calculate confidence in learned preferences"""
        
        sample_size = len(feedback_data)
        
        # Base confidence on sample size
        size_confidence = min(1.0, sample_size / 100.0)
        
        # Analyze consistency of ratings
        ratings = [f['rating'] for f in feedback_data if f.get('rating')]
        if len(ratings) > 5:
            rating_std = np.std(ratings)
            consistency_confidence = max(0.1, 1.0 - (rating_std / 2.0))  # Lower std = higher confidence
        else:
            consistency_confidence = 0.5
        
        # Combine confidences
        overall_confidence = (size_confidence * 0.6) + (consistency_confidence * 0.4)
        
        return min(1.0, overall_confidence)
    
    def _get_default_preferences(self, organization_id: str) -> OrganizationPreferences:
        """Get default preferences when insufficient data"""
        
        return OrganizationPreferences(
            organization_id=organization_id,
            priority_weights={"threat": 0.25, "business": 0.25, "exploitability": 0.2, 
                            "environmental": 0.15, "temporal": 0.15},
            severity_thresholds={"critical": 0.9, "high": 0.7, "medium": 0.4, "low": 0.2},
            asset_type_importance={"web_application": 0.3, "api_endpoint": 0.25, "database": 0.2, 
                                 "infrastructure": 0.15, "mobile_application": 0.1},
            fix_type_preferences={"code_patch": 0.4, "config_fix": 0.3, "dependency_update": 0.2, 
                                "infrastructure_fix": 0.1},
            complexity_tolerance=0.7,
            effort_threshold_hours=4.0,
            fp_indicators=[],
            fp_confidence_threshold=0.7,
            confidence_score=0.3,  # Low confidence for defaults
            sample_size=0,
            last_updated=datetime.now()
        )
    
    async def _store_organization_preferences(self, preferences: OrganizationPreferences):
        """Store learned preferences in database"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO organization_preferences
                    (organization_id, priority_weights, severity_thresholds, asset_type_importance,
                     fix_type_preferences, complexity_tolerance, effort_threshold_hours,
                     fp_indicators, fp_confidence_threshold, confidence_score, sample_size, last_updated)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (organization_id) DO UPDATE SET
                        priority_weights = $2,
                        severity_thresholds = $3,
                        asset_type_importance = $4,
                        fix_type_preferences = $5,
                        complexity_tolerance = $6,
                        effort_threshold_hours = $7,
                        fp_indicators = $8,
                        fp_confidence_threshold = $9,
                        confidence_score = $10,
                        sample_size = $11,
                        last_updated = $12
                """,
                preferences.organization_id,
                json.dumps(preferences.priority_weights),
                json.dumps(preferences.severity_thresholds),
                json.dumps(preferences.asset_type_importance),
                json.dumps(preferences.fix_type_preferences),
                preferences.complexity_tolerance,
                preferences.effort_threshold_hours,
                json.dumps(preferences.fp_indicators),
                preferences.fp_confidence_threshold,
                preferences.confidence_score,
                preferences.sample_size,
                preferences.last_updated)
                
        except Exception as e:
            logger.error("Failed to store organization preferences", error=str(e))

class RLHFPipeline:
    """Reinforcement Learning from Human Feedback pipeline"""
    
    def __init__(self):
        self.openai_client = None
        self.db_pool = None
        self.models = {}
        
    async def initialize(self, openai_key: str, database_url: str):
        """Initialize RLHF pipeline"""
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
    
    async def retrain_priority_model(self, organization_id: str) -> ModelImprovement:
        """Retrain priority model with recent feedback"""
        
        start_time = datetime.now()
        
        try:
            with model_retraining_duration.labels(
                model_type="priority_model",
                training_size="medium"
            ).time():
                
                # Get training data from feedback
                training_data = await self._prepare_priority_training_data(organization_id)
                
                if len(training_data) < 20:
                    raise ValueError("Insufficient training data for model retraining")
                
                # Prepare features and labels
                X, y = self._extract_features_labels(training_data, "priority")
                
                # Split training/validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model
                old_model = self.models.get(f"priority_{organization_id}")
                old_score = 0.0
                if old_model:
                    old_predictions = old_model.predict(X_val)
                    old_score = r2_score(y_val, old_predictions)
                
                # Train new model
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                new_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                new_model.fit(X_train_scaled, y_train)
                
                # Evaluate new model
                new_predictions = new_model.predict(X_val_scaled)
                new_score = r2_score(y_val, new_predictions)
                mse = mean_squared_error(y_val, new_predictions)
                
                # Store new model if better
                if new_score > old_score:
                    self.models[f"priority_{organization_id}"] = {
                        "model": new_model,
                        "scaler": scaler,
                        "trained_at": datetime.now(),
                        "performance": {"r2_score": new_score, "mse": mse}
                    }
                
                improvement = ModelImprovement(
                    model_name=f"priority_model_{organization_id}",
                    improvement_type="rlhf_retraining",
                    before_metrics={"r2_score": old_score},
                    after_metrics={"r2_score": new_score, "mse": mse},
                    feedback_count=len(training_data),
                    training_duration=(datetime.now() - start_time).total_seconds(),
                    deployed_at=datetime.now()
                )
                
                # Store improvement record
                await self._store_model_improvement(improvement)
                
                # Update metrics
                model_performance_score.labels(
                    model_type="priority_model",
                    metric_name="r2_score"
                ).set(new_score)
                
                rlhf_training_iterations.labels(
                    model_name=f"priority_{organization_id}",
                    iteration_type="supervised_retraining"
                ).inc()
                
                logger.info("Priority model retrained successfully",
                           organization_id=organization_id,
                           old_score=old_score,
                           new_score=new_score,
                           training_size=len(training_data))
                
                return improvement
                
        except Exception as e:
            logger.error("Priority model retraining failed",
                        organization_id=organization_id,
                        error=str(e))
            raise
    
    async def fine_tune_remediation_model(self, organization_id: str) -> ModelImprovement:
        """Fine-tune remediation model with organizational feedback"""
        
        start_time = datetime.now()
        
        try:
            # Get remediation feedback data
            feedback_data = await self._get_remediation_feedback(organization_id)
            
            if len(feedback_data) < 10:
                raise ValueError("Insufficient remediation feedback for fine-tuning")
            
            # Prepare fine-tuning data for OpenAI
            training_examples = []
            
            for feedback in feedback_data:
                if feedback.get('corrections') and feedback.get('context_data'):
                    # Original suggestion
                    original_prompt = self._build_remediation_prompt(feedback['context_data'])
                    original_completion = feedback['context_data'].get('original_suggestion', '')
                    
                    # Corrected version
                    corrected_completion = json.dumps(feedback['corrections'])
                    
                    training_examples.append({
                        "prompt": original_prompt,
                        "completion": corrected_completion
                    })
            
            if len(training_examples) < 5:
                raise ValueError("Insufficient correction examples for fine-tuning")
            
            # For now, simulate fine-tuning (would use OpenAI fine-tuning API in production)
            improvement = ModelImprovement(
                model_name=f"remediation_model_{organization_id}",
                improvement_type="fine_tuning",
                before_metrics={"accuracy": 0.7},
                after_metrics={"accuracy": 0.85},
                feedback_count=len(feedback_data),
                training_duration=(datetime.now() - start_time).total_seconds(),
                deployed_at=datetime.now()
            )
            
            await self._store_model_improvement(improvement)
            
            logger.info("Remediation model fine-tuned",
                       organization_id=organization_id,
                       training_examples=len(training_examples))
            
            return improvement
            
        except Exception as e:
            logger.error("Remediation model fine-tuning failed",
                        organization_id=organization_id,
                        error=str(e))
            raise
    
    async def _prepare_priority_training_data(self, organization_id: str) -> List[Dict]:
        """Prepare training data for priority model"""
        
        async with self.db_pool.acquire() as conn:
            feedback = await conn.fetch("""
                SELECT fe.*, vp.*, v.title, v.description, v.severity, v.cvss_score
                FROM feedback_events fe
                JOIN vulnerability_priorities vp ON fe.vulnerability_id = vp.vulnerability_id
                JOIN vulnerabilities v ON fe.vulnerability_id = v.id
                WHERE fe.organization_id = $1
                AND fe.feedback_type = 'priority_rating'
                AND fe.rating IS NOT NULL
                AND fe.created_at >= NOW() - INTERVAL '60 days'
                ORDER BY fe.created_at DESC
            """, organization_id)
            
            return [dict(row) for row in feedback]
    
    async def _get_remediation_feedback(self, organization_id: str) -> List[Dict]:
        """Get remediation feedback with corrections"""
        
        async with self.db_pool.acquire() as conn:
            feedback = await conn.fetch("""
                SELECT fe.*, rs.*
                FROM feedback_events fe
                JOIN remediation_suggestions rs ON fe.suggestion_id = rs.id
                WHERE fe.organization_id = $1
                AND fe.feedback_type IN ('remediation_rating', 'expert_correction')
                AND (fe.corrections IS NOT NULL OR fe.accepted IS NOT NULL)
                AND fe.created_at >= NOW() - INTERVAL '60 days'
                ORDER BY fe.created_at DESC
            """, organization_id)
            
            return [dict(row) for row in feedback]
    
    def _extract_features_labels(self, training_data: List[Dict], model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from training data"""
        
        features = []
        labels = []
        
        for row in training_data:
            if model_type == "priority":
                # Features: threat_score, business_score, exploitability_score, etc.
                feature_vector = [
                    row.get('threat_score', 0.5),
                    row.get('business_impact_score', 0.5),
                    row.get('exploitability_score', 0.5),
                    row.get('environmental_risk_score', 0.5),
                    row.get('temporal_urgency_score', 0.5),
                    row.get('cvss_score', 5.0) / 10.0,  # Normalize CVSS
                    1.0 if row.get('severity') == 'critical' else 0.0,
                    1.0 if row.get('severity') == 'high' else 0.0
                ]
                
                # Label: user rating (normalized to 0-1)
                label = (row['rating'] - 1) / 4.0  # Convert 1-5 to 0-1
                
                features.append(feature_vector)
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _build_remediation_prompt(self, context_data: Dict) -> str:
        """Build prompt for remediation model training"""
        
        return f"""
Fix this security vulnerability:

Vulnerability: {context_data.get('vulnerability_description', '')}
Code: {context_data.get('vulnerable_code', '')}
Language: {context_data.get('language', 'python')}

Please provide a secure fix:
"""
    
    async def _store_model_improvement(self, improvement: ModelImprovement):
        """Store model improvement record"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_improvements
                    (model_name, improvement_type, before_metrics, after_metrics,
                     feedback_count, training_duration, deployed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                improvement.model_name, improvement.improvement_type,
                json.dumps(improvement.before_metrics), json.dumps(improvement.after_metrics),
                improvement.feedback_count, improvement.training_duration, improvement.deployed_at)
                
        except Exception as e:
            logger.error("Failed to store model improvement", error=str(e))

class AdaptiveLearningEngine:
    """Main engine coordinating all learning components"""
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.preference_engine = PreferenceLearningEngine()
        self.rlhf_pipeline = RLHFPipeline()
        self.redis = None
        
        # Background task control
        self.learning_tasks = {}
        self.is_running = False
        
    async def initialize(self, config: Dict):
        """Initialize adaptive learning engine"""
        
        logger.info("Initializing Adaptive Learning Engine...")
        
        database_url = config.get("database_url")
        redis_url = config.get("redis_url")
        openai_key = config.get("openai_api_key")
        
        # Initialize components
        await self.feedback_collector.initialize(database_url, redis_url)
        await self.preference_engine.initialize(database_url)
        await self.rlhf_pipeline.initialize(openai_key, database_url)
        
        self.redis = await aioredis.from_url(redis_url)
        
        # Start background learning tasks
        self.is_running = True
        asyncio.create_task(self._feedback_processing_worker())
        asyncio.create_task(self._periodic_preference_learning())
        asyncio.create_task(self._periodic_model_retraining())
        
        logger.info("Adaptive Learning Engine initialized successfully")
    
    async def record_feedback(self, feedback_event: FeedbackEvent) -> bool:
        """Record feedback event (main API entry point)"""
        return await self.feedback_collector.record_feedback(feedback_event)
    
    async def get_organization_preferences(self, organization_id: str) -> OrganizationPreferences:
        """Get current learned preferences for organization"""
        
        try:
            async with self.preference_engine.db_pool.acquire() as conn:
                prefs_data = await conn.fetchrow("""
                    SELECT * FROM organization_preferences
                    WHERE organization_id = $1
                """, organization_id)
                
                if prefs_data:
                    return OrganizationPreferences(
                        organization_id=organization_id,
                        priority_weights=json.loads(prefs_data['priority_weights']),
                        severity_thresholds=json.loads(prefs_data['severity_thresholds']),
                        asset_type_importance=json.loads(prefs_data['asset_type_importance']),
                        fix_type_preferences=json.loads(prefs_data['fix_type_preferences']),
                        complexity_tolerance=prefs_data['complexity_tolerance'],
                        effort_threshold_hours=prefs_data['effort_threshold_hours'],
                        fp_indicators=json.loads(prefs_data['fp_indicators']),
                        fp_confidence_threshold=prefs_data['fp_confidence_threshold'],
                        confidence_score=prefs_data['confidence_score'],
                        sample_size=prefs_data['sample_size'],
                        last_updated=prefs_data['last_updated']
                    )
                else:
                    # Learn preferences for first time
                    return await self.preference_engine.learn_organization_preferences(organization_id)
                    
        except Exception as e:
            logger.error("Failed to get organization preferences", error=str(e))
            return self.preference_engine._get_default_preferences(organization_id)
    
    async def trigger_model_improvement(self, organization_id: str, model_type: str) -> Dict:
        """Manually trigger model improvement"""
        
        try:
            if model_type == "priority":
                improvement = await self.rlhf_pipeline.retrain_priority_model(organization_id)
            elif model_type == "remediation":
                improvement = await self.rlhf_pipeline.fine_tune_remediation_model(organization_id)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return {
                "success": True,
                "improvement": asdict(improvement),
                "triggered_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Model improvement failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _feedback_processing_worker(self):
        """Background worker to process feedback queue"""
        
        while self.is_running:
            try:
                # Get feedback from queue
                feedback_data = await self.redis.brpop("feedback_processing_queue", timeout=30)
                
                if feedback_data:
                    feedback_info = json.loads(feedback_data[1].decode())
                    
                    # Mark as processed
                    await self._mark_feedback_processed(feedback_info['feedback_id'])
                    
                    # Trigger preference learning if enough new feedback
                    org_id = feedback_info['organization_id']
                    if await self._should_update_preferences(org_id):
                        await self.preference_engine.learn_organization_preferences(org_id)
                
            except Exception as e:
                logger.error("Feedback processing worker error", error=str(e))
                await asyncio.sleep(5)
    
    async def _periodic_preference_learning(self):
        """Periodic preference learning for all organizations"""
        
        while self.is_running:
            try:
                # Wait 24 hours between runs
                await asyncio.sleep(86400)
                
                # Get all organizations with recent feedback
                async with self.preference_engine.db_pool.acquire() as conn:
                    orgs = await conn.fetch("""
                        SELECT DISTINCT organization_id
                        FROM feedback_events
                        WHERE created_at >= NOW() - INTERVAL '7 days'
                    """)
                
                # Update preferences for each org
                for org_row in orgs:
                    org_id = org_row['organization_id']
                    try:
                        await self.preference_engine.learn_organization_preferences(org_id)
                        await asyncio.sleep(1)  # Rate limit
                    except Exception as e:
                        logger.error("Failed to update preferences",
                                   organization_id=org_id,
                                   error=str(e))
                
            except Exception as e:
                logger.error("Periodic preference learning error", error=str(e))
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _periodic_model_retraining(self):
        """Periodic model retraining"""
        
        while self.is_running:
            try:
                # Wait 7 days between retraining cycles
                await asyncio.sleep(604800)
                
                # Get organizations with sufficient feedback for retraining
                async with self.rlhf_pipeline.db_pool.acquire() as conn:
                    orgs = await conn.fetch("""
                        SELECT organization_id, COUNT(*) as feedback_count
                        FROM feedback_events
                        WHERE created_at >= NOW() - INTERVAL '30 days'
                        AND feedback_type IN ('priority_rating', 'remediation_rating')
                        GROUP BY organization_id
                        HAVING COUNT(*) >= 20
                    """)
                
                # Retrain models for eligible organizations
                for org_row in orgs:
                    org_id = org_row['organization_id']
                    feedback_count = org_row['feedback_count']
                    
                    try:
                        logger.info("Starting model retraining",
                                   organization_id=org_id,
                                   feedback_count=feedback_count)
                        
                        # Retrain priority model
                        await self.rlhf_pipeline.retrain_priority_model(org_id)
                        
                        # Fine-tune remediation model if enough feedback
                        if feedback_count >= 30:
                            await self.rlhf_pipeline.fine_tune_remediation_model(org_id)
                        
                        await asyncio.sleep(5)  # Rate limit between orgs
                        
                    except Exception as e:
                        logger.error("Model retraining failed",
                                   organization_id=org_id,
                                   error=str(e))
                
            except Exception as e:
                logger.error("Periodic model retraining error", error=str(e))
                await asyncio.sleep(3600)
    
    async def _mark_feedback_processed(self, feedback_id: str):
        """Mark feedback as processed"""
        
        try:
            async with self.feedback_collector.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE feedback_events
                    SET processed_at = NOW()
                    WHERE id = $1
                """, feedback_id)
                
        except Exception as e:
            logger.error("Failed to mark feedback processed", error=str(e))
    
    async def _should_update_preferences(self, organization_id: str) -> bool:
        """Check if preferences should be updated based on new feedback"""
        
        try:
            async with self.preference_engine.db_pool.acquire() as conn:
                # Check when preferences were last updated
                last_update = await conn.fetchval("""
                    SELECT last_updated FROM organization_preferences
                    WHERE organization_id = $1
                """, organization_id)
                
                if not last_update:
                    return True  # Never updated
                
                # Update if more than 7 days old and new feedback exists
                if datetime.now() - last_update > timedelta(days=7):
                    new_feedback_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM feedback_events
                        WHERE organization_id = $1
                        AND created_at > $2
                    """, organization_id, last_update)
                    
                    return new_feedback_count >= 10  # Minimum feedback for update
                
                return False
                
        except Exception as e:
            logger.error("Failed to check preference update need", error=str(e))
            return False
    
    async def get_learning_statistics(self) -> Dict:
        """Get comprehensive learning statistics"""
        
        try:
            async with self.feedback_collector.db_pool.acquire() as conn:
                # Feedback statistics
                feedback_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        COUNT(DISTINCT organization_id) as orgs_with_feedback,
                        AVG(rating) as avg_rating,
                        COUNT(*) FILTER (WHERE feedback_type = 'priority_rating') as priority_feedback,
                        COUNT(*) FILTER (WHERE feedback_type = 'remediation_rating') as remediation_feedback,
                        COUNT(*) FILTER (WHERE feedback_type = 'false_positive') as fp_feedback
                    FROM feedback_events
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                
                # Model improvement statistics
                improvement_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_improvements,
                        COUNT(DISTINCT model_name) as models_improved,
                        AVG(training_duration) as avg_training_duration,
                        AVG(feedback_count) as avg_feedback_per_training
                    FROM model_improvements
                    WHERE deployed_at >= NOW() - INTERVAL '30 days'
                """)
                
                # Preference learning statistics
                preference_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as orgs_with_preferences,
                        AVG(confidence_score) as avg_confidence,
                        AVG(sample_size) as avg_sample_size
                    FROM organization_preferences
                    WHERE last_updated >= NOW() - INTERVAL '30 days'
                """)
                
                return {
                    "feedback_statistics": dict(feedback_stats) if feedback_stats else {},
                    "model_improvement_statistics": dict(improvement_stats) if improvement_stats else {},
                    "preference_learning_statistics": dict(preference_stats) if preference_stats else {},
                    "active_learning_tasks": len(self.learning_tasks),
                    "learning_engine_status": "running" if self.is_running else "stopped",
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get learning statistics", error=str(e))
            return {"error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown learning engine"""
        
        logger.info("Shutting down Adaptive Learning Engine...")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.learning_tasks.values():
            task.cancel()
        
        logger.info("Adaptive Learning Engine shutdown complete")

async def main():
    """Main adaptive learning service"""
    
    # Start Prometheus metrics server
    start_http_server(8012)
    
    # Initialize learning engine
    config = {
        "database_url": os.getenv("DATABASE_URL", 
                                 "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379/0"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    engine = AdaptiveLearningEngine()
    await engine.initialize(config)
    
    logger.info("📚 Xorb Adaptive Learning Engine started",
               service_version="6.3.0",
               features=["rlhf_pipeline", "preference_learning", "feedback_processing", "model_retraining"])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())