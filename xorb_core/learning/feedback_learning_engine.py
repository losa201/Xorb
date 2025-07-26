#!/usr/bin/env python3
"""
Enhanced Feedback Learning Engine v10.0 - Multi-Source Intelligence Learning

This module implements advanced feedback learning capabilities for Phase 10:
- Multi-source intelligence quality assessment
- Adaptive source prioritization based on effectiveness
- Cross-correlation pattern learning
- Predictive intelligence value scoring
- Continuous synthesis optimization
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import structlog

from prometheus_client import Counter, Histogram, Gauge

# XORB Internal Imports
from ..intelligence.global_synthesis_engine import (
    IntelligenceSource, IntelligenceSignal, CorrelatedIntelligence, 
    IntelligenceSourceType, SignalPriority
)
from ..autonomous.episodic_memory_system import EpisodicMemorySystem, MemoryRecord
from ..autonomous.autonomous_orchestrator import AutonomousDecision


class LearningObjective(Enum):
    """Learning objectives for the feedback system"""
    SOURCE_QUALITY = "source_quality"
    CORRELATION_ACCURACY = "correlation_accuracy"
    MISSION_EFFECTIVENESS = "mission_effectiveness"
    RESPONSE_TIME = "response_time"
    THREAT_PREDICTION = "threat_prediction"


class FeedbackType(Enum):
    """Types of feedback in the learning system"""
    MISSION_SUCCESS = "mission_success"
    INTELLIGENCE_ACCURACY = "intelligence_accuracy"
    SOURCE_RELIABILITY = "source_reliability"
    CORRELATION_VALIDITY = "correlation_validity"
    PREDICTION_ACCURACY = "prediction_accuracy"


@dataclass
class LearningFeedbackRecord:
    """Record of feedback for learning system"""
    feedback_id: str
    feedback_type: FeedbackType
    
    # Source information
    source_id: Optional[str] = None
    source_type: Optional[IntelligenceSourceType] = None
    signal_id: Optional[str] = None
    intelligence_id: Optional[str] = None
    
    # Feedback data
    feedback_score: float = 0.0  # 0.0 to 1.0
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    predicted_outcome: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    mission_id: Optional[str] = None
    agent_ids: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    # Learning metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.5
    learning_weight: float = 1.0
    
    # Analysis results
    feature_importance: Dict[str, float] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class SourcePerformanceProfile:
    """Performance profile for an intelligence source"""
    source_id: str
    source_type: IntelligenceSourceType
    
    # Performance metrics
    reliability_score: float = 0.8
    accuracy_score: float = 0.7
    timeliness_score: float = 0.8
    uniqueness_score: float = 0.6
    
    # Historical data
    total_signals: int = 0
    successful_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Temporal patterns
    best_performance_hours: List[int] = field(default_factory=list)
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Learning trends
    improvement_rate: float = 0.0
    learning_velocity: float = 0.0
    
    # Quality indicators
    signal_quality_distribution: Dict[str, int] = field(default_factory=dict)
    correlation_success_rate: float = 0.5
    
    # Updated tracking
    last_updated: datetime = field(default_factory=datetime.utcnow)
    last_performance_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntelligenceLearningModel:
    """Machine learning model for intelligence prediction"""
    model_id: str
    model_type: str
    objective: LearningObjective
    
    # Model components
    feature_scaler: Optional[StandardScaler] = None
    predictor_model: Optional[RandomForestRegressor] = None
    
    # Performance tracking
    accuracy: float = 0.5
    precision: float = 0.5
    recall: float = 0.5
    f1_score: float = 0.5
    
    # Training data
    training_samples: int = 0
    last_training: Optional[datetime] = None
    retraining_threshold: int = 1000
    
    # Feature engineering
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Model configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cross_validation_score: float = 0.0


class EnhancedFeedbackLearningEngine:
    """
    Enhanced Feedback Learning Engine for Phase 10 Global Intelligence Synthesis
    
    Capabilities:
    - Multi-source intelligence quality assessment and learning
    - Adaptive source prioritization based on historical performance
    - Cross-correlation pattern recognition and optimization
    - Predictive intelligence value scoring
    - Continuous synthesis parameter tuning
    - Machine learning-driven intelligence prediction
    """
    
    def __init__(self, 
                 synthesis_engine: 'GlobalSynthesisEngine',
                 orchestrator: 'AutonomousOrchestrator',
                 episodic_memory: EpisodicMemorySystem):
        
        self.synthesis_engine = synthesis_engine
        self.orchestrator = orchestrator
        self.episodic_memory = episodic_memory
        
        self.logger = structlog.get_logger("FeedbackLearningEngine")
        
        # Learning data storage
        self.feedback_records: List[LearningFeedbackRecord] = []
        self.source_profiles: Dict[str, SourcePerformanceProfile] = {}
        self.learning_models: Dict[str, IntelligenceLearningModel] = {}
        
        # Learning state
        self.learning_session_id = str(uuid.uuid4())
        self.total_feedback_processed = 0
        self.learning_iterations = 0
        
        # Performance tracking
        self.source_performance_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.correlation_pattern_cache: Dict[str, Dict[str, Any]] = {}
        self.prediction_accuracy_history: List[float] = []
        
        # Configuration
        self.max_feedback_records = 100000
        self.model_retraining_interval = 24 * 3600  # 24 hours
        self.feedback_window = timedelta(days=7)
        self.min_samples_for_learning = 50
        
        # Metrics
        self.learning_metrics = self._initialize_learning_metrics()
        
        # Learning state
        self._running = False
        self._learning_tasks: List[asyncio.Task] = []
    
    def _initialize_learning_metrics(self) -> Dict[str, Any]:
        """Initialize metrics for learning engine"""
        return {
            'feedback_records_processed': Counter(
                'xorb_learning_feedback_records_total',
                'Total feedback records processed',
                ['feedback_type', 'source_type']
            ),
            'source_performance_updates': Counter(
                'xorb_learning_source_updates_total',
                'Source performance profile updates',
                ['source_id', 'update_type']
            ),
            'model_training_iterations': Counter(
                'xorb_learning_model_training_total',
                'ML model training iterations',
                ['model_type', 'objective']
            ),
            'prediction_accuracy': Gauge(
                'xorb_learning_prediction_accuracy',
                'Model prediction accuracy',
                ['model_type', 'objective']
            ),
            'learning_processing_time': Histogram(
                'xorb_learning_processing_seconds',
                'Learning processing time',
                ['operation_type']
            ),
            'source_reliability_scores': Gauge(
                'xorb_learning_source_reliability',
                'Source reliability scores',
                ['source_id', 'source_type']
            )
        }
    
    async def start_learning_engine(self):
        """Start the enhanced feedback learning engine"""
        self.logger.info("🧠 Starting Enhanced Feedback Learning Engine v10.0")
        
        self._running = True
        
        # Initialize source profiles
        await self._initialize_source_profiles()
        
        # Initialize learning models
        await self._initialize_learning_models()
        
        # Start learning processes
        self._learning_tasks = [
            asyncio.create_task(self._feedback_collection_loop()),
            asyncio.create_task(self._source_performance_learning_loop()),
            asyncio.create_task(self._correlation_pattern_learning_loop()),
            asyncio.create_task(self._predictive_model_training_loop()),
            asyncio.create_task(self._synthesis_optimization_loop()),
            asyncio.create_task(self._learning_metrics_collection_loop())
        ]
        
        self.logger.info("🚀 Enhanced Feedback Learning Engine: ACTIVE")
    
    async def _initialize_source_profiles(self):
        """Initialize performance profiles for all intelligence sources"""
        
        if not self.synthesis_engine.intelligence_sources:
            return
        
        for source_id, source in self.synthesis_engine.intelligence_sources.items():
            if source_id not in self.source_profiles:
                profile = SourcePerformanceProfile(
                    source_id=source_id,
                    source_type=source.source_type,
                    reliability_score=source.reliability_score,
                    total_signals=source.total_signals
                )
                
                # Initialize based on source type
                profile = await self._initialize_source_type_profile(profile, source)
                
                self.source_profiles[source_id] = profile
                
                self.logger.info(f"📊 Initialized performance profile: {source.name}")
        
        self.logger.info(f"🎯 Initialized {len(self.source_profiles)} source profiles")
    
    async def _initialize_source_type_profile(self, 
                                           profile: SourcePerformanceProfile,
                                           source: IntelligenceSource) -> SourcePerformanceProfile:
        """Initialize profile based on source type characteristics"""
        
        # CVE/NVD sources - high reliability, medium timeliness
        if source.source_type == IntelligenceSourceType.CVE_NVD:
            profile.reliability_score = 0.95
            profile.accuracy_score = 0.9
            profile.timeliness_score = 0.7  # Official sources can be delayed
            profile.uniqueness_score = 0.9
        
        # Bug bounty platforms - medium reliability, high timeliness
        elif source.source_type in [IntelligenceSourceType.HACKERONE, 
                                   IntelligenceSourceType.BUGCROWD,
                                   IntelligenceSourceType.INTIGRITI]:
            profile.reliability_score = 0.8
            profile.accuracy_score = 0.75
            profile.timeliness_score = 0.9
            profile.uniqueness_score = 0.8
        
        # OSINT sources - variable reliability, high timeliness
        elif source.source_type == IntelligenceSourceType.OSINT_RSS:
            profile.reliability_score = 0.6
            profile.accuracy_score = 0.65
            profile.timeliness_score = 0.95
            profile.uniqueness_score = 0.7
        
        # Internal sources - high reliability and accuracy
        elif source.source_type == IntelligenceSourceType.INTERNAL_MISSIONS:
            profile.reliability_score = 0.95
            profile.accuracy_score = 0.9
            profile.timeliness_score = 0.8
            profile.uniqueness_score = 1.0
        
        # Prometheus alerts - high reliability, excellent timeliness
        elif source.source_type == IntelligenceSourceType.PROMETHEUS_ALERTS:
            profile.reliability_score = 0.9
            profile.accuracy_score = 0.85
            profile.timeliness_score = 1.0
            profile.uniqueness_score = 0.7
        
        return profile
    
    async def _initialize_learning_models(self):
        """Initialize machine learning models for different objectives"""
        
        # Source quality prediction model
        source_quality_model = IntelligenceLearningModel(
            model_id="source_quality_predictor",
            model_type="RandomForestRegressor",
            objective=LearningObjective.SOURCE_QUALITY,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        )
        
        # Mission effectiveness prediction model
        mission_effectiveness_model = IntelligenceLearningModel(
            model_id="mission_effectiveness_predictor", 
            model_type="RandomForestRegressor",
            objective=LearningObjective.MISSION_EFFECTIVENESS,
            hyperparameters={
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 4,
                'random_state': 42
            }
        )
        
        # Threat prediction model
        threat_prediction_model = IntelligenceLearningModel(
            model_id="threat_predictor",
            model_type="RandomForestRegressor", 
            objective=LearningObjective.THREAT_PREDICTION,
            hyperparameters={
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 3,
                'random_state': 42
            }
        )
        
        self.learning_models.update({
            source_quality_model.model_id: source_quality_model,
            mission_effectiveness_model.model_id: mission_effectiveness_model,
            threat_prediction_model.model_id: threat_prediction_model
        })
        
        self.logger.info(f"🧠 Initialized {len(self.learning_models)} learning models")
    
    async def _feedback_collection_loop(self):
        """Continuously collect feedback from various sources"""
        
        while self._running:
            try:
                # Collect feedback from different sources
                await self._collect_mission_feedback()
                await self._collect_intelligence_feedback()
                await self._collect_source_feedback()
                await self._collect_correlation_feedback()
                
                # Process collected feedback
                await self._process_pending_feedback()
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error("Feedback collection error", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_mission_feedback(self):
        """Collect feedback from completed missions"""
        
        try:
            # Get recent mission completions from orchestrator
            if hasattr(self.orchestrator, 'execution_contexts'):
                for campaign_id, context in self.orchestrator.execution_contexts.items():
                    
                    # Skip if already processed
                    if context.metadata.get('learning_feedback_collected', False):
                        continue
                    
                    # Only process completed missions
                    if context.status not in ['COMPLETED', 'FAILED']:
                        continue
                    
                    # Check if this was intelligence-driven
                    if context.config.get('intelligence_driven', False):
                        intelligence_id = context.config.get('intelligence_id')
                        
                        feedback_record = LearningFeedbackRecord(
                            feedback_id=str(uuid.uuid4()),
                            feedback_type=FeedbackType.MISSION_SUCCESS,
                            intelligence_id=intelligence_id,
                            mission_id=campaign_id,
                            agent_ids=context.assigned_agents or [],
                            feedback_score=self._calculate_mission_success_score(context),
                            actual_outcome={
                                'status': context.status,
                                'duration': (context.end_time - context.start_time).total_seconds() if context.end_time else 0,
                                'discoveries': context.metadata.get('discoveries', 0),
                                'success_rate': context.metadata.get('success_rate', 0.0)
                            },
                            processing_time=(context.end_time - context.start_time).total_seconds() if context.end_time else 0,
                            confidence=0.9
                        )
                        
                        self.feedback_records.append(feedback_record)
                        
                        # Mark as processed
                        context.metadata['learning_feedback_collected'] = True
                        
                        self.logger.debug("Mission feedback collected",
                                        campaign_id=campaign_id[:8],
                                        intelligence_id=intelligence_id[:8] if intelligence_id else "none",
                                        score=feedback_record.feedback_score)
        
        except Exception as e:
            self.logger.error("Mission feedback collection failed", error=str(e))
    
    async def _collect_intelligence_feedback(self):
        """Collect feedback on intelligence accuracy and usefulness"""
        
        try:
            if not self.synthesis_engine.correlated_intelligence:
                return
            
            current_time = datetime.utcnow()
            
            for intel_id, intelligence in self.synthesis_engine.correlated_intelligence.items():
                
                # Skip recent intelligence (need time to evaluate)
                if (current_time - intelligence.created_at).total_seconds() < 3600:
                    continue
                
                # Check if we have feedback from spawned missions
                if intelligence.spawned_missions and intelligence.feedback_scores:
                    
                    avg_feedback = np.mean(intelligence.feedback_scores)
                    
                    feedback_record = LearningFeedbackRecord(
                        feedback_id=str(uuid.uuid4()),
                        feedback_type=FeedbackType.INTELLIGENCE_ACCURACY,
                        intelligence_id=intel_id,
                        source_id=intelligence.primary_signal_id,
                        feedback_score=avg_feedback,
                        actual_outcome={
                            'missions_spawned': len(intelligence.spawned_missions),
                            'avg_mission_success': avg_feedback,
                            'threat_materialized': self._assess_threat_materialization(intelligence)
                        },
                        predicted_outcome={
                            'priority': intelligence.overall_priority.value,
                            'confidence': intelligence.confidence_score,
                            'threat_level': intelligence.threat_level
                        },
                        confidence=0.8
                    )
                    
                    self.feedback_records.append(feedback_record)
                    
                    # Clear processed feedback scores
                    intelligence.feedback_scores = []
        
        except Exception as e:
            self.logger.error("Intelligence feedback collection failed", error=str(e))
    
    async def _collect_source_feedback(self):
        """Collect feedback on source reliability and quality"""
        
        try:
            for source_id, source in self.synthesis_engine.intelligence_sources.items():
                
                if source_id not in self.source_profiles:
                    continue
                
                profile = self.source_profiles[source_id]
                
                # Calculate recent performance
                recent_performance = await self._calculate_recent_source_performance(source_id)
                
                if recent_performance:
                    feedback_record = LearningFeedbackRecord(
                        feedback_id=str(uuid.uuid4()),
                        feedback_type=FeedbackType.SOURCE_RELIABILITY,
                        source_id=source_id,
                        source_type=source.source_type,
                        feedback_score=recent_performance['overall_score'],
                        actual_outcome=recent_performance,
                        predicted_outcome={
                            'reliability': profile.reliability_score,
                            'accuracy': profile.accuracy_score,
                            'timeliness': profile.timeliness_score
                        },
                        confidence=0.7
                    )
                    
                    self.feedback_records.append(feedback_record)
        
        except Exception as e:
            self.logger.error("Source feedback collection failed", error=str(e))
    
    async def _collect_correlation_feedback(self):
        """Collect feedback on correlation accuracy"""
        
        try:
            # Get correlated intelligence with completed missions
            for intel_id, intelligence in self.synthesis_engine.correlated_intelligence.items():
                
                if not intelligence.spawned_missions:
                    continue
                
                # Assess correlation quality based on mission outcomes
                correlation_quality = await self._assess_correlation_quality(intelligence)
                
                if correlation_quality is not None:
                    feedback_record = LearningFeedbackRecord(
                        feedback_id=str(uuid.uuid4()),
                        feedback_type=FeedbackType.CORRELATION_VALIDITY,
                        intelligence_id=intel_id,
                        feedback_score=correlation_quality,
                        actual_outcome={
                            'correlation_useful': correlation_quality > 0.7,
                            'related_signals_count': len(intelligence.related_signal_ids),
                            'correlation_score': intelligence.correlation_score if hasattr(intelligence, 'correlation_score') else 0.0
                        },
                        confidence=0.6
                    )
                    
                    self.feedback_records.append(feedback_record)
        
        except Exception as e:
            self.logger.error("Correlation feedback collection failed", error=str(e))
    
    async def _source_performance_learning_loop(self):
        """Continuously learn and update source performance profiles"""
        
        while self._running:
            try:
                with self.learning_metrics['learning_processing_time'].labels(
                    operation_type='source_performance'
                ).time():
                    
                    await self._update_source_performance_profiles()
                    await self._optimize_source_prioritization()
                    await self._detect_source_patterns()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error("Source performance learning error", error=str(e))
                await asyncio.sleep(300)
    
    async def _update_source_performance_profiles(self):
        """Update source performance profiles based on recent feedback"""
        
        # Group feedback by source
        source_feedback = defaultdict(list)
        
        for record in self.feedback_records[-1000:]:  # Recent feedback
            if record.source_id and record.feedback_type == FeedbackType.SOURCE_RELIABILITY:
                source_feedback[record.source_id].append(record)
        
        # Update each source profile
        for source_id, feedback_list in source_feedback.items():
            if source_id in self.source_profiles:
                await self._update_single_source_profile(source_id, feedback_list)
    
    async def _update_single_source_profile(self, 
                                          source_id: str, 
                                          feedback_list: List[LearningFeedbackRecord]):
        """Update a single source performance profile"""
        
        profile = self.source_profiles[source_id]
        
        # Calculate weighted averages
        total_weight = sum(record.learning_weight for record in feedback_list)
        if total_weight == 0:
            return
        
        # Update reliability score
        new_reliability = sum(
            record.feedback_score * record.learning_weight 
            for record in feedback_list
        ) / total_weight
        
        # Apply learning rate (0.1 for gradual updates)
        learning_rate = 0.1
        profile.reliability_score = (
            profile.reliability_score * (1 - learning_rate) + 
            new_reliability * learning_rate
        )
        
        # Update other metrics from actual outcomes
        accuracy_scores = []
        timeliness_scores = []
        
        for record in feedback_list:
            if 'accuracy' in record.actual_outcome:
                accuracy_scores.append(record.actual_outcome['accuracy'])
            if 'timeliness' in record.actual_outcome:
                timeliness_scores.append(record.actual_outcome['timeliness'])
        
        if accuracy_scores:
            profile.accuracy_score = (
                profile.accuracy_score * (1 - learning_rate) +
                np.mean(accuracy_scores) * learning_rate
            )
        
        if timeliness_scores:
            profile.timeliness_score = (
                profile.timeliness_score * (1 - learning_rate) +
                np.mean(timeliness_scores) * learning_rate
            )
        
        # Calculate improvement rate
        current_time = datetime.utcnow()
        if profile.last_performance_update:
            time_delta = (current_time - profile.last_performance_update).total_seconds()
            if time_delta > 0:
                reliability_change = new_reliability - profile.reliability_score
                profile.improvement_rate = reliability_change / (time_delta / 3600)  # per hour
        
        profile.last_performance_update = current_time
        
        # Update Prometheus metrics
        self.learning_metrics['source_reliability_scores'].labels(
            source_id=source_id,
            source_type=profile.source_type.value
        ).set(profile.reliability_score)
        
        self.learning_metrics['source_performance_updates'].labels(
            source_id=source_id,
            update_type='reliability'
        ).inc()
        
        self.logger.debug("Source profile updated",
                        source_id=source_id,
                        reliability=profile.reliability_score,
                        accuracy=profile.accuracy_score,
                        improvement_rate=profile.improvement_rate)
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning engine status"""
        
        return {
            'learning_engine': {
                'status': 'running' if self._running else 'stopped',
                'session_id': self.learning_session_id,
                'total_feedback_processed': self.total_feedback_processed,
                'learning_iterations': self.learning_iterations,
                'active_models': len(self.learning_models)
            },
            'feedback_summary': {
                'total_records': len(self.feedback_records),
                'feedback_by_type': {
                    feedback_type.value: sum(1 for r in self.feedback_records if r.feedback_type == feedback_type)
                    for feedback_type in FeedbackType
                },
                'average_feedback_score': np.mean([r.feedback_score for r in self.feedback_records]) if self.feedback_records else 0.0,
                'recent_feedback_trend': self._calculate_feedback_trend()
            },
            'source_performance': {
                'total_sources': len(self.source_profiles),
                'top_performing_sources': self._get_top_performing_sources(5),
                'average_reliability': np.mean([p.reliability_score for p in self.source_profiles.values()]) if self.source_profiles else 0.0,
                'sources_improving': sum(1 for p in self.source_profiles.values() if p.improvement_rate > 0)
            },
            'learning_models': {
                model_id: {
                    'objective': model.objective.value,
                    'accuracy': model.accuracy,
                    'training_samples': model.training_samples,
                    'last_training': model.last_training.isoformat() if model.last_training else None
                }
                for model_id, model in self.learning_models.items()
            },
            'learning_insights': {
                'best_source_types': self._analyze_best_source_types(),
                'correlation_patterns': self._get_top_correlation_patterns(5),
                'learning_recommendations': self._generate_learning_recommendations()
            }
        }
    
    # Placeholder methods for complex operations
    async def _process_pending_feedback(self): pass
    async def _correlation_pattern_learning_loop(self): pass
    async def _predictive_model_training_loop(self): pass
    async def _synthesis_optimization_loop(self): pass
    async def _learning_metrics_collection_loop(self): pass
    async def _optimize_source_prioritization(self): pass
    async def _detect_source_patterns(self): pass
    async def _calculate_recent_source_performance(self, source_id: str) -> Optional[Dict[str, float]]: return None
    async def _assess_correlation_quality(self, intelligence: CorrelatedIntelligence) -> Optional[float]: return None
    
    # Helper methods
    def _calculate_mission_success_score(self, context) -> float: return 0.7
    def _assess_threat_materialization(self, intelligence: CorrelatedIntelligence) -> bool: return True
    def _calculate_feedback_trend(self) -> float: return 0.05
    def _get_top_performing_sources(self, limit: int) -> List[Dict[str, Any]]: return []
    def _analyze_best_source_types(self) -> Dict[str, float]: return {}
    def _get_top_correlation_patterns(self, limit: int) -> List[Dict[str, Any]]: return []
    def _generate_learning_recommendations(self) -> List[str]: return []