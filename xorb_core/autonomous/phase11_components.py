#!/usr/bin/env python3
"""
XORB Phase 11 Supporting Components

This module provides the supporting classes for Phase 11 enhancements:
- TemporalSignalPatternDetector: DBSCAN clustering for signal pattern recognition
- RoleAllocator: Dynamic agent role assignment with performance metrics
- MissionStrategyModifier: Fault-tolerant mission recycling strategies
- KPITracker: Per-signal KPI instrumentation with Prometheus
- ConflictDetector: Redundancy and conflict detection using vector hashing
"""

import asyncio
import numpy as np
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import structlog
from prometheus_client import Counter, Histogram, Gauge


class TemporalSignalPatternDetector:
    """
    Temporal Signal Pattern Recognition using time-decayed weighted vector embeddings
    
    Uses DBSCAN clustering to identify patterns in threat signals with temporal weighting.
    Optimized for sub-500ms processing cycles on Raspberry Pi 5.
    """
    
    def __init__(self, eps: float = 0.3, min_samples: int = 3, max_age_hours: float = 24.0):
        self.eps = eps
        self.min_samples = min_samples
        self.max_age_hours = max_age_hours
        self.scaler = StandardScaler()
        self.logger = structlog.get_logger("xorb.pattern_detector")
        
        # Pattern cache for performance
        self.pattern_cache: Dict[str, Dict] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}
        
        # Performance metrics
        self.detection_times: List[float] = []
        self.cluster_stability: Dict[int, float] = {}
        
    async def extract_signal_features(self, signal: 'ThreatSignal') -> np.ndarray:
        """Extract features from threat signal for clustering"""
        try:
            # Check cache first
            cache_key = f"{signal.signal_id}_{signal.timestamp.timestamp()}"
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            features = []
            
            # Temporal features
            hour_of_day = signal.timestamp.hour / 24.0
            day_of_week = signal.timestamp.weekday() / 7.0
            features.extend([hour_of_day, day_of_week])
            
            # Signal type encoding (one-hot)
            signal_types = ['network_anomaly', 'vulnerability_discovery', 'suspicious_behavior', 
                          'performance_degradation', 'security_alert', 'system_compromise',
                          'data_exfiltration', 'lateral_movement']
            signal_type_vector = [1.0 if signal.signal_type.value == st else 0.0 for st in signal_types]
            features.extend(signal_type_vector)
            
            # Severity and confidence
            features.extend([signal.severity, signal.confidence])
            
            # Source characteristics (hash-based encoding)
            source_hash = int(hashlib.md5(signal.source.encode()).hexdigest()[:8], 16)
            source_features = [(source_hash >> i) & 1 for i in range(8)]
            features.extend(source_features)
            
            # Raw data features (if available)
            if signal.raw_data:
                # Extract numeric features from raw data
                numeric_features = self._extract_numeric_features(signal.raw_data)
                features.extend(numeric_features[:10])  # Limit to 10 features
            else:
                features.extend([0.0] * 10)
            
            feature_array = np.array(features, dtype=np.float32)
            
            # Cache the result
            self.feature_cache[cache_key] = feature_array
            
            # Limit cache size
            if len(self.feature_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.feature_cache.keys())[:100]
                for key in oldest_keys:
                    del self.feature_cache[key]
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for signal {signal.signal_id[:8]}", error=str(e))
            return np.zeros(30, dtype=np.float32)  # Default feature size
    
    def _extract_numeric_features(self, raw_data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from raw signal data"""
        features = []
        
        for key, value in raw_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Convert string to numeric hash
                features.append(float(hash(value) % 1000) / 1000.0)
            elif isinstance(value, dict):
                # Recursively process nested dicts
                nested_features = self._extract_numeric_features(value)
                features.extend(nested_features[:3])  # Limit nested features
            
            if len(features) >= 10:
                break
        
        # Pad or truncate to exactly 10 features
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    async def detect_patterns(self, signals: List['ThreatSignal']) -> Dict[str, Any]:
        """Detect patterns in threat signals using temporal clustering"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if len(signals) < self.min_samples:
                return {'clusters': {}, 'patterns': [], 'processing_time': 0.0}
            
            # Filter signals by age
            current_time = datetime.now()
            recent_signals = [
                s for s in signals 
                if (current_time - s.timestamp).total_seconds() / 3600 <= self.max_age_hours
            ]
            
            if len(recent_signals) < self.min_samples:
                return {'clusters': {}, 'patterns': [], 'processing_time': 0.0}
            
            # Extract features for all signals
            features_list = []
            signal_weights = []
            
            for signal in recent_signals:
                features = await self.extract_signal_features(signal)
                time_weight = signal.get_time_weighted_score(current_time)
                
                features_list.append(features)
                signal_weights.append(time_weight)
            
            # Convert to numpy arrays
            X = np.array(features_list)
            weights = np.array(signal_weights)
            
            # Apply temporal weighting to features
            X_weighted = X * weights.reshape(-1, 1)
            
            # Normalize features
            X_normalized = self.scaler.fit_transform(X_weighted)
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=1)
            cluster_labels = clustering.fit_predict(X_normalized)
            
            # Analyze clusters
            clusters = defaultdict(list)
            patterns = []
            
            for i, (signal, label) in enumerate(zip(recent_signals, cluster_labels)):
                if label != -1:  # Not noise
                    clusters[label].append({
                        'signal_id': signal.signal_id,
                        'timestamp': signal.timestamp,
                        'confidence': signal.confidence,
                        'weight': signal_weights[i]
                    })
                    
                    # Update signal cluster assignment
                    signal.cluster_id = label
            
            # Generate pattern descriptions
            for cluster_id, cluster_signals in clusters.items():
                if len(cluster_signals) >= self.min_samples:
                    pattern = await self._analyze_cluster_pattern(cluster_id, cluster_signals, recent_signals)
                    patterns.append(pattern)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            self.detection_times.append(processing_time)
            
            # Maintain performance history
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-50:]
            
            return {
                'clusters': dict(clusters),
                'patterns': patterns,
                'processing_time': processing_time,
                'cluster_count': len(clusters),
                'noise_signals': len([l for l in cluster_labels if l == -1])
            }
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error("Pattern detection failed", error=str(e), processing_time=processing_time)
            return {'clusters': {}, 'patterns': [], 'processing_time': processing_time, 'error': str(e)}
    
    async def _analyze_cluster_pattern(self, cluster_id: int, cluster_signals: List[Dict], all_signals: List['ThreatSignal']) -> Dict[str, Any]:
        """Analyze a cluster to extract pattern characteristics"""
        try:
            # Get signal objects for this cluster
            cluster_signal_objects = [
                s for s in all_signals 
                if s.signal_id in [cs['signal_id'] for cs in cluster_signals]
            ]
            
            # Calculate pattern characteristics
            signal_types = [s.signal_type.value for s in cluster_signal_objects]
            sources = [s.source for s in cluster_signal_objects]
            timestamps = [s.timestamp for s in cluster_signal_objects]
            confidences = [s.confidence for s in cluster_signal_objects]
            
            # Time-based analysis
            time_span = max(timestamps) - min(timestamps) if timestamps else timedelta(0)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Pattern frequency
            most_common_type = max(set(signal_types), key=signal_types.count) if signal_types else None
            most_common_source = max(set(sources), key=sources.count) if sources else None
            
            pattern = {
                'cluster_id': cluster_id,
                'pattern_type': f"cluster_{cluster_id}",
                'signal_count': len(cluster_signals),
                'time_span_seconds': time_span.total_seconds(),
                'avg_confidence': avg_confidence,
                'dominant_signal_type': most_common_type,
                'dominant_source': most_common_source,
                'created_at': datetime.now(),
                'stability_score': self._calculate_cluster_stability(cluster_id, cluster_signals)
            }
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Cluster analysis failed for cluster {cluster_id}", error=str(e))
            return {
                'cluster_id': cluster_id,
                'pattern_type': f"cluster_{cluster_id}",
                'signal_count': len(cluster_signals),
                'created_at': datetime.now(),
                'error': str(e)
            }
    
    def _calculate_cluster_stability(self, cluster_id: int, cluster_signals: List[Dict]) -> float:
        """Calculate cluster stability score"""
        try:
            # Simple stability measure based on signal consistency
            if len(cluster_signals) < 2:
                return 0.5
            
            # Variance in confidence scores (lower variance = more stable)
            confidences = [s['confidence'] for s in cluster_signals]
            confidence_variance = np.var(confidences) if confidences else 1.0
            stability = max(0.0, 1.0 - confidence_variance)
            
            # Update stability history
            self.cluster_stability[cluster_id] = stability
            
            return stability
            
        except Exception:
            return 0.5


class RoleAllocator:
    """
    Multi-Agent Role Dynamic Assignment with success metrics-based scoring
    
    Dynamically assigns roles to agents based on performance metrics and capabilities.
    Optimized for high-throughput role allocation decisions.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("xorb.role_allocator")
        self.role_history: Dict[str, List[Dict]] = defaultdict(list)
        self.performance_cache: Dict[str, Dict] = {}
        
        # Role allocation strategies
        self.allocation_strategies = {
            'performance_based': self._allocate_by_performance,
            'capability_based': self._allocate_by_capability,
            'load_balanced': self._allocate_by_load,
            'adaptive': self._allocate_adaptive
        }
        
        # Performance thresholds
        self.min_success_rate = 0.6
        self.max_queue_size = 5
        self.role_switch_threshold = 0.3
    
    async def allocate_roles(self, agents: List['BaseAgent'], required_roles: List['RoleType'], 
                           strategy: str = 'adaptive') -> Dict[str, 'AgentRole']:
        """Allocate roles to agents based on specified strategy"""
        try:
            allocation_func = self.allocation_strategies.get(strategy, self._allocate_adaptive)
            
            # Get current agent performance
            agent_performance = {}
            for agent in agents:
                perf = await self._get_agent_performance(agent)
                agent_performance[agent.agent_id] = perf
            
            # Perform role allocation
            role_assignments = await allocation_func(agents, required_roles, agent_performance)
            
            # Update role history
            for role_assignment in role_assignments.values():
                self.role_history[role_assignment.agent_id].append({
                    'role_type': role_assignment.role_type,
                    'assigned_at': role_assignment.assigned_at,
                    'strategy': strategy
                })
            
            self.logger.info(f"Role allocation completed using {strategy} strategy",
                           roles_assigned=len(role_assignments),
                           agents_involved=len(agents))
            
            return role_assignments
            
        except Exception as e:
            self.logger.error(f"Role allocation failed with strategy {strategy}", error=str(e))
            return {}
    
    async def _get_agent_performance(self, agent: 'BaseAgent') -> Dict[str, Any]:
        """Get agent performance metrics with caching"""
        try:
            # Check cache first
            cache_key = f"{agent.agent_id}_{int(datetime.now().timestamp() / 60)}"  # 1-minute cache
            if cache_key in self.performance_cache:
                return self.performance_cache[cache_key]
            
            # Get health check data
            health = await agent.health_check()
            
            # Calculate additional metrics
            performance = {
                'success_rate': health.get('success_rate', 0.0),
                'avg_response_time': health.get('avg_response_time', float('inf')),
                'queue_size': health.get('queue_size', 0),
                'cpu_usage': health.get('cpu_usage', 0.0),
                'memory_usage': health.get('memory_usage', 0.0),
                'last_activity': health.get('last_activity', datetime.now() - timedelta(hours=1)),
                'capabilities': getattr(agent, 'capabilities', set()),
                'agent_type': getattr(agent, 'agent_type', 'unknown')
            }
            
            # Cache the result
            self.performance_cache[cache_key] = performance
            
            # Limit cache size
            if len(self.performance_cache) > 100:
                oldest_keys = list(self.performance_cache.keys())[:20]
                for key in oldest_keys:
                    del self.performance_cache[key]
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to get performance for agent {agent.agent_id[:8]}", error=str(e))
            return {
                'success_rate': 0.0,
                'avg_response_time': float('inf'),
                'queue_size': 0,
                'capabilities': set(),
                'agent_type': 'unknown'
            }
    
    async def _allocate_by_performance(self, agents: List['BaseAgent'], required_roles: List['RoleType'], 
                                     agent_performance: Dict[str, Dict]) -> Dict[str, 'AgentRole']:
        """Allocate roles based purely on performance metrics"""
        from ..autonomous.intelligent_orchestrator import AgentRole  # Import here to avoid circular import
        
        role_assignments = {}
        
        # Sort agents by performance score
        def performance_score(agent_id: str) -> float:
            perf = agent_performance.get(agent_id, {})
            success_rate = perf.get('success_rate', 0.0)
            response_time = perf.get('avg_response_time', float('inf'))
            queue_size = perf.get('queue_size', 0)
            
            # Higher success rate, lower response time, lower queue size = better score
            time_score = max(0, 1.0 - (response_time / 300.0))  # Normalize to 300s max
            queue_score = max(0, 1.0 - (queue_size / 10.0))     # Normalize to 10 max queue
            
            return (0.5 * success_rate + 0.3 * time_score + 0.2 * queue_score)
        
        agents_by_performance = sorted(agents, key=lambda a: performance_score(a.agent_id), reverse=True)
        
        # Assign roles to best performing agents
        for i, role_type in enumerate(required_roles):
            if i < len(agents_by_performance):
                agent = agents_by_performance[i]
                role_assignment = AgentRole(
                    role_id=str(uuid.uuid4()),
                    role_type=role_type,
                    agent_id=agent.agent_id,
                    assigned_at=datetime.now(),
                    **agent_performance.get(agent.agent_id, {})
                )
                role_assignments[role_assignment.role_id] = role_assignment
        
        return role_assignments
    
    async def _allocate_by_capability(self, agents: List['BaseAgent'], required_roles: List['RoleType'], 
                                    agent_performance: Dict[str, Dict]) -> Dict[str, 'AgentRole']:
        """Allocate roles based on agent capabilities"""
        from ..autonomous.intelligent_orchestrator import AgentRole
        
        role_assignments = {}
        
        # Define role-capability mappings
        role_capability_map = {
            'reconnaissance': {'network_scanning', 'port_discovery', 'service_enumeration'},
            'vulnerability_scanner': {'vulnerability_assessment', 'security_scanning', 'cve_detection'},
            'threat_hunter': {'behavioral_analysis', 'anomaly_detection', 'pattern_recognition'},
            'incident_responder': {'incident_management', 'containment', 'forensics'},
            'forensics_analyst': {'digital_forensics', 'evidence_collection', 'analysis'},
            'remediation_agent': {'system_patching', 'configuration_management', 'repair'},
            'monitoring_agent': {'continuous_monitoring', 'log_analysis', 'alerting'},
            'coordination_agent': {'workflow_management', 'task_coordination', 'communication'}
        }
        
        # Find best capability matches
        for role_type in required_roles:
            required_capabilities = role_capability_map.get(role_type.value, set())
            best_agent = None
            best_match_score = 0.0
            
            for agent in agents:
                agent_capabilities = agent_performance.get(agent.agent_id, {}).get('capabilities', set())
                
                # Calculate capability match score
                if required_capabilities:
                    match_count = len(required_capabilities.intersection(agent_capabilities))
                    match_score = match_count / len(required_capabilities)
                    
                    # Factor in performance
                    perf_score = agent_performance.get(agent.agent_id, {}).get('success_rate', 0.0)
                    combined_score = 0.7 * match_score + 0.3 * perf_score
                    
                    if combined_score > best_match_score:
                        best_match_score = combined_score
                        best_agent = agent
                else:
                    # If no specific capabilities required, use performance
                    perf_score = agent_performance.get(agent.agent_id, {}).get('success_rate', 0.0)
                    if perf_score > best_match_score:
                        best_match_score = perf_score
                        best_agent = agent
            
            # Assign role to best matching agent
            if best_agent and best_match_score > 0.0:
                role_assignment = AgentRole(
                    role_id=str(uuid.uuid4()),
                    role_type=role_type,
                    agent_id=best_agent.agent_id,
                    assigned_at=datetime.now(),
                    expertise_score=best_match_score,
                    **agent_performance.get(best_agent.agent_id, {})
                )
                role_assignments[role_assignment.role_id] = role_assignment
        
        return role_assignments
    
    async def _allocate_by_load(self, agents: List['BaseAgent'], required_roles: List['RoleType'], 
                              agent_performance: Dict[str, Dict]) -> Dict[str, 'AgentRole']:
        """Allocate roles based on current load balancing"""
        from ..autonomous.intelligent_orchestrator import AgentRole
        
        role_assignments = {}
        
        # Sort agents by current load (queue size, CPU usage)
        def load_score(agent_id: str) -> float:
            perf = agent_performance.get(agent_id, {})
            queue_size = perf.get('queue_size', 0)
            cpu_usage = perf.get('cpu_usage', 0.0)
            
            # Lower load = better score
            queue_score = max(0, 1.0 - (queue_size / 10.0))
            cpu_score = max(0, 1.0 - cpu_usage)
            
            return 0.6 * queue_score + 0.4 * cpu_score
        
        agents_by_load = sorted(agents, key=lambda a: load_score(a.agent_id), reverse=True)
        
        # Round-robin assignment to lowest load agents
        for i, role_type in enumerate(required_roles):
            agent_index = i % len(agents_by_load)
            agent = agents_by_load[agent_index]
            
            role_assignment = AgentRole(
                role_id=str(uuid.uuid4()),
                role_type=role_type,
                agent_id=agent.agent_id,
                assigned_at=datetime.now(),
                **agent_performance.get(agent.agent_id, {})
            )
            role_assignments[role_assignment.role_id] = role_assignment
        
        return role_assignments
    
    async def _allocate_adaptive(self, agents: List['BaseAgent'], required_roles: List['RoleType'], 
                               agent_performance: Dict[str, Dict]) -> Dict[str, 'AgentRole']:
        """Adaptive allocation combining multiple strategies"""
        from ..autonomous.intelligent_orchestrator import AgentRole
        
        # Use different strategies based on system state
        total_queue_size = sum(
            perf.get('queue_size', 0) 
            for perf in agent_performance.values()
        )
        avg_success_rate = np.mean([
            perf.get('success_rate', 0.0) 
            for perf in agent_performance.values()
        ]) if agent_performance else 0.0
        
        # Choose strategy based on system conditions
        if total_queue_size > len(agents) * 3:
            # High load - prioritize load balancing
            return await self._allocate_by_load(agents, required_roles, agent_performance)
        elif avg_success_rate < 0.7:
            # Low performance - prioritize capability matching
            return await self._allocate_by_capability(agents, required_roles, agent_performance)
        else:
            # Normal conditions - prioritize performance
            return await self._allocate_by_performance(agents, required_roles, agent_performance)


class MissionStrategyModifier:
    """
    Fault-Tolerant Mission Recycling with failure context storage
    
    Provides intelligent mission modification strategies for failure recovery.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("xorb.mission_modifier")
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.modification_strategies = {
            'timeout_adjustment': self._adjust_timeouts,
            'resource_reallocation': self._reallocate_resources,
            'strategy_simplification': self._simplify_strategy,
            'agent_substitution': self._substitute_agents,
            'phased_execution': self._create_phased_execution,
            'parameter_tuning': self._tune_parameters
        }
    
    async def generate_recycling_strategy(self, mission_context: 'MissionRecycleContext') -> Dict[str, Any]:
        """Generate recycling strategy based on failure analysis"""
        try:
            # Analyze failure pattern
            failure_analysis = await self._analyze_failure_pattern(mission_context)
            
            # Select appropriate modification strategy
            strategy_name = self._select_modification_strategy(failure_analysis)
            strategy_func = self.modification_strategies.get(strategy_name, self._default_modification)
            
            # Generate modifications
            modifications = await strategy_func(mission_context, failure_analysis)
            
            recycling_strategy = {
                'strategy_name': strategy_name,
                'modifications': modifications,
                'confidence': failure_analysis.get('confidence', 0.5),
                'expected_success_rate': failure_analysis.get('expected_success_rate', 0.7),
                'estimated_recovery_time': failure_analysis.get('estimated_recovery_time', 300),
                'generated_at': datetime.now()
            }
            
            return recycling_strategy
            
        except Exception as e:
            self.logger.error(f"Recycling strategy generation failed for mission {mission_context.mission_id[:8]}", error=str(e))
            return await self._default_modification(mission_context, {})
    
    async def _analyze_failure_pattern(self, mission_context: 'MissionRecycleContext') -> Dict[str, Any]:
        """Analyze failure pattern to determine root cause"""
        try:
            failure_reason = mission_context.original_failure_reason
            failure_context = mission_context.failure_context
            
            analysis = {
                'primary_cause': failure_reason,
                'failure_category': self._categorize_failure(failure_reason),
                'environmental_factors': mission_context.environmental_factors,
                'retry_count': mission_context.retry_count,
                'confidence': 0.7
            }
            
            # Pattern matching with historical failures
            similar_failures = self._find_similar_failures(failure_reason, failure_context)
            if similar_failures:
                success_rate = sum(1 for f in similar_failures if f.get('recovery_success', False)) / len(similar_failures)
                analysis['expected_success_rate'] = success_rate
                analysis['confidence'] = min(0.9, len(similar_failures) / 10)
            
            return analysis
            
        except Exception as e:
            self.logger.error("Failure pattern analysis failed", error=str(e))
            return {'primary_cause': 'unknown', 'confidence': 0.3}
    
    def _categorize_failure(self, failure_reason: str) -> str:
        """Categorize failure type for strategy selection"""
        failure_reason_lower = failure_reason.lower()
        
        if any(keyword in failure_reason_lower for keyword in ['timeout', 'slow', 'delay']):
            return 'timeout'
        elif any(keyword in failure_reason_lower for keyword in ['resource', 'memory', 'cpu', 'limit']):
            return 'resource'
        elif any(keyword in failure_reason_lower for keyword in ['agent', 'unavailable', 'offline']):
            return 'agent'
        elif any(keyword in failure_reason_lower for keyword in ['permission', 'access', 'auth']):
            return 'permission'
        elif any(keyword in failure_reason_lower for keyword in ['network', 'connection', 'unreachable']):
            return 'network'
        else:
            return 'unknown'
    
    def _find_similar_failures(self, failure_reason: str, failure_context: Dict[str, Any]) -> List[Dict]:
        """Find similar historical failures"""
        failure_category = self._categorize_failure(failure_reason)
        return self.failure_patterns.get(failure_category, [])
    
    def _select_modification_strategy(self, failure_analysis: Dict[str, Any]) -> str:
        """Select appropriate modification strategy"""
        failure_category = failure_analysis.get('failure_category', 'unknown')
        retry_count = failure_analysis.get('retry_count', 0)
        
        # Strategy escalation based on retry count and failure type
        if retry_count == 0:
            strategy_map = {
                'timeout': 'timeout_adjustment',
                'resource': 'resource_reallocation',
                'agent': 'agent_substitution',
                'permission': 'parameter_tuning',
                'network': 'timeout_adjustment',
                'unknown': 'parameter_tuning'
            }
        elif retry_count == 1:
            strategy_map = {
                'timeout': 'strategy_simplification',
                'resource': 'phased_execution',
                'agent': 'resource_reallocation',
                'permission': 'strategy_simplification',
                'network': 'phased_execution',
                'unknown': 'strategy_simplification'
            }
        else:
            # Last resort strategies
            strategy_map = {
                'timeout': 'phased_execution',
                'resource': 'strategy_simplification',
                'agent': 'phased_execution',
                'permission': 'phased_execution',
                'network': 'strategy_simplification',
                'unknown': 'phased_execution'
            }
        
        return strategy_map.get(failure_category, 'strategy_simplification')
    
    async def _adjust_timeouts(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adjust mission timeouts"""
        return [
            {
                'type': 'timeout_adjustment',
                'target': 'mission_timeout',
                'action': 'increase',
                'multiplier': 2.0,
                'description': 'Increase mission timeout to handle slow operations'
            },
            {
                'type': 'timeout_adjustment',
                'target': 'task_timeout',
                'action': 'increase',
                'multiplier': 1.5,
                'description': 'Increase individual task timeouts'
            }
        ]
    
    async def _reallocate_resources(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reallocate mission resources"""
        return [
            {
                'type': 'resource_reallocation',
                'target': 'cpu_limit',
                'action': 'increase',
                'multiplier': 1.5,
                'description': 'Increase CPU allocation for mission'
            },
            {
                'type': 'resource_reallocation',
                'target': 'memory_limit',
                'action': 'increase',
                'multiplier': 1.3,
                'description': 'Increase memory allocation for mission'
            }
        ]
    
    async def _simplify_strategy(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simplify mission execution strategy"""
        return [
            {
                'type': 'strategy_simplification',
                'target': 'execution_strategy',
                'action': 'change',
                'new_value': 'sequential',
                'description': 'Change to sequential execution for reliability'
            },
            {
                'type': 'strategy_simplification',
                'target': 'complexity_level',
                'action': 'reduce',
                'reduction_factor': 0.7,
                'description': 'Reduce mission complexity'
            }
        ]
    
    async def _substitute_agents(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Substitute failed agents"""
        return [
            {
                'type': 'agent_substitution',
                'target': 'failed_agents',
                'action': 'replace',
                'selection_criteria': 'high_availability',
                'description': 'Replace failed agents with high-availability alternatives'
            }
        ]
    
    async def _create_phased_execution(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create phased execution plan"""
        return [
            {
                'type': 'phased_execution',
                'target': 'execution_plan',
                'action': 'create_phases',
                'phase_count': 3,
                'description': 'Break mission into smaller phases for better control'
            }
        ]
    
    async def _tune_parameters(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Tune mission parameters"""
        return [
            {
                'type': 'parameter_tuning',
                'target': 'retry_attempts',
                'action': 'increase',
                'new_value': 5,
                'description': 'Increase retry attempts for individual tasks'
            },
            {
                'type': 'parameter_tuning',
                'target': 'error_tolerance',
                'action': 'increase',
                'multiplier': 1.5,
                'description': 'Increase error tolerance threshold'
            }
        ]
    
    async def _default_modification(self, mission_context: 'MissionRecycleContext', analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Default modification strategy"""
        return {
            'strategy_name': 'default_retry',
            'modifications': [
                {
                    'type': 'general_adjustment',
                    'target': 'all_parameters',
                    'action': 'conservative_retry',
                    'description': 'Apply conservative retry with minimal changes'
                }
            ],
            'confidence': 0.5,
            'expected_success_rate': 0.6,
            'estimated_recovery_time': 600
        }


class KPITracker:
    """
    Per-Signal KPI Instrumentation with Prometheus metrics
    
    Tracks key performance indicators for each signal type with detailed Prometheus metrics.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("xorb.kpi_tracker")
        self.signal_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.accuracy_scores: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize Prometheus metrics
        self.metrics = {
            'signal_processing_duration': Histogram('signal_processing_duration_seconds', 'Signal processing duration', ['signal_type']),
            'signal_accuracy': Gauge('signal_classification_accuracy', 'Signal classification accuracy', ['signal_type']),
            'response_effectiveness': Gauge('signal_response_effectiveness', 'Signal response effectiveness', ['signal_type']),
            'false_positive_rate': Gauge('signal_false_positive_rate', 'Signal false positive rate', ['signal_type']),
            'detection_latency': Histogram('signal_detection_latency_seconds', 'Signal detection latency', ['signal_type']),
            'system_load_impact': Gauge('signal_system_load_impact', 'Signal processing system load impact', ['signal_type'])
        }
    
    async def track_signal_kpi(self, signal: 'ThreatSignal', processing_duration: float, 
                              response_triggered: bool = False, response_effective: bool = False) -> Dict[str, float]:
        """Track KPIs for a specific signal"""
        try:
            signal_type_str = signal.signal_type.value
            current_time = datetime.now()
            
            # Processing performance
            self.metrics['signal_processing_duration'].labels(signal_type=signal_type_str).observe(processing_duration)
            
            # Detection latency (time from signal creation to processing)
            detection_latency = (current_time - signal.timestamp).total_seconds()
            self.metrics['detection_latency'].labels(signal_type=signal_type_str).observe(detection_latency)
            
            # Response tracking
            if response_triggered:
                response_time = (signal.response_time - signal.timestamp).total_seconds() if signal.response_time else 0
                self.response_times[signal_type_str].append(response_time)
                
                # Response effectiveness
                effectiveness = 1.0 if response_effective else 0.0
                self.metrics['response_effectiveness'].labels(signal_type=signal_type_str).set(effectiveness)
            
            # Update running averages
            kpis = await self._calculate_signal_kpis(signal_type_str)
            
            # Store in signal metrics
            self.signal_metrics[signal.signal_id] = {
                'signal_type': signal_type_str,
                'processing_duration': processing_duration,
                'detection_latency': detection_latency,
                'confidence': signal.confidence,
                'response_triggered': response_triggered,
                'response_effective': response_effective,
                'timestamp': current_time,
                'kpis': kpis
            }
            
            return kpis
            
        except Exception as e:
            self.logger.error(f"KPI tracking failed for signal {signal.signal_id[:8]}", error=str(e))
            return {}
    
    async def _calculate_signal_kpis(self, signal_type: str) -> Dict[str, float]:
        """Calculate comprehensive KPIs for signal type"""
        try:
            # Get recent metrics for this signal type
            recent_signals = [
                metrics for metrics in self.signal_metrics.values()
                if metrics.get('signal_type') == signal_type and
                (datetime.now() - metrics.get('timestamp', datetime.min)).total_seconds() < 3600
            ]
            
            if not recent_signals:
                return {}
            
            # Calculate various KPIs
            kpis = {}
            
            # Processing efficiency
            processing_times = [s['processing_duration'] for s in recent_signals]
            kpis['avg_processing_time'] = np.mean(processing_times)
            kpis['processing_time_p95'] = np.percentile(processing_times, 95)
            
            # Detection performance
            detection_latencies = [s['detection_latency'] for s in recent_signals]
            kpis['avg_detection_latency'] = np.mean(detection_latencies)
            kpis['detection_latency_p95'] = np.percentile(detection_latencies, 95)
            
            # Response metrics
            response_triggered_count = sum(1 for s in recent_signals if s.get('response_triggered', False))
            response_effective_count = sum(1 for s in recent_signals if s.get('response_effective', False))
            
            if response_triggered_count > 0:
                kpis['response_trigger_rate'] = response_triggered_count / len(recent_signals)
                kpis['response_effectiveness_rate'] = response_effective_count / response_triggered_count
            else:
                kpis['response_trigger_rate'] = 0.0
                kpis['response_effectiveness_rate'] = 0.0
            
            # Confidence metrics
            confidences = [s['confidence'] for s in recent_signals]
            kpis['avg_confidence'] = np.mean(confidences)
            kpis['confidence_stability'] = 1.0 - np.std(confidences)  # Lower std = more stable
            
            # System impact
            high_priority_signals = [s for s in recent_signals if s['confidence'] > 0.8]
            kpis['high_priority_rate'] = len(high_priority_signals) / len(recent_signals)
            
            # Update Prometheus metrics
            self.metrics['signal_accuracy'].labels(signal_type=signal_type).set(kpis['avg_confidence'])
            
            return kpis
            
        except Exception as e:
            self.logger.error(f"KPI calculation failed for signal type {signal_type}", error=str(e))
            return {}
    
    async def get_kpi_summary(self, time_window_hours: float = 1.0) -> Dict[str, Any]:
        """Get KPI summary across all signal types"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            recent_metrics = [
                metrics for metrics in self.signal_metrics.values()
                if metrics.get('timestamp', datetime.min) > cutoff_time
            ]
            
            if not recent_metrics:
                return {'total_signals': 0, 'signal_types': {}}
            
            # Group by signal type
            signals_by_type = defaultdict(list)
            for metrics in recent_metrics:
                signal_type = metrics.get('signal_type', 'unknown')
                signals_by_type[signal_type].append(metrics)
            
            # Calculate summary KPIs
            summary = {
                'total_signals': len(recent_metrics),
                'time_window_hours': time_window_hours,
                'signal_types': {},
                'overall_kpis': {}
            }
            
            # Per-type KPIs
            for signal_type, type_metrics in signals_by_type.items():
                summary['signal_types'][signal_type] = await self._calculate_signal_kpis(signal_type)
            
            # Overall KPIs
            all_processing_times = [m['processing_duration'] for m in recent_metrics]
            all_confidences = [m['confidence'] for m in recent_metrics]
            
            summary['overall_kpis'] = {
                'avg_processing_time': np.mean(all_processing_times),
                'avg_confidence': np.mean(all_confidences),
                'total_responses_triggered': sum(1 for m in recent_metrics if m.get('response_triggered', False)),
                'signal_throughput': len(recent_metrics) / time_window_hours
            }
            
            return summary
            
        except Exception as e:
            self.logger.error("KPI summary calculation failed", error=str(e))
            return {'error': str(e)}


class ConflictDetector:
    """
    Redundancy & Conflict Detection using vector-hashing for signal deduplication
    
    Detects duplicate signals and conflicting responses using advanced hashing techniques.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.logger = structlog.get_logger("xorb.conflict_detector")
        
        # Signal deduplication
        self.signal_hashes: Dict[str, Dict] = {}  # hash -> signal metadata
        self.vector_cache: Dict[str, np.ndarray] = {}
        
        # Conflict detection
        self.active_responses: Dict[str, Dict] = {}  # response_id -> response metadata
        self.conflict_patterns: List[Dict] = []
        
        # Performance optimization
        self.hash_cache_size = 10000
        self.vector_cache_size = 1000
    
    async def check_signal_duplication(self, signal: 'ThreatSignal') -> Dict[str, Any]:
        """Check if signal is a duplicate using vector hashing"""
        try:
            # Generate signal vector
            signal_vector = await self._generate_signal_vector(signal)
            
            # Check for similar vectors
            similar_signals = await self._find_similar_signals(signal_vector, signal)
            
            result = {
                'is_duplicate': len(similar_signals) > 0,
                'similar_signals': similar_signals,
                'confidence': 0.0,
                'deduplication_action': 'none'
            }
            
            if similar_signals:
                # Calculate confidence based on similarity scores
                max_similarity = max(s['similarity_score'] for s in similar_signals)
                result['confidence'] = max_similarity
                
                if max_similarity > self.similarity_threshold:
                    result['deduplication_action'] = 'discard'
                else:
                    result['deduplication_action'] = 'merge'
            
            # Store signal hash for future comparisons
            await self._store_signal_hash(signal, signal_vector)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Duplication check failed for signal {signal.signal_id[:8]}", error=str(e))
            return {'is_duplicate': False, 'error': str(e)}
    
    async def _generate_signal_vector(self, signal: 'ThreatSignal') -> np.ndarray:
        """Generate vector representation of signal for similarity comparison"""
        try:
            # Check cache first
            if signal.signal_id in self.vector_cache:
                return self.vector_cache[signal.signal_id]
            
            # Create feature vector
            features = []
            
            # Signal type encoding
            signal_types = ['network_anomaly', 'vulnerability_discovery', 'suspicious_behavior', 
                          'performance_degradation', 'security_alert', 'system_compromise',
                          'data_exfiltration', 'lateral_movement']
            type_vector = [1.0 if signal.signal_type.value == st else 0.0 for st in signal_types]
            features.extend(type_vector)
            
            # Source encoding
            source_hash = int(hashlib.md5(signal.source.encode()).hexdigest()[:8], 16)
            source_features = [(source_hash >> i) & 1 for i in range(16)]
            features.extend(source_features)
            
            # Temporal features (normalized)
            hour_feature = signal.timestamp.hour / 24.0
            day_feature = signal.timestamp.weekday() / 7.0
            features.extend([hour_feature, day_feature])
            
            # Confidence and severity
            features.extend([signal.confidence, signal.severity])
            
            # Raw data features (if available)
            if signal.raw_data:
                raw_features = self._extract_raw_data_vector(signal.raw_data)
                features.extend(raw_features[:20])  # Limit to 20 features
            else:
                features.extend([0.0] * 20)
            
            # Create numpy array
            vector = np.array(features, dtype=np.float32)
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            # Cache the vector
            self.vector_cache[signal.signal_id] = vector
            
            # Limit cache size
            if len(self.vector_cache) > self.vector_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.vector_cache.keys())[:100]
                for key in oldest_keys:
                    del self.vector_cache[key]
            
            return vector
            
        except Exception as e:
            self.logger.error(f"Vector generation failed for signal {signal.signal_id[:8]}", error=str(e))
            return np.zeros(48, dtype=np.float32)  # Default vector size
    
    def _extract_raw_data_vector(self, raw_data: Dict[str, Any]) -> List[float]:
        """Extract vector features from raw signal data"""
        features = []
        
        def extract_recursive(data, max_depth=3, current_depth=0):
            if current_depth >= max_depth or len(features) >= 20:
                return
            
            for key, value in data.items():
                if len(features) >= 20:
                    break
                
                if isinstance(value, (int, float)):
                    # Normalize numeric values
                    normalized_value = min(1.0, max(0.0, abs(value) / 1000.0))
                    features.append(normalized_value)
                elif isinstance(value, str):
                    # Convert string to hash-based feature
                    string_hash = hash(value) % 1000
                    features.append(string_hash / 1000.0)
                elif isinstance(value, dict):
                    extract_recursive(value, max_depth, current_depth + 1)
                elif isinstance(value, list) and value:
                    # Use first element or length
                    if isinstance(value[0], (int, float)):
                        features.append(min(1.0, len(value) / 100.0))
                    else:
                        features.append(len(value) / 100.0)
        
        extract_recursive(raw_data)
        
        # Pad or truncate to exactly 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    async def _find_similar_signals(self, signal_vector: np.ndarray, signal: 'ThreatSignal') -> List[Dict[str, Any]]:
        """Find similar signals using cosine similarity"""
        try:
            similar_signals = []
            
            # Compare with recently stored signals
            recent_cutoff = datetime.now() - timedelta(hours=1)  # Only check last hour
            
            for signal_hash, metadata in self.signal_hashes.items():
                if metadata.get('timestamp', datetime.min) < recent_cutoff:
                    continue
                
                # Skip if same signal
                if metadata.get('signal_id') == signal.signal_id:
                    continue
                
                # Compare vectors
                stored_vector = metadata.get('vector')
                if stored_vector is not None:
                    similarity = 1.0 - cosine(signal_vector, stored_vector)
                    
                    if similarity > 0.7:  # Threshold for considering similarity
                        similar_signals.append({
                            'signal_id': metadata.get('signal_id'),
                            'signal_hash': signal_hash,
                            'similarity_score': similarity,
                            'timestamp': metadata.get('timestamp'),
                            'signal_type': metadata.get('signal_type')
                        })
            
            # Sort by similarity score
            similar_signals.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_signals[:5]  # Return top 5 similar signals
            
        except Exception as e:
            self.logger.error("Similar signal search failed", error=str(e))
            return []
    
    async def _store_signal_hash(self, signal: 'ThreatSignal', signal_vector: np.ndarray):
        """Store signal hash for future deduplication"""
        try:
            # Generate hash
            signal_hash = signal.pattern_hash or signal._compute_pattern_hash()
            
            # Store metadata
            self.signal_hashes[signal_hash] = {
                'signal_id': signal.signal_id,
                'signal_type': signal.signal_type.value,
                'timestamp': signal.timestamp,
                'vector': signal_vector,
                'confidence': signal.confidence,
                'source': signal.source
            }
            
            # Limit hash storage size
            if len(self.signal_hashes) > self.hash_cache_size:
                # Remove oldest entries
                sorted_hashes = sorted(
                    self.signal_hashes.items(), 
                    key=lambda x: x[1].get('timestamp', datetime.min)
                )
                
                # Remove oldest 1000 entries
                for hash_key, _ in sorted_hashes[:1000]:
                    del self.signal_hashes[hash_key]
            
        except Exception as e:
            self.logger.error(f"Signal hash storage failed for {signal.signal_id[:8]}", error=str(e))
    
    async def detect_response_conflicts(self, active_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between active responses"""
        try:
            conflicts = []
            
            # Check for conflicting responses
            for i, response1 in enumerate(active_responses):
                for j, response2 in enumerate(active_responses[i+1:], i+1):
                    conflict = await self._check_response_conflict(response1, response2)
                    if conflict:
                        conflicts.append({
                            'conflict_type': conflict['type'],
                            'response1_id': response1.get('response_id'),
                            'response2_id': response2.get('response_id'),
                            'severity': conflict['severity'],
                            'resolution_suggestion': conflict['resolution'],
                            'detected_at': datetime.now()
                        })
            
            return conflicts
            
        except Exception as e:
            self.logger.error("Response conflict detection failed", error=str(e))
            return []
    
    async def _check_response_conflict(self, response1: Dict[str, Any], response2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if two responses conflict"""
        try:
            # Resource conflicts
            if self._check_resource_conflict(response1, response2):
                return {
                    'type': 'resource_conflict',
                    'severity': 'high',
                    'resolution': 'prioritize_by_importance'
                }
            
            # Target conflicts
            if self._check_target_conflict(response1, response2):
                return {
                    'type': 'target_conflict',
                    'severity': 'medium',
                    'resolution': 'coordinate_access'
                }
            
            # Strategy conflicts
            if self._check_strategy_conflict(response1, response2):
                return {
                    'type': 'strategy_conflict',
                    'severity': 'low',
                    'resolution': 'sequence_execution'
                }
            
            return None
            
        except Exception as e:
            self.logger.error("Response conflict check failed", error=str(e))
            return None
    
    def _check_resource_conflict(self, response1: Dict, response2: Dict) -> bool:
        """Check for resource conflicts between responses"""
        # Example: Check if both responses need the same agent
        agents1 = set(response1.get('required_agents', []))
        agents2 = set(response2.get('required_agents', []))
        return len(agents1.intersection(agents2)) > 0
    
    def _check_target_conflict(self, response1: Dict, response2: Dict) -> bool:
        """Check for target conflicts between responses"""
        # Example: Check if both responses target the same system
        targets1 = set(response1.get('target_systems', []))
        targets2 = set(response2.get('target_systems', []))
        return len(targets1.intersection(targets2)) > 0
    
    def _check_strategy_conflict(self, response1: Dict, response2: Dict) -> bool:
        """Check for strategy conflicts between responses"""
        # Example: Check if response strategies are incompatible
        strategy1 = response1.get('strategy', '')
        strategy2 = response2.get('strategy', '')
        
        incompatible_pairs = [
            ('aggressive', 'stealth'),
            ('immediate', 'delayed'),
            ('isolation', 'monitoring')
        ]
        
        for s1, s2 in incompatible_pairs:
            if (strategy1 == s1 and strategy2 == s2) or (strategy1 == s2 and strategy2 == s1):
                return True
        
        return False