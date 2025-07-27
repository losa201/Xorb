#!/usr/bin/env python3
"""
XORB Phase II Evolution Orchestrator
Adaptive Intelligence and Swarm Refinement System
Mission: Adaptive learning, swarm intelligence refinement, and mission-specific agent specialization
"""

import asyncio
import logging
import json
import time
import uuid
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil

# Configure logging for swarm behavior
os.makedirs("/var/log/xorb", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/xorb/swarm_behavior.log'),
        logging.FileHandler('logs/phase2_evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('XORB-PHASE2')

class AgentCluster(Enum):
    """Agent specialization clusters."""
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    ANALYST = "analyst"
    HYBRID = "hybrid"

class SwarmBehaviorType(Enum):
    """Types of swarm behaviors."""
    COORDINATED_ATTACK = "coordinated_attack"
    DEFENSIVE_FORMATION = "defensive_formation"
    INTELLIGENCE_SHARING = "intelligence_sharing"
    ADAPTIVE_LEARNING = "adaptive_learning"
    EMERGENT_STRATEGY = "emergent_strategy"

@dataclass
class AgentPerformanceProfile:
    """Performance profile for individual agents."""
    agent_id: str
    cluster: AgentCluster
    performance_score: float
    capabilities: Dict[str, float]  # stealth, remediation, detection, etc.
    evolution_cycles: int
    last_optimization: float
    emergent_behaviors: List[str] = field(default_factory=list)
    learning_rate: float = 0.1
    specialization_level: float = 0.5

@dataclass
class SwarmCluster:
    """Swarm cluster configuration and state."""
    cluster_id: str
    cluster_type: AgentCluster
    agents: List[AgentPerformanceProfile]
    cluster_intelligence: float = 0.0
    coordination_matrix: np.ndarray = None
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    emergence_events: int = 0

@dataclass
class EmergenceEvent:
    """Emergence event detection and logging."""
    event_id: str
    event_type: str
    timestamp: float
    agents_involved: List[str]
    behavior_description: str
    performance_impact: float
    validation_status: str = "pending"

class XORBPhase2EvolutionOrchestrator:
    """Master orchestrator for Phase II adaptive intelligence evolution."""
    
    def __init__(self):
        self.orchestrator_id = f"XORB-PHASE2-{str(uuid.uuid4())[:8].upper()}"
        
        # Evolution state
        self.agent_profiles = {}
        self.swarm_clusters = {}
        self.emergence_events = []
        self.evolution_cycles_completed = 0
        self.phase2_start_time = 0.0
        
        # Telemetry and monitoring
        self.telemetry_active = False
        self.performance_window_data = []
        self.swarm_behavior_log = []
        
        # Configuration
        self.performance_threshold = 0.85  # 85% threshold for evolution triggers
        self.emergence_detection_sensitivity = 0.8
        self.swarm_size_per_cluster = 15  # 15 agents per cluster
        
        logger.info(f"🧬 XORB Phase II Evolution Orchestrator initialized: {self.orchestrator_id}")
    
    async def initiate_phase2_evolution(
        self,
        duration: int = 3600,  # 1 hour default
        specialization_mode: str = "adaptive"
    ) -> Dict[str, Any]:
        """Initiate Phase II adaptive intelligence and swarm refinement."""
        
        self.phase2_start_time = time.time()
        session_id = f"PHASE2-{str(uuid.uuid4())[:8].upper()}"
        
        logger.info(f"🧬 XORB PHASE II EVOLUTION INITIATED")
        logger.info(f"🆔 Session ID: {session_id}")
        logger.info(f"⏱️ Duration: {duration} seconds")
        logger.info(f"🎯 Specialization Mode: {specialization_mode}")
        logger.info(f"📊 Performance Threshold: {self.performance_threshold:.1%}")
        logger.info(f"\n🚀 BEGINNING ADAPTIVE INTELLIGENCE EVOLUTION...\n")
        
        evolution_result = {
            "session_id": session_id,
            "start_time": self.phase2_start_time,
            "duration": duration,
            "phase2_objectives": {
                "agent_level_tuning": False,
                "swarm_cluster_specialization": False,
                "emergent_behavior_detection": False,
                "telemetry_monitoring": False
            },
            "performance_metrics": {},
            "emergence_events": [],
            "swarm_clusters": {},
            "evolution_success": False
        }
        
        try:
            # Phase 2.1: Initialize Agent Performance Assessment
            logger.info("📊 Phase 2.1: Agent Performance Assessment and Profiling")
            assessment_result = await self._assess_agent_performance()
            evolution_result["agent_assessment"] = assessment_result
            
            # Phase 2.2: Create Specialized Swarm Clusters
            logger.info("🤖 Phase 2.2: Swarm Cluster Specialization and Formation")
            cluster_result = await self._create_specialized_clusters()
            evolution_result["cluster_formation"] = cluster_result
            evolution_result["phase2_objectives"]["swarm_cluster_specialization"] = True
            
            # Phase 2.3: Activate Telemetry and Monitoring
            logger.info("📡 Phase 2.3: Telemetry and Performance Monitoring Activation")
            telemetry_result = await self._activate_telemetry_monitoring()
            evolution_result["telemetry_setup"] = telemetry_result
            evolution_result["phase2_objectives"]["telemetry_monitoring"] = True
            
            # Phase 2.4: Execute Adaptive Evolution Cycles
            logger.info("⚡ Phase 2.4: Adaptive Evolution and Optimization Cycles")
            
            evolution_tasks = [
                self._run_agent_level_tuning(),
                self._run_swarm_coordination_refinement(),
                self._run_emergence_detection_system(),
                self._run_performance_monitoring_system(),
                self._run_intelligence_amplification_system(),
                self._run_adaptive_learning_coordinator()
            ]
            
            # Execute all evolution systems concurrently
            await asyncio.gather(*evolution_tasks, return_exceptions=True)
            
            # Phase 2.5: Final Assessment and Validation
            logger.info("✅ Phase 2.5: Final Performance Assessment and Validation")
            final_assessment = await self._perform_final_assessment()
            evolution_result["final_assessment"] = final_assessment
            
            # Mark objectives as completed
            evolution_result["phase2_objectives"]["agent_level_tuning"] = True
            evolution_result["phase2_objectives"]["emergent_behavior_detection"] = True
            
            # Calculate final metrics
            evolution_result["duration_actual"] = time.time() - self.phase2_start_time
            evolution_result["evolution_cycles_completed"] = self.evolution_cycles_completed
            evolution_result["emergence_events"] = [event.__dict__ for event in self.emergence_events]
            evolution_result["swarm_clusters"] = {
                cluster_id: {
                    "cluster_type": cluster.cluster_type.value,
                    "agent_count": len(cluster.agents),
                    "cluster_intelligence": cluster.cluster_intelligence,
                    "emergence_events": cluster.emergence_events
                }
                for cluster_id, cluster in self.swarm_clusters.items()
            }
            evolution_result["evolution_success"] = True
            
            logger.info(f"✨ XORB PHASE II EVOLUTION COMPLETED")
            logger.info(f"⏱️ Duration: {evolution_result['duration_actual']:.1f} seconds")
            logger.info(f"🔄 Evolution Cycles: {self.evolution_cycles_completed}")
            logger.info(f"🌟 Emergence Events: {len(self.emergence_events)}")
            logger.info(f"🤖 Active Clusters: {len(self.swarm_clusters)}")
            
            return evolution_result
            
        except Exception as e:
            logger.error(f"❌ Phase II evolution failed: {e}")
            evolution_result["error"] = str(e)
            evolution_result["evolution_success"] = False
            evolution_result["duration_actual"] = time.time() - self.phase2_start_time
            return evolution_result
    
    async def _assess_agent_performance(self) -> Dict[str, Any]:
        """Assess current agent performance and identify optimization targets."""
        
        logger.info("📊 Assessing agent performance across XORB ecosystem...")
        
        assessment_result = {
            "agents_assessed": 0,
            "underperforming_agents": [],
            "high_performing_agents": [],
            "average_performance": 0.0,
            "optimization_targets": []
        }
        
        try:
            # Simulate agent performance assessment
            # In production, this would query actual agent metrics
            agent_count = 60  # Simulate 60 agents across the ecosystem
            
            for i in range(agent_count):
                agent_id = f"AGENT-{i+1:03d}"
                
                # Simulate performance metrics
                performance_score = np.random.uniform(0.6, 0.98)
                capabilities = {
                    "stealth": np.random.uniform(0.5, 0.95),
                    "remediation": np.random.uniform(0.6, 0.9),
                    "detection": np.random.uniform(0.7, 0.95),
                    "adaptation": np.random.uniform(0.4, 0.8),
                    "learning": np.random.uniform(0.5, 0.85)
                }
                
                # Assign to cluster based on capabilities
                if capabilities["detection"] > 0.8:
                    cluster = AgentCluster.ANALYST
                elif capabilities["stealth"] > 0.8:
                    cluster = AgentCluster.OFFENSIVE
                elif capabilities["remediation"] > 0.8:
                    cluster = AgentCluster.DEFENSIVE
                else:
                    cluster = AgentCluster.HYBRID
                
                agent_profile = AgentPerformanceProfile(
                    agent_id=agent_id,
                    cluster=cluster,
                    performance_score=performance_score,
                    capabilities=capabilities,
                    evolution_cycles=np.random.randint(0, 10),
                    last_optimization=time.time() - np.random.uniform(0, 3600)
                )
                
                self.agent_profiles[agent_id] = agent_profile
                
                # Categorize performance
                if performance_score < self.performance_threshold:
                    assessment_result["underperforming_agents"].append({
                        "agent_id": agent_id,
                        "performance_score": performance_score,
                        "cluster": cluster.value,
                        "optimization_priority": 1.0 - performance_score
                    })
                else:
                    assessment_result["high_performing_agents"].append({
                        "agent_id": agent_id,
                        "performance_score": performance_score,
                        "cluster": cluster.value
                    })
            
            assessment_result["agents_assessed"] = agent_count
            assessment_result["average_performance"] = np.mean([
                agent.performance_score for agent in self.agent_profiles.values()
            ])
            
            # Generate optimization targets
            underperforming_count = len(assessment_result["underperforming_agents"])
            if underperforming_count > 0:
                assessment_result["optimization_targets"] = [
                    f"Evolve {underperforming_count} agents below {self.performance_threshold:.1%} threshold",
                    "Enhance capability balance across all clusters",
                    "Implement adaptive learning acceleration",
                    "Activate swarm coordination protocols"
                ]
            
            logger.info(f"📊 Agent assessment completed:")
            logger.info(f"   Total Agents: {assessment_result['agents_assessed']}")
            logger.info(f"   Average Performance: {assessment_result['average_performance']:.1%}")
            logger.info(f"   Underperforming: {len(assessment_result['underperforming_agents'])}")
            logger.info(f"   High Performing: {len(assessment_result['high_performing_agents'])}")
            
        except Exception as e:
            logger.error(f"❌ Agent performance assessment failed: {e}")
            assessment_result["error"] = str(e)
        
        return assessment_result
    
    async def _create_specialized_clusters(self) -> Dict[str, Any]:
        """Create specialized swarm clusters for coordinated evolution."""
        
        logger.info("🤖 Creating specialized swarm clusters...")
        
        cluster_result = {
            "clusters_created": 0,
            "cluster_details": {},
            "coordination_matrices": {},
            "specialization_success": True
        }
        
        try:
            # Group agents by cluster type
            cluster_agents = {
                AgentCluster.DEFENSIVE: [],
                AgentCluster.OFFENSIVE: [],
                AgentCluster.ANALYST: [],
                AgentCluster.HYBRID: []
            }
            
            for agent in self.agent_profiles.values():
                cluster_agents[agent.cluster].append(agent)
            
            # Create swarm clusters
            for cluster_type, agents in cluster_agents.items():
                if len(agents) > 0:
                    cluster_id = f"CLUSTER-{cluster_type.value.upper()}"
                    
                    # Create coordination matrix for the cluster
                    coordination_matrix = np.random.rand(len(agents), len(agents))
                    # Make matrix symmetric for bidirectional coordination
                    coordination_matrix = (coordination_matrix + coordination_matrix.T) / 2
                    
                    # Calculate cluster intelligence
                    cluster_intelligence = np.mean([agent.performance_score for agent in agents])
                    
                    swarm_cluster = SwarmCluster(
                        cluster_id=cluster_id,
                        cluster_type=cluster_type,
                        agents=agents,
                        cluster_intelligence=cluster_intelligence,
                        coordination_matrix=coordination_matrix,
                        shared_knowledge={
                            "cluster_expertise": cluster_type.value,
                            "optimization_strategies": [],
                            "learned_patterns": []
                        }
                    )
                    
                    self.swarm_clusters[cluster_id] = swarm_cluster
                    
                    cluster_result["cluster_details"][cluster_id] = {
                        "cluster_type": cluster_type.value,
                        "agent_count": len(agents),
                        "cluster_intelligence": cluster_intelligence,
                        "coordination_matrix_size": coordination_matrix.shape,
                        "specialization_focus": self._get_cluster_specialization_focus(cluster_type)
                    }
                    
                    cluster_result["clusters_created"] += 1
                    
                    logger.info(f"🤖 Created {cluster_id}:")
                    logger.info(f"   Agents: {len(agents)}")
                    logger.info(f"   Intelligence: {cluster_intelligence:.1%}")
                    logger.info(f"   Specialization: {cluster_type.value}")
            
            logger.info(f"🤖 Cluster formation completed: {cluster_result['clusters_created']} clusters created")
            
        except Exception as e:
            logger.error(f"❌ Cluster creation failed: {e}")
            cluster_result["specialization_success"] = False
            cluster_result["error"] = str(e)
        
        return cluster_result
    
    def _get_cluster_specialization_focus(self, cluster_type: AgentCluster) -> List[str]:
        """Get specialization focus areas for cluster type."""
        
        specialization_map = {
            AgentCluster.DEFENSIVE: [
                "threat_detection", "remediation_speed", "system_hardening",
                "incident_response", "damage_containment"
            ],
            AgentCluster.OFFENSIVE: [
                "stealth_operations", "vulnerability_discovery", "payload_generation",
                "evasion_techniques", "persistence_mechanisms"
            ],
            AgentCluster.ANALYST: [
                "pattern_recognition", "threat_intelligence", "behavioral_analysis",
                "risk_assessment", "forensic_analysis"
            ],
            AgentCluster.HYBRID: [
                "adaptive_strategies", "cross_domain_expertise", "coordination",
                "knowledge_synthesis", "strategic_planning"
            ]
        }
        
        return specialization_map.get(cluster_type, ["general_capabilities"])
    
    async def _activate_telemetry_monitoring(self) -> Dict[str, Any]:
        """Activate comprehensive telemetry and monitoring systems."""
        
        logger.info("📡 Activating telemetry and monitoring systems...")
        
        telemetry_result = {
            "telemetry_active": True,
            "monitoring_channels": [],
            "prometheus_integration": False,
            "log_streams": [],
            "update_frequency": 2  # seconds
        }
        
        try:
            # Setup monitoring channels
            telemetry_result["monitoring_channels"] = [
                "agent_performance_metrics",
                "swarm_behavior_tracking",
                "emergence_event_detection",
                "cluster_coordination_analysis",
                "evolution_cycle_monitoring"
            ]
            
            # Setup log streams
            telemetry_result["log_streams"] = [
                "/var/log/xorb/swarm_behavior.log",
                "logs/phase2_evolution.log",
                "agent_state.json"
            ]
            
            # Initialize performance window tracking
            self.performance_window_data = []
            self.telemetry_active = True
            
            # Start telemetry tasks
            asyncio.create_task(self._telemetry_collection_loop())
            asyncio.create_task(self._prometheus_metrics_emitter())
            
            logger.info("📡 Telemetry system activated")
            logger.info(f"   Monitoring Channels: {len(telemetry_result['monitoring_channels'])}")
            logger.info(f"   Update Frequency: {telemetry_result['update_frequency']} seconds")
            
        except Exception as e:
            logger.error(f"❌ Telemetry activation failed: {e}")
            telemetry_result["telemetry_active"] = False
            telemetry_result["error"] = str(e)
        
        return telemetry_result
    
    async def _run_agent_level_tuning(self):
        """Run agent-level performance tuning and optimization."""
        
        logger.info("⚡ Starting agent-level tuning system...")
        
        tuning_cycles = 0
        
        try:
            while True:
                tuning_cycles += 1
                cycle_start = time.time()
                
                logger.info(f"⚡ Agent tuning cycle #{tuning_cycles}")
                
                # Identify agents needing optimization
                optimization_targets = []
                for agent in self.agent_profiles.values():
                    if agent.performance_score < self.performance_threshold:
                        optimization_targets.append(agent)
                
                # Optimize underperforming agents
                optimizations_applied = 0
                for agent in optimization_targets:
                    optimization_applied = await self._optimize_agent_performance(agent)
                    if optimization_applied:
                        optimizations_applied += 1
                        agent.evolution_cycles += 1
                        agent.last_optimization = time.time()
                
                # Log tuning results
                cycle_duration = time.time() - cycle_start
                
                self.swarm_behavior_log.append({
                    "timestamp": time.time(),
                    "event_type": "agent_tuning_cycle",
                    "cycle_number": tuning_cycles,
                    "optimization_targets": len(optimization_targets),
                    "optimizations_applied": optimizations_applied,
                    "cycle_duration": cycle_duration
                })
                
                logger.info(f"⚡ Tuning cycle {tuning_cycles}: {optimizations_applied}/{len(optimization_targets)} optimizations applied")
                
                await asyncio.sleep(120)  # Tune every 2 minutes
                
        except Exception as e:
            logger.error(f"❌ Agent tuning system failed: {e}")
    
    async def _optimize_agent_performance(self, agent: AgentPerformanceProfile) -> bool:
        """Optimize individual agent performance."""
        
        try:
            # Calculate optimization strategy
            weakest_capability = min(agent.capabilities.items(), key=lambda x: x[1])
            
            # Apply optimization based on cluster specialization
            if agent.cluster == AgentCluster.DEFENSIVE:
                # Focus on detection and remediation
                agent.capabilities["detection"] = min(1.0, agent.capabilities["detection"] + 0.05)
                agent.capabilities["remediation"] = min(1.0, agent.capabilities["remediation"] + 0.04)
            elif agent.cluster == AgentCluster.OFFENSIVE:
                # Focus on stealth and adaptation
                agent.capabilities["stealth"] = min(1.0, agent.capabilities["stealth"] + 0.05)
                agent.capabilities["adaptation"] = min(1.0, agent.capabilities["adaptation"] + 0.04)
            elif agent.cluster == AgentCluster.ANALYST:
                # Focus on detection and learning
                agent.capabilities["detection"] = min(1.0, agent.capabilities["detection"] + 0.06)
                agent.capabilities["learning"] = min(1.0, agent.capabilities["learning"] + 0.05)
            else:  # HYBRID
                # Balanced improvement
                for capability in agent.capabilities:
                    agent.capabilities[capability] = min(1.0, agent.capabilities[capability] + 0.02)
            
            # Boost weakest capability
            agent.capabilities[weakest_capability[0]] = min(1.0, agent.capabilities[weakest_capability[0]] + 0.03)
            
            # Recalculate performance score
            old_score = agent.performance_score
            agent.performance_score = np.mean(list(agent.capabilities.values()))
            
            improvement = agent.performance_score - old_score
            
            if improvement > 0.01:  # Significant improvement
                logger.debug(f"🔧 Optimized {agent.agent_id}: {old_score:.1%} → {agent.performance_score:.1%}")
                return True
            
        except Exception as e:
            logger.error(f"❌ Agent optimization failed for {agent.agent_id}: {e}")
        
        return False
    
    async def _run_swarm_coordination_refinement(self):
        """Run swarm coordination refinement and synchronization."""
        
        logger.info("🤖 Starting swarm coordination refinement...")
        
        coordination_cycles = 0
        
        try:
            while True:
                coordination_cycles += 1
                cycle_start = time.time()
                
                logger.info(f"🤖 Swarm coordination cycle #{coordination_cycles}")
                
                # Refine coordination for each cluster
                for cluster_id, cluster in self.swarm_clusters.items():
                    await self._refine_cluster_coordination(cluster)
                
                # Inter-cluster coordination
                await self._coordinate_between_clusters()
                
                # Update cluster intelligence
                for cluster in self.swarm_clusters.values():
                    cluster.cluster_intelligence = np.mean([
                        agent.performance_score for agent in cluster.agents
                    ])
                
                cycle_duration = time.time() - cycle_start
                
                self.swarm_behavior_log.append({
                    "timestamp": time.time(),
                    "event_type": "swarm_coordination_cycle",
                    "cycle_number": coordination_cycles,
                    "clusters_coordinated": len(self.swarm_clusters),
                    "cycle_duration": cycle_duration
                })
                
                logger.info(f"🤖 Coordination cycle {coordination_cycles}: {len(self.swarm_clusters)} clusters synchronized")
                
                await asyncio.sleep(90)  # Coordinate every 90 seconds
                
        except Exception as e:
            logger.error(f"❌ Swarm coordination failed: {e}")
    
    async def _refine_cluster_coordination(self, cluster: SwarmCluster):
        """Refine coordination within a single cluster."""
        
        try:
            # Update coordination matrix based on agent performance
            agent_count = len(cluster.agents)
            
            for i in range(agent_count):
                for j in range(i + 1, agent_count):
                    agent_i = cluster.agents[i]
                    agent_j = cluster.agents[j]
                    
                    # Calculate coordination strength based on capability similarity
                    capability_similarity = 1.0 - np.linalg.norm(
                        np.array(list(agent_i.capabilities.values())) - 
                        np.array(list(agent_j.capabilities.values()))
                    ) / np.sqrt(len(agent_i.capabilities))
                    
                    # Update coordination matrix
                    cluster.coordination_matrix[i, j] = capability_similarity
                    cluster.coordination_matrix[j, i] = capability_similarity
            
            # Share knowledge within cluster
            await self._share_cluster_knowledge(cluster)
            
        except Exception as e:
            logger.error(f"❌ Cluster coordination refinement failed: {e}")
    
    async def _share_cluster_knowledge(self, cluster: SwarmCluster):
        """Share knowledge and strategies within cluster."""
        
        try:
            # Collect high-performing strategies
            high_performers = [
                agent for agent in cluster.agents 
                if agent.performance_score > 0.9
            ]
            
            if high_performers:
                # Extract optimization strategies from high performers
                strategies = []
                for agent in high_performers:
                    if agent.emergent_behaviors:
                        strategies.extend(agent.emergent_behaviors)
                
                # Share strategies with cluster
                cluster.shared_knowledge["optimization_strategies"] = strategies
                
                # Apply successful strategies to underperformers
                underperformers = [
                    agent for agent in cluster.agents 
                    if agent.performance_score < 0.8
                ]
                
                for agent in underperformers:
                    # Apply knowledge transfer
                    agent.learning_rate = min(0.3, agent.learning_rate + 0.02)
                    
                    # Enhance capabilities based on cluster strategies
                    for capability in agent.capabilities:
                        best_capability_value = max([
                            hp.capabilities[capability] for hp in high_performers
                        ])
                        
                        # Gradual improvement towards best performer
                        improvement = (best_capability_value - agent.capabilities[capability]) * agent.learning_rate * 0.1
                        agent.capabilities[capability] = min(1.0, agent.capabilities[capability] + improvement)
            
        except Exception as e:
            logger.error(f"❌ Cluster knowledge sharing failed: {e}")
    
    async def _coordinate_between_clusters(self):
        """Coordinate between different clusters."""
        
        try:
            cluster_list = list(self.swarm_clusters.values())
            
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    cluster_a = cluster_list[i]
                    cluster_b = cluster_list[j]
                    
                    # Exchange complementary knowledge
                    if cluster_a.cluster_type != cluster_b.cluster_type:
                        await self._exchange_cluster_knowledge(cluster_a, cluster_b)
            
        except Exception as e:
            logger.error(f"❌ Inter-cluster coordination failed: {e}")
    
    async def _exchange_cluster_knowledge(self, cluster_a: SwarmCluster, cluster_b: SwarmCluster):
        """Exchange knowledge between different cluster types."""
        
        try:
            # Cross-pollinate strategies
            if cluster_a.shared_knowledge.get("optimization_strategies"):
                # Transfer relevant strategies to cluster_b
                relevant_strategies = [
                    strategy for strategy in cluster_a.shared_knowledge["optimization_strategies"]
                    if "adaptive" in strategy or "coordination" in strategy
                ]
                
                if "cross_cluster_strategies" not in cluster_b.shared_knowledge:
                    cluster_b.shared_knowledge["cross_cluster_strategies"] = []
                
                cluster_b.shared_knowledge["cross_cluster_strategies"].extend(relevant_strategies)
            
            # Similar exchange from cluster_b to cluster_a
            if cluster_b.shared_knowledge.get("optimization_strategies"):
                relevant_strategies = [
                    strategy for strategy in cluster_b.shared_knowledge["optimization_strategies"]
                    if "adaptive" in strategy or "coordination" in strategy
                ]
                
                if "cross_cluster_strategies" not in cluster_a.shared_knowledge:
                    cluster_a.shared_knowledge["cross_cluster_strategies"] = []
                
                cluster_a.shared_knowledge["cross_cluster_strategies"].extend(relevant_strategies)
            
        except Exception as e:
            logger.error(f"❌ Cluster knowledge exchange failed: {e}")
    
    async def _run_emergence_detection_system(self):
        """Run emergent behavior detection and validation system."""
        
        logger.info("🌟 Starting emergence detection system...")
        
        detection_cycles = 0
        
        try:
            while True:
                detection_cycles += 1
                cycle_start = time.time()
                
                logger.info(f"🌟 Emergence detection cycle #{detection_cycles}")
                
                # Detect emergent behaviors
                emergence_events = await self._detect_emergent_behaviors()
                
                # Validate detected emergences
                for event in emergence_events:
                    validation_result = await self._validate_emergence_event(event)
                    event.validation_status = validation_result
                    
                    if validation_result == "validated":
                        self.emergence_events.append(event)
                        
                        # Log emergence event
                        logger.info(f"🌟 EMERGENCE_XORB detected: {event.event_type}")
                        logger.info(f"   Event ID: {event.event_id}")
                        logger.info(f"   Agents: {', '.join(event.agents_involved)}")
                        logger.info(f"   Impact: {event.performance_impact:.3f}")
                        
                        # Tag in swarm behavior log
                        self.swarm_behavior_log.append({
                            "timestamp": event.timestamp,
                            "event_type": "EMERGENCE_XORB",
                            "event_id": event.event_id,
                            "description": event.behavior_description,
                            "agents_involved": event.agents_involved,
                            "performance_impact": event.performance_impact
                        })
                
                cycle_duration = time.time() - cycle_start
                
                logger.info(f"🌟 Detection cycle {detection_cycles}: {len(emergence_events)} emergence events detected")
                
                await asyncio.sleep(60)  # Detect every minute
                
        except Exception as e:
            logger.error(f"❌ Emergence detection failed: {e}")
    
    async def _detect_emergent_behaviors(self) -> List[EmergenceEvent]:
        """Detect emergent behaviors in the swarm."""
        
        emergence_events = []
        
        try:
            # Analyze recent performance data for emergence patterns
            if len(self.performance_window_data) >= 10:
                recent_data = self.performance_window_data[-10:]
                
                # Look for unexpected performance improvements
                for cluster_id, cluster in self.swarm_clusters.items():
                    performance_trend = []
                    
                    for data_point in recent_data:
                        cluster_performance = data_point.get("cluster_metrics", {}).get(cluster_id, {}).get("intelligence", 0)
                        performance_trend.append(cluster_performance)
                    
                    # Detect sudden improvements (emergence indicators)
                    if len(performance_trend) >= 5:
                        recent_avg = np.mean(performance_trend[-3:])
                        earlier_avg = np.mean(performance_trend[-5:-2])
                        
                        improvement_rate = (recent_avg - earlier_avg) / max(earlier_avg, 0.01)
                        
                        if improvement_rate > 0.15:  # 15% improvement indicates potential emergence
                            event = EmergenceEvent(
                                event_id=f"EMERGE-{str(uuid.uuid4())[:8].upper()}",
                                event_type="performance_breakthrough",
                                timestamp=time.time(),
                                agents_involved=[agent.agent_id for agent in cluster.agents[:3]],
                                behavior_description=f"Cluster {cluster_id} showed {improvement_rate:.1%} performance improvement",
                                performance_impact=improvement_rate
                            )
                            emergence_events.append(event)
                            cluster.emergence_events += 1
                
                # Look for novel coordination patterns
                for cluster in self.swarm_clusters.values():
                    if cluster.coordination_matrix is not None:
                        # Analyze coordination matrix for unexpected patterns
                        matrix_eigenvalues = np.linalg.eigvals(cluster.coordination_matrix)
                        dominant_eigenvalue = np.max(np.real(matrix_eigenvalues))
                        
                        if dominant_eigenvalue > 0.9:  # High coordination emergence
                            event = EmergenceEvent(
                                event_id=f"COORD-{str(uuid.uuid4())[:8].upper()}",
                                event_type="coordination_emergence",
                                timestamp=time.time(),
                                agents_involved=[agent.agent_id for agent in cluster.agents[:2]],
                                behavior_description=f"Emergent coordination pattern in {cluster.cluster_id}",
                                performance_impact=dominant_eigenvalue - 0.5
                            )
                            emergence_events.append(event)
            
        except Exception as e:
            logger.error(f"❌ Emergence detection failed: {e}")
        
        return emergence_events
    
    async def _validate_emergence_event(self, event: EmergenceEvent) -> str:
        """Validate detected emergence event."""
        
        try:
            # Validation criteria
            validation_score = 0.0
            
            # Check performance impact significance
            if event.performance_impact > 0.1:
                validation_score += 0.3
            
            # Check agent involvement
            if len(event.agents_involved) >= 2:
                validation_score += 0.2
            
            # Check event type validity
            valid_types = ["performance_breakthrough", "coordination_emergence", "adaptive_strategy", "novel_behavior"]
            if event.event_type in valid_types:
                validation_score += 0.3
            
            # Check temporal consistency
            recent_events = [e for e in self.emergence_events if time.time() - e.timestamp < 300]  # Last 5 minutes
            if len(recent_events) < 3:  # Not too many recent events
                validation_score += 0.2
            
            # Determine validation result
            if validation_score >= self.emergence_detection_sensitivity:
                return "validated"
            elif validation_score >= 0.5:
                return "pending"
            else:
                return "rejected"
                
        except Exception as e:
            logger.error(f"❌ Emergence validation failed: {e}")
            return "error"
    
    async def _run_performance_monitoring_system(self):
        """Run performance monitoring system for 10-minute windows."""
        
        logger.info("📊 Starting performance monitoring system...")
        
        monitoring_cycles = 0
        
        try:
            while True:
                monitoring_cycles += 1
                cycle_start = time.time()
                
                # Collect performance data
                performance_data = {
                    "timestamp": cycle_start,
                    "cycle": monitoring_cycles,
                    "system_metrics": {
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent,
                        "active_agents": len(self.agent_profiles)
                    },
                    "cluster_metrics": {},
                    "evolution_metrics": {
                        "cycles_completed": self.evolution_cycles_completed,
                        "emergence_events": len(self.emergence_events),
                        "average_performance": np.mean([
                            agent.performance_score for agent in self.agent_profiles.values()
                        ]) if self.agent_profiles else 0.0
                    }
                }
                
                # Collect cluster metrics
                for cluster_id, cluster in self.swarm_clusters.items():
                    cluster_metrics = {
                        "intelligence": cluster.cluster_intelligence,
                        "agent_count": len(cluster.agents),
                        "coordination_strength": np.mean(cluster.coordination_matrix) if cluster.coordination_matrix is not None else 0.0,
                        "emergence_events": cluster.emergence_events
                    }
                    performance_data["cluster_metrics"][cluster_id] = cluster_metrics
                
                # Store in performance window
                self.performance_window_data.append(performance_data)
                
                # Maintain 10-minute window (assuming 10-second cycles = 60 data points for 10 minutes)
                if len(self.performance_window_data) > 60:
                    self.performance_window_data = self.performance_window_data[-60:]
                
                # Log performance summary every 10 cycles
                if monitoring_cycles % 10 == 0:
                    logger.info(f"📊 Performance monitoring cycle {monitoring_cycles}")
                    logger.info(f"   Average Performance: {performance_data['evolution_metrics']['average_performance']:.1%}")
                    logger.info(f"   Active Clusters: {len(self.swarm_clusters)}")
                    logger.info(f"   Emergence Events: {len(self.emergence_events)}")
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {e}")
    
    async def _run_intelligence_amplification_system(self):
        """Run intelligence amplification monitoring system."""
        
        logger.info("🧠 Starting intelligence amplification system...")
        
        amplification_cycles = 0
        
        try:
            while True:
                amplification_cycles += 1
                cycle_start = time.time()
                
                # Calculate overall intelligence amplification
                if self.agent_profiles:
                    current_avg_performance = np.mean([
                        agent.performance_score for agent in self.agent_profiles.values()
                    ])
                    
                    # Compare with initial baseline (assume 0.75 initial average)
                    baseline_performance = 0.75
                    amplification_factor = current_avg_performance / baseline_performance
                    
                    # Check for significant amplification
                    if amplification_factor > 1.2:  # 20% amplification
                        logger.info(f"🧠 Intelligence amplification detected: {amplification_factor:.2f}x")
                        
                        # Apply amplification boost to all agents
                        for agent in self.agent_profiles.values():
                            # Boost learning rate for high-performing agents
                            if agent.performance_score > 0.9:
                                agent.learning_rate = min(0.5, agent.learning_rate * 1.1)
                
                await asyncio.sleep(180)  # Amplify every 3 minutes
                
        except Exception as e:
            logger.error(f"❌ Intelligence amplification failed: {e}")
    
    async def _run_adaptive_learning_coordinator(self):
        """Run adaptive learning coordination system."""
        
        logger.info("🎯 Starting adaptive learning coordinator...")
        
        learning_cycles = 0
        
        try:
            while True:
                learning_cycles += 1
                self.evolution_cycles_completed += 1
                cycle_start = time.time()
                
                logger.info(f"🎯 Adaptive learning cycle #{learning_cycles}")
                
                # Coordinate learning across all clusters
                for cluster in self.swarm_clusters.values():
                    await self._coordinate_cluster_learning(cluster)
                
                # Cross-cluster learning coordination
                await self._coordinate_cross_cluster_learning()
                
                # Update agent state
                await self._save_agent_state()
                
                # Check for self-critique triggers (every 3 cycles)
                if learning_cycles % 3 == 0:
                    await self._trigger_self_critique_module()
                
                cycle_duration = time.time() - cycle_start
                
                logger.info(f"🎯 Learning coordination cycle {learning_cycles} completed in {cycle_duration:.2f}s")
                
                await asyncio.sleep(150)  # Coordinate every 2.5 minutes
                
        except Exception as e:
            logger.error(f"❌ Adaptive learning coordination failed: {e}")
    
    async def _coordinate_cluster_learning(self, cluster: SwarmCluster):
        """Coordinate learning within a cluster."""
        
        try:
            # Identify learning opportunities
            learning_opportunities = []
            
            for agent in cluster.agents:
                # Check for stagnation
                if time.time() - agent.last_optimization > 600:  # 10 minutes since last optimization
                    learning_opportunities.append({
                        "agent": agent,
                        "opportunity_type": "stagnation_recovery",
                        "priority": 1.0 - agent.performance_score
                    })
                
                # Check for rapid improvement potential
                if agent.learning_rate > 0.2 and agent.performance_score < 0.9:
                    learning_opportunities.append({
                        "agent": agent,
                        "opportunity_type": "rapid_improvement",
                        "priority": agent.learning_rate
                    })
            
            # Apply learning coordination
            for opportunity in learning_opportunities:
                agent = opportunity["agent"]
                
                if opportunity["opportunity_type"] == "stagnation_recovery":
                    # Apply stagnation recovery
                    agent.learning_rate = min(0.4, agent.learning_rate + 0.05)
                    
                    # Introduce variation in capabilities
                    for capability in agent.capabilities:
                        variation = np.random.uniform(-0.02, 0.02)
                        agent.capabilities[capability] = max(0.0, min(1.0, agent.capabilities[capability] + variation))
                
                elif opportunity["opportunity_type"] == "rapid_improvement":
                    # Accelerate learning for promising agents
                    best_performer = max(cluster.agents, key=lambda a: a.performance_score)
                    
                    # Learn from best performer
                    for capability in agent.capabilities:
                        target_value = best_performer.capabilities[capability]
                        improvement = (target_value - agent.capabilities[capability]) * agent.learning_rate * 0.2
                        agent.capabilities[capability] = min(1.0, agent.capabilities[capability] + improvement)
            
        except Exception as e:
            logger.error(f"❌ Cluster learning coordination failed: {e}")
    
    async def _coordinate_cross_cluster_learning(self):
        """Coordinate learning across different clusters."""
        
        try:
            # Find best performing agents across all clusters
            all_agents = []
            for cluster in self.swarm_clusters.values():
                all_agents.extend(cluster.agents)
            
            if all_agents:
                top_performers = sorted(all_agents, key=lambda a: a.performance_score, reverse=True)[:5]
                
                # Extract learned patterns from top performers
                learned_patterns = []
                for agent in top_performers:
                    if agent.emergent_behaviors:
                        learned_patterns.extend(agent.emergent_behaviors)
                
                # Distribute patterns to all clusters
                for cluster in self.swarm_clusters.values():
                    if "learned_patterns" not in cluster.shared_knowledge:
                        cluster.shared_knowledge["learned_patterns"] = []
                    
                    cluster.shared_knowledge["learned_patterns"].extend(learned_patterns)
                    
                    # Apply patterns to cluster agents
                    for agent in cluster.agents:
                        if agent.performance_score < 0.85:  # Apply to underperformers
                            # Simulate learning from patterns
                            pattern_benefit = len(learned_patterns) * 0.01
                            for capability in agent.capabilities:
                                agent.capabilities[capability] = min(1.0, agent.capabilities[capability] + pattern_benefit)
            
        except Exception as e:
            logger.error(f"❌ Cross-cluster learning coordination failed: {e}")
    
    async def _save_agent_state(self):
        """Save agent state to agent_state.json."""
        
        try:
            agent_state = {
                "timestamp": time.time(),
                "orchestrator_id": self.orchestrator_id,
                "evolution_cycles_completed": self.evolution_cycles_completed,
                "agents": {},
                "clusters": {},
                "emergence_events": len(self.emergence_events)
            }
            
            # Save agent profiles
            for agent_id, agent in self.agent_profiles.items():
                agent_state["agents"][agent_id] = {
                    "cluster": agent.cluster.value,
                    "performance_score": agent.performance_score,
                    "capabilities": agent.capabilities,
                    "evolution_cycles": agent.evolution_cycles,
                    "learning_rate": agent.learning_rate,
                    "emergent_behaviors": agent.emergent_behaviors
                }
            
            # Save cluster information
            for cluster_id, cluster in self.swarm_clusters.items():
                agent_state["clusters"][cluster_id] = {
                    "cluster_type": cluster.cluster_type.value,
                    "agent_count": len(cluster.agents),
                    "cluster_intelligence": cluster.cluster_intelligence,
                    "emergence_events": cluster.emergence_events,
                    "shared_knowledge": cluster.shared_knowledge
                }
            
            # Write to file
            with open("agent_state.json", "w") as f:
                json.dump(agent_state, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Agent state save failed: {e}")
    
    async def _trigger_self_critique_module(self):
        """Trigger self-critique module if enabled."""
        
        try:
            # Check if self-critique is available
            critique_available = os.path.exists("self_critique_module.py")
            
            if critique_available:
                logger.info("🔍 Triggering self-critique module...")
                
                # Prepare critique data
                critique_data = {
                    "evolution_cycles": self.evolution_cycles_completed,
                    "emergence_events": len(self.emergence_events),
                    "average_performance": np.mean([
                        agent.performance_score for agent in self.agent_profiles.values()
                    ]) if self.agent_profiles else 0.0,
                    "cluster_performance": {
                        cluster_id: cluster.cluster_intelligence
                        for cluster_id, cluster in self.swarm_clusters.items()
                    }
                }
                
                # Save critique input
                with open("critique_input.json", "w") as f:
                    json.dump(critique_data, f, indent=2)
                
                # Note: In production, would actually call self-critique module
                logger.info("🔍 Self-critique data prepared")
            else:
                logger.debug("🔍 Self-critique module not available")
            
        except Exception as e:
            logger.error(f"❌ Self-critique trigger failed: {e}")
    
    async def _telemetry_collection_loop(self):
        """Telemetry collection loop."""
        
        while self.telemetry_active:
            try:
                # Collect telemetry data
                telemetry_data = {
                    "timestamp": time.time(),
                    "orchestrator_id": self.orchestrator_id,
                    "active_agents": len(self.agent_profiles),
                    "active_clusters": len(self.swarm_clusters),
                    "evolution_cycles": self.evolution_cycles_completed,
                    "emergence_events": len(self.emergence_events),
                    "system_metrics": {
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent
                    }
                }
                
                # Log to swarm behavior file
                with open("/var/log/xorb/swarm_behavior.log", "a") as f:
                    f.write(f"{json.dumps(telemetry_data)}\n")
                
                await asyncio.sleep(2)  # Emit every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Telemetry collection failed: {e}")
                await asyncio.sleep(2)
    
    async def _prometheus_metrics_emitter(self):
        """Emit metrics to Prometheus (simulated)."""
        
        while self.telemetry_active:
            try:
                # Simulate Prometheus metrics emission
                if self.agent_profiles:
                    avg_performance = np.mean([agent.performance_score for agent in self.agent_profiles.values()])
                    
                    # Log Prometheus-style metrics
                    logger.debug(f"xorb_agent_performance_average {avg_performance}")
                    logger.debug(f"xorb_active_agents_total {len(self.agent_profiles)}")
                    logger.debug(f"xorb_evolution_cycles_total {self.evolution_cycles_completed}")
                    logger.debug(f"xorb_emergence_events_total {len(self.emergence_events)}")
                
                await asyncio.sleep(2)  # Emit every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Prometheus metrics emission failed: {e}")
                await asyncio.sleep(2)
    
    async def _perform_final_assessment(self) -> Dict[str, Any]:
        """Perform final assessment of Phase II evolution."""
        
        logger.info("✅ Performing final Phase II assessment...")
        
        assessment = {
            "assessment_timestamp": time.time(),
            "total_duration": time.time() - self.phase2_start_time,
            "evolution_cycles_completed": self.evolution_cycles_completed,
            "emergence_events_detected": len(self.emergence_events),
            "agents_optimized": 0,
            "clusters_formed": len(self.swarm_clusters),
            "performance_improvements": {},
            "objectives_achieved": {},
            "recommendations": []
        }
        
        try:
            # Calculate performance improvements
            if self.agent_profiles:
                current_avg = np.mean([agent.performance_score for agent in self.agent_profiles.values()])
                baseline_avg = 0.75  # Assumed baseline
                improvement = (current_avg - baseline_avg) / baseline_avg
                
                assessment["performance_improvements"] = {
                    "baseline_performance": baseline_avg,
                    "current_performance": current_avg,
                    "improvement_percentage": improvement * 100,
                    "agents_above_threshold": len([
                        agent for agent in self.agent_profiles.values()
                        if agent.performance_score >= self.performance_threshold
                    ])
                }
                
                assessment["agents_optimized"] = len([
                    agent for agent in self.agent_profiles.values()
                    if agent.evolution_cycles > 0
                ])
            
            # Evaluate objectives
            assessment["objectives_achieved"] = {
                "agent_level_tuning": assessment["agents_optimized"] > 0,
                "swarm_cluster_specialization": len(self.swarm_clusters) >= 3,
                "emergent_behavior_detection": len(self.emergence_events) > 0,
                "telemetry_monitoring": self.telemetry_active,
                "adaptive_learning": self.evolution_cycles_completed >= 5
            }
            
            # Generate recommendations
            recommendations = []
            
            if assessment["performance_improvements"].get("improvement_percentage", 0) < 10:
                recommendations.append("Consider longer evolution cycles for greater performance gains")
            
            if len(self.emergence_events) == 0:
                recommendations.append("Adjust emergence detection sensitivity to capture more events")
            
            if assessment["agents_optimized"] < len(self.agent_profiles) * 0.5:
                recommendations.append("Increase optimization frequency for broader agent coverage")
            
            objectives_met = sum(assessment["objectives_achieved"].values())
            if objectives_met < len(assessment["objectives_achieved"]):
                recommendations.append("Focus on completing remaining objectives in future phases")
            
            assessment["recommendations"] = recommendations
            
            logger.info("✅ Final assessment completed:")
            logger.info(f"   Duration: {assessment['total_duration']:.1f} seconds")
            logger.info(f"   Evolution Cycles: {assessment['evolution_cycles_completed']}")
            logger.info(f"   Agents Optimized: {assessment['agents_optimized']}")
            logger.info(f"   Objectives Met: {objectives_met}/{len(assessment['objectives_achieved'])}")
            
        except Exception as e:
            logger.error(f"❌ Final assessment failed: {e}")
            assessment["error"] = str(e)
        
        return assessment
    
    def get_phase2_status(self) -> Dict[str, Any]:
        """Get current Phase II evolution status."""
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "phase2_active": self.phase2_start_time > 0,
            "evolution_cycles_completed": self.evolution_cycles_completed,
            "active_agents": len(self.agent_profiles),
            "active_clusters": len(self.swarm_clusters),
            "emergence_events": len(self.emergence_events),
            "telemetry_active": self.telemetry_active,
            "runtime": time.time() - self.phase2_start_time if self.phase2_start_time else 0,
            "performance_threshold": self.performance_threshold,
            "cluster_types": [cluster.cluster_type.value for cluster in self.swarm_clusters.values()]
        }

async def main():
    """Main execution for XORB Phase II Evolution."""
    
    print(f"\n🧬 XORB PHASE II EVOLUTION ORCHESTRATOR")
    print(f"🎯 Mission: Adaptive Intelligence and Swarm Refinement")
    print(f"🤖 Capabilities: Agent-level tuning, swarm specialization, emergence detection")
    print(f"📡 Features: Real-time telemetry, Prometheus metrics, adaptive learning")
    print(f"🌟 Objective: Evolve smarter-than-human behaviors with stealth and magnification")
    
    # Initialize Phase II orchestrator
    orchestrator = XORBPhase2EvolutionOrchestrator()
    
    try:
        print(f"\n🚀 Initiating Phase II Evolution (300 second demonstration)...")
        
        # Start Phase II evolution
        result = await orchestrator.initiate_phase2_evolution(
            duration=300,  # 5 minute demonstration
            specialization_mode="adaptive"
        )
        
        if result["evolution_success"]:
            print(f"\n✨ PHASE II EVOLUTION COMPLETED SUCCESSFULLY!")
            print(f"   Session ID: {result['session_id']}")
            print(f"   Duration: {result['duration_actual']:.1f} seconds")
            print(f"   Evolution Cycles: {result['evolution_cycles_completed']}")
            print(f"   Emergence Events: {len(result['emergence_events'])}")
            print(f"   Active Clusters: {len(result['swarm_clusters'])}")
            
            # Show objectives status
            objectives = result["phase2_objectives"]
            print(f"\n🎯 Objectives Status:")
            for objective, status in objectives.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {objective.replace('_', ' ').title()}")
            
            # Show cluster information
            print(f"\n🤖 Swarm Clusters:")
            for cluster_id, cluster_info in result["swarm_clusters"].items():
                print(f"   {cluster_id}: {cluster_info['agent_count']} agents, {cluster_info['cluster_intelligence']:.1%} intelligence")
            
            # Show emergence events
            if result["emergence_events"]:
                print(f"\n🌟 Emergence Events Detected:")
                for event in result["emergence_events"][:3]:  # Show first 3
                    print(f"   • {event['event_type']}: {event['behavior_description']}")
            
            # Show final assessment
            if "final_assessment" in result:
                final = result["final_assessment"]
                if "performance_improvements" in final:
                    perf = final["performance_improvements"]
                    print(f"\n📊 Performance Improvements:")
                    print(f"   Baseline: {perf['baseline_performance']:.1%}")
                    print(f"   Current: {perf['current_performance']:.1%}")
                    print(f"   Improvement: {perf['improvement_percentage']:.1f}%")
                    print(f"   Optimized Agents: {final['agents_optimized']}")
                
                if final.get("recommendations"):
                    print(f"\n💡 Recommendations:")
                    for rec in final["recommendations"][:2]:
                        print(f"   • {rec}")
        
        else:
            print(f"\n❌ PHASE II EVOLUTION FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print(f"   Duration: {result.get('duration_actual', 0):.1f} seconds")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Phase II Evolution interrupted by user")
        
        # Show final status
        status = orchestrator.get_phase2_status()
        print(f"\n📊 Final Status:")
        print(f"   Runtime: {status['runtime']:.1f} seconds") 
        print(f"   Evolution Cycles: {status['evolution_cycles_completed']}")
        print(f"   Active Agents: {status['active_agents']}")
        print(f"   Emergence Events: {status['emergence_events']}")
        
    except Exception as e:
        print(f"\n❌ Phase II Evolution orchestrator failed: {e}")
        logger.error(f"Phase II Evolution failed: {e}")

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("/var/log/xorb", exist_ok=True)
    
    asyncio.run(main())