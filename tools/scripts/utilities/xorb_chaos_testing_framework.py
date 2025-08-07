#!/usr/bin/env python3
"""
XORB Chaos Testing & Fault Injection Framework
Comprehensive chaos engineering and fault injection system for testing error handling
"""

import asyncio
import json
import logging
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import psutil
import socket

# Import error handling framework
from xorb_error_handling_framework import (
    XORBErrorHandler, ErrorCategory, ErrorSeverity, get_error_handler
)

class FaultType(Enum):
    """Types of faults that can be injected"""
    NETWORK_DELAY = "network_delay"
    NETWORK_FAILURE = "network_failure"
    DATABASE_SLOW = "database_slow"
    DATABASE_FAILURE = "database_failure"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    SERVICE_CRASH = "service_crash"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_FAILURE = "authentication_failure"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"
    DEPENDENCY_FAILURE = "dependency_failure"
    DATA_CORRUPTION = "data_corruption"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

class FaultScope(Enum):
    """Scope of fault injection"""
    SINGLE_REQUEST = "single_request"
    SINGLE_SERVICE = "single_service"
    MULTIPLE_SERVICES = "multiple_services"
    SYSTEM_WIDE = "system_wide"
    USER_SPECIFIC = "user_specific"
    TIME_BASED = "time_based"

class ChaosExperimentStatus(Enum):
    """Status of chaos experiment"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    PAUSED = "paused"

@dataclass
class FaultInjectionRule:
    """Rule for fault injection"""
    rule_id: str
    name: str
    description: str
    fault_type: FaultType
    scope: FaultScope
    target_services: List[str]
    target_functions: List[str]
    probability: float  # 0.0 to 1.0
    duration_seconds: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    safety_checks: List[str] = field(default_factory=list)
    recovery_action: Optional[str] = None

@dataclass
class ChaosExperiment:
    """Chaos experiment definition"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    fault_rules: List[FaultInjectionRule]
    duration_minutes: int
    steady_state_checks: List[str]
    rollback_criteria: List[str]
    status: ChaosExperimentStatus = ChaosExperimentStatus.PLANNED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)

@dataclass
class InjectedFault:
    """Active fault injection"""
    fault_id: str
    rule: FaultInjectionRule
    injected_at: datetime
    expires_at: datetime
    target_service: str
    target_function: str
    active: bool = True
    impact_metrics: Dict[str, Any] = field(default_factory=dict)

class NetworkFaultInjector:
    """Network-related fault injection"""
    
    def __init__(self):
        self.active_delays: Dict[str, float] = {}
        self.blocked_endpoints: Set[str] = set()
        
    async def inject_network_delay(self, target: str, delay_ms: int):
        """Inject network delay"""
        self.active_delays[target] = delay_ms / 1000.0
        
    async def inject_network_failure(self, target: str):
        """Inject network failure"""
        self.blocked_endpoints.add(target)
        
    async def clear_network_faults(self, target: str = None):
        """Clear network faults"""
        if target:
            self.active_delays.pop(target, None)
            self.blocked_endpoints.discard(target)
        else:
            self.active_delays.clear()
            self.blocked_endpoints.clear()
    
    def should_inject_delay(self, target: str) -> float:
        """Check if delay should be injected"""
        return self.active_delays.get(target, 0.0)
    
    def should_block_request(self, target: str) -> bool:
        """Check if request should be blocked"""
        return target in self.blocked_endpoints

class DatabaseFaultInjector:
    """Database-related fault injection"""
    
    def __init__(self):
        self.query_delays: Dict[str, float] = {}
        self.connection_failures: Set[str] = set()
        self.corruption_patterns: Dict[str, str] = {}
        
    async def inject_slow_queries(self, database: str, delay_ms: int):
        """Inject slow database queries"""
        self.query_delays[database] = delay_ms / 1000.0
        
    async def inject_connection_failure(self, database: str):
        """Inject database connection failure"""
        self.connection_failures.add(database)
        
    async def inject_data_corruption(self, database: str, pattern: str):
        """Inject data corruption"""
        self.corruption_patterns[database] = pattern
        
    async def clear_database_faults(self, database: str = None):
        """Clear database faults"""
        if database:
            self.query_delays.pop(database, None)
            self.connection_failures.discard(database)
            self.corruption_patterns.pop(database, None)
        else:
            self.query_delays.clear()
            self.connection_failures.clear()
            self.corruption_patterns.clear()

class ResourceFaultInjector:
    """System resource fault injection"""
    
    def __init__(self):
        self.memory_pressure_active = False
        self.cpu_spike_active = False
        self.disk_full_simulation = False
        self.memory_hog_task = None
        self.cpu_hog_task = None
        
    async def inject_memory_pressure(self, target_mb: int, duration_seconds: int):
        """Inject memory pressure"""
        if self.memory_pressure_active:
            return
            
        self.memory_pressure_active = True
        
        # Start memory hogging task
        self.memory_hog_task = asyncio.create_task(
            self._memory_hog_worker(target_mb, duration_seconds)
        )
        
    async def inject_cpu_spike(self, target_percent: int, duration_seconds: int):
        """Inject CPU spike"""
        if self.cpu_spike_active:
            return
            
        self.cpu_spike_active = True
        
        # Start CPU hogging task
        self.cpu_hog_task = asyncio.create_task(
            self._cpu_hog_worker(target_percent, duration_seconds)
        )
        
    async def inject_disk_full(self, enabled: bool = True):
        """Simulate disk full condition"""
        self.disk_full_simulation = enabled
        
    async def _memory_hog_worker(self, target_mb: int, duration_seconds: int):
        """Worker to consume memory"""
        try:
            # Allocate memory
            memory_blocks = []
            block_size = 1024 * 1024  # 1MB blocks
            
            for _ in range(target_mb):
                memory_blocks.append(bytearray(block_size))
                await asyncio.sleep(0.01)  # Yield control
            
            # Hold memory for duration
            await asyncio.sleep(duration_seconds)
            
        except Exception as e:
            logging.error(f"Memory pressure injection failed: {e}")
        finally:
            self.memory_pressure_active = False
            
    async def _cpu_hog_worker(self, target_percent: int, duration_seconds: int):
        """Worker to consume CPU"""
        try:
            end_time = time.time() + duration_seconds
            work_time = target_percent / 100.0
            sleep_time = 1.0 - work_time
            
            while time.time() < end_time:
                # Do CPU-intensive work
                start = time.time()
                while time.time() - start < work_time:
                    _ = sum(range(1000))  # Busy work
                
                # Sleep to control CPU usage
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logging.error(f"CPU spike injection failed: {e}")
        finally:
            self.cpu_spike_active = False
            
    async def clear_resource_faults(self):
        """Clear all resource faults"""
        self.memory_pressure_active = False
        self.cpu_spike_active = False
        self.disk_full_simulation = False
        
        if self.memory_hog_task:
            self.memory_hog_task.cancel()
        if self.cpu_hog_task:
            self.cpu_hog_task.cancel()

class ChaosTestingFramework:
    """Main chaos testing and fault injection framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.error_handler = get_error_handler("chaos_testing")
        
        # Fault injectors
        self.network_injector = NetworkFaultInjector()
        self.database_injector = DatabaseFaultInjector()
        self.resource_injector = ResourceFaultInjector()
        
        # Experiment management
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_faults: Dict[str, InjectedFault] = {}
        self.fault_rules: Dict[str, FaultInjectionRule] = {}
        
        # Safety mechanisms
        self.safety_enabled = True
        self.max_concurrent_faults = 5
        self.emergency_stop = False
        
        # Metrics and monitoring
        self.metrics_collector = MetricsCollector()
        self.experiment_results: List[Dict[str, Any]] = []
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="chaos-testing")
        
        # Initialize default fault rules
        self._initialize_default_fault_rules()
        
        # Start background monitoring
        self._start_background_tasks()
        
        self.logger = logging.getLogger("xorb.chaos_testing")
        self.logger.info("XORB Chaos Testing Framework initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "enabled": True,
            "safety_checks_enabled": True,
            "max_concurrent_experiments": 3,
            "default_experiment_duration": 300,  # 5 minutes
            "metrics_collection_interval": 10,  # seconds
            "safety_thresholds": {
                "max_error_rate": 0.5,  # 50% error rate
                "max_response_time": 10.0,  # 10 seconds
                "min_success_rate": 0.1  # 10% success rate
            },
            "target_services": [
                "neural_orchestrator",
                "learning_service",
                "threat_detection"
            ]
        }
    
    def _initialize_default_fault_rules(self):
        """Initialize default fault injection rules"""
        default_rules = [
            FaultInjectionRule(
                rule_id="network_delay_light",
                name="Light Network Delay",
                description="Inject 100-500ms network delay",
                fault_type=FaultType.NETWORK_DELAY,
                scope=FaultScope.SINGLE_SERVICE,
                target_services=["neural_orchestrator"],
                target_functions=["orchestrate_agents"],
                probability=0.1,
                duration_seconds=30,
                parameters={"delay_ms": [100, 500]},
                safety_checks=["check_error_rate", "check_response_time"]
            ),
            FaultInjectionRule(
                rule_id="database_slow_queries",
                name="Slow Database Queries",
                description="Inject database query delays",
                fault_type=FaultType.DATABASE_SLOW,
                scope=FaultScope.SINGLE_SERVICE,
                target_services=["learning_service"],
                target_functions=["store_knowledge", "retrieve_knowledge"],
                probability=0.05,
                duration_seconds=60,
                parameters={"delay_ms": [200, 1000]},
                safety_checks=["check_database_health"]
            ),
            FaultInjectionRule(
                rule_id="memory_pressure",
                name="Memory Pressure",
                description="Create memory pressure on system",
                fault_type=FaultType.MEMORY_PRESSURE,
                scope=FaultScope.SYSTEM_WIDE,
                target_services=["*"],
                target_functions=["*"],
                probability=0.02,
                duration_seconds=120,
                parameters={"target_mb": 512},
                safety_checks=["check_system_resources"]
            ),
            FaultInjectionRule(
                rule_id="timeout_errors",
                name="Timeout Errors",
                description="Inject timeout errors",
                fault_type=FaultType.TIMEOUT_ERROR,
                scope=FaultScope.SINGLE_REQUEST,
                target_services=["neural_orchestrator", "learning_service"],
                target_functions=["*"],
                probability=0.03,
                duration_seconds=10,
                parameters={"timeout_ms": 1000},
                safety_checks=["check_error_rate"]
            ),
            FaultInjectionRule(
                rule_id="validation_errors",
                name="Validation Errors",
                description="Inject validation errors",
                fault_type=FaultType.VALIDATION_ERROR,
                scope=FaultScope.SINGLE_REQUEST,
                target_services=["*"],
                target_functions=["*"],
                probability=0.05,
                duration_seconds=5,
                parameters={"error_message": "Chaos-injected validation error"},
                safety_checks=["check_error_rate"]
            )
        ]
        
        for rule in default_rules:
            self.fault_rules[rule.rule_id] = rule
        
        self.logger.info(f"Initialized {len(default_rules)} default fault rules")
    
    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks"""
        self.executor.submit(self._fault_monitor_loop)
        self.executor.submit(self._metrics_collection_loop)
        self.executor.submit(self._safety_monitor_loop)
    
    async def create_chaos_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Create a new chaos experiment"""
        try:
            # Validate experiment configuration
            if not self._validate_experiment_config(experiment_config):
                raise ValueError("Invalid experiment configuration")
            
            # Create experiment
            experiment = ChaosExperiment(
                experiment_id=str(uuid.uuid4()),
                name=experiment_config["name"],
                description=experiment_config["description"],
                hypothesis=experiment_config.get("hypothesis", "System should remain stable under fault conditions"),
                fault_rules=[
                    self.fault_rules[rule_id] for rule_id in experiment_config.get("fault_rules", [])
                    if rule_id in self.fault_rules
                ],
                duration_minutes=experiment_config.get("duration_minutes", 5),
                steady_state_checks=experiment_config.get("steady_state_checks", [
                    "check_error_rate", "check_response_time", "check_service_health"
                ]),
                rollback_criteria=experiment_config.get("rollback_criteria", [
                    "error_rate > 0.5", "response_time > 10", "service_unavailable"
                ])
            )
            
            # Safety checks
            if self.safety_enabled and not self._pre_experiment_safety_check(experiment):
                raise ValueError("Experiment failed safety checks")
            
            # Store experiment
            self.experiments[experiment.experiment_id] = experiment
            
            self.logger.info(f"Created chaos experiment: {experiment.name} (ID: {experiment.experiment_id})")
            return experiment.experiment_id
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH,
                context={"operation": "create_chaos_experiment", "config": experiment_config}
            )
            raise e
    
    async def start_chaos_experiment(self, experiment_id: str) -> bool:
        """Start a chaos experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ChaosExperimentStatus.PLANNED:
                raise ValueError(f"Experiment not in planned state: {experiment.status}")
            
            # Final safety check
            if self.safety_enabled and not self._pre_experiment_safety_check(experiment):
                raise ValueError("Experiment failed pre-execution safety checks")
            
            # Collect baseline metrics
            experiment.metrics_before = await self.metrics_collector.collect_all_metrics()
            
            # Start experiment
            experiment.status = ChaosExperimentStatus.RUNNING
            experiment.start_time = datetime.now()
            
            # Schedule fault injections
            await self._schedule_fault_injections(experiment)
            
            # Start experiment monitoring task
            self.executor.submit(self._experiment_monitor_loop, experiment_id)
            
            self.logger.info(f"Started chaos experiment: {experiment.name}")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.HIGH,
                context={"operation": "start_chaos_experiment", "experiment_id": experiment_id}
            )
            return False
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> bool:
        """Validate experiment configuration"""
        required_fields = ["name", "description"]
        
        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate duration
        duration = config.get("duration_minutes", 5)
        if duration > 60:  # Max 1 hour
            self.logger.error(f"Experiment duration too long: {duration} minutes")
            return False
        
        return True
    
    def _pre_experiment_safety_check(self, experiment: ChaosExperiment) -> bool:
        """Perform safety checks before starting experiment"""
        try:
            # Check system health
            if not self._check_system_health():
                self.logger.error("System health check failed")
                return False
            
            # Check current load
            if self._get_current_load() > 0.8:  # 80% load threshold
                self.logger.error("System load too high for chaos experiment")
                return False
            
            # Check existing experiments
            running_experiments = [
                exp for exp in self.experiments.values()
                if exp.status == ChaosExperimentStatus.RUNNING
            ]
            
            if len(running_experiments) >= self.config.get("max_concurrent_experiments", 3):
                self.logger.error("Too many concurrent experiments")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    async def _schedule_fault_injections(self, experiment: ChaosExperiment):
        """Schedule fault injections for an experiment"""
        try:
            for rule in experiment.fault_rules:
                if not rule.enabled:
                    continue
                
                # Calculate injection times
                experiment_duration = experiment.duration_minutes * 60
                injection_count = max(1, experiment_duration // rule.duration_seconds)
                
                for i in range(injection_count):
                    # Random timing within experiment duration
                    delay = random.uniform(0, experiment_duration - rule.duration_seconds)
                    
                    # Schedule injection
                    asyncio.create_task(
                        self._delayed_fault_injection(delay, rule, experiment.experiment_id)
                    )
            
        except Exception as e:
            self.logger.error(f"Fault injection scheduling failed: {e}")
    
    async def _delayed_fault_injection(self, delay: float, rule: FaultInjectionRule, experiment_id: str):
        """Execute delayed fault injection"""
        try:
            await asyncio.sleep(delay)
            
            # Check if experiment is still running
            experiment = self.experiments.get(experiment_id)
            if not experiment or experiment.status != ChaosExperimentStatus.RUNNING:
                return
            
            # Check probability
            if random.random() > rule.probability:
                return
            
            # Select target
            target_service = random.choice(rule.target_services) if rule.target_services != ["*"] else "system"
            target_function = random.choice(rule.target_functions) if rule.target_functions != ["*"] else "all"
            
            # Inject fault
            await self._inject_fault(rule, target_service, target_function, experiment_id)
            
        except Exception as e:
            self.logger.error(f"Delayed fault injection failed: {e}")
    
    async def _inject_fault(self, rule: FaultInjectionRule, target_service: str, target_function: str, experiment_id: str):
        """Inject a specific fault"""
        try:
            # Safety check
            if len(self.active_faults) >= self.max_concurrent_faults:
                self.logger.warning("Maximum concurrent faults reached, skipping injection")
                return
            
            # Create fault instance
            fault = InjectedFault(
                fault_id=str(uuid.uuid4()),
                rule=rule,
                injected_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=rule.duration_seconds),
                target_service=target_service,
                target_function=target_function
            )
            
            # Execute fault injection based on type
            success = await self._execute_fault_injection(fault)
            
            if success:
                self.active_faults[fault.fault_id] = fault
                self.logger.info(f"Injected fault: {rule.name} on {target_service}.{target_function}")
                
                # Schedule cleanup
                asyncio.create_task(self._cleanup_fault_after_delay(fault.fault_id, rule.duration_seconds))
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM,
                context={"operation": "inject_fault", "rule": rule.name}
            )
    
    async def _execute_fault_injection(self, fault: InjectedFault) -> bool:
        """Execute the actual fault injection"""
        try:
            rule = fault.rule
            params = rule.parameters
            
            if rule.fault_type == FaultType.NETWORK_DELAY:
                delay_range = params.get("delay_ms", [100, 500])
                delay = random.randint(delay_range[0], delay_range[1])
                await self.network_injector.inject_network_delay(fault.target_service, delay)
                
            elif rule.fault_type == FaultType.NETWORK_FAILURE:
                await self.network_injector.inject_network_failure(fault.target_service)
                
            elif rule.fault_type == FaultType.DATABASE_SLOW:
                delay_range = params.get("delay_ms", [200, 1000])
                delay = random.randint(delay_range[0], delay_range[1])
                await self.database_injector.inject_slow_queries(fault.target_service, delay)
                
            elif rule.fault_type == FaultType.DATABASE_FAILURE:
                await self.database_injector.inject_connection_failure(fault.target_service)
                
            elif rule.fault_type == FaultType.MEMORY_PRESSURE:
                target_mb = params.get("target_mb", 256)
                await self.resource_injector.inject_memory_pressure(target_mb, rule.duration_seconds)
                
            elif rule.fault_type == FaultType.CPU_SPIKE:
                target_percent = params.get("target_percent", 80)
                await self.resource_injector.inject_cpu_spike(target_percent, rule.duration_seconds)
                
            elif rule.fault_type == FaultType.TIMEOUT_ERROR:
                # This would be handled by middleware/decorators in actual services
                pass
                
            elif rule.fault_type == FaultType.VALIDATION_ERROR:
                # This would be handled by middleware/decorators in actual services
                pass
                
            else:
                self.logger.warning(f"Unsupported fault type: {rule.fault_type}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fault injection execution failed: {e}")
            return False
    
    async def _cleanup_fault_after_delay(self, fault_id: str, delay_seconds: int):
        """Cleanup fault after delay"""
        try:
            await asyncio.sleep(delay_seconds)
            await self.cleanup_fault(fault_id)
        except Exception as e:
            self.logger.error(f"Fault cleanup failed: {e}")
    
    async def cleanup_fault(self, fault_id: str):
        """Cleanup a specific fault"""
        try:
            if fault_id not in self.active_faults:
                return
            
            fault = self.active_faults[fault_id]
            fault.active = False
            
            # Cleanup based on fault type
            rule = fault.rule
            
            if rule.fault_type in [FaultType.NETWORK_DELAY, FaultType.NETWORK_FAILURE]:
                await self.network_injector.clear_network_faults(fault.target_service)
                
            elif rule.fault_type in [FaultType.DATABASE_SLOW, FaultType.DATABASE_FAILURE]:
                await self.database_injector.clear_database_faults(fault.target_service)
                
            elif rule.fault_type in [FaultType.MEMORY_PRESSURE, FaultType.CPU_SPIKE]:
                await self.resource_injector.clear_resource_faults()
            
            # Remove from active faults
            del self.active_faults[fault_id]
            
            self.logger.info(f"Cleaned up fault: {fault_id}")
            
        except Exception as e:
            self.logger.error(f"Fault cleanup failed: {e}")
    
    async def stop_chaos_experiment(self, experiment_id: str, reason: str = "manual_stop") -> bool:
        """Stop a chaos experiment"""
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ChaosExperimentStatus.RUNNING:
                return False
            
            # Update experiment status
            experiment.status = ChaosExperimentStatus.COMPLETED if reason == "natural_end" else ChaosExperimentStatus.ABORTED
            experiment.end_time = datetime.now()
            
            # Cleanup all faults for this experiment
            await self._cleanup_experiment_faults(experiment_id)
            
            # Collect final metrics
            experiment.metrics_after = await self.metrics_collector.collect_all_metrics()
            
            # Generate experiment results
            experiment.results = self._generate_experiment_results(experiment)
            
            # Store results
            self.experiment_results.append({
                "experiment_id": experiment_id,
                "name": experiment.name,
                "status": experiment.status.value,
                "duration": (experiment.end_time - experiment.start_time).total_seconds(),
                "results": experiment.results,
                "reason": reason
            })
            
            self.logger.info(f"Stopped chaos experiment: {experiment.name} (Reason: {reason})")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e, ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.HIGH,
                context={"operation": "stop_chaos_experiment", "experiment_id": experiment_id}
            )
            return False
    
    async def _cleanup_experiment_faults(self, experiment_id: str):
        """Cleanup all faults associated with an experiment"""
        try:
            experiment_faults = [
                fault_id for fault_id, fault in self.active_faults.items()
                # We'd need to track experiment association in the fault
            ]
            
            for fault_id in experiment_faults:
                await self.cleanup_fault(fault_id)
            
        except Exception as e:
            self.logger.error(f"Experiment fault cleanup failed: {e}")
    
    def _generate_experiment_results(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Generate experiment results and analysis"""
        try:
            # Calculate metrics deltas
            metrics_delta = {}
            for metric, after_value in experiment.metrics_after.items():
                before_value = experiment.metrics_before.get(metric, 0)
                metrics_delta[metric] = {
                    "before": before_value,
                    "after": after_value,
                    "delta": after_value - before_value,
                    "change_percent": ((after_value - before_value) / before_value * 100) if before_value > 0 else 0
                }
            
            # Analyze hypothesis
            hypothesis_validated = self._validate_experiment_hypothesis(experiment, metrics_delta)
            
            # Generate insights
            insights = self._generate_experiment_insights(experiment, metrics_delta)
            
            return {
                "hypothesis_validated": hypothesis_validated,
                "metrics_delta": metrics_delta,
                "insights": insights,
                "faults_injected": len([fault for fault in self.active_faults.values() if fault.rule in experiment.fault_rules]),
                "safety_violations": 0,  # Would track actual violations
                "recommendations": self._generate_experiment_recommendations(experiment, metrics_delta)
            }
            
        except Exception as e:
            self.logger.error(f"Experiment results generation failed: {e}")
            return {"error": "Failed to generate results"}
    
    def _validate_experiment_hypothesis(self, experiment: ChaosExperiment, metrics_delta: Dict[str, Any]) -> bool:
        """Validate experiment hypothesis against results"""
        try:
            # Simple validation - in practice this would be more sophisticated
            error_rate_change = metrics_delta.get("error_rate", {}).get("change_percent", 0)
            response_time_change = metrics_delta.get("response_time", {}).get("change_percent", 0)
            
            # Hypothesis: System should remain stable (error rate < 50% increase, response time < 100% increase)
            return error_rate_change < 50 and response_time_change < 100
            
        except Exception as e:
            self.logger.error(f"Hypothesis validation failed: {e}")
            return False
    
    def _generate_experiment_insights(self, experiment: ChaosExperiment, metrics_delta: Dict[str, Any]) -> List[str]:
        """Generate insights from experiment results"""
        insights = []
        
        try:
            # Analyze error rate changes
            error_rate_change = metrics_delta.get("error_rate", {}).get("change_percent", 0)
            if error_rate_change > 20:
                insights.append(f"Error rate increased by {error_rate_change:.1f}% during fault injection")
            
            # Analyze response time changes
            response_time_change = metrics_delta.get("response_time", {}).get("change_percent", 0)
            if response_time_change > 50:
                insights.append(f"Response time increased by {response_time_change:.1f}% during fault injection")
            
            # Analyze fault tolerance
            if error_rate_change < 10 and response_time_change < 25:
                insights.append("System demonstrated good fault tolerance")
            
            # Analyze specific fault impacts
            for rule in experiment.fault_rules:
                if rule.fault_type == FaultType.NETWORK_DELAY:
                    insights.append("Network delay fault injection tested service resilience to network issues")
                elif rule.fault_type == FaultType.DATABASE_SLOW:
                    insights.append("Database slowdown tested data layer resilience")
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
        
        return insights
    
    def _generate_experiment_recommendations(self, experiment: ChaosExperiment, metrics_delta: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment results"""
        recommendations = []
        
        try:
            error_rate_change = metrics_delta.get("error_rate", {}).get("change_percent", 0)
            
            if error_rate_change > 30:
                recommendations.append("Consider implementing more robust error handling")
                recommendations.append("Review circuit breaker thresholds")
            
            if error_rate_change > 50:
                recommendations.append("Critical: Implement better fault tolerance mechanisms")
                recommendations.append("Consider graceful degradation strategies")
            
            response_time_change = metrics_delta.get("response_time", {}).get("change_percent", 0)
            
            if response_time_change > 100:
                recommendations.append("Optimize timeout configurations")
                recommendations.append("Implement request queuing and throttling")
            
            # General recommendations
            recommendations.extend([
                "Continue regular chaos testing to improve system resilience",
                "Monitor error handling effectiveness over time",
                "Consider expanding fault injection scenarios"
            ])
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def _fault_monitor_loop(self):
        """Background task to monitor active faults"""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                current_time = datetime.now()
                expired_faults = []
                
                for fault_id, fault in self.active_faults.items():
                    if current_time > fault.expires_at:
                        expired_faults.append(fault_id)
                
                # Cleanup expired faults
                for fault_id in expired_faults:
                    asyncio.run(self.cleanup_fault(fault_id))
                
            except Exception as e:
                self.logger.error(f"Fault monitor loop failed: {e}")
                time.sleep(30)
    
    def _metrics_collection_loop(self):
        """Background task to collect metrics"""
        while True:
            try:
                interval = self.config.get("metrics_collection_interval", 10)
                time.sleep(interval)
                
                # Collect current metrics
                metrics = asyncio.run(self.metrics_collector.collect_all_metrics())
                
                # Store metrics with timestamp
                timestamp = datetime.now()
                # In practice, would store to time-series database
                
            except Exception as e:
                self.logger.error(f"Metrics collection loop failed: {e}")
                time.sleep(30)
    
    def _safety_monitor_loop(self):
        """Background task to monitor system safety"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.safety_enabled:
                    continue
                
                # Check safety thresholds
                if self._check_safety_violations():
                    self.logger.critical("Safety violation detected, stopping all experiments")
                    self._emergency_stop_all_experiments()
                
            except Exception as e:
                self.logger.error(f"Safety monitor loop failed: {e}")
                time.sleep(60)
    
    def _check_safety_violations(self) -> bool:
        """Check for safety violations"""
        try:
            thresholds = self.config.get("safety_thresholds", {})
            
            # Check error rate
            current_error_rate = self._get_current_error_rate()
            if current_error_rate > thresholds.get("max_error_rate", 0.5):
                return True
            
            # Check response time
            current_response_time = self._get_current_response_time()
            if current_response_time > thresholds.get("max_response_time", 10.0):
                return True
            
            # Check success rate
            current_success_rate = self._get_current_success_rate()
            if current_success_rate < thresholds.get("min_success_rate", 0.1):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Safety violation check failed: {e}")
            return True  # Err on the side of caution
    
    def _emergency_stop_all_experiments(self):
        """Emergency stop all running experiments"""
        try:
            self.emergency_stop = True
            
            running_experiments = [
                exp_id for exp_id, exp in self.experiments.items()
                if exp.status == ChaosExperimentStatus.RUNNING
            ]
            
            for exp_id in running_experiments:
                asyncio.run(self.stop_chaos_experiment(exp_id, "emergency_stop"))
            
            # Clear all active faults
            fault_ids = list(self.active_faults.keys())
            for fault_id in fault_ids:
                asyncio.run(self.cleanup_fault(fault_id))
            
            self.logger.critical("Emergency stop completed")
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
    
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_current_load(self) -> float:
        """Get current system load"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            memory_percent = psutil.virtual_memory().percent / 100.0
            
            # Simple load calculation
            return max(cpu_percent, memory_percent)
            
        except Exception:
            return 1.0  # Assume high load on error
    
    def _get_current_error_rate(self) -> float:
        """Get current error rate"""
        # In practice, would get from metrics system
        return random.uniform(0.0, 0.1)  # Simulate 0-10% error rate
    
    def _get_current_response_time(self) -> float:
        """Get current response time"""
        # In practice, would get from metrics system
        return random.uniform(0.1, 2.0)  # Simulate 0.1-2s response time
    
    def _get_current_success_rate(self) -> float:
        """Get current success rate"""
        # In practice, would get from metrics system
        return random.uniform(0.85, 0.99)  # Simulate 85-99% success rate
    
    def _experiment_monitor_loop(self, experiment_id: str):
        """Monitor a specific experiment"""
        try:
            experiment = self.experiments[experiment_id]
            end_time = experiment.start_time + timedelta(minutes=experiment.duration_minutes)
            
            while datetime.now() < end_time and experiment.status == ChaosExperimentStatus.RUNNING:
                time.sleep(30)  # Check every 30 seconds
                
                # Check rollback criteria
                if self._should_rollback_experiment(experiment):
                    asyncio.run(self.stop_chaos_experiment(experiment_id, "rollback_criteria"))
                    break
                
                # Check for emergency stop
                if self.emergency_stop:
                    break
            
            # Natural end of experiment
            if experiment.status == ChaosExperimentStatus.RUNNING:
                asyncio.run(self.stop_chaos_experiment(experiment_id, "natural_end"))
            
        except Exception as e:
            self.logger.error(f"Experiment monitor failed: {e}")
    
    def _should_rollback_experiment(self, experiment: ChaosExperiment) -> bool:
        """Check if experiment should be rolled back"""
        try:
            # Check rollback criteria
            for criteria in experiment.rollback_criteria:
                if self._evaluate_rollback_criteria(criteria):
                    self.logger.warning(f"Rollback criteria met: {criteria}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rollback criteria check failed: {e}")
            return True  # Err on the side of caution
    
    def _evaluate_rollback_criteria(self, criteria: str) -> bool:
        """Evaluate rollback criteria"""
        try:
            # Simple criteria evaluation
            if "error_rate > 0.5" in criteria:
                return self._get_current_error_rate() > 0.5
            elif "response_time > 10" in criteria:
                return self._get_current_response_time() > 10.0
            elif "service_unavailable" in criteria:
                return not self._check_system_health()
            
            return False
            
        except Exception:
            return True
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific experiment"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "duration_minutes": experiment.duration_minutes,
            "fault_rules_count": len(experiment.fault_rules),
            "active_faults": len([f for f in self.active_faults.values() if f.rule in experiment.fault_rules]),
            "results": experiment.results
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall chaos testing system status"""
        return {
            "chaos_testing_enabled": self.config.get("enabled", True),
            "safety_enabled": self.safety_enabled,
            "emergency_stop": self.emergency_stop,
            "active_experiments": len([exp for exp in self.experiments.values() if exp.status == ChaosExperimentStatus.RUNNING]),
            "active_faults": len(self.active_faults),
            "total_experiments": len(self.experiments),
            "fault_rules": len(self.fault_rules),
            "system_health": self._check_system_health(),
            "current_load": self._get_current_load(),
            "safety_thresholds": self.config.get("safety_thresholds", {})
        }

class MetricsCollector:
    """Metrics collection for chaos testing"""
    
    def __init__(self):
        self.last_metrics = {}
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all relevant metrics"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": await self._collect_system_metrics(),
                "service_metrics": await self._collect_service_metrics(),
                "error_metrics": await self._collect_error_metrics()
            }
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            logging.error(f"Metrics collection failed: {e}")
            return {}
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Collect service-specific metrics"""
        services = {}
        
        # Check service health
        service_endpoints = [
            ("neural_orchestrator", "http://localhost:8003/health"),
            ("learning_service", "http://localhost:8004/health"),
            ("threat_detection", "http://localhost:8005/health")
        ]
        
        for service_name, endpoint in service_endpoints:
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=5)
                response_time = time.time() - start_time
                
                services[service_name] = {
                    "available": response.status_code == 200,
                    "response_time": response_time,
                    "status_code": response.status_code
                }
                
            except Exception:
                services[service_name] = {
                    "available": False,
                    "response_time": 0.0,
                    "status_code": 0
                }
        
        return services
    
    async def _collect_error_metrics(self) -> Dict[str, Any]:
        """Collect error-related metrics"""
        # In practice, would integrate with error handling framework
        return {
            "error_rate": random.uniform(0.0, 0.1),
            "total_errors": random.randint(0, 50),
            "critical_errors": random.randint(0, 5)
        }

# Global chaos testing framework instance
_global_chaos_framework: Optional[ChaosTestingFramework] = None

def get_chaos_framework(config: Optional[Dict[str, Any]] = None) -> ChaosTestingFramework:
    """Get or create global chaos testing framework"""
    global _global_chaos_framework
    if _global_chaos_framework is None:
        _global_chaos_framework = ChaosTestingFramework(config)
    return _global_chaos_framework

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Starting XORB Chaos Testing Framework...")
    
    # Create chaos testing framework
    chaos_framework = ChaosTestingFramework()
    
    # Example experiment configuration
    experiment_config = {
        "name": "Network Resilience Test",
        "description": "Test system resilience to network delays and failures",
        "hypothesis": "System should maintain >90% availability under network stress",
        "duration_minutes": 5,
        "fault_rules": ["network_delay_light", "timeout_errors"]
    }
    
    async def run_test_experiment():
        # Create experiment
        experiment_id = await chaos_framework.create_chaos_experiment(experiment_config)
        print(f"Created experiment: {experiment_id}")
        
        # Start experiment
        success = await chaos_framework.start_chaos_experiment(experiment_id)
        if success:
            print("Experiment started successfully")
            
            # Monitor experiment
            while True:
                status = chaos_framework.get_experiment_status(experiment_id)
                print(f"Experiment status: {status['status']}")
                
                if status["status"] in ["completed", "failed", "aborted"]:
                    break
                
                await asyncio.sleep(10)
            
            print(f"Experiment completed with results: {status['results']}")
        else:
            print("Failed to start experiment")
    
    # Run test
    asyncio.run(run_test_experiment())
    
    # Get system status
    system_status = chaos_framework.get_system_status()
    print(f"System Status: {json.dumps(system_status, indent=2)}")
    
    print("✅ Chaos Testing Framework test completed")