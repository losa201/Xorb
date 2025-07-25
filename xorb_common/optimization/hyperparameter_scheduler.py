#!/usr/bin/env python3
"""
Automated Hyperparameter Scheduling and Management for Xorb 2.0

This module provides scheduling, monitoring, and lifecycle management 
for automated hyperparameter optimization in production environments.
"""

import asyncio
import logging
import json
import schedule
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import yaml

from .hyperparameter_tuner import EPYCHyperparameterTuner, HyperparameterSet


@dataclass
class OptimizationSchedule:
    """Schedule configuration for hyperparameter optimization"""
    name: str
    cron_expression: str  # e.g., "0 2 * * *" for daily at 2 AM
    enabled: bool = True
    max_trials: int = 50
    max_duration_minutes: int = 120
    workload_types: List[str] = None
    priority: str = "normal"  # "low", "normal", "high"
    
    def __post_init__(self):
        if self.workload_types is None:
            self.workload_types = ["all"]


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization run"""
    schedule_name: str
    start_time: datetime
    end_time: datetime
    best_score: float
    trials_completed: int
    hyperparameters: HyperparameterSet
    performance_improvement: Dict[str, float]
    applied_to_production: bool = False
    rollback_available: bool = True


class HyperparameterScheduler:
    """
    Production scheduler for automated hyperparameter optimization
    """
    
    def __init__(self, 
                 config_file: str = "hyperparameter_schedules.yaml",
                 results_directory: str = "optimization_results",
                 epyc_cores: int = 64):
        self.config_file = Path(config_file)
        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(exist_ok=True)
        self.epyc_cores = epyc_cores
        
        self.logger = logging.getLogger(__name__)
        self.schedules: List[OptimizationSchedule] = []
        self.running_optimizations: Dict[str, asyncio.Task] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # Production safety settings
        self.max_concurrent_optimizations = 2
        self.safety_threshold_score = 0.05  # Minimum improvement to apply changes
        self.rollback_window_hours = 24
        
        # Initialize tuner
        self.tuner = EPYCHyperparameterTuner(epyc_cores=epyc_cores)
        
    async def load_schedules(self):
        """Load optimization schedules from configuration file"""
        try:
            if self.config_file.exists():
                async with aiofiles.open(self.config_file, 'r') as f:
                    content = await f.read()
                    config = yaml.safe_load(content)
                    
                    self.schedules = [
                        OptimizationSchedule(**schedule_config)
                        for schedule_config in config.get('schedules', [])
                    ]
                    
                    self.logger.info(f"Loaded {len(self.schedules)} optimization schedules")
            else:
                # Create default configuration
                await self._create_default_config()
                
        except Exception as e:
            self.logger.error(f"Failed to load optimization schedules: {e}")
            self.schedules = []
            
    async def _create_default_config(self):
        """Create default optimization schedule configuration"""
        default_schedules = [
            OptimizationSchedule(
                name="daily_optimization",
                cron_expression="0 2 * * *",  # Daily at 2 AM
                enabled=True,
                max_trials=100,
                max_duration_minutes=180,
                workload_types=["all"],
                priority="normal"
            ),
            OptimizationSchedule(
                name="weekly_deep_optimization", 
                cron_expression="0 1 * * 0",  # Weekly on Sunday at 1 AM
                enabled=True,
                max_trials=500,
                max_duration_minutes=720,  # 12 hours
                workload_types=["ml_training", "high_throughput"],
                priority="high"
            ),
            OptimizationSchedule(
                name="hourly_quick_tune",
                cron_expression="0 * * * *",  # Every hour
                enabled=False,  # Disabled by default
                max_trials=20,
                max_duration_minutes=30,
                workload_types=["real_time"],
                priority="low"
            )
        ]
        
        config = {
            'schedules': [asdict(schedule) for schedule in default_schedules],
            'global_settings': {
                'max_concurrent_optimizations': self.max_concurrent_optimizations,
                'safety_threshold_score': self.safety_threshold_score,
                'rollback_window_hours': self.rollback_window_hours
            }
        }
        
        async with aiofiles.open(self.config_file, 'w') as f:
            await f.write(yaml.dump(config, indent=2))
            
        self.schedules = default_schedules
        self.logger.info(f"Created default optimization schedule configuration")
        
    async def start_scheduler(self):
        """Start the hyperparameter optimization scheduler"""
        self.logger.info("Starting hyperparameter optimization scheduler")
        
        await self.load_schedules()
        
        # Set up scheduled tasks
        for schedule_config in self.schedules:
            if schedule_config.enabled:
                await self._schedule_optimization(schedule_config)
                
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Hyperparameter scheduler started successfully")
        
    async def _schedule_optimization(self, schedule_config: OptimizationSchedule):
        """Schedule a specific optimization task"""
        # Convert cron to schedule library format
        cron_parts = schedule_config.cron_expression.split()
        
        if len(cron_parts) == 5:
            minute, hour, day, month, weekday = cron_parts
            
            # Simple cron parsing (extend as needed)
            if hour != "*" and minute != "*":
                schedule_time = f"{hour}:{minute.zfill(2)}"
                
                if weekday != "*":
                    # Weekly schedule
                    weekday_map = {"0": "sunday", "1": "monday", "2": "tuesday", 
                                 "3": "wednesday", "4": "thursday", "5": "friday", "6": "saturday"}
                    day_name = weekday_map.get(weekday, "sunday")
                    
                    schedule.every().week.at(schedule_time).do(
                        self._queue_optimization, schedule_config
                    ).tag(f"weekly_{day_name}")
                    
                elif day == "*" and month == "*":
                    # Daily schedule
                    schedule.every().day.at(schedule_time).do(
                        self._queue_optimization, schedule_config
                    ).tag("daily")
                    
            elif minute != "*" and hour == "*":
                # Hourly schedule
                schedule.every().hour.at(f":{minute.zfill(2)}").do(
                    self._queue_optimization, schedule_config
                ).tag("hourly")
                
        self.logger.info(f"Scheduled optimization '{schedule_config.name}' with cron: {schedule_config.cron_expression}")
        
    def _queue_optimization(self, schedule_config: OptimizationSchedule):
        """Queue an optimization task for execution"""
        if len(self.running_optimizations) >= self.max_concurrent_optimizations:
            self.logger.warning(f"Maximum concurrent optimizations reached, skipping {schedule_config.name}")
            return
            
        # Create and start optimization task
        task = asyncio.create_task(self._run_optimization(schedule_config))
        self.running_optimizations[schedule_config.name] = task
        
        self.logger.info(f"Queued optimization task: {schedule_config.name}")
        
    async def _run_optimization(self, schedule_config: OptimizationSchedule):
        """Run a single optimization task"""
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting optimization: {schedule_config.name}")
        
        try:
            # Configure tuner for this optimization
            self.tuner.optimization_budget = timedelta(minutes=schedule_config.max_duration_minutes)
            self.tuner.study_name = f"xorb_scheduled_{schedule_config.name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Run optimization
            best_hyperparams = await self.tuner.optimize_hyperparameters(
                n_trials=schedule_config.max_trials
            )
            
            end_time = datetime.utcnow()
            
            # Evaluate performance improvement
            performance_improvement = await self._evaluate_performance_improvement(best_hyperparams)
            
            # Create optimization result
            result = OptimizationResult(
                schedule_name=schedule_config.name,
                start_time=start_time,
                end_time=end_time,
                best_score=self.tuner.best_parameters and self.tuner.optimization_history[-1]['composite_score'] or 0.0,
                trials_completed=len(self.tuner.optimization_history),
                hyperparameters=best_hyperparams,
                performance_improvement=performance_improvement
            )
            
            # Save result
            await self._save_optimization_result(result)
            
            # Apply to production if improvement is significant
            if self._should_apply_to_production(result):
                await self._apply_to_production(result)
                
            self.optimization_history.append(result)
            
            self.logger.info(f"Optimization completed: {schedule_config.name}, score: {result.best_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {schedule_config.name}, error: {e}")
            
        finally:
            # Remove from running tasks
            if schedule_config.name in self.running_optimizations:
                del self.running_optimizations[schedule_config.name]
                
    async def _evaluate_performance_improvement(self, hyperparams: HyperparameterSet) -> Dict[str, float]:
        """Evaluate performance improvement from new hyperparameters"""
        # Compare against baseline or previous best
        improvement_metrics = {}
        
        if self.optimization_history:
            # Compare against last successful optimization
            last_result = self.optimization_history[-1]
            
            # Calculate relative improvements
            for objective in self.tuner.objectives:
                baseline_value = getattr(last_result.hyperparameters, f"{objective.name}_value", 0.5)
                current_value = objective.current_value
                
                if baseline_value > 0:
                    improvement = (current_value - baseline_value) / baseline_value
                    improvement_metrics[objective.name] = improvement
                    
        return improvement_metrics
        
    def _should_apply_to_production(self, result: OptimizationResult) -> bool:
        """Determine if optimization result should be applied to production"""
        # Safety checks
        if result.best_score < self.safety_threshold_score:
            self.logger.info(f"Optimization score {result.best_score:.4f} below safety threshold {self.safety_threshold_score}")
            return False
            
        # Check for minimum improvement
        avg_improvement = sum(result.performance_improvement.values()) / len(result.performance_improvement) if result.performance_improvement else 0.0
        
        if avg_improvement < 0.02:  # Minimum 2% improvement
            self.logger.info(f"Average improvement {avg_improvement:.4f} below minimum threshold")
            return False
            
        # Check for recent rollbacks (avoid instability)
        recent_rollbacks = [r for r in self.optimization_history[-10:] if not r.applied_to_production]
        if len(recent_rollbacks) > 3:
            self.logger.warning("Too many recent rollbacks, skipping production application")
            return False
            
        return True
        
    async def _apply_to_production(self, result: OptimizationResult):
        """Apply optimization result to production system"""
        self.logger.info(f"Applying optimization to production: {result.schedule_name}")
        
        try:
            # Store current configuration for rollback
            await self._create_rollback_point()
            
            # Apply new hyperparameters
            await self.tuner._apply_optimized_parameters(result.hyperparameters)
            
            # Mark as applied
            result.applied_to_production = True
            
            # Schedule rollback monitoring
            asyncio.create_task(self._monitor_production_deployment(result))
            
            self.logger.info("Optimization successfully applied to production")
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization to production: {e}")
            result.applied_to_production = False
            
    async def _create_rollback_point(self):
        """Create a rollback point with current configuration"""
        rollback_config = {
            'timestamp': datetime.utcnow().isoformat(),
            'current_hyperparameters': asdict(self.tuner.best_parameters) if self.tuner.best_parameters else {},
            'performance_metrics': self.tuner.optimization_history[-1] if self.tuner.optimization_history else {}
        }
        
        rollback_file = self.results_directory / f"rollback_point_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(rollback_file, 'w') as f:
            await f.write(json.dumps(rollback_config, indent=2))
            
        self.logger.info(f"Created rollback point: {rollback_file}")
        
    async def _monitor_production_deployment(self, result: OptimizationResult):
        """Monitor production deployment and rollback if needed"""
        monitoring_duration = timedelta(hours=self.rollback_window_hours)
        start_time = datetime.utcnow()
        
        while datetime.utcnow() - start_time < monitoring_duration:
            try:
                # Check system health metrics
                health_status = await self._check_system_health()
                
                if not health_status['healthy']:
                    self.logger.warning(f"System health degraded after optimization, initiating rollback")
                    await self._rollback_optimization(result)
                    return
                    
                # Check performance metrics
                performance_status = await self._check_performance_regression()
                
                if performance_status['regression_detected']:
                    self.logger.warning(f"Performance regression detected, initiating rollback")
                    await self._rollback_optimization(result)
                    return
                    
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring production deployment: {e}")
                await asyncio.sleep(600)  # Check every 10 minutes on error
                
        self.logger.info(f"Production monitoring completed successfully for {result.schedule_name}")
        
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        # This would integrate with monitoring systems
        # For now, return a placeholder
        return {
            'healthy': True,
            'cpu_utilization': 0.65,
            'memory_utilization': 0.70,
            'error_rate': 0.01
        }
        
    async def _check_performance_regression(self) -> Dict[str, Any]:
        """Check for performance regression"""
        # This would compare current metrics against baseline
        return {
            'regression_detected': False,
            'performance_delta': 0.05,
            'metrics_stable': True
        }
        
    async def _rollback_optimization(self, result: OptimizationResult):
        """Rollback optimization changes"""
        self.logger.warning(f"Rolling back optimization: {result.schedule_name}")
        
        try:
            # Find latest rollback point
            rollback_files = list(self.results_directory.glob("rollback_point_*.json"))
            if not rollback_files:
                self.logger.error("No rollback point found")
                return
                
            latest_rollback = max(rollback_files, key=lambda x: x.stat().st_mtime)
            
            async with aiofiles.open(latest_rollback, 'r') as f:
                rollback_config = json.loads(await f.read())
                
            # Apply rollback configuration
            if rollback_config['current_hyperparameters']:
                rollback_params = HyperparameterSet(**rollback_config['current_hyperparameters'])
                await self.tuner._apply_optimized_parameters(rollback_params)
                
            result.applied_to_production = False
            result.rollback_available = False
            
            self.logger.info("Optimization rollback completed successfully")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            
    async def _save_optimization_result(self, result: OptimizationResult):
        """Save optimization result to disk"""
        result_file = self.results_directory / f"optimization_result_{result.schedule_name}_{result.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        result_data = {
            'schedule_name': result.schedule_name,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'best_score': result.best_score,
            'trials_completed': result.trials_completed,
            'hyperparameters': asdict(result.hyperparameters),
            'performance_improvement': result.performance_improvement,
            'applied_to_production': result.applied_to_production,
            'rollback_available': result.rollback_available
        }
        
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(json.dumps(result_data, indent=2))
            
        self.logger.info(f"Saved optimization result: {result_file}")
        
    async def _monitoring_loop(self):
        """Main monitoring loop for scheduled tasks"""
        while True:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Clean up completed tasks
                completed_tasks = [name for name, task in self.running_optimizations.items() if task.done()]
                for task_name in completed_tasks:
                    del self.running_optimizations[task_name]
                    
                # Log status
                if self.running_optimizations:
                    self.logger.debug(f"Running optimizations: {list(self.running_optimizations.keys())}")
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
                
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'scheduler_running': True,
            'active_schedules': len([s for s in self.schedules if s.enabled]),
            'running_optimizations': list(self.running_optimizations.keys()),
            'recent_results': [
                {
                    'schedule_name': r.schedule_name,
                    'end_time': r.end_time.isoformat(),
                    'best_score': r.best_score,
                    'applied_to_production': r.applied_to_production
                }
                for r in self.optimization_history[-5:]
            ]
        }
        
    async def trigger_manual_optimization(self, 
                                        schedule_name: str,
                                        max_trials: int = 50,
                                        max_duration_minutes: int = 60) -> OptimizationResult:
        """Trigger manual optimization outside of schedule"""
        manual_schedule = OptimizationSchedule(
            name=f"manual_{schedule_name}",
            cron_expression="",  # Not used for manual
            max_trials=max_trials,
            max_duration_minutes=max_duration_minutes,
            priority="high"
        )
        
        await self._run_optimization(manual_schedule)
        
        return self.optimization_history[-1] if self.optimization_history else None


# Integration with Kubernetes CronJob for production deployment
async def create_k8s_optimization_cronjob():
    """Create Kubernetes CronJob for hyperparameter optimization"""
    cronjob_yaml = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: xorb-hyperparameter-optimization
  namespace: xorb-prod
  labels:
    app.kubernetes.io/name: xorb
    app.kubernetes.io/component: optimization
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: hyperparameter-optimizer
            image: xorb/optimization:latest
            command:
            - python
            - -m
            - xorb_core.optimization.hyperparameter_scheduler
            env:
            - name: EPYC_CORES
              value: "64"
            - name: OPTIMIZATION_BUDGET_MINUTES
              value: "180"
            - name: MAX_TRIALS
              value: "100"
            resources:
              requests:
                cpu: "4"
                memory: "8Gi"
              limits:
                cpu: "16"
                memory: "32Gi"
            volumeMounts:
            - name: optimization-results
              mountPath: /app/optimization_results
          volumes:
          - name: optimization-results
            persistentVolumeClaim:
              claimName: xorb-optimization-results
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
"""
    
    return cronjob_yaml


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        scheduler = HyperparameterScheduler(epyc_cores=64)
        await scheduler.start_scheduler()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            status = await scheduler.get_optimization_status()
            print(f"Scheduler status: {status}")
            
    asyncio.run(main())