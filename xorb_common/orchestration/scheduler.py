#!/usr/bin/env python3

import asyncio
import heapq
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ScheduleType(str, Enum):
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    RECURRING = "recurring"
    PRIORITY_BASED = "priority_based"


@dataclass
class ScheduledTask:
    campaign_id: str
    schedule_time: datetime
    priority: int = 0
    task_type: ScheduleType = ScheduleType.IMMEDIATE
    recurrence_interval: Optional[timedelta] = None
    metadata: Dict[str, Any] = None
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.schedule_time < other.schedule_time


class CampaignScheduler:
    def __init__(self):
        self.task_queue: List[ScheduledTask] = []
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
        
        self.priority_weights = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25
        }

    def start(self):
        if not self.running:
            self.running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.logger.info("Campaign scheduler started")

    async def stop(self):
        if self.running:
            self.running = False
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Campaign scheduler stopped")

    async def queue_campaign(self, campaign_id: str, priority: str = "medium", delay: Optional[timedelta] = None, recurrence: Optional[timedelta] = None) -> bool:
        schedule_time = datetime.utcnow()
        if delay:
            schedule_time += delay
        
        task_type = ScheduleType.IMMEDIATE
        if delay:
            task_type = ScheduleType.DELAYED
        if recurrence:
            task_type = ScheduleType.RECURRING
        
        priority_score = self.priority_weights.get(priority.lower(), 50)
        
        scheduled_task = ScheduledTask(
            campaign_id=campaign_id,
            schedule_time=schedule_time,
            priority=priority_score,
            task_type=task_type,
            recurrence_interval=recurrence,
            metadata={"priority_level": priority}
        )
        
        heapq.heappush(self.task_queue, scheduled_task)
        
        self.logger.info(f"Queued campaign {campaign_id} with priority {priority} for {schedule_time}")
        return True

    async def schedule_immediate(self, campaign_id: str, priority: str = "medium") -> bool:
        return await self.queue_campaign(campaign_id, priority)

    async def schedule_delayed(self, campaign_id: str, delay: timedelta, priority: str = "medium") -> bool:
        return await self.queue_campaign(campaign_id, priority, delay=delay)

    async def schedule_recurring(self, campaign_id: str, interval: timedelta, priority: str = "medium") -> bool:
        return await self.queue_campaign(campaign_id, priority, recurrence=interval)

    async def cancel_campaign(self, campaign_id: str) -> bool:
        original_length = len(self.task_queue)
        self.task_queue = [task for task in self.task_queue if task.campaign_id != campaign_id]
        heapq.heapify(self.task_queue)
        
        removed_count = original_length - len(self.task_queue)
        if removed_count > 0:
            self.logger.info(f"Cancelled {removed_count} scheduled tasks for campaign {campaign_id}")
            return True
        return False

    async def get_scheduled_campaigns(self) -> List[Dict[str, Any]]:
        scheduled = []
        for task in sorted(self.task_queue):
            scheduled.append({
                "campaign_id": task.campaign_id,
                "schedule_time": task.schedule_time.isoformat(),
                "priority": task.priority,
                "task_type": task.task_type.value,
                "recurrence_interval": task.recurrence_interval.total_seconds() if task.recurrence_interval else None,
                "metadata": task.metadata
            })
        return scheduled

    async def reschedule_campaign(self, campaign_id: str, new_schedule_time: datetime, priority: str = "medium") -> bool:
        await self.cancel_campaign(campaign_id)
        delay = new_schedule_time - datetime.utcnow()
        return await self.schedule_delayed(campaign_id, delay, priority)

    async def get_next_scheduled_time(self) -> Optional[datetime]:
        if self.task_queue:
            return self.task_queue[0].schedule_time
        return None

    async def _scheduler_loop(self):
        self.logger.debug("Scheduler loop started")
        
        while self.running:
            try:
                await self._process_scheduled_tasks()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)

    async def _process_scheduled_tasks(self):
        current_time = datetime.utcnow()
        processed_tasks = []
        
        while self.task_queue and self.task_queue[0].schedule_time <= current_time:
            task = heapq.heappop(self.task_queue)
            
            try:
                await self._execute_scheduled_task(task)
                processed_tasks.append(task)
                
                if task.task_type == ScheduleType.RECURRING and task.recurrence_interval:
                    next_run = current_time + task.recurrence_interval
                    recurring_task = ScheduledTask(
                        campaign_id=task.campaign_id,
                        schedule_time=next_run,
                        priority=task.priority,
                        task_type=task.task_type,
                        recurrence_interval=task.recurrence_interval,
                        metadata=task.metadata
                    )
                    heapq.heappush(self.task_queue, recurring_task)
                    
            except Exception as e:
                self.logger.error(f"Error executing scheduled task for campaign {task.campaign_id}: {e}")

    async def _execute_scheduled_task(self, task: ScheduledTask):
        self.logger.info(f"Executing scheduled task for campaign {task.campaign_id}")
        
        from .orchestrator import Orchestrator
        try:
            orchestrator = Orchestrator()
            await orchestrator.start_campaign(task.campaign_id)
        except Exception as e:
            self.logger.error(f"Failed to start campaign {task.campaign_id}: {e}")

    async def optimize_schedule(self) -> int:
        if not self.task_queue:
            return 0
        
        optimized_count = 0
        current_time = datetime.utcnow()
        
        for task in self.task_queue:
            if task.task_type == ScheduleType.PRIORITY_BASED:
                if task.priority > 75:
                    if task.schedule_time > current_time + timedelta(minutes=5):
                        task.schedule_time = current_time + timedelta(minutes=1)
                        optimized_count += 1
                elif task.priority < 30:
                    if task.schedule_time < current_time + timedelta(hours=1):
                        task.schedule_time = current_time + timedelta(hours=2)
                        optimized_count += 1
        
        if optimized_count > 0:
            heapq.heapify(self.task_queue)
            self.logger.info(f"Optimized {optimized_count} scheduled tasks")
        
        return optimized_count

    def get_queue_stats(self) -> Dict[str, Any]:
        if not self.task_queue:
            return {"total": 0, "by_priority": {}, "by_type": {}, "next_execution": None}
        
        by_priority = {}
        by_type = {}
        
        for task in self.task_queue:
            priority_level = task.metadata.get("priority_level", "unknown") if task.metadata else "unknown"
            by_priority[priority_level] = by_priority.get(priority_level, 0) + 1
            by_type[task.task_type.value] = by_type.get(task.task_type.value, 0) + 1
        
        return {
            "total": len(self.task_queue),
            "by_priority": by_priority,
            "by_type": by_type,
            "next_execution": self.task_queue[0].schedule_time.isoformat() if self.task_queue else None
        }