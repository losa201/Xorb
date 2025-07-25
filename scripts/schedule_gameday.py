#!/usr/bin/env python3
"""
Scheduled Game Day Automation
Automatically schedules and runs Game Day exercises on a regular basis
"""

import asyncio
import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameDayScheduler:
    """Schedules and manages automated Game Day exercises"""
    
    def __init__(self, config_file: str = "/opt/xorb/config/gameday_schedule.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.exercise_history = []
    
    def load_config(self) -> Dict:
        """Load Game Day schedule configuration"""
        default_config = {
            "enabled": True,
            "environment": "staging",
            "schedule": {
                "weekly_basic": {
                    "day": "saturday",
                    "time": "02:00",
                    "scenarios": ["database"],
                    "enabled": True
                },
                "monthly_comprehensive": {
                    "day_of_month": 15,
                    "time": "01:00",
                    "scenarios": ["all"],
                    "enabled": True
                },
                "quarterly_full_dr": {
                    "months": [3, 6, 9, 12],
                    "day_of_month": 1,
                    "time": "00:00",
                    "scenarios": ["all"],
                    "enabled": True,
                    "duration_minutes": 120
                }
            },
            "notifications": {
                "slack_webhook": "",
                "email_recipients": [],
                "enabled": True
            },
            "participants": [
                "engineering-team@xorb.ai",
                "devops-team@xorb.ai"
            ]
        }
        
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
            else:
                # Create default config file
                Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load config, using defaults: {e}")
            return default_config
    
    async def send_notification(self, message: str, level: str = "info"):
        """Send notification about Game Day exercise"""
        if not self.config["notifications"]["enabled"]:
            return
        
        try:
            # Slack notification
            webhook_url = self.config["notifications"].get("slack_webhook")
            if webhook_url:
                import aiohttp
                
                payload = {
                    "text": f"🎮 Xorb Game Day: {message}",
                    "color": "good" if level == "info" else "warning" if level == "warning" else "danger"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status == 200:
                            logger.info("Slack notification sent successfully")
                        else:
                            logger.error(f"Failed to send Slack notification: {response.status}")
            
            # Email notification (simple implementation)
            email_recipients = self.config["notifications"].get("email_recipients", [])
            if email_recipients:
                # This would integrate with your email service
                logger.info(f"Email notification would be sent to: {email_recipients}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def run_scheduled_exercise(self, exercise_type: str, scenarios: List[str], duration: int = 60):
        """Run a scheduled Game Day exercise"""
        exercise_start = datetime.now()
        
        logger.info(f"🎮 Starting scheduled {exercise_type} Game Day exercise")
        
        await self.send_notification(
            f"Starting {exercise_type} Game Day exercise with scenarios: {', '.join(scenarios)}",
            level="info"
        )
        
        try:
            # Import and run the Game Day orchestrator
            import subprocess
            
            cmd = [
                "python", "/opt/xorb/scripts/gameday_orchestrator.py",
                "--environment", self.config["environment"],
                "--scenario", scenarios[0] if len(scenarios) == 1 else "all",
                "--duration", str(duration),
                "--participants"] + self.config["participants"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            exercise_end = datetime.now()
            duration_minutes = (exercise_end - exercise_start).total_seconds() / 60
            
            # Record exercise in history
            exercise_record = {
                "type": exercise_type,
                "scenarios": scenarios,
                "start_time": exercise_start.isoformat(),
                "end_time": exercise_end.isoformat(),
                "duration_minutes": duration_minutes,
                "success": process.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else ""
            }
            
            self.exercise_history.append(exercise_record)
            
            # Save history
            history_file = "/opt/xorb/data/gameday_history.json"
            Path(history_file).parent.mkdir(parents=True, exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump(self.exercise_history, f, indent=2)
            
            if process.returncode == 0:
                await self.send_notification(
                    f"✅ {exercise_type} Game Day exercise completed successfully in {duration_minutes:.1f} minutes",
                    level="info"
                )
                logger.info(f"✅ {exercise_type} exercise completed successfully")
            else:
                await self.send_notification(
                    f"❌ {exercise_type} Game Day exercise failed after {duration_minutes:.1f} minutes",
                    level="error"
                )
                logger.error(f"❌ {exercise_type} exercise failed")
            
        except Exception as e:
            await self.send_notification(
                f"💥 {exercise_type} Game Day exercise crashed: {str(e)}",
                level="error"
            )
            logger.error(f"Game Day exercise crashed: {e}")
    
    def schedule_exercises(self):
        """Set up the exercise schedule"""
        if not self.config["enabled"]:
            logger.info("Game Day scheduling is disabled")
            return
        
        logger.info("Setting up Game Day exercise schedule...")
        
        # Weekly basic exercises
        weekly_config = self.config["schedule"]["weekly_basic"]
        if weekly_config["enabled"]:
            day_map = {
                "monday": schedule.every().monday,
                "tuesday": schedule.every().tuesday,
                "wednesday": schedule.every().wednesday,
                "thursday": schedule.every().thursday,
                "friday": schedule.every().friday,
                "saturday": schedule.every().saturday,
                "sunday": schedule.every().sunday
            }
            
            day_scheduler = day_map.get(weekly_config["day"].lower(), schedule.every().saturday)
            day_scheduler.at(weekly_config["time"]).do(
                lambda: asyncio.create_task(self.run_scheduled_exercise(
                    "weekly_basic",
                    weekly_config["scenarios"],
                    60
                ))
            )
            logger.info(f"Scheduled weekly basic exercise: {weekly_config['day']} at {weekly_config['time']}")
        
        # Monthly comprehensive exercises
        monthly_config = self.config["schedule"]["monthly_comprehensive"]
        if monthly_config["enabled"]:
            # For monthly scheduling, we'll check daily and run if it's the right day
            schedule.every().day.at(monthly_config["time"]).do(
                lambda: self.check_and_run_monthly(monthly_config)
            )
            logger.info(f"Scheduled monthly comprehensive exercise: day {monthly_config['day_of_month']} at {monthly_config['time']}")
        
        # Quarterly full DR exercises
        quarterly_config = self.config["schedule"]["quarterly_full_dr"]
        if quarterly_config["enabled"]:
            schedule.every().day.at(quarterly_config["time"]).do(
                lambda: self.check_and_run_quarterly(quarterly_config)
            )
            logger.info(f"Scheduled quarterly DR exercise: months {quarterly_config['months']} day {quarterly_config['day_of_month']}")
    
    def check_and_run_monthly(self, config):
        """Check if today is the monthly exercise day"""
        today = datetime.now()
        if today.day == config["day_of_month"]:
            asyncio.create_task(self.run_scheduled_exercise(
                "monthly_comprehensive",
                config["scenarios"],
                config.get("duration_minutes", 90)
            ))
    
    def check_and_run_quarterly(self, config):
        """Check if today is the quarterly exercise day"""
        today = datetime.now()
        if today.month in config["months"] and today.day == config["day_of_month"]:
            asyncio.create_task(self.run_scheduled_exercise(
                "quarterly_full_dr",
                config["scenarios"],
                config.get("duration_minutes", 120)
            ))
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("🚀 Game Day scheduler started")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
                # Clean up old history (keep last 100 exercises)
                if len(self.exercise_history) > 100:
                    self.exercise_history = self.exercise_history[-100:]
                    
            except KeyboardInterrupt:
                logger.info("Game Day scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def get_next_exercises(self, days: int = 30) -> List[Dict]:
        """Get list of upcoming exercises in the next N days"""
        upcoming = []
        
        # This is a simplified implementation
        # In a real system, you'd calculate based on the actual schedule
        
        if self.config["schedule"]["weekly_basic"]["enabled"]:
            upcoming.append({
                "type": "weekly_basic",
                "next_run": "Next Saturday 02:00",
                "scenarios": self.config["schedule"]["weekly_basic"]["scenarios"]
            })
        
        if self.config["schedule"]["monthly_comprehensive"]["enabled"]:
            upcoming.append({
                "type": "monthly_comprehensive",
                "next_run": f"15th of this month 01:00",
                "scenarios": self.config["schedule"]["monthly_comprehensive"]["scenarios"]
            })
        
        return upcoming
    
    def generate_status_report(self) -> Dict:
        """Generate a status report of the Game Day system"""
        recent_exercises = [ex for ex in self.exercise_history 
                          if datetime.fromisoformat(ex["start_time"]) > datetime.now() - timedelta(days=30)]
        
        return {
            "scheduler_enabled": self.config["enabled"],
            "environment": self.config["environment"],
            "total_exercises_run": len(self.exercise_history),
            "recent_exercises": len(recent_exercises),
            "success_rate": len([ex for ex in recent_exercises if ex["success"]]) / max(len(recent_exercises), 1),
            "upcoming_exercises": self.get_next_exercises(),
            "last_exercise": self.exercise_history[-1] if self.exercise_history else None,
            "config": self.config
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Game Day Scheduler")
    parser.add_argument("--config", default="/opt/xorb/config/gameday_schedule.json",
                       help="Path to configuration file")
    parser.add_argument("--status", action="store_true",
                       help="Show status report and exit")
    parser.add_argument("--test", action="store_true",
                       help="Run a test exercise immediately")
    
    args = parser.parse_args()
    
    scheduler = GameDayScheduler(config_file=args.config)
    
    if args.status:
        report = scheduler.generate_status_report()
        print(json.dumps(report, indent=2))
        return
    
    if args.test:
        asyncio.run(scheduler.run_scheduled_exercise("test", ["database"], 30))
        return
    
    # Set up and run the scheduler
    scheduler.schedule_exercises()
    scheduler.run_scheduler()

if __name__ == "__main__":
    main()