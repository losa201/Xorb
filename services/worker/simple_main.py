#!/usr/bin/env python3
"""
Xorb PTaaS Worker Service - Simple Version
Minimal worker without conflicting imports
"""

import os
import sys
import asyncio
from datetime import datetime

class SimpleWorker:
    """Simple worker implementation"""
    
    def __init__(self):
        self.running = False
        self.tasks_processed = 0
        
    async def start(self):
        """Start the worker"""
        self.running = True
        print(f"🚀 Starting Xorb PTaaS Worker - {datetime.now().isoformat()}")
        print("📋 Worker ready to process tasks")
        
        try:
            while self.running:
                # Simulate task processing
                await asyncio.sleep(30)
                print(f"💼 Worker heartbeat - Tasks processed: {self.tasks_processed}")
                
        except KeyboardInterrupt:
            print("🛑 Worker stopped by user")
        except Exception as e:
            print(f"❌ Worker error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the worker"""
        self.running = False
        print("📴 Worker shutting down gracefully")
        
    async def process_task(self, task):
        """Process a task"""
        print(f"🔄 Processing task: {task}")
        await asyncio.sleep(1)  # Simulate work
        self.tasks_processed += 1
        print(f"✅ Task completed: {task}")

async def main():
    """Main worker function"""
    worker = SimpleWorker()
    
    try:
        await worker.start()
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🏗️  Xorb PTaaS Worker Service v2.0.0")
    print("=" * 50)
    
    # Check environment
    database_url = os.getenv("DATABASE_URL", "not_configured")
    redis_url = os.getenv("REDIS_URL", "not_configured")
    nats_url = os.getenv("NATS_URL", "not_configured")
    
    print(f"🗄️  Database: {database_url}")
    print(f"🔴 Redis: {redis_url}")
    print(f"📡 NATS: {nats_url}")
    print("=" * 50)
    
    asyncio.run(main())