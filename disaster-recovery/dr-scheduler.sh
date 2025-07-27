#!/bin/bash

# XORB Disaster Recovery Scheduler
# Automated scheduling of backups and recovery tests

echo "⏰ XORB Disaster Recovery Scheduler"
echo "=================================="

ACTION="${1:-status}"

case "$ACTION" in
    "setup")
        echo "🔧 Setting up disaster recovery schedule..."
        
        # Create cron jobs for automated backups
        (crontab -l 2>/dev/null; echo "0 2 * * * /root/Xorb/disaster-recovery/backup-system.sh") | crontab -
        (crontab -l 2>/dev/null; echo "0 4 * * 0 /root/Xorb/disaster-recovery/tests/dr-test-suite.sh") | crontab -
        
        echo "✅ Scheduled:"
        echo "  - Daily backups at 2:00 AM"
        echo "  - Weekly DR tests on Sunday at 4:00 AM"
        ;;
        
    "status")
        echo "📋 Current DR schedule:"
        crontab -l | grep -E "(backup-system|dr-test-suite)" || echo "No DR schedules found"
        ;;
        
    "remove")
        echo "🗑️  Removing DR schedule..."
        crontab -l | grep -v -E "(backup-system|dr-test-suite)" | crontab -
        echo "✅ DR schedule removed"
        ;;
        
    *)
        echo "Usage: $0 {setup|status|remove}"
        exit 1
        ;;
esac
