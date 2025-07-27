# XORB Disaster Recovery Runbook

## Overview
This runbook provides step-by-step procedures for XORB disaster recovery scenarios.

## Emergency Contacts
- Primary: System Administrator
- Secondary: DevOps Team
- Escalation: XORB Platform Team

## Recovery Time Objectives (RTO)
- Critical Services: 30 minutes
- Database: 1 hour
- Full Platform: 2 hours

## Recovery Point Objectives (RPO)
- Database: 15 minutes (continuous replication)
- Configuration: 24 hours (daily backups)
- Code: 24 hours (daily backups)

## Disaster Scenarios

### Scenario 1: Database Failure
1. Assess damage: `docker logs xorb_postgres_1`
2. Stop affected services: `docker stop xorb_postgres_1`
3. Restore from backup: `/root/Xorb/disaster-recovery/recovery-system.sh database`
4. Verify recovery: `curl http://localhost:8000/health`
5. Resume operations

### Scenario 2: Complete System Failure
1. Assess infrastructure: `docker ps -a`
2. Full recovery: `/root/Xorb/disaster-recovery/recovery-system.sh full`
3. Verify all services: `/root/Xorb/disaster-recovery/tests/dr-test-suite.sh`
4. Resume operations

### Scenario 3: Data Corruption
1. Stop affected services
2. Restore from point-in-time backup
3. Validate data integrity
4. Resume operations

### Scenario 4: Security Breach
1. Isolate affected systems
2. Rotate all secrets: `/root/Xorb/secrets/rotate-secrets.sh`
3. Restore from clean backup
4. Apply security patches
5. Resume operations with monitoring

## Backup Verification
- Daily: Automated backup integrity checks
- Weekly: Full recovery test in isolated environment
- Monthly: Cross-region backup verification

## Post-Incident Actions
1. Document incident details
2. Update recovery procedures
3. Test improvements
4. Brief stakeholders

## Testing Schedule
- Daily: Backup creation and verification
- Weekly: Partial recovery tests
- Monthly: Full disaster recovery drill
- Quarterly: Cross-region failover test

## Monitoring and Alerts
- Backup failure alerts
- Recovery test failure alerts
- RTO/RPO threshold alerts
- Capacity and performance alerts
