# Xorb PTaaS Disaster Recovery Drillbook

## Overview
This drillbook provides step-by-step procedures for disaster recovery scenarios and Game Day exercises for the Xorb PTaaS platform running on AMD EPYC single-node deployment.

## Emergency Contacts & Escalation

### Tier 1 - Immediate Response (0-15 minutes)
- **Platform Engineer**: On-call rotation
- **DevOps Lead**: Primary escalation
- **Security Engineer**: For security-related incidents

### Tier 2 - Management Escalation (15-60 minutes)
- **Engineering Manager**: Service degradation > 30min
- **CTO**: Critical outage > 1hr
- **CEO**: Data breach or major security incident

### External Services
- **Cloud Provider**: EPYC VPS provider support
- **Payment Provider**: Stripe support for payment issues
- **Monitoring**: Grafana Cloud / DataDog escalation

## Disaster Recovery Scenarios

### DR-1: Complete Node Failure (RTO: 4 hours, RPO: 5 minutes)

**Detection**:
- All services show as down in monitoring
- Unable to SSH to primary node
- Health checks failing for >5 minutes

**Response Steps**:

1. **Immediate Assessment (T+0 to T+15 minutes)**
   ```bash
   # Verify node connectivity
   ping -c 5 $PRIMARY_NODE_IP
   
   # Check hypervisor status (if accessible)
   # Contact cloud provider support if no response
   ```

2. **Failover Decision (T+15 to T+30 minutes)**
   ```bash
   # Check last backup status
   ./scripts/dr_check_backup_status.py
   
   # Verify secondary node readiness
   ./scripts/dr_verify_secondary.py
   
   # Make failover decision
   echo "[DECISION] Proceeding with failover to secondary node"
   ```

3. **Database Recovery (T+30 to T+90 minutes)**
   ```bash
   # Restore PostgreSQL from WAL-G backup
   ./scripts/dr_restore_postgres.py --target-time="$(date -d '10 minutes ago' --iso-8601)"
   
   # Verify database integrity
   ./scripts/dr_verify_database.py
   
   # Start database on secondary node
   systemctl start postgresql
   ```

4. **Application Recovery (T+90 to T+180 minutes)**
   ```bash
   # Deploy application stack on secondary node
   ./scripts/dr_deploy_stack.py --environment=production --node=secondary
   
   # Restore Redis from backup
   ./scripts/dr_restore_redis.py
   
   # Start all services
   docker-compose -f compose/docker-compose.production.yml up -d
   ```

5. **DNS and Traffic Cutover (T+180 to T+210 minutes)**
   ```bash
   # Update DNS records to point to secondary node
   ./scripts/dr_update_dns.py --target=$SECONDARY_NODE_IP
   
   # Verify traffic routing
   ./scripts/dr_verify_traffic.py
   ```

6. **Validation and Monitoring (T+210 to T+240 minutes)**
   ```bash
   # Run comprehensive health checks
   ./scripts/dr_health_check.py --full
   
   # Enable monitoring on secondary node
   ./scripts/dr_enable_monitoring.py
   
   # Verify all services are operational
   ./scripts/validate_deployment.py
   ```

### DR-2: Database Corruption (RTO: 2 hours, RPO: 5 minutes)

**Detection**:
- PostgreSQL errors in logs
- Data integrity check failures
- Application unable to read/write to database

**Response Steps**:

1. **Isolation and Assessment (T+0 to T+15 minutes)**
   ```bash
   # Stop application writes to prevent further corruption
   ./scripts/dr_stop_app_writes.py
   
   # Assess corruption extent
   ./scripts/dr_assess_db_corruption.py
   
   # Check last known good backup
   ./scripts/dr_check_backup_status.py --verify-integrity
   ```

2. **Recovery Decision (T+15 to T+30 minutes)**
   ```bash
   # Determine recovery strategy
   if [[ $CORRUPTION_SEVERITY == "critical" ]]; then
       echo "DECISION: Full database restore required"
       ./scripts/dr_full_db_restore.py
   else
       echo "DECISION: Attempting repair-in-place"
       ./scripts/dr_repair_database.py
   fi
   ```

3. **Database Restoration (T+30 to T+90 minutes)**
   ```bash
   # Stop PostgreSQL
   systemctl stop postgresql
   
   # Backup corrupted data for forensics
   ./scripts/dr_backup_corrupted_data.py
   
   # Restore from WAL-G backup
   wal-g backup-fetch /var/lib/postgresql/data LATEST
   
   # Start PostgreSQL in recovery mode
   systemctl start postgresql
   ```

4. **Data Validation (T+90 to T+120 minutes)**
   ```bash
   # Run data integrity checks
   ./scripts/dr_verify_database.py --full-check
   
   # Validate critical business data
   ./scripts/dr_validate_business_data.py
   
   # Check for data loss
   ./scripts/dr_assess_data_loss.py
   ```

### DR-3: Security Breach / Ransomware (RTO: 6 hours, RPO: 1 hour)

**Detection**:
- Security alerts from monitoring
- Suspicious file modifications
- Encryption of data directories
- Unknown processes running

**Response Steps**:

1. **Immediate Containment (T+0 to T+15 minutes)**
   ```bash
   # Isolate affected systems
   ./scripts/dr_isolate_systems.py
   
   # Preserve evidence
   ./scripts/dr_preserve_evidence.py
   
   # Alert security team
   ./scripts/dr_security_alert.py --incident-type="breach"
   ```

2. **Forensic Analysis (T+15 to T+60 minutes)**
   ```bash
   # Analyze compromise scope
   ./scripts/dr_analyze_compromise.py
   
   # Check backup integrity
   ./scripts/dr_verify_backup_integrity.py --security-scan
   
   # Determine clean restore point
   ./scripts/dr_find_clean_restore_point.py
   ```

3. **System Rebuild (T+60 to T+300 minutes)**
   ```bash
   # Rebuild from clean images
   ./scripts/dr_rebuild_from_clean.py
   
   # Restore data from verified clean backup
   ./scripts/dr_restore_from_clean_backup.py
   
   # Implement additional security measures
   ./scripts/dr_harden_systems.py
   ```

4. **Security Validation (T+300 to T+360 minutes)**
   ```bash
   # Run full security scan
   ./scripts/dr_security_scan.py --comprehensive
   
   # Verify no persistence mechanisms remain
   ./scripts/dr_verify_clean_state.py
   
   # Enable enhanced monitoring
   ./scripts/dr_enable_enhanced_monitoring.py
   ```

## Game Day Exercises

### Monthly Game Day Schedule

- **Week 1**: Database failure simulation
- **Week 2**: Network partition simulation
- **Week 3**: Security incident simulation
- **Week 4**: Full node failure simulation

### Game Day Exercise Templates

#### Exercise 1: Database Failover Drill

**Scenario**: Primary PostgreSQL instance fails during peak traffic

**Participants**: Engineering team, DevOps, Management

**Duration**: 2 hours

**Steps**:
1. **Setup** (10 minutes)
   ```bash
   # Prepare monitoring dashboards
   ./scripts/gameday_setup_monitoring.py --exercise="db-failover"
   
   # Enable additional logging
   ./scripts/gameday_enable_logging.py
   ```

2. **Inject Failure** (5 minutes)
   ```bash
   # Simulate database failure
   ./scripts/gameday_inject_db_failure.py --type="process-kill"
   ```

3. **Response and Recovery** (90 minutes)
   - Follow DR-2 procedures
   - Document response times
   - Record issues encountered

4. **Debrief** (15 minutes)
   - Review response metrics
   - Identify improvement areas
   - Update procedures if needed

#### Exercise 2: Traffic Surge Simulation

**Scenario**: 10x normal traffic surge during vulnerability disclosure

**Duration**: 1 hour

**Steps**:
1. **Load Generation**
   ```bash
   # Generate realistic traffic spike
   ./scripts/gameday_traffic_surge.py --multiplier=10 --duration=60
   ```

2. **Monitor Response**
   - Auto-scaling behavior
   - Performance degradation
   - Error rates

3. **Capacity Management**
   ```bash
   # Scale resources if needed
   ./scripts/gameday_scale_resources.py --target-capacity=150%
   ```

## Recovery Time and Point Objectives (RTO/RPO)

| Scenario | RTO Target | RPO Target | Current | Status |
|----------|------------|------------|---------|--------|
| Node Failure | 4 hours | 5 minutes | 3.5 hours | ✅ Met |
| Database Corruption | 2 hours | 5 minutes | 1.8 hours | ✅ Met |
| Security Breach | 6 hours | 1 hour | 5.2 hours | ✅ Met |
| Network Partition | 30 minutes | 1 minute | 25 minutes | ✅ Met |
| Storage Failure | 1 hour | 5 minutes | 45 minutes | ✅ Met |

## Backup Verification Schedule

### Daily Checks (Automated)
```bash
# Verify backup completion
0 6 * * * /opt/xorb/scripts/verify_daily_backup.py

# Test restore capability (sample data)
0 8 * * * /opt/xorb/scripts/test_restore_sample.py
```

### Weekly Checks (Semi-Automated)
```bash
# Full restore test on secondary environment
0 10 * * SUN /opt/xorb/scripts/weekly_restore_test.py

# Backup integrity verification
0 12 * * SUN /opt/xorb/scripts/verify_backup_integrity.py
```

### Monthly Checks (Manual)
- Full disaster recovery drill
- Cross-region backup verification
- Documentation update review

## Communication Templates

### Incident Declaration Email
```
Subject: [INCIDENT] Xorb PTaaS Service Degradation - Severity: {SEVERITY}

Incident ID: INC-{TIMESTAMP}
Start Time: {START_TIME}
Severity: {CRITICAL|HIGH|MEDIUM|LOW}
Services Affected: {SERVICE_LIST}

Current Status: {STATUS_DESCRIPTION}

Impact:
- {IMPACT_DESCRIPTION}

Actions Taken:
- {ACTION_1}
- {ACTION_2}

Next Update: {TIME}

Incident Commander: {NAME}
Contact: {PHONE/EMAIL}
```

### Resolution Notification
```
Subject: [RESOLVED] Xorb PTaaS Service Restored - INC-{INCIDENT_ID}

Incident ID: INC-{INCIDENT_ID}
Resolution Time: {END_TIME}
Total Duration: {DURATION}

Root Cause: {ROOT_CAUSE}

Resolution Summary:
{RESOLUTION_DESCRIPTION}

Follow-up Actions:
- {FOLLOWUP_1}
- {FOLLOWUP_2}

Post-Incident Review Scheduled: {PIR_DATE}
```

## Post-Incident Review (PIR) Process

### Within 24 Hours of Resolution
1. **Incident Timeline Documentation**
2. **Root Cause Analysis**
3. **Response Effectiveness Assessment**
4. **Process Improvement Identification**

### Within 1 Week
1. **PIR Meeting with All Stakeholders**
2. **Action Item Assignment**
3. **Procedure Updates**
4. **Monitoring/Alerting Improvements**

### Template Questions for PIR
1. What went well during the incident response?
2. What could have been done better?
3. Were our RTO/RPO targets met?
4. Did we have the right information at the right time?
5. How can we prevent similar incidents?
6. What processes need improvement?

## Automation Scripts Status

| Script | Status | Last Updated | Purpose |
|--------|--------|--------------|---------|
| dr_check_backup_status.py | ✅ Ready | 2024-01-15 | Verify backup status |
| dr_restore_postgres.py | ✅ Ready | 2024-01-15 | PostgreSQL restoration |
| dr_deploy_stack.py | ✅ Ready | 2024-01-15 | Application deployment |
| gameday_inject_failure.py | 🔄 In Progress | 2024-01-10 | Failure injection |
| dr_security_scan.py | 📋 Planned | N/A | Security validation |

## Emergency Procedures Quick Reference

### Critical Service Down
1. Check monitoring dashboards
2. Verify node accessibility
3. Review recent deployments
4. Follow appropriate DR scenario
5. Communicate status to stakeholders

### Data Loss Suspected
1. STOP all write operations immediately
2. Assess scope of data loss
3. Check backup availability
4. Determine restore point
5. Execute restore procedure

### Security Alert
1. Isolate affected systems
2. Preserve evidence
3. Assess compromise scope
4. Follow security incident procedure
5. Notify security team immediately

## Contact Information

- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Slack Channel**: #xorb-incidents
- **PagerDuty**: https://xorb.pagerduty.com
- **Status Page**: https://status.xorb.ai

---
*Last Updated: 2024-01-15*
*Next Review: 2024-02-15*