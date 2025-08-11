# 🔮 XORB Cyber Range Evolution Roadmap - Red • Blue • Purple (SEaaS)

## Executive Summary

This is a phased, execution-ready roadmap transforming the XORB wargame from "good lab" to a self-adapting, zero-day-sniffing cyber ecosystem. Built for enterprise-scale Security-as-a-Service (SEaaS) with NVIDIA Qwen3-235b-a22b MoE orchestration.

## Phase Overview

| Phase | Timeline | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| Phase 0 | Week 0-1 | Stabilize & Instrument | Purple Generator v1, Telemetry spine, Safety rails |
| Phase 1 | Week 2-4 | Replay-Driven RL Core | BC→IQL training loop, League manager |
| Phase 2 | Week 5-7 | Quality-Diversity + Curriculum | Role personas, MAP-Elites archives |
| Phase 3 | Week 8-10 | Offensive & Defensive Leaps | Multi-stage adversary, Adaptive defenses |
| Phase 4 | Week 11-13 | Campaign Mode | Persistent worlds, Threat hunting |
| Phase 5 | Week 14-16 | Shadow League & Compliance | 24/7 AI league, Compliance dynamics |

## 30/60/90 Day Milestones

### Day 30 (Phase 0-1)
- ✅ Purple Generator v1 with tenant isolation & snapshots
- ✅ Telemetry spine (OTel, Zeek/PCAP, SIEM integration)
- ✅ BC + IQL offline training loop
- ✅ Basic league with promotion mechanics

### Day 60 (Phase 2)
- ✅ Role/interest personas with MAP-Elites archives
- ✅ Curriculum tiers (T1-T2 operational)
- ✅ Deception mesh v1 with honey deployment
- ✅ JWKS/endpoint rotation capabilities

### Day 90 (Phase 3-4)
- ✅ Multi-stage Red Team with fuzzing integration
- ✅ Auto-PR system with replay validation gates
- ✅ Campaign mode pilot deployment
- ✅ Shadow league operational

## Architecture Components

### 1. Replay Store Infrastructure
```
replay-store/
├── schemas/
│   ├── events.jsonl.zst      # Compressed event streams
│   ├── metrics.parquet       # Performance metrics
│   ├── pcap/                 # Network captures
│   └── manifest.json         # Episode metadata
├── ingestion/                # Real-time event ingestion
├── analytics/                # ML feature extraction
└── storage/                  # Distributed storage layer
```

### 2. League Management System
```
league/
├── managers/
│   ├── promotion_engine.py   # Policy promotion logic
│   ├── champion_archive.py   # Historical champions
│   └── shadow_league.py      # 24/7 AI-only arena
├── policies/
│   ├── bc_trainer.py         # Behavioral cloning
│   ├── iql_trainer.py        # Implicit Q-Learning
│   └── map_elites.py         # Quality-diversity archives
└── evaluation/
    ├── arena.py              # Head-to-head evaluation
    └── metrics.py            # KPI calculation
```

### 3. Safety & Governance
```
safety/
├── roe_contracts.py          # Rules of engagement
├── kill_switch.py            # Emergency shutdown
├── scope_enforcer.py         # Action boundary enforcement
├── safety_critic.py          # AI safety monitor
└── compliance/
    ├── hipaa.py              # Healthcare compliance
    ├── pci_dss.py            # Payment card compliance
    └── gdpr.py               # Data protection compliance
```

## Current Implementation Status

### ✅ Phase 0 Complete
- Purple Generator v1 with tenant isolation
- Telemetry spine with OTel integration
- Safety rails with RoE contracts
- Replay store with JSONL/Parquet schemas

### 🚧 Phase 1 In Progress
- BC→IQL training pipeline (80% complete)
- League manager with promotion logic (60% complete)
- Policy gateway with shadow/live flags (40% complete)

### 📋 Upcoming Phases
- Phase 2: Role personas and MAP-Elites (planned)
- Phase 3: Multi-stage adversary engine (designed)
- Phase 4: Campaign mode architecture (specified)
- Phase 5: Shadow league deployment (roadmapped)

## Key Performance Indicators

### Learning Metrics
- **Episodes/day**: Target ≥100 (current: 45)
- **Promotion cadence**: 1-2/week (current: bi-weekly)
- **Archive coverage**: Target ≥70% (current: 35%)
- **Novelty index**: Trending upward ✅

### Defense Metrics
- **Detection latency p95**: Target ↓30% (baseline: 45s)
- **False positive rate**: Target <5% (current: 12%)
- **Residual risk score**: Target ↓40% (tracking)
- **Replay pass-rate**: Target ≥95% (current: 78%)

### Business Metrics
- **MTTR/MTTC**: Mean time to recovery/containment
- **Time-to-patch**: From PR creation to deployment
- **SLA adherence**: Uptime during attack simulations
- **Zero-day findings**: Target ≥3/month from fuzzing

## Integration with NVIDIA MoE

### Orchestrator Brain
- **NVIDIA Qwen3-235b-a22b**: Executive planning and audit
- **Specialist models**: Coder (diffs/tests), Generalist (verification)
- **Bandit routing**: Dynamic model selection by role/interest
- **Verifier gates**: All PR blueprints require verification

### AI Safety Architecture
- **Safety Critic**: Real-time action monitoring
- **Verifier chains**: Multi-model consensus for critical decisions
- **Rollback mechanisms**: Automatic policy reversion on KPI breach
- **Human oversight**: Escalation paths for edge cases

## Risk Management

### Technical Risks
- **Overfitting to Purple**: Mitigation via weekly stack rotation
- **Cost creep**: Episode budgets with warm-pool optimization
- **Signal dilution**: Strict schemas with novelty penalties

### Operational Risks
- **Context sprawl**: Data contracts with schema linting
- **Model drift**: Continuous validation with benchmark suites
- **Security exposure**: Sandboxed execution with network isolation

## Quick Start Commands

```bash
# Phase 0: Initialize infrastructure
./scripts/setup-phase0.sh

# Deploy replay store
docker-compose -f compose/replay-store.yml up -d

# Start telemetry spine
./scripts/start-telemetry.sh

# Phase 1: Launch training loop
python league/managers/training_orchestrator.py

# Monitor KPIs
grafana-cli dashboard import dashboards/cyber-range-kpis.json
```

## Documentation Structure

```
docs/
├── architecture/             # System design documents
├── deployment/              # Deployment guides
├── api/                     # API documentation
├── tutorials/               # Getting started guides
├── compliance/              # Regulatory requirements
└── troubleshooting/         # Common issues and solutions
```

## Contact & Support

- **Architecture Team**: cyber-range-arch@xorb.ai
- **DevOps Support**: devops@xorb.ai
- **Security Review**: security@xorb.ai
- **Compliance**: compliance@xorb.ai

---

**Status**: Phase 0 Complete ✅ | Phase 1 In Progress 🚧 | Next Milestone: Day 60
**Last Updated**: 2025-08-11
**Version**: 1.0.0-alpha