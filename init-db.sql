-- XORB Phase 12 Database Initialization
-- Creates all necessary tables for the autonomous agent ecosystem

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'xorb_agent') THEN
        CREATE ROLE xorb_agent WITH LOGIN PASSWORD 'agent_secure_pass_2023';
    END IF;
END
$$;

-- Grant permissions
GRANT CONNECT ON DATABASE xorb TO xorb_agent;
GRANT USAGE ON SCHEMA public TO xorb_agent;
GRANT CREATE ON SCHEMA public TO xorb_agent;

-- Orchestrator Agent Tables
CREATE TABLE IF NOT EXISTS orchestration_cycles (
    cycle_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(100) NOT NULL,
    cycle_number BIGINT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    agents_discovered INTEGER DEFAULT 0,
    tasks_executed INTEGER DEFAULT 0,
    consensus_achieved BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id VARCHAR(100) PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    capabilities JSONB,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    metrics JSONB,
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Evolutionary Defense Agent Tables
CREATE TABLE IF NOT EXISTS defense_protocols (
    protocol_id VARCHAR(100) PRIMARY KEY,
    generation INTEGER NOT NULL,
    protocol_data JSONB NOT NULL,
    fitness_score REAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evolution_metrics (
    id SERIAL PRIMARY KEY,
    generation INTEGER NOT NULL,
    metrics_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS protocol_deployments (
    id SERIAL PRIMARY KEY,
    protocol_id VARCHAR(100) NOT NULL,
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) NOT NULL,
    fitness_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS protocol_feedback (
    id SERIAL PRIMARY KEY,
    protocol_id VARCHAR(100) NOT NULL,
    feedback_data JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Threat Propagation Agent Tables
CREATE TABLE IF NOT EXISTS network_nodes (
    node_id VARCHAR(100) PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,
    state VARCHAR(20) NOT NULL,
    infection_time TIMESTAMP WITH TIME ZONE,
    recovery_time TIMESTAMP WITH TIME ZONE,
    vulnerability_score REAL NOT NULL,
    connectivity INTEGER NOT NULL,
    criticality REAL NOT NULL,
    security_controls JSONB NOT NULL,
    latitude REAL,
    longitude REAL,
    active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS network_edges (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(100) NOT NULL,
    target_id VARCHAR(100) NOT NULL,
    connection_type VARCHAR(50) NOT NULL,
    weight REAL NOT NULL,
    bandwidth REAL NOT NULL,
    latency REAL NOT NULL,
    security_level VARCHAR(20) NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_id, target_id)
);

CREATE TABLE IF NOT EXISTS propagation_scenarios (
    scenario_id VARCHAR(100) PRIMARY KEY,
    threat_type VARCHAR(50) NOT NULL,
    scenario_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_results (
    id SERIAL PRIMARY KEY,
    scenario_id VARCHAR(100) NOT NULL,
    final_infection_rate REAL NOT NULL,
    peak_infection_time INTEGER NOT NULL,
    total_infected_nodes INTEGER NOT NULL,
    result_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS containment_strategies (
    strategy_id VARCHAR(100) PRIMARY KEY,
    strategy_type VARCHAR(50) NOT NULL,
    effectiveness_score REAL NOT NULL,
    implementation_cost REAL NOT NULL,
    strategy_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Autonomous Response Agent Tables
CREATE TABLE IF NOT EXISTS response_executions (
    execution_id VARCHAR(100) PRIMARY KEY,
    plan_id VARCHAR(100) NOT NULL,
    signal_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    execution_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS response_execution_archive (
    id SERIAL PRIMARY KEY,
    execution_id VARCHAR(100) NOT NULL,
    plan_id VARCHAR(100) NOT NULL,
    signal_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    effectiveness_score REAL NOT NULL,
    execution_data JSONB NOT NULL,
    archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ecosystem Integration Agent Tables
CREATE TABLE IF NOT EXISTS integration_partners (
    partner_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    partner_type VARCHAR(50) NOT NULL,
    partner_data JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS threat_indicators (
    indicator_id VARCHAR(100) PRIMARY KEY,
    indicator_type VARCHAR(50) NOT NULL,
    value VARCHAR(500) NOT NULL,
    confidence REAL NOT NULL,
    severity VARCHAR(20) NOT NULL,
    source VARCHAR(100) NOT NULL,
    tlp_marking VARCHAR(10) NOT NULL,
    indicator_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS compliance_mappings (
    id SERIAL PRIMARY KEY,
    framework VARCHAR(50) NOT NULL,
    control_id VARCHAR(50) NOT NULL,
    control_name VARCHAR(200) NOT NULL,
    xorb_capability VARCHAR(100) NOT NULL,
    implementation_status VARCHAR(20) NOT NULL,
    compliance_score REAL NOT NULL,
    mapping_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(framework, control_id)
);

-- Shared Tables
CREATE TABLE IF NOT EXISTS agent_communications (
    id SERIAL PRIMARY KEY,
    from_agent VARCHAR(100) NOT NULL,
    to_agent VARCHAR(100) NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    message_data JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    agent_id VARCHAR(100),
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    target VARCHAR(200),
    outcome VARCHAR(20) NOT NULL,
    details JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_orchestration_cycles_agent ON orchestration_cycles(agent_id);
CREATE INDEX IF NOT EXISTS idx_orchestration_cycles_start_time ON orchestration_cycles(start_time);
CREATE INDEX IF NOT EXISTS idx_agent_registry_type ON agent_registry(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_registry_status ON agent_registry(status);

CREATE INDEX IF NOT EXISTS idx_protocols_generation ON defense_protocols(generation);
CREATE INDEX IF NOT EXISTS idx_protocols_fitness ON defense_protocols(fitness_score);
CREATE INDEX IF NOT EXISTS idx_deployments_protocol ON protocol_deployments(protocol_id);
CREATE INDEX IF NOT EXISTS idx_feedback_unprocessed ON protocol_feedback(processed, created_at);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON network_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_state ON network_nodes(state);
CREATE INDEX IF NOT EXISTS idx_edges_source ON network_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON network_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_scenarios_threat ON propagation_scenarios(threat_type);
CREATE INDEX IF NOT EXISTS idx_results_scenario ON simulation_results(scenario_id);

CREATE INDEX IF NOT EXISTS idx_executions_status ON response_executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_signal ON response_executions(signal_id);
CREATE INDEX IF NOT EXISTS idx_archive_effectiveness ON response_execution_archive(effectiveness_score);

CREATE INDEX IF NOT EXISTS idx_partners_type ON integration_partners(partner_type);
CREATE INDEX IF NOT EXISTS idx_indicators_type ON threat_indicators(indicator_type);
CREATE INDEX IF NOT EXISTS idx_indicators_source ON threat_indicators(source);
CREATE INDEX IF NOT EXISTS idx_indicators_tlp ON threat_indicators(tlp_marking);
CREATE INDEX IF NOT EXISTS idx_compliance_framework ON compliance_mappings(framework);

CREATE INDEX IF NOT EXISTS idx_communications_to_agent ON agent_communications(to_agent, processed);
CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_agent_time ON audit_logs(agent_id, timestamp);

-- Grant permissions to agent user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO xorb_agent;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO xorb_agent;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_agent_registry_updated_at BEFORE UPDATE ON agent_registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_defense_protocols_updated_at BEFORE UPDATE ON defense_protocols FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_response_executions_updated_at BEFORE UPDATE ON response_executions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_integration_partners_updated_at BEFORE UPDATE ON integration_partners FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_threat_indicators_updated_at BEFORE UPDATE ON threat_indicators FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_compliance_mappings_updated_at BEFORE UPDATE ON compliance_mappings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data
INSERT INTO agent_registry (agent_id, agent_type, status, capabilities, configuration) VALUES
('orchestrator-001', 'orchestrator', 'active', '["coordination", "consensus", "discovery"]', '{"cycle_interval": 500, "max_agents": 32}'),
('evolutionary-defense-001', 'evolutionary_defense', 'active', '["protocol_evolution", "fitness_evaluation", "genetic_algorithms"]', '{"population_size": 50, "mutation_rate": 0.1}'),
('threat-propagation-001', 'threat_propagation', 'active', '["network_modeling", "simulation", "containment"]', '{"simulation_steps": 1000, "monte_carlo_runs": 100}'),
('autonomous-response-001', 'autonomous_response', 'active', '["threat_response", "coordination", "remediation"]', '{"confidence_threshold": 0.72, "max_concurrent": 10}'),
('ecosystem-integration-001', 'ecosystem_integration', 'active', '["threat_intel", "standards_compliance", "federation"]', '{"sync_interval": 3600, "max_operations": 20}')
ON CONFLICT (agent_id) DO NOTHING;

-- Log initialization
INSERT INTO audit_logs (agent_id, action, outcome, details) VALUES
('system', 'database_initialization', 'success', '{"tables_created": 20, "indexes_created": 25, "triggers_created": 6}');

COMMIT;