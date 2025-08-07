-- XORB Database Initialization Script
-- Creates database schema for the XORB cybersecurity platform

-- Connect to default database first
\c postgres;

-- Create temporal database
SELECT 'CREATE DATABASE temporal' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'temporal')\gexec

-- Connect to xorb database
\c xorb;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS xorb_core;
CREATE SCHEMA IF NOT EXISTS xorb_agents;
CREATE SCHEMA IF NOT EXISTS xorb_ptaas;
CREATE SCHEMA IF NOT EXISTS xorb_analytics;

-- Core system tables
CREATE TABLE IF NOT EXISTS xorb_core.system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS xorb_core.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(255),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent management tables
CREATE TABLE IF NOT EXISTS xorb_agents.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB NOT NULL DEFAULT '{}',
    capabilities TEXT[],
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- PTaaS tables
CREATE TABLE IF NOT EXISTS xorb_ptaas.companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    industry VARCHAR(100),
    contact_email VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS xorb_ptaas.researchers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    reputation_score INTEGER DEFAULT 0,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS xorb_ptaas.vulnerabilities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(50) DEFAULT 'submitted',
    researcher_id UUID REFERENCES xorb_ptaas.researchers(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert demo data
INSERT INTO xorb_ptaas.companies (name, domain, industry, contact_email) VALUES 
('TechCorp Solutions', 'techcorp.com', 'Technology', 'security@techcorp.com'),
('SecureBank Inc', 'securebank.com', 'Financial Services', 'security@securebank.com')
ON CONFLICT DO NOTHING;

INSERT INTO xorb_ptaas.researchers (username, email, first_name, last_name, reputation_score, verified) VALUES 
('alexchen', 'alex@example.com', 'Alex', 'Chen', 950, true),
('sarahj', 'sarah@example.com', 'Sarah', 'Johnson', 875, true)
ON CONFLICT DO NOTHING;