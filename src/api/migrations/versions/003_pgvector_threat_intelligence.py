"""Add pgvector support for AI-powered threat intelligence

Revision ID: 003_pgvector_support
Revises: 002_security_setup
Create Date: 2025-08-09 14:00:00.000000

This migration adds vector embedding support for:
1. Threat indicator similarity search
2. Vulnerability correlation analysis
3. Behavioral pattern recognition
4. Automated threat hunting
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '003_pgvector_support'
down_revision = '002_security_setup'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add pgvector columns and intelligence functions"""

    # Add vector embedding column (1536 dimensions for OpenAI embeddings)
    vector_columns_sql = """
    -- Add vector column to embedding_vectors table
    ALTER TABLE embedding_vectors
    ADD COLUMN IF NOT EXISTS embedding vector(1536);

    -- Create vector similarity index for fast searches
    CREATE INDEX IF NOT EXISTS idx_embedding_vectors_cosine
        ON embedding_vectors USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

    -- Create additional indexes for hybrid search
    CREATE INDEX IF NOT EXISTS idx_embedding_vectors_tenant_model
        ON embedding_vectors (tenant_id, embedding_model);
    """

    op.execute(text(vector_columns_sql))

    # Create threat intelligence tables
    threat_intel_tables_sql = """
    -- Vulnerability database
    CREATE TABLE IF NOT EXISTS vulnerabilities (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        cve_id varchar(50) UNIQUE,
        title varchar(500) NOT NULL,
        description text,
        severity varchar(20) NOT NULL DEFAULT 'UNKNOWN',
        cvss_score float,
        cvss_vector varchar(200),
        published_date timestamp with time zone,
        modified_date timestamp with time zone,
        affected_products jsonb DEFAULT '[]',
        tags jsonb DEFAULT '[]',
        references jsonb DEFAULT '[]',
        created_at timestamp with time zone DEFAULT NOW(),
        updated_at timestamp with time zone DEFAULT NOW()
    );

    -- Enable RLS on vulnerabilities
    ALTER TABLE vulnerabilities ENABLE ROW LEVEL SECURITY;
    CREATE POLICY tenant_isolation_vulnerabilities ON vulnerabilities
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

    -- Attack patterns and TTPs (MITRE ATT&CK)
    CREATE TABLE IF NOT EXISTS attack_patterns (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        mitre_id varchar(20),
        name varchar(200) NOT NULL,
        description text,
        tactic varchar(100),
        technique varchar(100),
        sub_technique varchar(100),
        platforms jsonb DEFAULT '[]',
        data_sources jsonb DEFAULT '[]',
        defenses_bypassed jsonb DEFAULT '[]',
        permissions_required jsonb DEFAULT '[]',
        created_at timestamp with time zone DEFAULT NOW(),
        updated_at timestamp with time zone DEFAULT NOW()
    );

    -- Enable RLS on attack patterns
    ALTER TABLE attack_patterns ENABLE ROW LEVEL SECURITY;
    CREATE POLICY tenant_isolation_attack_patterns ON attack_patterns
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

    -- Threat actor profiles
    CREATE TABLE IF NOT EXISTS threat_actors (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        name varchar(200) NOT NULL,
        aliases jsonb DEFAULT '[]',
        description text,
        sophistication_level varchar(20) DEFAULT 'UNKNOWN',
        primary_motivation varchar(50),
        secondary_motivations jsonb DEFAULT '[]',
        capabilities jsonb DEFAULT '[]',
        attributed_attacks jsonb DEFAULT '[]',
        infrastructure jsonb DEFAULT '[]',
        created_at timestamp with time zone DEFAULT NOW(),
        updated_at timestamp with time zone DEFAULT NOW()
    );

    -- Enable RLS on threat actors
    ALTER TABLE threat_actors ENABLE ROW LEVEL SECURITY;
    CREATE POLICY tenant_isolation_threat_actors ON threat_actors
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

    -- Create indexes for threat intelligence queries
    CREATE INDEX IF NOT EXISTS idx_vulnerabilities_cve ON vulnerabilities (cve_id);
    CREATE INDEX IF NOT EXISTS idx_vulnerabilities_severity ON vulnerabilities (severity, cvss_score DESC);
    CREATE INDEX IF NOT EXISTS idx_vulnerabilities_published ON vulnerabilities (published_date DESC);

    CREATE INDEX IF NOT EXISTS idx_attack_patterns_mitre ON attack_patterns (mitre_id);
    CREATE INDEX IF NOT EXISTS idx_attack_patterns_tactic ON attack_patterns (tactic, technique);

    CREATE INDEX IF NOT EXISTS idx_threat_actors_name ON threat_actors (name);
    """

    op.execute(text(threat_intel_tables_sql))

    # Create AI-powered analysis functions
    ai_functions_sql = """
    -- Function to find similar threats using vector similarity
    CREATE OR REPLACE FUNCTION find_similar_threats(
        tenant_uuid uuid,
        query_embedding vector(1536),
        similarity_threshold float DEFAULT 0.8,
        max_results int DEFAULT 10
    )
    RETURNS TABLE(
        content_id uuid,
        content_type varchar(50),
        similarity_score float,
        metadata jsonb
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT
            ev.content_id,
            ev.content_type,
            (1 - (ev.embedding <=> query_embedding))::float as similarity_score,
            ev.vector_metadata
        FROM embedding_vectors ev
        WHERE ev.tenant_id = tenant_uuid
          AND (1 - (ev.embedding <=> query_embedding)) > similarity_threshold
        ORDER BY ev.embedding <=> query_embedding
        LIMIT max_results;
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;

    -- Function to analyze threat landscape for a tenant
    CREATE OR REPLACE FUNCTION analyze_threat_landscape(tenant_uuid uuid)
    RETURNS jsonb AS $$
    DECLARE
        result jsonb;
    BEGIN
        SELECT jsonb_build_object(
            'total_indicators', (
                SELECT COUNT(*) FROM threat_indicators
                WHERE tenant_id = tenant_uuid
            ),
            'high_confidence_indicators', (
                SELECT COUNT(*) FROM threat_indicators
                WHERE tenant_id = tenant_uuid AND confidence_score > 0.8
            ),
            'critical_vulnerabilities', (
                SELECT COUNT(*) FROM vulnerabilities
                WHERE tenant_id = tenant_uuid AND severity = 'CRITICAL'
            ),
            'active_incidents', (
                SELECT COUNT(*) FROM security_incidents
                WHERE tenant_id = tenant_uuid AND status IN ('OPEN', 'INVESTIGATING')
            ),
            'threat_categories', (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'category', indicator_type,
                        'count', cnt
                    )
                ) FROM (
                    SELECT indicator_type, COUNT(*) as cnt
                    FROM threat_indicators
                    WHERE tenant_id = tenant_uuid
                    GROUP BY indicator_type
                    ORDER BY cnt DESC
                    LIMIT 10
                ) categories
            ),
            'recent_activity', (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'date', date_trunc('day', created_at),
                        'count', cnt
                    )
                ) FROM (
                    SELECT date_trunc('day', created_at) as day, COUNT(*) as cnt
                    FROM threat_indicators
                    WHERE tenant_id = tenant_uuid
                      AND created_at > NOW() - INTERVAL '30 days'
                    GROUP BY day
                    ORDER BY day DESC
                ) activity
            )
        ) INTO result;

        RETURN result;
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;

    -- Function to correlate threats across different data sources
    CREATE OR REPLACE FUNCTION correlate_threats(
        tenant_uuid uuid,
        indicator_value text,
        correlation_window interval DEFAULT '7 days'
    )
    RETURNS jsonb AS $$
    DECLARE
        result jsonb;
    BEGIN
        SELECT jsonb_build_object(
            'primary_indicator', indicator_value,
            'correlation_timeframe', correlation_window,
            'related_indicators', (
                SELECT jsonb_agg(DISTINCT ti.indicator_value)
                FROM threat_indicators ti
                JOIN security_incidents si ON si.indicators ? ti.id::text
                WHERE ti.tenant_id = tenant_uuid
                  AND si.tenant_id = tenant_uuid
                  AND ti.created_at > NOW() - correlation_window
                  AND ti.indicator_value != indicator_value
                  AND EXISTS (
                      SELECT 1 FROM security_incidents si2
                      WHERE si2.indicators ? (
                          SELECT id::text FROM threat_indicators
                          WHERE indicator_value = correlate_threats.indicator_value
                          AND tenant_id = tenant_uuid
                      )
                  )
            ),
            'associated_incidents', (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'incident_id', si.id,
                        'title', si.title,
                        'severity', si.severity,
                        'status', si.status,
                        'created_at', si.created_at
                    )
                )
                FROM security_incidents si
                WHERE si.tenant_id = tenant_uuid
                  AND si.indicators ? (
                      SELECT id::text FROM threat_indicators
                      WHERE indicator_value = correlate_threats.indicator_value
                      AND tenant_id = tenant_uuid
                  )
                  AND si.created_at > NOW() - correlation_window
            )
        ) INTO result;

        RETURN result;
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;

    -- Grant permissions to application role
    GRANT EXECUTE ON FUNCTION find_similar_threats(uuid, vector(1536), float, int) TO xorb_app_service;
    GRANT EXECUTE ON FUNCTION analyze_threat_landscape(uuid) TO xorb_app_service;
    GRANT EXECUTE ON FUNCTION correlate_threats(uuid, text, interval) TO xorb_app_service;
    """

    op.execute(text(ai_functions_sql))

    # Create materialized views for performance
    materialized_views_sql = """
    -- Materialized view for threat intelligence dashboard
    CREATE MATERIALIZED VIEW IF NOT EXISTS threat_intel_dashboard AS
    SELECT
        t.id as tenant_id,
        t.name as tenant_name,
        COUNT(DISTINCT ti.id) as total_indicators,
        COUNT(DISTINCT CASE WHEN ti.confidence_score > 0.8 THEN ti.id END) as high_confidence_indicators,
        COUNT(DISTINCT v.id) as total_vulnerabilities,
        COUNT(DISTINCT CASE WHEN v.severity = 'CRITICAL' THEN v.id END) as critical_vulnerabilities,
        COUNT(DISTINCT si.id) as active_incidents,
        MAX(ti.last_seen) as last_threat_activity,
        jsonb_agg(DISTINCT ti.indicator_type) FILTER (WHERE ti.indicator_type IS NOT NULL) as threat_types
    FROM tenants t
    LEFT JOIN threat_indicators ti ON t.id = ti.tenant_id
    LEFT JOIN vulnerabilities v ON t.id = v.tenant_id
    LEFT JOIN security_incidents si ON t.id = si.tenant_id AND si.status IN ('OPEN', 'INVESTIGATING')
    GROUP BY t.id, t.name;

    -- Create unique index for concurrent refresh
    CREATE UNIQUE INDEX IF NOT EXISTS idx_threat_intel_dashboard_tenant
        ON threat_intel_dashboard (tenant_id);

    -- Grant select permission
    GRANT SELECT ON threat_intel_dashboard TO xorb_app_service;
    GRANT SELECT ON threat_intel_dashboard TO xorb_analytics;
    """

    op.execute(text(materialized_views_sql))


def downgrade() -> None:
    """Remove pgvector threat intelligence features"""

    # Drop materialized views
    op.execute(text("DROP MATERIALIZED VIEW IF EXISTS threat_intel_dashboard"))

    # Drop functions
    op.execute(text("DROP FUNCTION IF EXISTS find_similar_threats(uuid, vector(1536), float, int)"))
    op.execute(text("DROP FUNCTION IF EXISTS analyze_threat_landscape(uuid)"))
    op.execute(text("DROP FUNCTION IF EXISTS correlate_threats(uuid, text, interval)"))

    # Drop threat intelligence tables
    op.execute(text("DROP TABLE IF EXISTS threat_actors"))
    op.execute(text("DROP TABLE IF EXISTS attack_patterns"))
    op.execute(text("DROP TABLE IF EXISTS vulnerabilities"))

    # Drop vector indexes and columns
    op.execute(text("DROP INDEX IF EXISTS idx_embedding_vectors_cosine"))
    op.execute(text("DROP INDEX IF EXISTS idx_embedding_vectors_tenant_model"))

    # Remove vector column
    op.execute(text("ALTER TABLE embedding_vectors DROP COLUMN IF EXISTS embedding"))
