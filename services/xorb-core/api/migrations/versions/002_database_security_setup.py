"""Database security setup with roles and permissions

Revision ID: 002_security_setup
Revises: 001_initial_schema
Create Date: 2025-08-09 13:45:00.000000

This migration sets up production-ready database security:
1. Database roles for different access levels
2. Connection pooling optimization
3. Performance monitoring views
4. Backup and maintenance procedures
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '002_security_setup'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Setup database security and performance optimization"""

    # Create database roles for different access levels
    security_setup_sql = """
    -- Create application roles
    DO $$
    BEGIN
        -- Application service role (used by API)
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xorb_app_service') THEN
            CREATE ROLE xorb_app_service WITH LOGIN;
        END IF;

        -- Read-only analytics role
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xorb_analytics') THEN
            CREATE ROLE xorb_analytics WITH LOGIN;
        END IF;

        -- Migration role (for schema changes)
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xorb_migration') THEN
            CREATE ROLE xorb_migration WITH LOGIN CREATEROLE;
        END IF;

        -- Authenticated user role (for RLS policies)
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated_user') THEN
            CREATE ROLE authenticated_user;
        END IF;
    END
    $$;

    -- Grant appropriate permissions

    -- Application service permissions (full CRUD within RLS constraints)
    GRANT USAGE ON SCHEMA public TO xorb_app_service;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO xorb_app_service;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO xorb_app_service;
    GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO xorb_app_service;

    -- Make xorb_app_service a member of authenticated_user for RLS
    GRANT authenticated_user TO xorb_app_service;

    -- Analytics role permissions (read-only)
    GRANT USAGE ON SCHEMA public TO xorb_analytics;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO xorb_analytics;
    GRANT authenticated_user TO xorb_analytics;

    -- Migration role permissions
    GRANT ALL PRIVILEGES ON SCHEMA public TO xorb_migration;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO xorb_migration;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO xorb_migration;
    GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO xorb_migration;

    -- Set default privileges for future objects
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO xorb_app_service;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO xorb_app_service;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO xorb_app_service;

    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO xorb_analytics;
    """

    op.execute(text(security_setup_sql))

    # Create performance monitoring views
    monitoring_views_sql = """
    -- Tenant activity monitoring view
    CREATE OR REPLACE VIEW tenant_activity_stats AS
    SELECT
        t.id as tenant_id,
        t.name as tenant_name,
        t.plan,
        t.status,
        COUNT(DISTINCT u.id) as active_users,
        COUNT(DISTINCT ti.id) as threat_indicators_count,
        COUNT(DISTINCT si.id) as open_incidents,
        MAX(al.timestamp) as last_activity,
        COUNT(al.id) as daily_actions
    FROM tenants t
    LEFT JOIN users u ON t.id = u.tenant_id AND u.is_active = true
    LEFT JOIN threat_indicators ti ON t.id = ti.tenant_id
    LEFT JOIN security_incidents si ON t.id = si.tenant_id AND si.status = 'OPEN'
    LEFT JOIN audit_logs al ON t.id = al.tenant_id AND al.timestamp > NOW() - INTERVAL '24 hours'
    GROUP BY t.id, t.name, t.plan, t.status;

    -- Performance monitoring view
    CREATE OR REPLACE VIEW performance_metrics AS
    SELECT
        'threat_indicators' as table_name,
        COUNT(*) as row_count,
        pg_size_pretty(pg_total_relation_size('threat_indicators')) as table_size
    FROM threat_indicators
    UNION ALL
    SELECT
        'audit_logs' as table_name,
        COUNT(*) as row_count,
        pg_size_pretty(pg_total_relation_size('audit_logs')) as table_size
    FROM audit_logs
    UNION ALL
    SELECT
        'embedding_vectors' as table_name,
        COUNT(*) as row_count,
        pg_size_pretty(pg_total_relation_size('embedding_vectors')) as table_size
    FROM embedding_vectors;

    -- Grant access to monitoring views
    GRANT SELECT ON tenant_activity_stats TO xorb_analytics;
    GRANT SELECT ON performance_metrics TO xorb_analytics;
    """

    op.execute(text(monitoring_views_sql))

    # Create maintenance functions
    maintenance_functions_sql = """
    -- Function to cleanup old audit logs
    CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(retention_days INTEGER DEFAULT 90)
    RETURNS INTEGER AS $$
    DECLARE
        deleted_count INTEGER;
    BEGIN
        DELETE FROM audit_logs
        WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;

        GET DIAGNOSTICS deleted_count = ROW_COUNT;

        -- Log the cleanup action
        INSERT INTO audit_logs (tenant_id, action, resource_type, details)
        VALUES (
            '00000000-0000-0000-0000-000000000000'::uuid,
            'CLEANUP_AUDIT_LOGS',
            'SYSTEM',
            json_build_object('deleted_count', deleted_count, 'retention_days', retention_days)
        );

        RETURN deleted_count;
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;

    -- Function to analyze tenant storage usage
    CREATE OR REPLACE FUNCTION analyze_tenant_storage(tenant_uuid uuid)
    RETURNS TABLE(
        resource_type text,
        count bigint,
        avg_size numeric,
        total_size text
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 'threat_indicators'::text,
               COUNT(*)::bigint,
               AVG(LENGTH(indicator_value))::numeric,
               pg_size_pretty(SUM(LENGTH(indicator_value))::bigint)
        FROM threat_indicators
        WHERE tenant_id = tenant_uuid
        UNION ALL
        SELECT 'security_incidents'::text,
               COUNT(*)::bigint,
               AVG(LENGTH(COALESCE(description, '')))::numeric,
               pg_size_pretty(SUM(LENGTH(COALESCE(description, '')))::bigint)
        FROM security_incidents
        WHERE tenant_id = tenant_uuid
        UNION ALL
        SELECT 'audit_logs'::text,
               COUNT(*)::bigint,
               AVG(LENGTH(details::text))::numeric,
               pg_size_pretty(SUM(LENGTH(details::text))::bigint)
        FROM audit_logs
        WHERE tenant_id = tenant_uuid;
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;

    -- Grant execute permissions
    GRANT EXECUTE ON FUNCTION cleanup_old_audit_logs(INTEGER) TO xorb_app_service;
    GRANT EXECUTE ON FUNCTION analyze_tenant_storage(uuid) TO xorb_analytics;
    """

    op.execute(text(maintenance_functions_sql))

    # Create database configuration optimizations
    optimization_sql = """
    -- Enable query performance tracking
    -- (These would typically be set at the database level)
    -- ALTER SYSTEM SET track_activities = on;
    -- ALTER SYSTEM SET track_counts = on;
    -- ALTER SYSTEM SET track_io_timing = on;

    -- Create indexes for common query patterns
    CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_action_timestamp
        ON audit_logs (tenant_id, action, timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_threat_indicators_tenant_confidence
        ON threat_indicators (tenant_id, confidence_score DESC);

    CREATE INDEX IF NOT EXISTS idx_security_incidents_tenant_severity_status
        ON security_incidents (tenant_id, severity, status);

    -- Partial indexes for active records
    CREATE INDEX IF NOT EXISTS idx_users_active
        ON users (tenant_id, email) WHERE is_active = true;

    CREATE INDEX IF NOT EXISTS idx_threat_feeds_active
        ON threat_feeds (tenant_id, feed_type) WHERE is_active = true;
    """

    op.execute(text(optimization_sql))


def downgrade() -> None:
    """Remove database security setup"""

    # Drop maintenance functions
    op.execute(text("DROP FUNCTION IF EXISTS cleanup_old_audit_logs(INTEGER)"))
    op.execute(text("DROP FUNCTION IF EXISTS analyze_tenant_storage(uuid)"))

    # Drop monitoring views
    op.execute(text("DROP VIEW IF EXISTS tenant_activity_stats"))
    op.execute(text("DROP VIEW IF EXISTS performance_metrics"))

    # Drop additional indexes
    op.execute(text("DROP INDEX IF EXISTS idx_audit_logs_tenant_action_timestamp"))
    op.execute(text("DROP INDEX IF EXISTS idx_threat_indicators_tenant_confidence"))
    op.execute(text("DROP INDEX IF EXISTS idx_security_incidents_tenant_severity_status"))
    op.execute(text("DROP INDEX IF EXISTS idx_users_active"))
    op.execute(text("DROP INDEX IF EXISTS idx_threat_feeds_active"))

    # Note: We don't drop roles as they might be in use
    # This would be done manually in production
    """
    DROP ROLE IF EXISTS xorb_app_service;
    DROP ROLE IF EXISTS xorb_analytics;
    DROP ROLE IF EXISTS xorb_migration;
    DROP ROLE IF EXISTS authenticated_user;
    """
