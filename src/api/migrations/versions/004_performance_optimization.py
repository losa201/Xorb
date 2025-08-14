"""Performance optimization for high-volume PTaaS scanning

Revision ID: 004_performance_optimization
Revises: 003_pgvector_threat_intelligence
Create Date: 2025-08-10 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '004_performance_optimization'
down_revision = '003_pgvector_threat_intelligence'
branch_labels = None
depends_on = None

def upgrade():
    """Apply performance optimizations for PTaaS scanning"""

    # Create scan sessions table for PTaaS orchestration
    op.create_table(
        'scan_sessions',
        sa.Column('session_id', sa.String(length=128), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=128), nullable=False),
        sa.Column('scan_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False, default=1),
        sa.Column('targets_count', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKey('session_id')
    )

    # Create scan targets table
    op.create_table(
        'scan_targets',
        sa.Column('target_id', sa.String(length=128), nullable=False),
        sa.Column('session_id', sa.String(length=128), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('host', sa.String(length=255), nullable=False),
        sa.Column('ports', postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column('scan_profile', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('scan_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scan_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('constraints', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKey('target_id'),
        sa.ForeignKey('scan_sessions.session_id', ondelete='CASCADE')
    )

    # Create scan results table for detailed findings
    op.create_table(
        'scan_results',
        sa.Column('result_id', sa.String(length=128), nullable=False),
        sa.Column('target_id', sa.String(length=128), nullable=False),
        sa.Column('session_id', sa.String(length=128), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('scanner', sa.String(length=50), nullable=False),
        sa.Column('vulnerability_name', sa.String(length=500), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('port', sa.Integer(), nullable=True),
        sa.Column('service', sa.String(length=100), nullable=True),
        sa.Column('cve_id', sa.String(length=20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('remediation', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKey('result_id'),
        sa.ForeignKey('scan_targets.target_id', ondelete='CASCADE'),
        sa.ForeignKey('scan_sessions.session_id', ondelete='CASCADE')
    )

    # Create threat indicators table for enhanced threat intelligence
    op.create_table(
        'threat_indicators',
        sa.Column('indicator_id', sa.String(length=128), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('ioc_type', sa.String(length=20), nullable=False),
        sa.Column('ioc_value', sa.String(length=2000), nullable=False),
        sa.Column('ioc_hash', sa.String(length=64), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('sources', postgresql.ARRAY(sa.String(length=100)), nullable=False),
        sa.Column('tags', postgresql.ARRAY(sa.String(length=50)), nullable=True),
        sa.Column('first_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKey('indicator_id')
    )

    # Create threat feeds table
    op.create_table(
        'threat_feeds',
        sa.Column('feed_id', sa.String(length=128), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('url', sa.String(length=1000), nullable=False),
        sa.Column('feed_type', sa.String(length=50), nullable=False),
        sa.Column('format_type', sa.String(length=20), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('update_interval', sa.Integer(), nullable=False, default=3600),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('indicators_count', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKey('feed_id')
    )

    # Create API keys table for authentication
    op.create_table(
        'api_keys',
        sa.Column('key_id', sa.String(length=128), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('user_id', sa.String(length=128), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('permissions', postgresql.ARRAY(sa.String(length=100)), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKey('key_id')
    )

    # High-performance indexes for scanning workloads

    # Scan sessions indexes
    op.create_index(
        'idx_scan_sessions_tenant_status',
        'scan_sessions',
        ['tenant_id', 'status'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_scan_sessions_created_at',
        'scan_sessions',
        ['created_at'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_scan_sessions_priority_status',
        'scan_sessions',
        ['priority', 'status'],
        postgresql_using='btree'
    )

    # Scan targets indexes
    op.create_index(
        'idx_scan_targets_session_status',
        'scan_targets',
        ['session_id', 'status'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_scan_targets_host_tenant',
        'scan_targets',
        ['host', 'tenant_id'],
        postgresql_using='btree'
    )

    # Scan results indexes for fast vulnerability queries
    op.create_index(
        'idx_scan_results_tenant_severity',
        'scan_results',
        ['tenant_id', 'severity', 'created_at'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_scan_results_cve_id',
        'scan_results',
        ['cve_id'],
        postgresql_using='btree',
        postgresql_where=sa.text('cve_id IS NOT NULL')
    )

    op.create_index(
        'idx_scan_results_scanner_tenant',
        'scan_results',
        ['scanner', 'tenant_id'],
        postgresql_using='btree'
    )

    # GIN indexes for JSONB columns
    op.create_index(
        'idx_scan_results_evidence_gin',
        'scan_results',
        ['evidence'],
        postgresql_using='gin'
    )

    op.create_index(
        'idx_scan_results_raw_data_gin',
        'scan_results',
        ['raw_data'],
        postgresql_using='gin'
    )

    # Threat indicators indexes
    op.create_index(
        'idx_threat_indicators_ioc_type_tenant',
        'threat_indicators',
        ['ioc_type', 'tenant_id'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_threat_indicators_ioc_hash',
        'threat_indicators',
        ['ioc_hash'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_threat_indicators_severity_confidence',
        'threat_indicators',
        ['severity', 'confidence'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_threat_indicators_last_seen',
        'threat_indicators',
        ['last_seen'],
        postgresql_using='btree'
    )

    # GIN index for tags array
    op.create_index(
        'idx_threat_indicators_tags_gin',
        'threat_indicators',
        ['tags'],
        postgresql_using='gin'
    )

    # API keys indexes
    op.create_index(
        'idx_api_keys_key_hash',
        'api_keys',
        ['key_hash'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_api_keys_user_tenant',
        'api_keys',
        ['user_id', 'tenant_id'],
        postgresql_using='btree'
    )

    op.create_index(
        'idx_api_keys_active_expires',
        'api_keys',
        ['is_active', 'expires_at'],
        postgresql_using='btree'
    )

    # Partitioning setup for large tables

    # Create partitioned tables for scan results by month
    op.execute("""
        -- Create monthly partitions for scan_results
        CREATE OR REPLACE FUNCTION create_scan_results_partition()
        RETURNS TRIGGER AS $$
        DECLARE
            partition_date TEXT;
            partition_name TEXT;
            start_date DATE;
            end_date DATE;
        BEGIN
            partition_date := to_char(NEW.created_at, 'YYYY_MM');
            partition_name := 'scan_results_' || partition_date;
            start_date := date_trunc('month', NEW.created_at);
            end_date := start_date + INTERVAL '1 month';

            -- Create partition if it doesn't exist
            PERFORM 1 FROM pg_class WHERE relname = partition_name;
            IF NOT FOUND THEN
                EXECUTE format('CREATE TABLE %I PARTITION OF scan_results
                               FOR VALUES FROM (%L) TO (%L)',
                               partition_name, start_date, end_date);

                -- Create indexes on the new partition
                EXECUTE format('CREATE INDEX %I ON %I (tenant_id, severity, created_at)',
                               'idx_' || partition_name || '_tenant_severity', partition_name);
                EXECUTE format('CREATE INDEX %I ON %I (scanner, tenant_id)',
                               'idx_' || partition_name || '_scanner_tenant', partition_name);
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Performance optimization functions

    # Function to clean old scan data
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_old_scan_data(days_to_keep INTEGER DEFAULT 90)
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            -- Delete old scan sessions and cascade to related data
            DELETE FROM scan_sessions
            WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;

            GET DIAGNOSTICS deleted_count = ROW_COUNT;

            -- Vacuum analyze to reclaim space
            PERFORM pg_advisory_lock(12345);
            EXECUTE 'VACUUM ANALYZE scan_sessions, scan_targets, scan_results';
            PERFORM pg_advisory_unlock(12345);

            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Function to get scan statistics
    op.execute("""
        CREATE OR REPLACE FUNCTION get_scan_statistics(tenant_uuid UUID, days_back INTEGER DEFAULT 30)
        RETURNS JSONB AS $$
        DECLARE
            stats JSONB;
        BEGIN
            SELECT jsonb_build_object(
                'total_scans', COUNT(*),
                'completed_scans', COUNT(*) FILTER (WHERE status = 'completed'),
                'failed_scans', COUNT(*) FILTER (WHERE status = 'failed'),
                'avg_duration_minutes',
                    COALESCE(AVG(EXTRACT(EPOCH FROM (completed_at - started_at))/60.0)
                    FILTER (WHERE status = 'completed'), 0),
                'total_vulnerabilities', (
                    SELECT COUNT(*)
                    FROM scan_results sr
                    WHERE sr.tenant_id = tenant_uuid
                    AND sr.created_at > NOW() - INTERVAL '1 day' * days_back
                ),
                'critical_vulnerabilities', (
                    SELECT COUNT(*)
                    FROM scan_results sr
                    WHERE sr.tenant_id = tenant_uuid
                    AND sr.severity = 'critical'
                    AND sr.created_at > NOW() - INTERVAL '1 day' * days_back
                ),
                'high_vulnerabilities', (
                    SELECT COUNT(*)
                    FROM scan_results sr
                    WHERE sr.tenant_id = tenant_uuid
                    AND sr.severity = 'high'
                    AND sr.created_at > NOW() - INTERVAL '1 day' * days_back
                )
            ) INTO stats
            FROM scan_sessions ss
            WHERE ss.tenant_id = tenant_uuid
            AND ss.created_at > NOW() - INTERVAL '1 day' * days_back;

            RETURN COALESCE(stats, '{}'::jsonb);
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Materialized view for dashboard statistics
    op.execute("""
        CREATE MATERIALIZED VIEW scan_dashboard_stats AS
        SELECT
            tenant_id,
            DATE(created_at) as scan_date,
            COUNT(*) as total_scans,
            COUNT(*) FILTER (WHERE status = 'completed') as completed_scans,
            COUNT(*) FILTER (WHERE status = 'failed') as failed_scans,
            AVG(EXTRACT(EPOCH FROM (completed_at - started_at))/60.0)
                FILTER (WHERE status = 'completed') as avg_duration_minutes,
            MAX(created_at) as last_updated
        FROM scan_sessions
        GROUP BY tenant_id, DATE(created_at);

        CREATE UNIQUE INDEX idx_scan_dashboard_stats_unique
        ON scan_dashboard_stats (tenant_id, scan_date);

        CREATE INDEX idx_scan_dashboard_stats_date
        ON scan_dashboard_stats (scan_date);
    """)

    # Function to refresh dashboard stats
    op.execute("""
        CREATE OR REPLACE FUNCTION refresh_scan_dashboard_stats()
        RETURNS VOID AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY scan_dashboard_stats;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Row-level security for multi-tenancy

    # Enable RLS on new tables
    op.execute("ALTER TABLE scan_sessions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE scan_targets ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE scan_results ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE threat_indicators ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY")

    # Create RLS policies
    op.execute("""
        CREATE POLICY tenant_isolation_scan_sessions ON scan_sessions
        USING (tenant_id = current_setting('app.tenant_id')::uuid)
    """)

    op.execute("""
        CREATE POLICY tenant_isolation_scan_targets ON scan_targets
        USING (tenant_id = current_setting('app.tenant_id')::uuid)
    """)

    op.execute("""
        CREATE POLICY tenant_isolation_scan_results ON scan_results
        USING (tenant_id = current_setting('app.tenant_id')::uuid)
    """)

    op.execute("""
        CREATE POLICY tenant_isolation_threat_indicators ON threat_indicators
        USING (tenant_id = current_setting('app.tenant_id')::uuid)
    """)

    op.execute("""
        CREATE POLICY tenant_isolation_api_keys ON api_keys
        USING (tenant_id = current_setting('app.tenant_id')::uuid)
    """)

def downgrade():
    """Remove performance optimizations"""

    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS scan_dashboard_stats")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS cleanup_old_scan_data(INTEGER)")
    op.execute("DROP FUNCTION IF EXISTS get_scan_statistics(UUID, INTEGER)")
    op.execute("DROP FUNCTION IF EXISTS refresh_scan_dashboard_stats()")
    op.execute("DROP FUNCTION IF EXISTS create_scan_results_partition()")

    # Drop tables (will cascade and remove indexes)
    op.drop_table('api_keys')
    op.drop_table('threat_feeds')
    op.drop_table('threat_indicators')
    op.drop_table('scan_results')
    op.drop_table('scan_targets')
    op.drop_table('scan_sessions')
