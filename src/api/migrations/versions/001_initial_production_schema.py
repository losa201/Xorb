"""Initial production schema with all XORB entities

Revision ID: 001_initial_production
Revises:
Create Date: 2025-01-10 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_production'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial production database schema"""

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('username', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('roles', postgresql.ARRAY(sa.String()), nullable=False, default=sa.text("'{}'::character varying[]")),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('auth_provider', sa.String(length=50), nullable=True, default='local'),
        sa.Column('external_id', sa.String(length=255), nullable=True),
        sa.Column('first_name', sa.String(length=100), nullable=True),
        sa.Column('last_name', sa.String(length=100), nullable=True),
        sa.Column('phone', sa.String(length=20), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True, default='UTC'),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=True, default=0),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('mfa_enabled', sa.Boolean(), nullable=True, default=False),
        sa.Column('mfa_secret', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )

    # Create indexes for users
    op.create_index('idx_users_username_active', 'users', ['username', 'is_active'])
    op.create_index('idx_users_email_active', 'users', ['email', 'is_active'])
    op.create_index('idx_users_auth_provider', 'users', ['auth_provider'])

    # Create organizations table
    op.create_table('organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=True),
        sa.Column('plan_type', sa.String(length=50), nullable=False, default='free'),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('subscription_id', sa.String(length=255), nullable=True),
        sa.Column('billing_email', sa.String(length=255), nullable=True),
        sa.Column('trial_ends_at', sa.DateTime(), nullable=True),
        sa.Column('settings', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('features', postgresql.ARRAY(sa.String()), nullable=True, default=sa.text("'{}'::character varying[]")),
        sa.Column('api_quota_monthly', sa.Integer(), nullable=True, default=1000),
        sa.Column('api_usage_current', sa.Integer(), nullable=True, default=0),
        sa.Column('user_limit', sa.Integer(), nullable=True, default=5),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.UniqueConstraint('slug')
    )

    # Create indexes for organizations
    op.create_index('idx_organizations_name_active', 'organizations', ['name', 'is_active'])
    op.create_index('idx_organizations_plan_active', 'organizations', ['plan_type', 'is_active'])

    # Create user_organizations junction table
    op.create_table('user_organizations',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False, default='member'),
        sa.Column('permissions', postgresql.ARRAY(sa.String()), nullable=True, default=sa.text("'{}'::character varying[]")),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('joined_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.Column('invited_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('user_id', 'organization_id')
    )

    # Create auth_tokens table
    op.create_table('auth_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token', sa.String(length=255), nullable=True),
        sa.Column('token_hash', sa.String(length=255), nullable=True),
        sa.Column('token_type', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, default=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoked_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('scopes', postgresql.ARRAY(sa.String()), nullable=True, default=sa.text("'{}'::character varying[]")),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token'),
        sa.UniqueConstraint('token_hash')
    )

    # Create indexes for auth_tokens
    op.create_index('idx_auth_tokens_user_active', 'auth_tokens', ['user_id', 'is_revoked'])
    op.create_index('idx_auth_tokens_type_active', 'auth_tokens', ['token_type', 'is_revoked'])
    op.create_index('idx_auth_tokens_expires', 'auth_tokens', ['expires_at'])

    # Create embedding_requests table
    op.create_table('embedding_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('texts', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('input_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('processing_time', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('batch_id', sa.String(length=255), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True, default=5),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for embedding_requests
    op.create_index('idx_embedding_requests_user_status', 'embedding_requests', ['user_id', 'status'])
    op.create_index('idx_embedding_requests_org_created', 'embedding_requests', ['org_id', 'created_at'])
    op.create_index('idx_embedding_requests_status_priority', 'embedding_requests', ['status', 'priority'])

    # Create embedding_results table
    op.create_table('embedding_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('embeddings', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('model_used', sa.String(length=100), nullable=False),
        sa.Column('processing_time', sa.Integer(), nullable=False),
        sa.Column('similarity_scores', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('confidence_scores', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('vector_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('cache_key', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['request_id'], ['embedding_requests.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('request_id')
    )

    # Create indexes for embedding_results
    op.create_index('idx_embedding_results_request', 'embedding_results', ['request_id'])
    op.create_index('idx_embedding_results_cache', 'embedding_results', ['cache_key'])
    op.create_index('idx_embedding_results_expires', 'embedding_results', ['expires_at'])

    # Create discovery_workflows table
    op.create_table('discovery_workflows',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('workflow_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=False),
        sa.Column('workflow_type', sa.String(length=50), nullable=True, default='discovery'),
        sa.Column('status', sa.String(length=50), nullable=False, default='pending'),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('parameters', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('results', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('progress', sa.Integer(), nullable=True, default=0),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('workflow_id')
    )

    # Create indexes for discovery_workflows
    op.create_index('idx_discovery_workflows_user_status', 'discovery_workflows', ['user_id', 'status'])
    op.create_index('idx_discovery_workflows_domain', 'discovery_workflows', ['domain'])
    op.create_index('idx_discovery_workflows_type_status', 'discovery_workflows', ['workflow_type', 'status'])

    # Create ptaas_scan_sessions table
    op.create_table('ptaas_scan_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('targets', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('scan_type', sa.String(length=50), nullable=False),
        sa.Column('scan_profile', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, default='pending'),
        sa.Column('progress', sa.Integer(), nullable=True, default=0),
        sa.Column('results', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('vulnerabilities', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('compliance_results', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('estimated_duration', sa.Integer(), nullable=True),
        sa.Column('actual_duration', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True, default=sa.text("'{}'::character varying[]")),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )

    # Create indexes for ptaas_scan_sessions
    op.create_index('idx_ptaas_sessions_user_status', 'ptaas_scan_sessions', ['user_id', 'status'])
    op.create_index('idx_ptaas_sessions_org_created', 'ptaas_scan_sessions', ['organization_id', 'created_at'])
    op.create_index('idx_ptaas_sessions_type_status', 'ptaas_scan_sessions', ['scan_type', 'status'])
    op.create_index('idx_ptaas_sessions_session_id', 'ptaas_scan_sessions', ['session_id'])

    # Create security_events table
    op.create_table('security_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False, default='info'),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for security_events
    op.create_index('idx_security_events_type_created', 'security_events', ['event_type', 'created_at'])
    op.create_index('idx_security_events_user_created', 'security_events', ['user_id', 'created_at'])
    op.create_index('idx_security_events_severity_created', 'security_events', ['severity', 'created_at'])
    op.create_index('idx_security_events_ip_created', 'security_events', ['ip_address', 'created_at'])

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('value', sa.String(length=255), nullable=False),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True, default=sa.text("'{}'::json")),
        sa.Column('service_name', sa.String(length=100), nullable=True),
        sa.Column('instance_id', sa.String(length=100), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for system_metrics
    op.create_index('idx_system_metrics_name_timestamp', 'system_metrics', ['metric_name', 'timestamp'])
    op.create_index('idx_system_metrics_service_timestamp', 'system_metrics', ['service_name', 'timestamp'])
    op.create_index('idx_system_metrics_type_timestamp', 'system_metrics', ['metric_type', 'timestamp'])

    # Insert default data

    # Create default super admin user
    op.execute("""
        INSERT INTO users (id, username, email, password_hash, roles, is_active, created_at)
        VALUES (
            gen_random_uuid(),
            'admin',
            'admin@xorb.com',
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVMstdMWT0StZvF6',  -- password: admin123
            ARRAY['super_admin', 'admin', 'user'],
            true,
            NOW()
        )
    """)

    # Create default organization
    op.execute("""
        INSERT INTO organizations (id, name, plan_type, is_active, created_at)
        VALUES (
            gen_random_uuid(),
            'XORB Default Organization',
            'enterprise',
            true,
            NOW()
        )
    """)

    # Associate admin user with default organization
    op.execute("""
        INSERT INTO user_organizations (user_id, organization_id, role, joined_at)
        SELECT u.id, o.id, 'admin', NOW()
        FROM users u, organizations o
        WHERE u.username = 'admin' AND o.name = 'XORB Default Organization'
    """)


def downgrade():
    """Drop all tables"""
    op.drop_table('system_metrics')
    op.drop_table('security_events')
    op.drop_table('ptaas_scan_sessions')
    op.drop_table('discovery_workflows')
    op.drop_table('embedding_results')
    op.drop_table('embedding_requests')
    op.drop_table('auth_tokens')
    op.drop_table('user_organizations')
    op.drop_table('organizations')
    op.drop_table('users')
