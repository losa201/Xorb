"""Initial multi-tenant schema with pgvector support

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-08-09 13:30:00.000000

This migration creates the foundational multi-tenant architecture
for the XORB cybersecurity platform with:

1. Tenant isolation using Row Level Security (RLS)
2. pgvector extension for threat intelligence embeddings
3. Audit logging and compliance tracking
4. Performance-optimized indexes
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial multi-tenant schema"""
    
    # Enable pgvector extension for embeddings
    op.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    op.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    op.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
    
    # Create custom types
    tenant_status_enum = postgresql.ENUM(
        'ACTIVE', 'SUSPENDED', 'DEPROVISIONING', 'ARCHIVED', 
        name='tenant_status'
    )
    tenant_plan_enum = postgresql.ENUM(
        'STARTER', 'PROFESSIONAL', 'ENTERPRISE', 
        name='tenant_plan'
    )
    
    tenant_status_enum.create(op.get_bind())
    tenant_plan_enum.create(op.get_bind())
    
    # Core tenants table (foundation of multi-tenancy)
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('status', tenant_status_enum, nullable=False, default='ACTIVE'),
        sa.Column('plan', tenant_plan_enum, nullable=False, default='PROFESSIONAL'),
        sa.Column('settings', sa.JSON, nullable=False, default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create indexes for tenant operations
    op.create_index('idx_tenants_slug', 'tenants', ['slug'])
    op.create_index('idx_tenants_status', 'tenants', ['status'])
    
    # Users table with tenant association
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(320), nullable=False),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('is_superuser', sa.Boolean, nullable=False, default=False),
        sa.Column('roles', sa.JSON, nullable=False, default='["user"]'),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Unique constraint on email per tenant
    op.create_index('idx_users_tenant_email', 'users', ['tenant_id', 'email'], unique=True)
    op.create_index('idx_users_tenant_username', 'users', ['tenant_id', 'username'], unique=True)
    
    # Organizations table for hierarchical tenancy
    op.create_table(
        'organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('plan_type', tenant_plan_enum, nullable=False, default='PROFESSIONAL'),
        sa.Column('settings', sa.JSON, nullable=False, default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Threat Intelligence tables
    op.create_table(
        'threat_feeds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('feed_name', sa.String(200), nullable=False),
        sa.Column('feed_type', sa.String(50), nullable=False),
        sa.Column('source_url', sa.String(500), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('config', sa.JSON, nullable=False, default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Threat indicators with vector embeddings
    op.create_table(
        'threat_indicators',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('indicator_type', sa.String(50), nullable=False),
        sa.Column('indicator_value', sa.Text, nullable=False),
        sa.Column('confidence_score', sa.Float, nullable=False, default=0.5),
        sa.Column('threat_level', sa.String(20), nullable=False, default='UNKNOWN'),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('tags', sa.JSON, nullable=False, default='[]'),
        sa.Column('context', sa.JSON, nullable=False, default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for threat intelligence queries
    op.create_index('idx_threat_indicators_tenant_type', 'threat_indicators', 
                   ['tenant_id', 'indicator_type'])
    op.create_index('idx_threat_indicators_threat_level', 'threat_indicators', 
                   ['threat_level'])
    op.create_index('idx_threat_indicators_last_seen', 'threat_indicators', 
                   ['last_seen'])
    
    # Vector embeddings table for AI-powered threat correlation
    op.create_table(
        'embedding_vectors',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_type', sa.String(50), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=False),
        sa.Column('vector_metadata', sa.JSON, nullable=False, default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Note: vector column will be added after pgvector is confirmed
    # op.add_column('embedding_vectors', sa.Column('embedding', vector(1536)))
    
    # Incidents and workflow tracking
    op.create_table(
        'security_incidents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('severity', sa.String(20), nullable=False, default='MEDIUM'),
        sa.Column('status', sa.String(20), nullable=False, default='OPEN'),
        sa.Column('assigned_to', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('indicators', sa.JSON, nullable=False, default='[]'),
        sa.Column('timeline', sa.JSON, nullable=False, default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assigned_to'], ['users.id'], ondelete='SET NULL'),
    )
    
    # Audit log table for compliance
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=text('gen_random_uuid()')),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('details', sa.JSON, nullable=False, default='{}'),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=text('NOW()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    )
    
    # Indexes for audit log queries
    op.create_index('idx_audit_logs_tenant_timestamp', 'audit_logs', 
                   ['tenant_id', 'timestamp'])
    op.create_index('idx_audit_logs_user_timestamp', 'audit_logs', 
                   ['user_id', 'timestamp'])
    op.create_index('idx_audit_logs_action', 'audit_logs', ['action'])
    
    # Create Row Level Security (RLS) policies
    enable_rls_sql = """
    -- Enable RLS on all tenant-aware tables
    ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
    ALTER TABLE users ENABLE ROW LEVEL SECURITY;
    ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
    ALTER TABLE threat_feeds ENABLE ROW LEVEL SECURITY;
    ALTER TABLE threat_indicators ENABLE ROW LEVEL SECURITY;
    ALTER TABLE embedding_vectors ENABLE ROW LEVEL SECURITY;
    ALTER TABLE security_incidents ENABLE ROW LEVEL SECURITY;
    ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
    
    -- Create policies for tenant isolation
    -- Users can only see their own tenant's data
    CREATE POLICY tenant_isolation_users ON users
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    CREATE POLICY tenant_isolation_organizations ON organizations
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    CREATE POLICY tenant_isolation_threat_feeds ON threat_feeds
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    CREATE POLICY tenant_isolation_threat_indicators ON threat_indicators
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    CREATE POLICY tenant_isolation_embedding_vectors ON embedding_vectors
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    CREATE POLICY tenant_isolation_security_incidents ON security_incidents
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    CREATE POLICY tenant_isolation_audit_logs ON audit_logs
        FOR ALL TO authenticated_user
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
    
    -- Tenants table policy - users can only see their own tenant
    CREATE POLICY tenant_self_access ON tenants
        FOR ALL TO authenticated_user
        USING (id = current_setting('app.current_tenant_id')::uuid);
    """
    
    op.execute(text(enable_rls_sql))
    
    # Create function to set tenant context
    tenant_context_function = """
    CREATE OR REPLACE FUNCTION set_tenant_context(tenant_uuid uuid)
    RETURNS void AS $$
    BEGIN
        PERFORM set_config('app.current_tenant_id', tenant_uuid::text, false);
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;
    
    -- Grant execute permission to authenticated users
    GRANT EXECUTE ON FUNCTION set_tenant_context(uuid) TO authenticated_user;
    """
    
    op.execute(text(tenant_context_function))


def downgrade() -> None:
    """Remove multi-tenant schema"""
    
    # Drop RLS policies
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_users ON users"))
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_organizations ON organizations"))
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_threat_feeds ON threat_feeds"))
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_threat_indicators ON threat_indicators"))
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_embedding_vectors ON embedding_vectors"))
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_security_incidents ON security_incidents"))
    op.execute(text("DROP POLICY IF EXISTS tenant_isolation_audit_logs ON audit_logs"))
    op.execute(text("DROP POLICY IF EXISTS tenant_self_access ON tenants"))
    
    # Drop function
    op.execute(text("DROP FUNCTION IF EXISTS set_tenant_context(uuid)"))
    
    # Drop tables in reverse order
    op.drop_table('audit_logs')
    op.drop_table('security_incidents')
    op.drop_table('embedding_vectors')
    op.drop_table('threat_indicators')
    op.drop_table('threat_feeds')
    op.drop_table('organizations')
    op.drop_table('users')
    op.drop_table('tenants')
    
    # Drop custom types
    op.execute(text("DROP TYPE IF EXISTS tenant_plan"))
    op.execute(text("DROP TYPE IF EXISTS tenant_status"))
    
    # Note: We don't drop extensions as they might be used by other applications
    # op.execute(text("DROP EXTENSION IF EXISTS vector"))
    # op.execute(text("DROP EXTENSION IF EXISTS pgcrypto"))
    # op.execute(text("DROP EXTENSION IF EXISTS \"uuid-ossp\""))