"""Create RLS policies for tenant isolation

Revision ID: 002_tenant_rls
Revises: 001_tenant_isolation
Create Date: 2024-08-09 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '002_tenant_rls'
down_revision = '001_tenant_isolation'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Enable RLS and create tenant isolation policies."""
    
    # Enable RLS on tenant-scoped tables
    op.execute("ALTER TABLE evidence ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE findings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE embedding_vectors ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tenant_users ENABLE ROW LEVEL SECURITY")
    
    # Create RLS policies for evidence table
    op.execute("""
        CREATE POLICY evidence_tenant_isolation ON evidence
        USING (tenant_id::text = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true))
    """)
    
    # Create RLS policies for findings table
    op.execute("""
        CREATE POLICY findings_tenant_isolation ON findings
        USING (tenant_id::text = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true))
    """)
    
    # Create RLS policies for embedding_vectors table
    op.execute("""
        CREATE POLICY embedding_vectors_tenant_isolation ON embedding_vectors
        USING (tenant_id::text = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true))
    """)
    
    # Create RLS policies for tenant_users table
    op.execute("""
        CREATE POLICY tenant_users_tenant_isolation ON tenant_users
        USING (tenant_id::text = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true))
    """)
    
    # Create a function to set tenant context (for testing and admin operations)
    op.execute("""
        CREATE OR REPLACE FUNCTION set_tenant_context(tenant_uuid UUID)
        RETURNS void AS $$
        BEGIN
            PERFORM set_config('app.tenant_id', tenant_uuid::text, false);
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Create a function to get current tenant context
    op.execute("""
        CREATE OR REPLACE FUNCTION get_current_tenant_id()
        RETURNS UUID AS $$
        DECLARE
            tenant_id_text text;
        BEGIN
            tenant_id_text := current_setting('app.tenant_id', true);
            IF tenant_id_text IS NULL OR tenant_id_text = '' THEN
                RETURN NULL;
            END IF;
            RETURN tenant_id_text::UUID;
        EXCEPTION
            WHEN OTHERS THEN
                RETURN NULL;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Create function to bypass RLS for super admin operations
    op.execute("""
        CREATE OR REPLACE FUNCTION bypass_rls_for_user(user_role text)
        RETURNS boolean AS $$
        BEGIN
            -- Allow super_admin role to bypass RLS
            RETURN user_role = 'super_admin';
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)
    
    # Create policies that allow super admin to bypass RLS
    op.execute("""
        CREATE POLICY evidence_super_admin ON evidence
        USING (bypass_rls_for_user(current_setting('app.user_role', true)))
        WITH CHECK (bypass_rls_for_user(current_setting('app.user_role', true)))
    """)
    
    op.execute("""
        CREATE POLICY findings_super_admin ON findings
        USING (bypass_rls_for_user(current_setting('app.user_role', true)))
        WITH CHECK (bypass_rls_for_user(current_setting('app.user_role', true)))
    """)
    
    op.execute("""
        CREATE POLICY embedding_vectors_super_admin ON embedding_vectors
        USING (bypass_rls_for_user(current_setting('app.user_role', true)))
        WITH CHECK (bypass_rls_for_user(current_setting('app.user_role', true)))
    """)
    
    op.execute("""
        CREATE POLICY tenant_users_super_admin ON tenant_users
        USING (bypass_rls_for_user(current_setting('app.user_role', true)))
        WITH CHECK (bypass_rls_for_user(current_setting('app.user_role', true)))
    """)


def downgrade() -> None:
    """Remove RLS policies and disable RLS."""
    
    # Drop super admin policies
    op.execute("DROP POLICY IF EXISTS evidence_super_admin ON evidence")
    op.execute("DROP POLICY IF EXISTS findings_super_admin ON findings")
    op.execute("DROP POLICY IF EXISTS embedding_vectors_super_admin ON embedding_vectors")
    op.execute("DROP POLICY IF EXISTS tenant_users_super_admin ON tenant_users")
    
    # Drop tenant isolation policies
    op.execute("DROP POLICY IF EXISTS evidence_tenant_isolation ON evidence")
    op.execute("DROP POLICY IF EXISTS findings_tenant_isolation ON findings")
    op.execute("DROP POLICY IF EXISTS embedding_vectors_tenant_isolation ON embedding_vectors")
    op.execute("DROP POLICY IF EXISTS tenant_users_tenant_isolation ON tenant_users")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS bypass_rls_for_user(text)")
    op.execute("DROP FUNCTION IF EXISTS get_current_tenant_id()")
    op.execute("DROP FUNCTION IF EXISTS set_tenant_context(UUID)")
    
    # Disable RLS
    op.execute("ALTER TABLE tenant_users DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE embedding_vectors DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE findings DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE evidence DISABLE ROW LEVEL SECURITY")