"""Add RBAC system tables

Revision ID: 006_rbac_system
Revises: 005_enhanced_production_schema
Create Date: 2025-01-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006_rbac_system'
down_revision = '005_enhanced_production_schema'
branch_labels = None
depends_on = None


def upgrade():
    """Create RBAC system tables"""
    
    # Create roles table
    op.create_table('rbac_roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('display_name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_system_role', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('level', sa.Integer(), nullable=False, default=0),  # For hierarchy
        sa.Column('parent_role_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['parent_role_id'], ['rbac_roles.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create permissions table
    op.create_table('rbac_permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('display_name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('resource', sa.String(length=100), nullable=False),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('is_system_permission', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.text('NOW()')),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.UniqueConstraint('resource', 'action')
    )
    
    # Create role_permissions junction table
    op.create_table('rbac_role_permissions',
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('permission_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('granted_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('granted_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(['role_id'], ['rbac_roles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['permission_id'], ['rbac_permissions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('role_id', 'permission_id')
    )
    
    # Create user_roles junction table
    op.create_table('rbac_user_roles',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True),  # For tenant-specific roles
        sa.Column('granted_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('granted_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['role_id'], ['rbac_roles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tenant_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('user_id', 'role_id', 'tenant_id')
    )
    
    # Create user_permissions table for direct permission assignment
    op.create_table('rbac_user_permissions',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('permission_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('granted_at', sa.DateTime(), nullable=False, default=sa.text('NOW()')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('granted_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['permission_id'], ['rbac_permissions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tenant_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('user_id', 'permission_id', 'tenant_id')
    )
    
    # Create indexes
    op.create_index('idx_rbac_roles_name_active', 'rbac_roles', ['name', 'is_active'])
    op.create_index('idx_rbac_roles_level', 'rbac_roles', ['level'])
    op.create_index('idx_rbac_roles_parent', 'rbac_roles', ['parent_role_id'])
    
    op.create_index('idx_rbac_permissions_resource_action', 'rbac_permissions', ['resource', 'action'])
    op.create_index('idx_rbac_permissions_name_active', 'rbac_permissions', ['name', 'is_active'])
    
    op.create_index('idx_rbac_role_permissions_role', 'rbac_role_permissions', ['role_id'])
    op.create_index('idx_rbac_role_permissions_permission', 'rbac_role_permissions', ['permission_id'])
    
    op.create_index('idx_rbac_user_roles_user_active', 'rbac_user_roles', ['user_id', 'is_active'])
    op.create_index('idx_rbac_user_roles_role_active', 'rbac_user_roles', ['role_id', 'is_active'])
    op.create_index('idx_rbac_user_roles_tenant', 'rbac_user_roles', ['tenant_id'])
    op.create_index('idx_rbac_user_roles_expires', 'rbac_user_roles', ['expires_at'])
    
    op.create_index('idx_rbac_user_permissions_user_active', 'rbac_user_permissions', ['user_id', 'is_active'])
    op.create_index('idx_rbac_user_permissions_permission_active', 'rbac_user_permissions', ['permission_id', 'is_active'])
    op.create_index('idx_rbac_user_permissions_tenant', 'rbac_user_permissions', ['tenant_id'])
    op.create_index('idx_rbac_user_permissions_expires', 'rbac_user_permissions', ['expires_at'])
    
    # Insert system roles
    op.execute("""
        INSERT INTO rbac_roles (name, display_name, description, is_system_role, level, created_at) VALUES
        ('super_admin', 'Super Administrator', 'Full system access across all tenants', true, 100, NOW()),
        ('tenant_admin', 'Tenant Administrator', 'Full access within tenant', true, 90, NOW()),
        ('security_manager', 'Security Manager', 'Security operations and user management', true, 80, NOW()),
        ('security_analyst', 'Security Analyst', 'Security analysis and scanning operations', true, 70, NOW()),
        ('pentester', 'Penetration Tester', 'Penetration testing and vulnerability assessment', true, 60, NOW()),
        ('compliance_officer', 'Compliance Officer', 'Compliance monitoring and reporting', true, 60, NOW()),
        ('auditor', 'Auditor', 'Read-only audit and compliance access', true, 50, NOW()),
        ('viewer', 'Viewer', 'Read-only access to basic resources', true, 30, NOW()),
        ('user', 'User', 'Basic authenticated user access', true, 20, NOW())
    """)
    
    # Insert system permissions
    op.execute("""
        INSERT INTO rbac_permissions (name, display_name, description, resource, action, is_system_permission, created_at) VALUES
        -- User management
        ('user:create', 'Create Users', 'Create new user accounts', 'user', 'create', true, NOW()),
        ('user:read', 'Read Users', 'View user accounts and profiles', 'user', 'read', true, NOW()),
        ('user:update', 'Update Users', 'Modify user accounts and profiles', 'user', 'update', true, NOW()),
        ('user:delete', 'Delete Users', 'Delete user accounts', 'user', 'delete', true, NOW()),
        ('user:manage_roles', 'Manage User Roles', 'Assign and revoke user roles', 'user', 'manage_roles', true, NOW()),
        
        -- Organization/Tenant management
        ('organization:create', 'Create Organizations', 'Create new organizations/tenants', 'organization', 'create', true, NOW()),
        ('organization:read', 'Read Organizations', 'View organization details', 'organization', 'read', true, NOW()),
        ('organization:update', 'Update Organizations', 'Modify organization settings', 'organization', 'update', true, NOW()),
        ('organization:delete', 'Delete Organizations', 'Delete organizations', 'organization', 'delete', true, NOW()),
        
        -- PTaaS operations
        ('ptaas:scan:create', 'Create Scans', 'Initiate security scans', 'ptaas', 'scan:create', true, NOW()),
        ('ptaas:scan:read', 'Read Scans', 'View scan results and reports', 'ptaas', 'scan:read', true, NOW()),
        ('ptaas:scan:update', 'Update Scans', 'Modify scan configurations', 'ptaas', 'scan:update', true, NOW()),
        ('ptaas:scan:delete', 'Delete Scans', 'Delete scan sessions and results', 'ptaas', 'scan:delete', true, NOW()),
        ('ptaas:scan:cancel', 'Cancel Scans', 'Cancel running scans', 'ptaas', 'scan:cancel', true, NOW()),
        ('ptaas:workflow:manage', 'Manage Workflows', 'Create and manage PTaaS workflows', 'ptaas', 'workflow:manage', true, NOW()),
        
        -- Intelligence operations
        ('intelligence:read', 'Read Intelligence', 'Access threat intelligence data', 'intelligence', 'read', true, NOW()),
        ('intelligence:analyze', 'Analyze Intelligence', 'Perform intelligence analysis', 'intelligence', 'analyze', true, NOW()),
        ('intelligence:manage', 'Manage Intelligence', 'Manage intelligence sources and feeds', 'intelligence', 'manage', true, NOW()),
        
        -- Agent management
        ('agent:read', 'Read Agents', 'View agent status and configurations', 'agent', 'read', true, NOW()),
        ('agent:create', 'Create Agents', 'Deploy new autonomous agents', 'agent', 'create', true, NOW()),
        ('agent:update', 'Update Agents', 'Modify agent configurations', 'agent', 'update', true, NOW()),
        ('agent:delete', 'Delete Agents', 'Remove agents from system', 'agent', 'delete', true, NOW()),
        ('agent:control', 'Control Agents', 'Start, stop, and control agents', 'agent', 'control', true, NOW()),
        
        -- System administration
        ('system:admin', 'System Administration', 'Full system administration access', 'system', 'admin', true, NOW()),
        ('system:monitor', 'System Monitoring', 'Monitor system health and performance', 'system', 'monitor', true, NOW()),
        ('system:config', 'System Configuration', 'Modify system configurations', 'system', 'config', true, NOW()),
        
        -- Audit and compliance
        ('audit:read', 'Read Audit Logs', 'Access audit logs and security events', 'audit', 'read', true, NOW()),
        ('audit:export', 'Export Audit Data', 'Export audit logs and reports', 'audit', 'export', true, NOW()),
        ('compliance:read', 'Read Compliance', 'View compliance status and reports', 'compliance', 'read', true, NOW()),
        ('compliance:manage', 'Manage Compliance', 'Configure compliance frameworks', 'compliance', 'manage', true, NOW()),
        
        -- Evidence and storage
        ('evidence:read', 'Read Evidence', 'Access evidence and findings', 'evidence', 'read', true, NOW()),
        ('evidence:write', 'Write Evidence', 'Create and modify evidence', 'evidence', 'write', true, NOW()),
        ('evidence:delete', 'Delete Evidence', 'Delete evidence and findings', 'evidence', 'delete', true, NOW()),
        
        -- Jobs and orchestration
        ('job:read', 'Read Jobs', 'View job status and results', 'job', 'read', true, NOW()),
        ('job:create', 'Create Jobs', 'Submit new jobs', 'job', 'create', true, NOW()),
        ('job:cancel', 'Cancel Jobs', 'Cancel running jobs', 'job', 'cancel', true, NOW()),
        ('job:priority', 'Manage Job Priority', 'Set job priority levels', 'job', 'priority', true, NOW()),
        
        -- Telemetry and metrics
        ('telemetry:read', 'Read Telemetry', 'Access telemetry and metrics data', 'telemetry', 'read', true, NOW()),
        ('telemetry:write', 'Write Telemetry', 'Submit telemetry data', 'telemetry', 'write', true, NOW())
    """)
    
    # Assign permissions to roles
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'super_admin'  -- Super admin gets all permissions
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'tenant_admin' AND p.name IN (
            'user:read', 'user:update', 'user:manage_roles',
            'organization:read', 'organization:update',
            'ptaas:scan:create', 'ptaas:scan:read', 'ptaas:scan:update', 'ptaas:scan:delete', 'ptaas:scan:cancel', 'ptaas:workflow:manage',
            'intelligence:read', 'intelligence:analyze', 'intelligence:manage',
            'agent:read', 'agent:create', 'agent:update', 'agent:delete', 'agent:control',
            'system:monitor', 'audit:read', 'audit:export', 'compliance:read', 'compliance:manage',
            'evidence:read', 'evidence:write', 'evidence:delete',
            'job:read', 'job:create', 'job:cancel', 'job:priority',
            'telemetry:read', 'telemetry:write'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'security_manager' AND p.name IN (
            'user:read', 'user:update', 'user:manage_roles',
            'organization:read',
            'ptaas:scan:create', 'ptaas:scan:read', 'ptaas:scan:update', 'ptaas:scan:cancel', 'ptaas:workflow:manage',
            'intelligence:read', 'intelligence:analyze', 'intelligence:manage',
            'agent:read', 'agent:create', 'agent:update', 'agent:control',
            'system:monitor', 'audit:read', 'compliance:read',
            'evidence:read', 'evidence:write',
            'job:read', 'job:create', 'job:cancel',
            'telemetry:read', 'telemetry:write'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'security_analyst' AND p.name IN (
            'user:read', 'organization:read',
            'ptaas:scan:create', 'ptaas:scan:read', 'ptaas:scan:update',
            'intelligence:read', 'intelligence:analyze',
            'agent:read', 'agent:update',
            'system:monitor', 'audit:read', 'compliance:read',
            'evidence:read', 'evidence:write',
            'job:read', 'job:create',
            'telemetry:read', 'telemetry:write'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'pentester' AND p.name IN (
            'user:read', 'organization:read',
            'ptaas:scan:create', 'ptaas:scan:read', 'ptaas:scan:update',
            'intelligence:read',
            'agent:read',
            'audit:read', 'compliance:read',
            'evidence:read', 'evidence:write',
            'job:read', 'job:create',
            'telemetry:read'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'compliance_officer' AND p.name IN (
            'user:read', 'organization:read',
            'ptaas:scan:read',
            'intelligence:read',
            'audit:read', 'audit:export', 'compliance:read', 'compliance:manage',
            'evidence:read',
            'job:read',
            'telemetry:read'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'auditor' AND p.name IN (
            'user:read', 'organization:read',
            'ptaas:scan:read',
            'intelligence:read',
            'audit:read', 'audit:export', 'compliance:read',
            'evidence:read',
            'job:read',
            'telemetry:read'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'viewer' AND p.name IN (
            'user:read', 'organization:read',
            'ptaas:scan:read',
            'intelligence:read',
            'audit:read', 'compliance:read',
            'evidence:read',
            'job:read',
            'telemetry:read'
        )
    """)
    
    op.execute("""
        INSERT INTO rbac_role_permissions (role_id, permission_id, granted_at)
        SELECT r.id, p.id, NOW()
        FROM rbac_roles r, rbac_permissions p
        WHERE r.name = 'user' AND p.name IN (
            'user:read', 'organization:read',
            'ptaas:scan:read',
            'evidence:read',
            'job:read',
            'telemetry:read'
        )
    """)
    
    # Assign super_admin role to existing admin user
    op.execute("""
        INSERT INTO rbac_user_roles (user_id, role_id, granted_at)
        SELECT u.id, r.id, NOW()
        FROM users u, rbac_roles r
        WHERE u.username = 'admin' AND r.name = 'super_admin'
    """)


def downgrade():
    """Drop RBAC system tables"""
    op.drop_table('rbac_user_permissions')
    op.drop_table('rbac_user_roles')
    op.drop_table('rbac_role_permissions')
    op.drop_table('rbac_permissions')
    op.drop_table('rbac_roles')