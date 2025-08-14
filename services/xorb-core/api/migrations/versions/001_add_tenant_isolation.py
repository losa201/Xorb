"""Add tenant isolation with RLS policies

Revision ID: 001_tenant_isolation
Revises:
Create Date: 2024-08-09 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_tenant_isolation'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply tenant isolation changes."""

    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('status', sa.String(50), nullable=False, default='active'),
        sa.Column('plan', sa.String(50), nullable=False, default='starter'),
        sa.Column('settings', postgresql.JSON, default={}),
        sa.Column('contact_email', sa.String(255)),
        sa.Column('contact_name', sa.String(255)),
        sa.Column('max_users', sa.Integer, default=10),
        sa.Column('max_storage_gb', sa.Integer, default=100),
        sa.Column('require_mfa', sa.Boolean, default=False),
        sa.Column('allowed_domains', postgresql.JSON, default=[]),
        sa.Column('logo_url', sa.String(500)),
        sa.Column('primary_color', sa.String(7)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # Create tenant_users table
    op.create_table(
        'tenant_users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('roles', postgresql.JSON, default=[]),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('invited_at', sa.DateTime(timezone=True)),
        sa.Column('joined_at', sa.DateTime(timezone=True)),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('tenant_id', 'user_id', name='uq_tenant_user'),
    )

    # Create evidence table with tenant isolation
    op.create_table(
        'evidence',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('content_type', sa.String(100)),
        sa.Column('size_bytes', sa.BigInteger),
        sa.Column('sha256_hash', sa.String(64)),
        sa.Column('storage_path', sa.String(1000)),
        sa.Column('storage_backend', sa.String(50), default='filesystem'),
        sa.Column('status', sa.String(50), default='uploaded'),
        sa.Column('processed_at', sa.DateTime(timezone=True)),
        sa.Column('uploaded_by', sa.String(255)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )

    # Create findings table with tenant isolation
    op.create_table(
        'findings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('severity', sa.String(20)),
        sa.Column('status', sa.String(50), default='open'),
        sa.Column('category', sa.String(100)),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('evidence_ids', postgresql.JSON, default=[]),
        sa.Column('attack_techniques', postgresql.JSON, default=[]),
        sa.Column('attack_tactics', postgresql.JSON, default=[]),
        sa.Column('created_by', sa.String(255)),
        sa.Column('assigned_to', sa.String(255)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )

    # Create embedding vectors table with tenant isolation
    op.create_table(
        'embedding_vectors',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_type', sa.String(50)),
        sa.Column('source_id', postgresql.UUID(as_uuid=True)),
        sa.Column('content_hash', sa.String(64)),
        sa.Column('embedding_model', sa.String(100)),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )

    # Add indexes for performance
    op.create_index('idx_tenant_users_tenant_id', 'tenant_users', ['tenant_id'])
    op.create_index('idx_tenant_users_user_id', 'tenant_users', ['user_id'])
    op.create_index('idx_evidence_tenant_id', 'evidence', ['tenant_id'])
    op.create_index('idx_evidence_uploaded_by', 'evidence', ['uploaded_by'])
    op.create_index('idx_findings_tenant_id', 'findings', ['tenant_id'])
    op.create_index('idx_findings_created_by', 'findings', ['created_by'])
    op.create_index('idx_findings_status', 'findings', ['status'])
    op.create_index('idx_embedding_vectors_tenant_id', 'embedding_vectors', ['tenant_id'])
    op.create_index('idx_embedding_vectors_source', 'embedding_vectors', ['source_type', 'source_id'])


def downgrade() -> None:
    """Remove tenant isolation changes."""

    # Drop indexes
    op.drop_index('idx_embedding_vectors_source')
    op.drop_index('idx_embedding_vectors_tenant_id')
    op.drop_index('idx_findings_status')
    op.drop_index('idx_findings_created_by')
    op.drop_index('idx_findings_tenant_id')
    op.drop_index('idx_evidence_uploaded_by')
    op.drop_index('idx_evidence_tenant_id')
    op.drop_index('idx_tenant_users_user_id')
    op.drop_index('idx_tenant_users_tenant_id')

    # Drop tables
    op.drop_table('embedding_vectors')
    op.drop_table('findings')
    op.drop_table('evidence')
    op.drop_table('tenant_users')
    op.drop_table('tenants')
