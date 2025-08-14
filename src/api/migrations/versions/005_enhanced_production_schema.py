"""Enhanced Production Schema - Comprehensive database schema for XORB Enterprise

Revision ID: 005_enhanced_production_schema
Revises: 004_performance_optimization
Create Date: 2025-01-10 22:15:55.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers
revision = '005_enhanced_production_schema'
down_revision = '004_performance_optimization'
branch_labels = None
depends_on = None


def upgrade():
    """Create enhanced production schema with comprehensive tables"""

    # Enhanced Users table with advanced features
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('username', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('first_name', sa.String(255)),
        sa.Column('last_name', sa.String(255)),
        sa.Column('roles', postgresql.JSONB, default=lambda: ['user']),
        sa.Column('permissions', postgresql.JSONB, default=lambda: {}),
        sa.Column('is_active', sa.Boolean, default=True, index=True),
        sa.Column('is_verified', sa.Boolean, default=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('failed_login_attempts', sa.Integer, default=0),
        sa.Column('locked_until', sa.DateTime(timezone=True)),
        sa.Column('password_changed_at', sa.DateTime(timezone=True)),
        sa.Column('mfa_enabled', sa.Boolean, default=False),
        sa.Column('mfa_secret', sa.String(255)),
        sa.Column('preferences', postgresql.JSONB, default=lambda: {}),
        sa.Column('metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Tenants table for multi-tenancy
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('slug', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('description', sa.Text),
        sa.Column('plan_type', sa.String(50), nullable=False, default='basic'),
        sa.Column('status', sa.String(50), nullable=False, default='active'),
        sa.Column('settings', postgresql.JSONB, default=lambda: {}),
        sa.Column('limits', postgresql.JSONB, default=lambda: {}),
        sa.Column('billing_info', postgresql.JSONB, default=lambda: {}),
        sa.Column('contact_info', postgresql.JSONB, default=lambda: {}),
        sa.Column('subscription_expires_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Enhanced scan sessions table
    op.create_table(
        'scan_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('scan_type', sa.String(100), nullable=False, index=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('targets_count', sa.Integer, default=0),
        sa.Column('scan_profile', sa.String(100), default='comprehensive'),
        sa.Column('stealth_mode', sa.Boolean, default=False),
        sa.Column('priority', sa.String(50), default='medium', index=True),
        sa.Column('estimated_duration', sa.Integer),  # seconds
        sa.Column('actual_duration', sa.Integer),  # seconds
        sa.Column('progress_percentage', sa.Float, default=0.0),
        sa.Column('current_stage', sa.String(100)),
        sa.Column('findings_count', sa.Integer, default=0),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('error_message', sa.Text),
        sa.Column('metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('compliance_framework', sa.String(100)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Scan targets table
    op.create_table(
        'scan_targets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('target_order', sa.Integer, nullable=False),
        sa.Column('host', sa.String(255), nullable=False),
        sa.Column('ports', postgresql.JSONB, default=lambda: []),
        sa.Column('scan_profile', sa.String(100), default='standard'),
        sa.Column('authorized', sa.Boolean, default=False),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('scan_results', postgresql.JSONB, default=lambda: {}),
        sa.Column('metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Scan results table
    op.create_table(
        'scan_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('target_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('scanner_type', sa.String(100), nullable=False),
        sa.Column('vulnerability_summary', postgresql.JSONB, default=lambda: {}),
        sa.Column('raw_results', postgresql.JSONB, default=lambda: {}),
        sa.Column('findings', postgresql.JSONB, default=lambda: []),
        sa.Column('risk_score', sa.Float, default=0.0),
        sa.Column('compliance_results', postgresql.JSONB, default=lambda: {}),
        sa.Column('recommendations', postgresql.JSONB, default=lambda: []),
        sa.Column('mitre_techniques', postgresql.JSONB, default=lambda: []),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Threat intelligence indicators table
    op.create_table(
        'threat_indicators',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('value', sa.String(1000), nullable=False, index=True),
        sa.Column('indicator_type', sa.String(100), nullable=False, index=True),
        sa.Column('threat_level', sa.String(50), nullable=False, index=True),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('source', sa.String(255), nullable=False),
        sa.Column('first_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('tags', postgresql.JSONB, default=lambda: []),
        sa.Column('context', postgresql.JSONB, default=lambda: {}),
        sa.Column('related_indicators', postgresql.JSONB, default=lambda: []),
        sa.Column('attribution', postgresql.JSONB, default=lambda: {}),
        sa.Column('is_active', sa.Boolean, default=True, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Threat analysis sessions table
    op.create_table(
        'threat_analysis_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('analysis_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('indicators_analyzed', sa.Integer, default=0),
        sa.Column('threat_level', sa.String(50), index=True),
        sa.Column('confidence_score', sa.Float),
        sa.Column('risk_score', sa.Float),
        sa.Column('analysis_summary', sa.Text),
        sa.Column('correlation_results', postgresql.JSONB, default=lambda: {}),
        sa.Column('behavioral_analysis', postgresql.JSONB, default=lambda: {}),
        sa.Column('attribution', postgresql.JSONB, default=lambda: {}),
        sa.Column('recommendations', postgresql.JSONB, default=lambda: []),
        sa.Column('mitre_techniques', postgresql.JSONB, default=lambda: []),
        sa.Column('analysis_metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Behavioral profiles table
    op.create_table(
        'behavioral_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('entity_id', sa.String(255), nullable=False, index=True),
        sa.Column('entity_type', sa.String(50), nullable=False, index=True),  # user, system, network
        sa.Column('profile_data', postgresql.JSONB, nullable=False),
        sa.Column('baseline_metrics', postgresql.JSONB, default=lambda: {}),
        sa.Column('anomaly_scores', postgresql.JSONB, default=lambda: {}),
        sa.Column('risk_factors', postgresql.JSONB, default=lambda: []),
        sa.Column('last_analysis', sa.DateTime(timezone=True)),
        sa.Column('confidence_score', sa.Float, default=0.0),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Threat hunting queries table
    op.create_table(
        'threat_hunting_queries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('query_text', sa.Text, nullable=False),
        sa.Column('query_language', sa.String(50), default='xorb_ql'),
        sa.Column('category', sa.String(100), index=True),
        sa.Column('severity', sa.String(50), index=True),
        sa.Column('mitre_techniques', postgresql.JSONB, default=lambda: []),
        sa.Column('tags', postgresql.JSONB, default=lambda: []),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('is_public', sa.Boolean, default=False),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('effectiveness_score', sa.Float, default=0.0),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Forensics evidence table
    op.create_table(
        'forensics_evidence',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('evidence_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('case_id', sa.String(255), index=True),
        sa.Column('collector_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evidence_type', sa.String(100), nullable=False, index=True),
        sa.Column('source_system', sa.String(255)),
        sa.Column('collection_method', sa.String(100)),
        sa.Column('file_hash', sa.String(255), index=True),
        sa.Column('file_size', sa.BigInteger),
        sa.Column('metadata', postgresql.JSONB, nullable=False),
        sa.Column('chain_of_custody', postgresql.JSONB, nullable=False),
        sa.Column('integrity_verified', sa.Boolean, default=False),
        sa.Column('encryption_status', sa.String(50)),
        sa.Column('legal_hold', sa.Boolean, default=False),
        sa.Column('retention_policy', sa.String(100)),
        sa.Column('tags', postgresql.JSONB, default=lambda: []),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Compliance assessments table
    op.create_table(
        'compliance_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('framework', sa.String(100), nullable=False, index=True),
        sa.Column('assessment_type', sa.String(100), nullable=False),
        sa.Column('target_scope', postgresql.JSONB, nullable=False),
        sa.Column('scan_session_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('status', sa.String(50), default='pending', index=True),
        sa.Column('compliance_score', sa.Float),
        sa.Column('controls_assessed', sa.Integer, default=0),
        sa.Column('controls_passed', sa.Integer, default=0),
        sa.Column('controls_failed', sa.Integer, default=0),
        sa.Column('findings', postgresql.JSONB, default=lambda: []),
        sa.Column('recommendations', postgresql.JSONB, default=lambda: []),
        sa.Column('gaps_identified', postgresql.JSONB, default=lambda: []),
        sa.Column('remediation_plan', postgresql.JSONB, default=lambda: {}),
        sa.Column('assessor_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('assessment_date', sa.DateTime(timezone=True)),
        sa.Column('next_assessment_due', sa.DateTime(timezone=True)),
        sa.Column('metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Workflow orchestration table
    op.create_table(
        'workflows',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('workflow_type', sa.String(100), nullable=False, index=True),
        sa.Column('definition', postgresql.JSONB, nullable=False),
        sa.Column('triggers', postgresql.JSONB, default=lambda: []),
        sa.Column('schedule', postgresql.JSONB),
        sa.Column('is_enabled', sa.Boolean, default=True, index=True),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('execution_count', sa.Integer, default=0),
        sa.Column('success_rate', sa.Float, default=0.0),
        sa.Column('average_duration', sa.Integer),  # seconds
        sa.Column('last_execution', sa.DateTime(timezone=True)),
        sa.Column('metadata', postgresql.JSONB, default=lambda: {}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Workflow executions table
    op.create_table(
        'workflow_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('execution_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('trigger_type', sa.String(100)),
        sa.Column('input_parameters', postgresql.JSONB, default=lambda: {}),
        sa.Column('current_stage', sa.String(255)),
        sa.Column('progress_percentage', sa.Float, default=0.0),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('duration', sa.Integer),  # seconds
        sa.Column('error_message', sa.Text),
        sa.Column('results', postgresql.JSONB, default=lambda: {}),
        sa.Column('execution_log', postgresql.JSONB, default=lambda: []),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Add foreign key constraints
    op.create_foreign_key('fk_users_tenant', 'users', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_scan_sessions_user', 'scan_sessions', 'users', ['user_id'], ['id'])
    op.create_foreign_key('fk_scan_sessions_tenant', 'scan_sessions', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_scan_targets_session', 'scan_targets', 'scan_sessions', ['session_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_scan_results_session', 'scan_results', 'scan_sessions', ['session_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_scan_results_target', 'scan_results', 'scan_targets', ['target_id'], ['id'])
    op.create_foreign_key('fk_threat_analysis_user', 'threat_analysis_sessions', 'users', ['user_id'], ['id'])
    op.create_foreign_key('fk_threat_analysis_tenant', 'threat_analysis_sessions', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_behavioral_profiles_tenant', 'behavioral_profiles', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_hunting_queries_author', 'threat_hunting_queries', 'users', ['author_id'], ['id'])
    op.create_foreign_key('fk_hunting_queries_tenant', 'threat_hunting_queries', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_forensics_evidence_collector', 'forensics_evidence', 'users', ['collector_id'], ['id'])
    op.create_foreign_key('fk_forensics_evidence_tenant', 'forensics_evidence', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_compliance_assessments_assessor', 'compliance_assessments', 'users', ['assessor_id'], ['id'])
    op.create_foreign_key('fk_compliance_assessments_tenant', 'compliance_assessments', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_compliance_assessments_session', 'compliance_assessments', 'scan_sessions', ['scan_session_id'], ['id'])
    op.create_foreign_key('fk_workflows_author', 'workflows', 'users', ['author_id'], ['id'])
    op.create_foreign_key('fk_workflows_tenant', 'workflows', 'tenants', ['tenant_id'], ['id'])
    op.create_foreign_key('fk_workflow_executions_workflow', 'workflow_executions', 'workflows', ['workflow_id'], ['id'])
    op.create_foreign_key('fk_workflow_executions_tenant', 'workflow_executions', 'tenants', ['tenant_id'], ['id'])

    # Create indexes for performance
    op.create_index('idx_users_tenant_active', 'users', ['tenant_id', 'is_active'])
    op.create_index('idx_scan_sessions_tenant_status', 'scan_sessions', ['tenant_id', 'status'])
    op.create_index('idx_scan_sessions_user_created', 'scan_sessions', ['user_id', 'created_at'])
    op.create_index('idx_threat_indicators_type_level', 'threat_indicators', ['indicator_type', 'threat_level'])
    op.create_index('idx_threat_indicators_value_hash', 'threat_indicators', [sa.text('md5(value)')])
    op.create_index('idx_behavioral_profiles_entity', 'behavioral_profiles', ['entity_id', 'entity_type'])
    op.create_index('idx_forensics_evidence_case', 'forensics_evidence', ['case_id'])
    op.create_index('idx_compliance_framework_date', 'compliance_assessments', ['framework', 'assessment_date'])
    op.create_index('idx_workflows_type_enabled', 'workflows', ['workflow_type', 'is_enabled'])
    op.create_index('idx_workflow_executions_status_created', 'workflow_executions', ['status', 'created_at'])

    # Enable Row Level Security (RLS) for multi-tenancy
    op.execute('ALTER TABLE users ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE scan_sessions ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE scan_targets ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE scan_results ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE threat_analysis_sessions ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE behavioral_profiles ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE threat_hunting_queries ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE forensics_evidence ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE compliance_assessments ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE workflows ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE workflow_executions ENABLE ROW LEVEL SECURITY')

    # Create RLS policies
    op.execute('''
        CREATE POLICY tenant_isolation_users ON users
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid)
    ''')

    op.execute('''
        CREATE POLICY tenant_isolation_scan_sessions ON scan_sessions
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid)
    ''')

    op.execute('''
        CREATE POLICY tenant_isolation_behavioral_profiles ON behavioral_profiles
        USING (tenant_id = current_setting('app.current_tenant_id')::uuid)
    ''')


def downgrade():
    """Remove enhanced production schema"""

    # Drop RLS policies
    op.execute('DROP POLICY IF EXISTS tenant_isolation_users ON users')
    op.execute('DROP POLICY IF EXISTS tenant_isolation_scan_sessions ON scan_sessions')
    op.execute('DROP POLICY IF EXISTS tenant_isolation_behavioral_profiles ON behavioral_profiles')

    # Disable RLS
    op.execute('ALTER TABLE users DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE scan_sessions DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE scan_targets DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE scan_results DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE threat_analysis_sessions DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE behavioral_profiles DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE threat_hunting_queries DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE forensics_evidence DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE compliance_assessments DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE workflows DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE workflow_executions DISABLE ROW LEVEL SECURITY')

    # Drop tables in reverse order of dependencies
    op.drop_table('workflow_executions')
    op.drop_table('workflows')
    op.drop_table('compliance_assessments')
    op.drop_table('forensics_evidence')
    op.drop_table('threat_hunting_queries')
    op.drop_table('behavioral_profiles')
    op.drop_table('threat_analysis_sessions')
    op.drop_table('threat_indicators')
    op.drop_table('scan_results')
    op.drop_table('scan_targets')
    op.drop_table('scan_sessions')
    op.drop_table('tenants')
    op.drop_table('users')
