"""
Add gamification features to researchers table
Revision ID: 004_gamification
Revises: 003_ptaas_v1
Create Date: 2024-01-15 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '004_gamification'
down_revision = '003_ptaas_v1'
branch_labels = None
depends_on = None

def upgrade():
    """Add gamification columns to researchers table"""
    
    # Add Glicko-2 rating columns
    op.add_column('researchers', sa.Column('rating', sa.Float(), nullable=False, default=1500.0))
    op.add_column('researchers', sa.Column('rd', sa.Float(), nullable=False, default=350.0))
    op.add_column('researchers', sa.Column('vol', sa.Float(), nullable=False, default=0.06))
    op.add_column('researchers', sa.Column('last_competition', sa.DateTime(timezone=True), nullable=True))
    
    # Add gamification stats
    op.add_column('researchers', sa.Column('total_findings', sa.Integer(), nullable=False, default=0))
    op.add_column('researchers', sa.Column('accepted_findings', sa.Integer(), nullable=False, default=0))
    op.add_column('researchers', sa.Column('duplicate_findings', sa.Integer(), nullable=False, default=0))
    op.add_column('researchers', sa.Column('false_positive_findings', sa.Integer(), nullable=False, default=0))
    op.add_column('researchers', sa.Column('total_earnings', sa.Numeric(precision=10, scale=2), nullable=False, default=0.0))
    
    # Add badge and tier information  
    op.add_column('researchers', sa.Column('current_tier', sa.String(50), nullable=False, default='Bronze'))
    op.add_column('researchers', sa.Column('badges_earned', postgresql.JSONB(), nullable=False, default='[]'))
    op.add_column('researchers', sa.Column('tier_history', postgresql.JSONB(), nullable=False, default='[]'))
    
    # Add leaderboard preferences
    op.add_column('researchers', sa.Column('leaderboard_visible', sa.Boolean(), nullable=False, default=True))
    op.add_column('researchers', sa.Column('handle_anonymous', sa.Boolean(), nullable=False, default=False))
    
    # Create badges table for badge definitions
    op.create_table('badges',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('svg_icon', sa.Text(), nullable=False),
        sa.Column('criteria', postgresql.JSONB(), nullable=False),
        sa.Column('tier_requirement', sa.String(50), nullable=True),
        sa.Column('rating_requirement', sa.Float(), nullable=True),
        sa.Column('rd_requirement', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=False, default=True)
    )
    
    # Create researcher_badges junction table
    op.create_table('researcher_badges',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('researcher_id', sa.UUID(), sa.ForeignKey('researchers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('badge_id', sa.String(50), sa.ForeignKey('badges.id', ondelete='CASCADE'), nullable=False),
        sa.Column('earned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('org_id', sa.UUID(), sa.ForeignKey('orgs.id', ondelete='CASCADE'), nullable=False),
        sa.UniqueConstraint('researcher_id', 'badge_id', name='unique_researcher_badge')
    )
    
    # Create rating_history table for tracking rating changes
    op.create_table('rating_history',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('researcher_id', sa.UUID(), sa.ForeignKey('researchers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('old_rating', sa.Float(), nullable=False),
        sa.Column('new_rating', sa.Float(), nullable=False),
        sa.Column('old_rd', sa.Float(), nullable=False),
        sa.Column('new_rd', sa.Float(), nullable=False),
        sa.Column('old_vol', sa.Float(), nullable=False),
        sa.Column('new_vol', sa.Float(), nullable=False),
        sa.Column('change_reason', sa.String(100), nullable=False),  # 'finding_accepted', 'time_decay', etc.
        sa.Column('finding_id', sa.UUID(), sa.ForeignKey('findings.id', ondelete='SET NULL'), nullable=True),
        sa.Column('rating_change', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('org_id', sa.UUID(), sa.ForeignKey('orgs.id', ondelete='CASCADE'), nullable=False)
    )
    
    # Add indexes for performance
    op.create_index('ix_researchers_rating', 'researchers', ['rating'], unique=False)
    op.create_index('ix_researchers_current_tier', 'researchers', ['current_tier'], unique=False)
    op.create_index('ix_researchers_leaderboard_visible', 'researchers', ['leaderboard_visible'], unique=False)
    op.create_index('ix_researcher_badges_researcher_id', 'researcher_badges', ['researcher_id'], unique=False)
    op.create_index('ix_rating_history_researcher_id', 'rating_history', ['researcher_id'], unique=False)
    op.create_index('ix_rating_history_created_at', 'rating_history', ['created_at'], unique=False)
    
    # Add RLS policies for gamification tables
    op.execute("""
        ALTER TABLE researcher_badges ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY researcher_badges_org_isolation ON researcher_badges
            FOR ALL TO authenticated_role
            USING (org_id = current_setting('app.current_org_id')::uuid);
    """)
    
    op.execute("""
        ALTER TABLE rating_history ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY rating_history_org_isolation ON rating_history
            FOR ALL TO authenticated_role
            USING (org_id = current_setting('app.current_org_id')::uuid);
    """)
    
    # Insert default badges
    op.execute("""
        INSERT INTO badges (id, name, description, svg_icon, criteria, tier_requirement, rating_requirement) VALUES
        ('first_finding', 'First Finding', 'Submitted your first security finding', 
         '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{"findings_submitted": 1}', NULL, NULL),
         
        ('bronze_tier', 'Bronze Researcher', 'Achieved Bronze tier rating', 
         '<svg viewBox="0 0 24 24" fill="#CD7F32"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{}', 'Bronze', 0),
         
        ('silver_tier', 'Silver Researcher', 'Achieved Silver tier rating', 
         '<svg viewBox="0 0 24 24" fill="#C0C0C0"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{}', 'Silver', 1200),
         
        ('gold_tier', 'Gold Researcher', 'Achieved Gold tier rating', 
         '<svg viewBox="0 0 24 24" fill="#FFD700"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{}', 'Gold', 1500),
         
        ('platinum_tier', 'Platinum Researcher', 'Achieved Platinum tier rating', 
         '<svg viewBox="0 0 24 24" fill="#E5E4E2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{}', 'Platinum', 1800),
         
        ('diamond_tier', 'Diamond Researcher', 'Achieved Diamond tier rating', 
         '<svg viewBox="0 0 24 24" fill="#B9F2FF"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{}', 'Diamond', 2100),
         
        ('master_tier', 'Master Researcher', 'Achieved Master tier rating', 
         '<svg viewBox="0 0 24 24" fill="#FF6B6B"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
         '{}', 'Master', 2500),
         
        ('reliable_researcher', 'Reliable Researcher', 'Maintained low rating deviation (RD < 60)', 
         '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
         '{}', NULL, NULL),
         
        ('high_value_hunter', 'High Value Hunter', 'Found 5+ critical vulnerabilities', 
         '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>',
         '{"critical_findings": 5}', NULL, NULL),
         
        ('quality_contributor', 'Quality Contributor', 'Maintained <5% false positive rate with 20+ findings', 
         '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
         '{"min_findings": 20, "max_false_positive_rate": 0.05}', NULL, NULL)
    """)


def downgrade():
    """Remove gamification features"""
    
    # Drop indexes
    op.drop_index('ix_rating_history_created_at', table_name='rating_history')
    op.drop_index('ix_rating_history_researcher_id', table_name='rating_history')
    op.drop_index('ix_researcher_badges_researcher_id', table_name='researcher_badges')
    op.drop_index('ix_researchers_leaderboard_visible', table_name='researchers')
    op.drop_index('ix_researchers_current_tier', table_name='researchers')
    op.drop_index('ix_researchers_rating', table_name='researchers')
    
    # Drop tables
    op.drop_table('rating_history')
    op.drop_table('researcher_badges')
    op.drop_table('badges')
    
    # Remove columns from researchers table
    op.drop_column('researchers', 'handle_anonymous')
    op.drop_column('researchers', 'leaderboard_visible')
    op.drop_column('researchers', 'tier_history')
    op.drop_column('researchers', 'badges_earned')
    op.drop_column('researchers', 'current_tier')
    op.drop_column('researchers', 'total_earnings')
    op.drop_column('researchers', 'false_positive_findings')
    op.drop_column('researchers', 'duplicate_findings')
    op.drop_column('researchers', 'accepted_findings')
    op.drop_column('researchers', 'total_findings')
    op.drop_column('researchers', 'last_competition')
    op.drop_column('researchers', 'vol')
    op.drop_column('researchers', 'rd')
    op.drop_column('researchers', 'rating')