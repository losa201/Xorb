"""Add HNSW index for findings embeddings

Revision ID: 005_hnsw_index
Revises: 004_gamification
Create Date: 2025-07-25 10:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '005_hnsw_index'
down_revision = '004_gamification'
branch_labels = None
depends_on = None


def upgrade():
    """Add HNSW index for vector similarity search on findings.embedding"""

    # Enable pgvector extension if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create HNSW index for faster similarity search
    # Using CONCURRENTLY to avoid locking the table during index creation
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS findings_embedding_hnsw
        ON findings
        USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # Create additional index for cosine distance if needed
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS findings_embedding_cosine_hnsw
        ON findings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # Vacuum analyze to update statistics
    op.execute("VACUUM ANALYZE findings;")


def downgrade():
    """Remove HNSW indexes"""

    op.execute("DROP INDEX IF EXISTS findings_embedding_hnsw;")
    op.execute("DROP INDEX IF EXISTS findings_embedding_cosine_hnsw;")
