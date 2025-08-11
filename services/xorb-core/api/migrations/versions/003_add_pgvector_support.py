"""Add pgvector support and embedding column

Revision ID: 003_pgvector_support
Revises: 002_tenant_rls
Create Date: 2024-08-09 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '003_pgvector_support'
down_revision = '002_tenant_rls'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add pgvector extension and embedding column."""
    
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Add vector column to embedding_vectors table
    op.execute("""
        ALTER TABLE embedding_vectors 
        ADD COLUMN IF NOT EXISTS embedding vector(1536)
    """)
    
    # Create HNSW index for fast similarity search
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding_vectors_embedding_hnsw
        ON embedding_vectors 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    
    # Create index on content_hash for deduplication
    op.create_index(
        'idx_embedding_vectors_content_hash',
        'embedding_vectors',
        ['content_hash']
    )
    
    # Create composite index for tenant + source lookups
    op.create_index(
        'idx_embedding_vectors_tenant_source',
        'embedding_vectors',
        ['tenant_id', 'source_type', 'source_id']
    )
    
    # Create index on embedding model for filtering
    op.create_index(
        'idx_embedding_vectors_model',
        'embedding_vectors',
        ['embedding_model']
    )
    
    # Add constraints
    op.execute("""
        ALTER TABLE embedding_vectors
        ADD CONSTRAINT chk_embedding_dimension 
        CHECK (vector_dims(embedding) = 1536)
    """)
    
    # Create function for embedding search optimization
    op.execute("""
        CREATE OR REPLACE FUNCTION search_embeddings(
            query_embedding vector(1536),
            search_tenant_id uuid,
            search_source_type text DEFAULT NULL,
            similarity_threshold float DEFAULT 0.0,
            result_limit int DEFAULT 10
        )
        RETURNS TABLE(
            id uuid,
            source_type text,
            source_id uuid,
            content_hash text,
            embedding_model text,
            metadata jsonb,
            similarity float
        ) AS $$
        BEGIN
            -- Set tenant context for RLS
            PERFORM set_config('app.tenant_id', search_tenant_id::text, true);
            
            RETURN QUERY
            SELECT 
                ev.id,
                ev.source_type,
                ev.source_id,
                ev.content_hash,
                ev.embedding_model,
                ev.metadata,
                1 - (ev.embedding <=> query_embedding) as similarity
            FROM embedding_vectors ev
            WHERE ev.tenant_id = search_tenant_id
                AND (search_source_type IS NULL OR ev.source_type = search_source_type)
                AND 1 - (ev.embedding <=> query_embedding) >= similarity_threshold
            ORDER BY ev.embedding <=> query_embedding
            LIMIT result_limit;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)


def downgrade() -> None:
    """Remove pgvector support."""
    
    # Drop function
    op.execute("DROP FUNCTION IF EXISTS search_embeddings")
    
    # Drop constraints
    op.execute("ALTER TABLE embedding_vectors DROP CONSTRAINT IF EXISTS chk_embedding_dimension")
    
    # Drop indexes
    op.drop_index('idx_embedding_vectors_model')
    op.drop_index('idx_embedding_vectors_tenant_source')
    op.drop_index('idx_embedding_vectors_content_hash')
    op.execute("DROP INDEX IF EXISTS idx_embedding_vectors_embedding_hnsw")
    
    # Drop vector column
    op.execute("ALTER TABLE embedding_vectors DROP COLUMN IF EXISTS embedding")
    
    # Note: We don't drop the vector extension as it might be used by other applications