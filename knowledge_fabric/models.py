#!/usr/bin/env python3

from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class AtomModel(Base):
    __tablename__ = "knowledge_atoms"
    
    id = Column(String(36), primary_key=True)
    atom_type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    confidence = Column(Float, default=0.5, nullable=False)
    predictive_score = Column(Float, default=0.0, nullable=False)
    tags = Column(Text, default="[]")
    sources = Column(Text, default="[]")
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Relationships
    source_relationships = relationship("RelationshipModel", foreign_keys="RelationshipModel.source_atom_id", back_populates="source_atom")
    target_relationships = relationship("RelationshipModel", foreign_keys="RelationshipModel.target_atom_id", back_populates="target_atom")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_atom_type', 'atom_type'),
        Index('idx_confidence', 'confidence'),
        Index('idx_predictive_score', 'predictive_score'),
        Index('idx_created_at', 'created_at'),
        Index('idx_expires_at', 'expires_at'),
        Index('idx_type_confidence', 'atom_type', 'confidence'),
        Index('idx_type_score', 'atom_type', 'predictive_score'),
    )


class RelationshipModel(Base):
    __tablename__ = "atom_relationships"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_atom_id = Column(String(36), ForeignKey("knowledge_atoms.id"), nullable=False)
    target_atom_id = Column(String(36), ForeignKey("knowledge_atoms.id"), nullable=False)
    relationship_type = Column(String(50), default="related", nullable=False)
    strength = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    atom_metadata = Column(Text, default="{}")
    
    # Relationships
    source_atom = relationship("AtomModel", foreign_keys=[source_atom_id], back_populates="source_relationships")
    target_atom = relationship("AtomModel", foreign_keys=[target_atom_id], back_populates="target_relationships")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_source_atom', 'source_atom_id'),
        Index('idx_target_atom', 'target_atom_id'),
        Index('idx_relationship_type', 'relationship_type'),
        Index('idx_relationship_strength', 'strength'),
        Index('idx_bidirectional', 'source_atom_id', 'target_atom_id'),
    )


class ValidationModel(Base):
    __tablename__ = "atom_validations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    atom_id = Column(String(36), ForeignKey("knowledge_atoms.id"), nullable=False)
    validation_method = Column(String(100), nullable=False)
    is_valid = Column(Integer, nullable=False)  # Using Integer as boolean for SQLite compatibility
    confidence_adjustment = Column(Float, default=0.0)
    validation_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, default="")
    validator_id = Column(String(100), nullable=True)
    atom_metadata = Column(Text, default="{}")
    
    # Relationship
    atom = relationship("AtomModel")
    
    # Indexes
    __table_args__ = (
        Index('idx_atom_validation', 'atom_id'),
        Index('idx_validation_method', 'validation_method'),
        Index('idx_validation_timestamp', 'validation_timestamp'),
        Index('idx_is_valid', 'is_valid'),
    )


class UsageStatsModel(Base):
    __tablename__ = "atom_usage_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    atom_id = Column(String(36), ForeignKey("knowledge_atoms.id"), nullable=False)
    usage_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    usage_context = Column(String(100), nullable=False)  # campaign_id, agent_id, etc.
    success = Column(Integer, nullable=False)  # Using Integer as boolean
    execution_time = Column(Float, default=0.0)
    error_details = Column(Text, nullable=True)
    atom_metadata = Column(Text, default="{}")
    
    # Relationship
    atom = relationship("AtomModel")
    
    # Indexes
    __table_args__ = (
        Index('idx_atom_usage', 'atom_id'),
        Index('idx_usage_timestamp', 'usage_timestamp'),
        Index('idx_usage_context', 'usage_context'),
        Index('idx_usage_success', 'success'),
    )


class KnowledgeGraphModel(Base):
    __tablename__ = "knowledge_graph_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_atoms = Column(Integer, default=0)
    total_relationships = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    avg_predictive_score = Column(Float, default=0.0)
    atom_types_distribution = Column(Text, default="{}")
    confidence_distribution = Column(Text, default="{}")
    
    # Indexes
    __table_args__ = (
        Index('idx_stat_date', 'stat_date'),
    )


async def create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)