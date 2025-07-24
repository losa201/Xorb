from sqlalchemy import Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Target(Base):
    __tablename__ = "targets"

    id = Column(Integer, primary_key=True, index=True)
    value = Column(String, index=True)
    target_type = Column(String)
    scope = Column(String)
    priority = Column(Integer)

class Finding(Base):
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    target_id = Column(Integer)
    finding_type = Column(String)
    severity = Column(String)
    confidence = Column(Float)
    embedding = Column(Vector(1536))
