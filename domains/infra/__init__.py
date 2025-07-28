"""
XORB Infrastructure Domain

Database connections, external integrations, and infrastructure management.
"""

from .database import DatabaseManager, db_manager

__all__ = [
    "db_manager",
    "DatabaseManager"
]
