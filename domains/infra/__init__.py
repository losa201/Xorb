"""
XORB Infrastructure Domain

Database connections, external integrations, and infrastructure management.
"""

from .database import db_manager, DatabaseManager

__all__ = [
    "db_manager",
    "DatabaseManager"
]