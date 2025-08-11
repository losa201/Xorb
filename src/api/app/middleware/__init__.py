"""Middleware package with fallback implementations"""

from typing import Optional

def get_current_tenant_id() -> Optional[str]:
    """Get current tenant ID (fallback implementation)"""
    return "default-tenant"

__all__ = ["get_current_tenant_id"]