"""
Legacy dependencies - kept for backward compatibility
"""

from fastapi import Depends, HTTPException

from .dependencies import get_current_user


def has_role(role: str):
    """Legacy role checker - use dependencies.require_role instead"""
    def _dep(current_user = Depends(get_current_user)):
        if not current_user.has_role(role):
            raise HTTPException(403, 'Forbidden')
        return {"username": current_user.username, "roles": current_user.roles}
    return _dep
