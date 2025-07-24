from fastapi import Depends, HTTPException
from .security import get_current_user, TokenData

def has_role(role: str):
    def _dep(current_user: TokenData = Depends(get_current_user)):
        if role not in current_user.roles and 'admin' not in current_user.roles:
            raise HTTPException(403, 'Forbidden')
        return {"username": current_user.username, "roles": current_user.roles}
    return _dep
