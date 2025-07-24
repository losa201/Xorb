from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from xorb_core.security.jwt import verify

_bearer=HTTPBearer()

def has_role(role:str):
  def _dep(cred=Depends(_bearer)):
      user=verify(cred.credentials)
      if not user or user['role'] not in (role,'admin'):
          raise HTTPException(403,'Forbidden')
      return user
  return _dep
