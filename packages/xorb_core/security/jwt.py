from jose import jwt
import os

SECRET = os.environ.get("JWT_SECRET", "a_default_secret")
ALGO = 'HS256'

def create_token(sub:str,role:str):
    return jwt.encode({'sub':sub,'role':role}, SECRET, algorithm=ALGO)

def verify(tok:str):
    from jose import JWTError
    try: return jwt.decode(tok, SECRET, algorithms=[ALGO])
    except JWTError: return None
