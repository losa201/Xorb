from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from .. import security

router = APIRouter()


@router.post("/auth/token", response_model=security.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real application, you would look up the user in a database
    # and verify the password.
    # For this example, we'll use a dummy user.
    if not security.verify_password(form_data.password, security.get_password_hash("secret")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=security.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": form_data.username, "roles": ["admin"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
