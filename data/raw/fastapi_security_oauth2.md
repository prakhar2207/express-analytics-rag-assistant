# FastAPI Security, OAuth2, and JWT Tokens

FastAPI provides several security utilities to handle authentication, authorization, and permission checking using standard OAuth2 specifications.

## OAuth2 with Password Flow and Bearer Tokens

The Password flow is one of the ways defined in OAuth2 to handle authentication and authorization.
The user sends a username and password to get an `access_token`. Then, for subsequent requests, the client sends this `access_token` in an `Authorization` header: `Authorization: Bearer <token>`.

### Implementation with `OAuth2PasswordBearer`

```python
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt

SECRET_KEY = "super-secret-key"
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    # Verify user credentials
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user
```

## Security Scopes

FastAPI supports OAuth2 scopes to grant fine-grained permissions (e.g. `read:users`, `write:items`):

```python
from fastapi.security import SecurityScopes

async def get_current_user(
    security_scopes: SecurityScopes,
    token: Annotated[str, Depends(oauth2_scheme)]
):
    for scope in security_scopes.scopes:
        if scope not in user_scopes:
            raise HTTPException(status_code=403, detail="Not enough permissions")
    return user
```
