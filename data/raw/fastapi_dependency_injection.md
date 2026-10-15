# FastAPI Dependencies and Dependency Injection

FastAPI has a very powerful but intuitive **Dependency Injection** system.
It is designed to be very simple to use, and to make it very easy for any developer to integrate other components with FastAPI.

## What is "Dependency Injection"

**"Dependency Injection"** means, in programming, that there is a way for your code (in this case, your *path operation functions*) to declare things that it requires to work and use: "dependencies".

And then, that system (in this case **FastAPI**) will take care of doing whatever is needed to provide your code with those needed dependencies ("inject" the dependencies).

This is very useful when you need to:
* Have shared logic (the same code logic over and over again).
* Share database connections.
* Enforce security, authentication, role requirements, etc.
* And minimize code duplication.

## First Steps: Simple Dependency

Let's see a simple example:

```python
from typing import Annotated
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons

@app.get("/users/")
async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
    return commons
```

### Explanation of `Depends`

You declare the dependency using `Depends`:
* `Depends` takes a callable (like a function).
* FastAPI calls the function with its declared parameters.
* The returned value is passed to your path operation parameter (`commons`).

## Sub-dependencies

You can create dependencies that themselves have dependencies:

```python
def query_extractor(q: str | None = None):
    return q

def query_or_cookie_extractor(
    q: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[str | None, Cookie()] = None,
):
    if not q:
        return last_query
    return q

@app.get("/items/")
async def read_query(query_or_default: Annotated[str, Depends(query_or_cookie_extractor)]):
    return {"q_or_cookie": query_or_default}
```

FastAPI will resolve the full dependency tree automatically and cache the result for that request (unless `use_cache=False` is set).

## Dependencies with `yield` (Cleanup & DB Sessions)

FastAPI supports dependencies that do some extra steps after finishing. To do this, use `yield` instead of `return`:

```python
async def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()
```

The code prior to `yield` runs before the path operation executes. The code after `yield` runs after the response is sent back to the client.

## Global Dependencies

You can apply dependencies across all routes in an APIRouter or FastAPI instance:

```python
app = FastAPI(dependencies=[Depends(verify_token), Depends(verify_key)])
```
