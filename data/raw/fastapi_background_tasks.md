# FastAPI Background Tasks

You can define background tasks to be run *after* returning a response. This is useful for operations that need to happen after a request, but where the client doesn't really have to be waiting for the operation to complete before receiving the response.

## Common Use Cases

* Sending email notifications after performing an action (e.g. verifying an account).
* Processing data, images, or generating reports.
* Updating search indices or logging analytics events.

## Using `BackgroundTasks`

First, import `BackgroundTasks` and define a parameter in your path operation function with a type declaration of `BackgroundTasks`:

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}\n"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

### How it Works

1. FastAPI creates an object of type `BackgroundTasks` and injects it into your route handler.
2. In your handler, call `background_tasks.add_task(func, *args, **kwargs)`.
3. FastAPI will return the response immediately to the client (`{"message": "Notification sent in the background"}`).
4. The background task (`write_notification`) runs right after the response has been sent.

## Background Tasks in Dependencies

You can also inject and add background tasks inside reusable dependencies:

```python
from fastapi import BackgroundTasks, Depends

def write_log(message: str):
    with open("app.log", "a") as f:
        f.write(message + "\n")

def get_query(background_tasks: BackgroundTasks, q: str | None = None):
    if q:
        message = f"Found query: {q}"
        background_tasks.add_task(write_log, message)
    return q

@app.post("/send/")
async def send_data(query: Annotated[str, Depends(get_query)]):
    return {"query": query}
```

## Technical Caveats & Scalability

* **In-process execution**: `BackgroundTasks` runs in the same Python process as FastAPI. If the server crashes or restarts before the task executes, the task is lost.
* **Heavy background processing**: For CPU-intensive jobs (machine learning, video encoding, large ETL) or mission-critical distributed tasks, use dedicated task queues like **Celery**, **ARQ**, or **Redis Queue (RQ)** instead of FastAPI's built-in `BackgroundTasks`.
