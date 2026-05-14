import asyncio
import json
import uuid
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .models import TaskStatus, TaskResponse, TaskSubmitResponse
from . import store
from .worker import worker_loop

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
HEARTBEAT_INTERVAL = 30

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

_subscribers: dict[str, set[asyncio.Event]] = {}
_sub_lock = asyncio.Lock()


async def _subscribe(task_id: str) -> asyncio.Event:
    evt = asyncio.Event()
    async with _sub_lock:
        _subscribers.setdefault(task_id, set()).add(evt)
    return evt


async def _unsubscribe(task_id: str, evt: asyncio.Event):
    async with _sub_lock:
        subs = _subscribers.get(task_id)
        if subs:
            subs.discard(evt)
            if not subs:
                del _subscribers[task_id]


def notify_task_update(task_id: str):
    subs = _subscribers.get(task_id)
    if subs:
        for evt in list(subs):
            evt.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.reset_stale_processing()
    worker = asyncio.create_task(worker_loop(on_update=notify_task_update))
    yield
    worker.cancel()


app = FastAPI(title="GLM-OCR Service", version="1.0.0", lifespan=lifespan)


@app.post("/api/tasks", response_model=TaskSubmitResponse)
async def submit_task(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    task_id = uuid.uuid4().hex
    save_dir = UPLOAD_DIR / task_id
    save_dir.mkdir(parents=True)
    save_path = save_dir / safe_name

    if not save_path.resolve().is_relative_to(save_dir.resolve()):
        raise HTTPException(400, "Invalid filename")

    total = 0
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                shutil.rmtree(save_dir, ignore_errors=True)
                raise HTTPException(413, "File too large (max 100MB)")
            f.write(chunk)

    store.create_task(task_id, safe_name, str(save_path))

    return TaskSubmitResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Subscribe to GET /api/tasks/{task_id}/events for real-time updates",
    )


@app.get("/api/tasks/{task_id}/events")
async def task_events_stream(task_id: str, request: Request):
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    async def event_generator():
        evt = await _subscribe(task_id)
        last_status = None
        try:
            while True:
                if await request.is_disconnected():
                    break

                evt.clear()
                current = await asyncio.to_thread(store.get_task, task_id)

                if current["status"] != last_status:
                    last_status = current["status"]
                    yield _sse(current["status"], current)

                if current["status"] in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break

                try:
                    await asyncio.wait_for(evt.wait(), timeout=HEARTBEAT_INTERVAL)
                except asyncio.TimeoutError:
                    yield _sse("heartbeat", {"task_id": task_id})
        finally:
            await _unsubscribe(task_id, evt)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@app.get("/api/tasks")
async def list_tasks(
    limit: int = Query(default=20, ge=1, le=200),
    status: TaskStatus | None = None,
):
    return store.list_tasks(limit=limit, status=status.value if status else None)


@app.get("/health")
async def health():
    return {"status": "ok"}
