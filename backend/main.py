"""
main.py — FastAPI application entry point.

Endpoints
---------
POST /tasks/              Upload a video; returns task_id
GET  /tasks/              List all tasks
GET  /tasks/{task_id}     Get task status
GET  /tasks/{task_id}/result  Get structured analysis JSON
DELETE /tasks/{task_id}   Cancel / remove a task
GET  /                    Serve the frontend HTML page
"""
from __future__ import annotations

import os
import uuid
import json
import logging
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from database import init_db, create_task, update_task_status, save_task_result, get_task, list_tasks
from schemas import TaskCreated, TaskStatus, TaskList, AnalysisResult
from processor import process_video

# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
KEYFRAMES_DIR = BASE_DIR / "keyframes"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Thread-pool for CPU-bound video processing (keeps event loop free)
executor = ThreadPoolExecutor(max_workers=2)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Video Object Analysis Service",
    description="Detects objects, classifies motion, and identifies person interactions in video files.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve keyframe images
app.mount("/keyframes", StaticFiles(directory=str(KEYFRAMES_DIR)), name="keyframes")
# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Initialise DB on startup
@app.on_event("startup")
def on_startup():
    init_db()
    log.info("Database initialised at %s", BASE_DIR / "data" / "tasks.db")


# ── Background video processing ───────────────────────────────────────────────

def _run_pipeline(task_id: str, video_path: str):
    """Synchronous wrapper executed in the thread-pool."""
    try:
        update_task_status(task_id, "processing")
        log.info("Pipeline started for task %s", task_id)

        result = process_video(
            video_path=video_path,
            keyframes_root=KEYFRAMES_DIR,
            task_id=task_id,
        )

        save_task_result(task_id, result)
        log.info("Pipeline completed for task %s", task_id)

    except Exception as exc:
        log.exception("Pipeline failed for task %s", task_id)
        update_task_status(task_id, "failed", error=str(exc))


async def _schedule_pipeline(task_id: str, video_path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _run_pipeline, task_id, video_path)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the single-page frontend."""
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        async with aiofiles.open(html_path, "r") as f:
            return await f.read()
    return HTMLResponse("<h1>Video Analysis API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.post("/tasks/", response_model=TaskCreated, status_code=202, tags=["Tasks"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (.mp4, .mov, .avi …)"),
):
    """
    Upload a video for asynchronous analysis.

    Returns a ``task_id`` which can be polled via ``GET /tasks/{task_id}``.
    """
    suffix = Path(file.filename or "video").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    task_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{task_id}{suffix}"

    # Stream upload to disk
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            await out.write(chunk)

    create_task(task_id, file.filename or dest.name)
    background_tasks.add_task(_schedule_pipeline, task_id, str(dest))

    return TaskCreated(
        task_id=task_id,
        message="Video accepted for processing. Poll /tasks/{task_id} for status.",
    )


@app.get("/tasks/", response_model=TaskList, tags=["Tasks"])
async def list_all_tasks():
    """Return a summary list of all tasks."""
    rows = list_tasks()
    tasks = [TaskStatus(**r) for r in rows]
    return TaskList(tasks=tasks, total=len(tasks))


@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["Tasks"])
async def get_task_status(task_id: str):
    """Return the current status of a specific task."""
    row = get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return TaskStatus(**{k: v for k, v in row.items() if k != "result_json"})


@app.get("/tasks/{task_id}/result", tags=["Tasks"])
async def get_task_result(task_id: str):
    """
    Return the full structured analysis JSON once the task is completed.

    Raises 404 if the task does not exist, 409 if it is not yet completed.
    """
    row = get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    if row["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Task is '{row['status']}', not yet completed.",
        )
    result = json.loads(row["result_json"])
    return JSONResponse(content=result)


@app.delete("/tasks/{task_id}", status_code=204, tags=["Tasks"])
async def delete_task(task_id: str):
    """Remove a task record (does not cancel an in-progress pipeline)."""
    from database import get_connection
    row = get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    with get_connection() as conn:
        conn.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "video-analysis"}
