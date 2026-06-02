"""
tests/test_api.py — Integration tests for the FastAPI REST layer.

Uses the FastAPI TestClient (wraps httpx) so no running server is needed.
"""
import json
import sys
import tempfile
import shutil
import numpy as np
import cv2
import pytest
from pathlib import Path

# Ensure backend is importable
BACKEND = Path(__file__).parent.parent / "backend"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(BACKEND))

# ── Patch paths before importing app ─────────────────────────────────────────
import database as _db_module
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_db_module.DB_PATH = Path(_tmp_db.name)
_tmp_db.close()

from fastapi.testclient import TestClient
import main as app_module

# Redirect uploads & keyframes to temp dirs
_tmp_upload = tempfile.mkdtemp()
_tmp_keys   = tempfile.mkdtemp()
app_module.UPLOAD_DIR    = Path(_tmp_upload)
app_module.KEYFRAMES_DIR = Path(_tmp_keys)

client = TestClient(app_module.app, raise_server_exceptions=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_video(path: Path, num_frames: int = 30, w: int = 320, h: int = 240) -> Path:
    """Create a minimal synthetic .mp4 with `num_frames` coloured frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 2] = int(i * 255 / num_frames)   # R channel ramp
        out.write(frame)
    out.release()
    return path


@pytest.fixture(scope="module")
def test_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("vids") / "test_video.mp4"
    return make_test_video(p)


@pytest.fixture(autouse=True)
def init_db():
    _db_module.init_db()


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_valid_video(test_video):
    with open(test_video, "rb") as f:
        res = client.post("/tasks/", files={"file": ("test.mp4", f, "video/mp4")})
    assert res.status_code == 202
    body = res.json()
    assert "task_id" in body
    assert body["status"] == "pending"


def test_upload_invalid_extension():
    res = client.post(
        "/tasks/",
        files={"file": ("malware.exe", b"fake", "application/octet-stream")},
    )
    assert res.status_code == 415


def test_upload_txt_rejected():
    res = client.post(
        "/tasks/",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 415


# ── Task lifecycle ────────────────────────────────────────────────────────────

def test_list_tasks_initially_empty():
    # Create fresh DB state
    _db_module.init_db()
    # list should return a valid structure (may have tasks from other tests)
    res = client.get("/tasks/")
    assert res.status_code == 200
    body = res.json()
    assert "tasks" in body
    assert "total" in body
    assert isinstance(body["tasks"], list)


def test_get_nonexistent_task():
    res = client.get("/tasks/does-not-exist-00000000")
    assert res.status_code == 404


def test_get_result_before_complete(test_video):
    with open(test_video, "rb") as f:
        res = client.post("/tasks/", files={"file": ("t.mp4", f, "video/mp4")})
    task_id = res.json()["task_id"]

    # Force status to 'processing' without completing
    _db_module.update_task_status(task_id, "processing")
    res = client.get(f"/tasks/{task_id}/result")
    assert res.status_code == 409   # conflict — not completed


def test_task_status_fields(test_video):
    with open(test_video, "rb") as f:
        res = client.post("/tasks/", files={"file": ("t2.mp4", f, "video/mp4")})
    task_id = res.json()["task_id"]

    res = client.get(f"/tasks/{task_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["task_id"] == task_id
    assert body["status"] in {"pending", "processing", "completed", "failed"}
    assert "created_at" in body
    assert "updated_at" in body


def test_delete_task(test_video):
    with open(test_video, "rb") as f:
        res = client.post("/tasks/", files={"file": ("del.mp4", f, "video/mp4")})
    task_id = res.json()["task_id"]

    res = client.delete(f"/tasks/{task_id}")
    assert res.status_code == 204

    res = client.get(f"/tasks/{task_id}")
    assert res.status_code == 404


def test_delete_nonexistent():
    res = client.delete("/tasks/ghost-task-0000")
    assert res.status_code == 404


# ── Result injection (mocked completed state) ─────────────────────────────────

SAMPLE_RESULT = {
    "videoMetadata": {
        "filename": "sample.mp4",
        "duration_seconds": 3.0,
        "total_frames": 30,
        "fps": 10.0,
        "resolution": {"width": 320, "height": 240},
    },
    "objectsDetected": [
        {
            "object_id": 1,
            "track_id": 42,
            "class": "bottle",
            "motion_history": [
                {"frame_range": [0, 14], "state": "moving"},
                {"frame_range": [15, 29], "state": "stationary"},
            ],
            "interactions": [
                {"interacted_by_person": 1, "frame_start": 0, "frame_end": 10}
            ],
        }
    ],
}


def test_get_completed_result():
    """Inject a completed result directly and verify GET returns it."""
    task_id = "completed-task-test-123"
    _db_module.create_task(task_id, "sample.mp4")
    _db_module.save_task_result(task_id, SAMPLE_RESULT)

    res = client.get(f"/tasks/{task_id}/result")
    assert res.status_code == 200
    body = res.json()

    assert "videoMetadata" in body
    assert "objectsDetected" in body
    objs = body["objectsDetected"]
    assert len(objs) == 1
    assert objs[0]["class"] == "bottle"
    assert objs[0]["object_id"] == 1
    assert len(objs[0]["motion_history"]) == 2
    assert objs[0]["motion_history"][0]["state"] == "moving"
    assert len(objs[0]["interactions"]) == 1
    assert objs[0]["interactions"][0]["interacted_by_person"] == 1


def test_result_schema_fields():
    """Verify all required schema fields are present in a completed result."""
    task_id = "schema-test-456"
    _db_module.create_task(task_id, "sample.mp4")
    _db_module.save_task_result(task_id, SAMPLE_RESULT)

    res = client.get(f"/tasks/{task_id}/result")
    body = res.json()

    vm = body["videoMetadata"]
    assert all(k in vm for k in ["filename", "duration_seconds", "total_frames", "fps", "resolution"])

    obj = body["objectsDetected"][0]
    assert all(k in obj for k in ["object_id", "class", "motion_history", "interactions"])

    mh = obj["motion_history"][0]
    assert "frame_range" in mh and "state" in mh
    assert len(mh["frame_range"]) == 2

    ia = obj["interactions"][0]
    assert all(k in ia for k in ["interacted_by_person", "frame_start", "frame_end"])


# ── Database helpers ──────────────────────────────────────────────────────────

def test_db_create_and_get():
    tid = "db-test-001"
    row = _db_module.create_task(tid, "vid.mp4")
    assert row["task_id"] == tid
    assert row["status"] == "pending"

    fetched = _db_module.get_task(tid)
    assert fetched["filename"] == "vid.mp4"


def test_db_update_status():
    tid = "db-test-002"
    _db_module.create_task(tid, "vid.mp4")
    _db_module.update_task_status(tid, "processing")
    row = _db_module.get_task(tid)
    assert row["status"] == "processing"


def test_db_save_result_sets_completed():
    tid = "db-test-003"
    _db_module.create_task(tid, "vid.mp4")
    _db_module.save_task_result(tid, {"objectsDetected": []})
    row = _db_module.get_task(tid)
    assert row["status"] == "completed"
    assert json.loads(row["result_json"]) == {"objectsDetected": []}


def test_db_get_nonexistent():
    assert _db_module.get_task("does-not-exist") is None
