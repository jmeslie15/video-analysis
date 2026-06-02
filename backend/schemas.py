"""
schemas.py — Pydantic models for request validation and response serialisation.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class MotionInterval(BaseModel):
    frame_range: list[int] = Field(..., min_length=2, max_length=2)
    state: str  # "moving" | "stationary"


class Interaction(BaseModel):
    interacted_by_person: int
    frame_start: int
    frame_end: int


class DetectedObject(BaseModel):
    object_id: int
    track_id: int
    class_: str = Field(..., alias="class")
    motion_history: list[MotionInterval]
    interactions: list[Interaction]

    class Config:
        populate_by_name = True


class VideoMetadata(BaseModel):
    filename: str
    duration_seconds: float
    total_frames: int
    fps: float
    resolution: dict[str, int]


class KeyFrame(BaseModel):
    frame: int
    reason: str
    path: str
    confidence: Optional[float] = None


class AnalysisResult(BaseModel):
    videoMetadata: VideoMetadata
    objectsDetected: list[DetectedObject]
    keyFrames: Optional[list[KeyFrame]] = None


class TaskStatus(BaseModel):
    task_id: str
    filename: str
    status: str          # pending | processing | completed | failed
    created_at: str
    updated_at: str
    error: Optional[str] = None


class TaskCreated(BaseModel):
    task_id: str
    message: str
    status: str = "pending"


class TaskList(BaseModel):
    tasks: list[TaskStatus]
    total: int
