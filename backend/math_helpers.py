"""
math_helpers.py — Pure geometry and signal-processing utilities.

All functions are stateless and unit-testable. No CV or ML imports here so
the module can be tested without heavy dependencies.
"""
from __future__ import annotations
import math
from typing import Sequence


# ── Bounding-box helpers ─────────────────────────────────────────────────────

def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Return the (cx, cy) centre of an axis-aligned bounding box."""
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Area of a bounding box; returns 0 for degenerate boxes."""
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Intersection-over-Union between two bounding boxes.

    Parameters
    ----------
    a, b : [x1, y1, x2, y2]

    Returns
    -------
    float in [0, 1]
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = bbox_area(*a) + bbox_area(*b) - inter
    return inter / union if union > 0 else 0.0


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ── Motion classification ────────────────────────────────────────────────────

def displacement_series(centers: list[tuple[float, float]]) -> list[float]:
    """
    Compute frame-to-frame displacement magnitudes from a list of centres.

    Returns a list of length ``len(centers) - 1``.
    """
    return [euclidean_distance(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]


def is_moving(
    centers: list[tuple[float, float]],
    threshold_px: float = 4.0,
    ratio: float = 0.3,
) -> bool:
    """
    Classify an object as *moving* if at least `ratio` of consecutive-frame
    displacements exceed `threshold_px` pixels.

    Parameters
    ----------
    centers        : ordered list of (cx, cy) observations
    threshold_px   : minimum pixel displacement to count as "moved"
    ratio          : fraction of frames that must exceed the threshold

    Returns
    -------
    True if the object is classified as moving.
    """
    if len(centers) < 2:
        return False
    disps = displacement_series(centers)
    moving_count = sum(1 for d in disps if d > threshold_px)
    return (moving_count / len(disps)) >= ratio


def classify_motion_history(
    frame_centers: dict[int, tuple[float, float]],
    window: int = 30,
    threshold_px: float = 4.0,
    ratio: float = 0.3,
) -> list[dict]:
    """
    Produce a run-length encoded motion history across frame ranges.

    Parameters
    ----------
    frame_centers : {frame_index: (cx, cy)}  — may have gaps
    window        : number of frames per analysis window
    threshold_px  : motion threshold forwarded to `is_moving`
    ratio         : motion ratio forwarded to `is_moving`

    Returns
    -------
    List of ``{"frame_range": [start, end], "state": "moving"|"stationary"}``
    entries with adjacent identical states merged.
    """
    if not frame_centers:
        return []

    frames = sorted(frame_centers)
    min_f, max_f = frames[0], frames[-1]

    raw: list[tuple[int, int, str]] = []  # (start, end, state)
    start = min_f
    while start <= max_f:
        end = min(start + window - 1, max_f)
        window_frames = [f for f in frames if start <= f <= end]
        centers = [frame_centers[f] for f in window_frames]
        state = "moving" if is_moving(centers, threshold_px, ratio) else "stationary"
        raw.append((start, end, state))
        start = end + 1

    # Merge consecutive windows with the same state
    merged: list[dict] = []
    for seg_start, seg_end, state in raw:
        if merged and merged[-1]["state"] == state:
            merged[-1]["frame_range"][1] = seg_end
        else:
            merged.append({"frame_range": [seg_start, seg_end], "state": state})

    return merged


# ── Interaction helpers ──────────────────────────────────────────────────────

def proximity_score(
    person_bbox: Sequence[float],
    object_bbox: Sequence[float],
    frame_diagonal: float,
) -> float:
    """
    Continuous proximity score in [0, 1].

    Score = 1 when boxes overlap (IoU > 0), decays with centre-to-centre
    distance, reaching 0 at ``frame_diagonal / 4``.

    Parameters
    ----------
    person_bbox, object_bbox : [x1, y1, x2, y2]
    frame_diagonal           : diagonal of the video frame in pixels
    """
    iou = bbox_iou(person_bbox, object_bbox)
    if iou > 0:
        return 1.0

    pc = bbox_center(*person_bbox)
    oc = bbox_center(*object_bbox)
    dist = euclidean_distance(pc, oc)
    max_dist = frame_diagonal / 4.0
    return max(0.0, 1.0 - dist / max_dist)


def run_length_encode_interactions(
    frame_flags: dict[int, bool],
) -> list[tuple[int, int]]:
    """
    Convert a {frame: bool} interaction map into ``[(start, end), …]`` intervals.

    Only *True* frames are returned.
    """
    if not frame_flags:
        return []

    frames = sorted(frame_flags)
    intervals: list[tuple[int, int]] = []
    seg_start: int | None = None

    for i, f in enumerate(frames):
        if frame_flags[f]:
            if seg_start is None:
                seg_start = f
        else:
            if seg_start is not None:
                intervals.append((seg_start, frames[i - 1]))
                seg_start = None

    if seg_start is not None:
        intervals.append((seg_start, frames[-1]))

    return intervals
