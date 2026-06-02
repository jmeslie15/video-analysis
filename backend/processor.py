"""
processor.py — Video analysis pipeline using OWLv2 for open-vocabulary detection.

Pipeline stages
---------------
1. VideoReader         – iterate frames, expose metadata
2. ObjectDetector      – OWLv2 (Google) open-vocabulary detector + IoU tracker
3. HumanTracker        – isolate person-proxy detections (LLM-supplied classes)
4. MotionClassifier    – stationary / moving per object via sliding window
5. InteractionDetector – proximity-based person <-> object interaction
6. KeyframeExtractor   – save JPEG keyframes on motion state transitions
7. ActionNarrative     – build LLM-readable action summary
8. ResultBuilder       – assemble structured JSON payload

OWLv2 advantages over YOLO-World:
- Accepts any free-form text query (no COCO class restriction)
- Understands "spectrophotometer", "ethernet cable", "lab coat" natively
- No alias hacks needed — query exactly what you want to find
- Transformer architecture fuses vision + language inside the model
"""
from __future__ import annotations

import cv2
import os
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Generator
from PIL import Image

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from math_helpers import (
    bbox_center,
    classify_motion_history,
    proximity_score,
    run_length_encode_interactions,
    is_moving,
)
from model_selector import select_config, PipelineConfig

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Large ensemble model = best accuracy; base model = faster
OWLV2_MODEL_ID    = "google/owlv2-base-patch16"
PERSON_CLASS_NAME    = "person"
CONFIDENCE_THRESHOLD = 0.10    # OWLv2 scores are lower than YOLO — 0.10 is appropriate

# Process every Nth frame for speed; positions interpolated for skipped frames
# 1 = every frame (slowest, most accurate)
# 3 = every 3rd frame (good balance on CPU)
FRAME_SAMPLE_RATE    = 6

DEFAULT_CLASSES = [
    "person", "cable", "wire", "laptop", "monitor",
    "keyboard", "bottle", "machine", "table", "chair",
]


# ── IoU tracker ───────────────────────────────────────────────────────────────

class _IoUTracker:
    """
    Lightweight IoU-based tracker that assigns stable integer IDs to detections
    across frames by matching bounding boxes via intersection-over-union.
    OWLv2 has no built-in tracker so we provide one here.
    """

    def __init__(self, iou_threshold: float = 0.30, max_lost: int = 20):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self._next_id      = 1
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _iou(a: list, b: list) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0

    def update(self, detections: list[dict]) -> list[dict]:
        # Age all existing tracks
        for t in self._tracks.values():
            t["lost"] += 1

        matched: set[int] = set()
        results: list[dict] = []

        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for tid, track in self._tracks.items():
                if tid in matched:
                    continue
                # Only match same class
                if track.get("class_name") != det["class_name"]:
                    continue
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                self._tracks[best_id].update({
                    "bbox": det["bbox"], "lost": 0,
                    "class_name": det["class_name"],
                })
                matched.add(best_id)
                track_id = best_id
            else:
                track_id = self._next_id
                self._tracks[track_id] = {
                    "bbox": det["bbox"], "lost": 0,
                    "class_name": det["class_name"],
                }
                self._next_id += 1

            results.append({**det, "track_id": track_id})

        # Remove stale tracks
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t["lost"] <= self.max_lost
        }
        return results


# ── Stage 1 – VideoReader ─────────────────────────────────────────────────────

class VideoReader:
    def __init__(self, video_path: str):
        self.path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0

    @property
    def diagonal(self) -> float:
        return float(np.sqrt(self.width ** 2 + self.height ** 2))

    def metadata(self) -> dict:
        return {
            "filename": os.path.basename(self.path),
            "duration_seconds": round(self.duration_seconds, 3),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 3),
            "resolution": {"width": self.width, "height": self.height},
        }

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def __del__(self):
        self._cap.release()


# ── Stage 2 – ObjectDetector (OWLv2) ─────────────────────────────────────────

class ObjectDetector:
    """
    OWLv2 open-vocabulary object detector.

    Takes a list of plain-English class names and detects them in each frame.
    No aliases or COCO restrictions — queries exactly what the LLM specifies.
    Model weights (~1.2GB) download automatically from HuggingFace on first run.
    """

    def __init__(
        self,
        classes: list[str],
        conf: float = CONFIDENCE_THRESHOLD,
        person_proxies: set[str] | None = None,
    ):
        self.conf           = conf
        self.person_proxies = person_proxies or {"person"}
        self.tracker        = _IoUTracker()
        self._load_model()
        self.set_classes(classes)

    def _load_model(self) -> None:
        log.info("Loading OWLv2 (%s)...", OWLV2_MODEL_ID)
        self.processor = Owlv2Processor.from_pretrained(OWLV2_MODEL_ID)
        self.model     = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL_ID)
        self.device    = "mps" if torch.backends.mps.is_available() else \
                         "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device) # type: ignore
        self.model.eval()
        log.info("OWLv2 loaded on %s", self.device)

    def set_classes(self, classes: list[str]) -> None:
        """
        Update detection vocabulary.
        Classes are passed as free-form text — no restrictions.
        Always ensure at least one person-like class is present.
        """
        full = list(dict.fromkeys(classes))
        if not any(c.lower() in self.person_proxies for c in full):
            full.insert(0, PERSON_CLASS_NAME)
        self.classes = full
        # Pre-tokenise text queries (done once, reused every frame)
        self._text_queries = [[f"a photo of a {c}" for c in self.classes]]
        log.info("OWLv2 classes (%d): %s", len(self.classes), self.classes)

    def _nms(self, detections: list[dict], iou_thresh: float = 0.5) -> list[dict]:
        """Non-maximum suppression across all classes."""
        if not detections:
            return []
        boxes  = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return [detections[i] for i in keep]

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run OWLv2 on one BGR frame.
        Returns list of dicts: track_id, class_name, bbox [x1,y1,x2,y2], confidence.
        """
        h, w = frame.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(
            text=self._text_queries,
            images=pil_image,
            return_tensors="pt", # type: ignore
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: convert to absolute pixel boxes
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.conf,
            target_sizes=target_sizes, # type: ignore
        )[0]

        detections: list[dict] = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            cls_idx = int(label_idx.item())
            if cls_idx >= len(self.classes):
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "track_id":   -1,
                "class_name": self.classes[cls_idx],
                "bbox":       [x1, y1, x2, y2],
                "confidence": float(score.item()),
            })

        detections = self._nms(detections)
        return self.tracker.update(detections)


# ── Stage 3 – HumanTracker ────────────────────────────────────────────────────

class HumanTracker:
    """
    Separates person detections from objects.
    Uses the LLM-supplied person_proxy_classes so "lab coat",
    "scientist", etc. are all treated as persons.
    """

    def __init__(self, person_proxy_classes: list[str] | None = None):
        raw = set(person_proxy_classes or [PERSON_CLASS_NAME])
        # Normalise to lowercase for robust matching
        self.proxy_classes = {c.lower().strip() for c in raw}
        self.proxy_classes.add("person")
        log.info("Person proxy classes: %s", self.proxy_classes)

    def _is_person(self, det: dict) -> bool:
        return det["class_name"].lower().strip() in self.proxy_classes

    def split(self, detections: list[dict]) -> tuple[list[dict], list[dict]]:
        persons = [d for d in detections if self._is_person(d)]
        objects = [d for d in detections if not self._is_person(d)]
        return persons, objects


# ── Stage 4 & 5 – per-object accumulators ────────────────────────────────────

class ObjectState:
    def __init__(self, track_id: int, class_name: str):
        self.track_id   = track_id
        self.class_name = class_name
        self.frame_centers: dict[int, tuple[float, float]] = {}
        self.frame_bboxes:  dict[int, list[float]]         = {}
        self.last_motion_state: str | None = None


class PersonState:
    def __init__(self, track_id: int):
        self.track_id      = track_id
        self.frame_bboxes:  dict[int, list[float]]         = {}
        self.frame_centers: dict[int, tuple[float, float]] = {}


# ── Stage 6 – KeyframeExtractor ──────────────────────────────────────────────

class KeyframeExtractor:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        task_dir.mkdir(parents=True, exist_ok=True)
        self.saved: list[dict] = []

    def save(self, frame: np.ndarray, frame_idx: int, reason: str) -> str:
        filename = f"frame_{frame_idx:06d}_{reason}.jpg"
        path     = self.task_dir / filename
        cv2.imwrite(str(path), frame)
        self.saved.append({"frame": frame_idx, "reason": reason, "path": str(path)})
        return str(path)


# ── Stage 7 – Action narrative ────────────────────────────────────────────────

def build_action_narrative(
    objects_detected: list[dict],
    persons: dict[int, PersonState],
    fps: float,
    total_frames: int,
) -> dict:
    """
    Build a structured action summary an LLM can read to understand
    what happened in the video without watching it.
    """

    def ft(f: int) -> str:
        return f"{f / fps:.2f}s (frame {f})" if fps else f"frame {f}"

    def dur(s: int, e: int) -> str:
        return f"{(e - s) / fps:.2f}s" if fps else f"{e - s} frames"

    motion_events: list[dict] = []
    for obj in objects_detected:
        for mh in obj["motion_history"]:
            if mh["state"] == "moving":
                motion_events.append({
                    "object":     obj["class"],
                    "object_id":  obj["object_id"],
                    "event":      "moving",
                    "from_frame": mh["frame_range"][0],
                    "to_frame":   mh["frame_range"][1],
                    "start_time": ft(mh["frame_range"][0]),
                    "end_time":   ft(mh["frame_range"][1]),
                    "duration":   dur(mh["frame_range"][0], mh["frame_range"][1]),
                })
    motion_events.sort(key=lambda e: e["from_frame"])

    interaction_events: list[dict] = []
    for obj in objects_detected:
        for ia in obj["interactions"]:
            interaction_events.append({
                "person_id":  ia["interacted_by_person"],
                "object":     obj["class"],
                "object_id":  obj["object_id"],
                "from_frame": ia["frame_start"],
                "to_frame":   ia["frame_end"],
                "start_time": ft(ia["frame_start"]),
                "end_time":   ft(ia["frame_end"]),
                "duration":   dur(ia["frame_start"], ia["frame_end"]),
            })
    interaction_events.sort(key=lambda e: e["from_frame"])

    person_summaries: list[dict] = []
    for pid, person in persons.items():
        touched = [e["object"] for e in interaction_events if e["person_id"] == pid]
        first   = min(person.frame_bboxes) if person.frame_bboxes else 0
        last    = max(person.frame_bboxes) if person.frame_bboxes else 0
        person_summaries.append({
            "person_id":               pid,
            "first_seen_time":         ft(first),
            "last_seen_time":          ft(last),
            "objects_interacted_with": list(dict.fromkeys(touched)),
            "total_interactions":      len(touched),
        })

    object_summaries: list[dict] = []
    for obj in objects_detected:
        states     = [mh["state"] for mh in obj["motion_history"]]
        first_f    = obj["motion_history"][0]["frame_range"][0] if obj["motion_history"] else 0
        last_f     = obj["motion_history"][-1]["frame_range"][1] if obj["motion_history"] else 0
        object_summaries.append({
            "object_id":           obj["object_id"],
            "class":               obj["class"],
            "first_seen":          ft(first_f),
            "last_seen":           ft(last_f),
            "was_moved":           "moving" in states,
            "was_interacted_with": len(obj["interactions"]) > 0,
            "motion_state_changes": len(obj["motion_history"]),
            "interaction_count":   len(obj["interactions"]),
        })

    # Plain-text summary for LLM prompt injection
    moving  = [s["class"] for s in object_summaries if s["was_moved"]]
    touched = [s["class"] for s in object_summaries if s["was_interacted_with"]]
    parts   = [
        f"Video duration: {total_frames / fps:.1f}s ({total_frames} frames at {fps}fps).",
        f"{len(person_summaries)} person(s) detected.",
        f"{len(objects_detected)} object(s) tracked: {', '.join(s['class'] for s in object_summaries)}." if objects_detected else "No objects tracked.",
    ]
    if moving:
        parts.append(f"Objects that moved: {', '.join(moving)}.")
    if touched:
        parts.append(f"Objects interacted with: {', '.join(touched)}.")
    for ie in interaction_events:
        parts.append(
            f"Person {ie['person_id']} interacted with {ie['object']} "
            f"from {ie['start_time']} to {ie['end_time']} ({ie['duration']})."
        )
    for me in motion_events:
        parts.append(
            f"{me['object']} was moving from {me['start_time']} "
            f"to {me['end_time']} ({me['duration']})."
        )

    return {
        "scene_summary":        " ".join(parts),
        "person_summaries":     person_summaries,
        "object_summaries":     object_summaries,
        "motion_timeline":      motion_events,
        "interaction_timeline": interaction_events,
    }


# ── Stage 8 – ResultBuilder ──────────────────────────────────────────────────

class ResultBuilder:
    @staticmethod
    def build(
        video_meta: dict,
        objects: dict[int, ObjectState],
        persons: dict[int, PersonState],
        keyframes: list[dict],
        frame_diagonal: float,
        config: PipelineConfig | None = None,
    ) -> dict:
        cfg = config or PipelineConfig()
        objects_detected = []
        obj_id_counter   = 1

        for track_id, obj in objects.items():
            if len(obj.frame_centers) < cfg.min_frame_persistence:
                log.debug(
                    "Dropping track %d (%s) — %d frames < min %d",
                    track_id, obj.class_name,
                    len(obj.frame_centers), cfg.min_frame_persistence,
                )
                continue

            motion_history = classify_motion_history(
                obj.frame_centers,
                window=cfg.motion_window,
                threshold_px=cfg.motion_threshold_px,
                ratio=cfg.motion_ratio,
            )

            interactions = []
            for person_tid, person in persons.items():
                flags: dict[int, bool] = {}
                for frame_idx, obj_bbox in obj.frame_bboxes.items():
                    p_bbox = person.frame_bboxes.get(frame_idx)
                    if p_bbox is None:
                        flags[frame_idx] = False
                        continue
                    score = proximity_score(p_bbox, obj_bbox, frame_diagonal)
                    flags[frame_idx] = score >= cfg.proximity_threshold

                for start, end in run_length_encode_interactions(flags):
                    interactions.append({
                        "interacted_by_person": person_tid,
                        "frame_start": start,
                        "frame_end":   end,
                    })

            objects_detected.append({
                "object_id":      obj_id_counter,
                "track_id":       track_id,
                "class":          obj.class_name,
                "motion_history": motion_history,
                "interactions":   interactions,
            })
            obj_id_counter += 1

        payload: dict = {
            "videoMetadata":   video_meta,
            "objectsDetected": objects_detected,
        }
        if keyframes:
            payload["keyFrames"] = keyframes
        return payload


# ── Top-level pipeline entry point ────────────────────────────────────────────

def process_video(
    video_path: str,
    keyframes_root: Path | None = None,
    task_id: str | None = None,
    progress_callback=None,
    api_key: str | None = None,
) -> dict:
    """
    Full pipeline: scene-analysis -> OWLv2 detect -> track -> classify -> output.
    """
    # Stage 0: LLM scene analysis -> config + custom class list
    log.info("Stage 0: Analysing first frame...")
    config = select_config(video_path, api_key=api_key)
    log.info("Domain: '%s'", config.domain)

    detection_classes = config.custom_classes if config.custom_classes else DEFAULT_CLASSES
    person_proxies    = set(c.lower() for c in (config.person_proxy_classes or ["person"]))
    log.info("Detection classes (%d): %s", len(detection_classes), detection_classes)
    log.info("Person proxies: %s", person_proxies)

    reader        = VideoReader(video_path)
    detector      = ObjectDetector(
        classes=detection_classes,
        conf=config.confidence,
        person_proxies=person_proxies,
    )
    human_tracker = HumanTracker(person_proxy_classes=config.person_proxy_classes)

    keyframe_extractor: KeyframeExtractor | None = None
    if keyframes_root and task_id:
        keyframe_extractor = KeyframeExtractor(Path(keyframes_root) / task_id)

    obj_states:    dict[int, ObjectState] = {}
    person_states: dict[int, PersonState] = {}

    total = reader.total_frames or 1
    log.info("Processing %d frames (sampling every %d)...", total, FRAME_SAMPLE_RATE)

    for frame_idx, frame in reader.frames():
        # Sample frames for speed; tracker handles continuity across gaps
        if frame_idx % FRAME_SAMPLE_RATE != 0:
            continue

        detections = detector.detect(frame)

        # Apply denylist
        detections = [
            d for d in detections
            if d["class_name"] not in config.class_denylist
        ]

        persons, objects = human_tracker.split(detections)

        for p in persons:
            pid = p["track_id"]
            if pid not in person_states:
                person_states[pid] = PersonState(pid)
            person_states[pid].frame_bboxes[frame_idx]  = p["bbox"]
            cx, cy = bbox_center(*p["bbox"])
            person_states[pid].frame_centers[frame_idx] = (cx, cy)

        for obj in objects:
            tid = obj["track_id"]
            if tid not in obj_states:
                obj_states[tid] = ObjectState(tid, obj["class_name"])
            cx, cy = bbox_center(*obj["bbox"])
            obj_states[tid].frame_centers[frame_idx] = (cx, cy)
            obj_states[tid].frame_bboxes[frame_idx]  = obj["bbox"]

        if keyframe_extractor:
            for tid, os_ in obj_states.items():
                if frame_idx in os_.frame_centers and len(os_.frame_centers) >= 2:
                    recent    = sorted(os_.frame_centers)[-10:]
                    centers   = [os_.frame_centers[f] for f in recent]
                    cur_state = "moving" if is_moving(
                        centers, config.motion_threshold_px, config.motion_ratio
                    ) else "stationary"
                    if os_.last_motion_state and os_.last_motion_state != cur_state:
                        keyframe_extractor.save(frame, frame_idx, f"transition_obj{tid}")
                    os_.last_motion_state = cur_state

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(int(frame_idx / total * 90))

    log.info(
        "Tracking complete — %d objects, %d persons detected",
        len(obj_states), len(person_states),
    )

    keyframes = keyframe_extractor.saved if keyframe_extractor else []
    result = ResultBuilder.build(
        video_meta=reader.metadata(),
        objects=obj_states,
        persons=person_states,
        keyframes=keyframes,
        frame_diagonal=reader.diagonal,
        config=config,
    )

    result["actionNarrative"] = build_action_narrative(
        objects_detected=result["objectsDetected"],
        persons=person_states,
        fps=reader.fps,
        total_frames=reader.total_frames,
    )

    result["sceneAnalysis"] = {
        "domain":      config.domain,
        "description": config.scene_description,
        "rationale":   config.rationale,
        "config": {
            "model":                 OWLV2_MODEL_ID,
            "confidence":            config.confidence,
            "detection_classes":     detection_classes,
            "person_proxy_classes":  config.person_proxy_classes,
            "frame_sample_rate":     FRAME_SAMPLE_RATE,
            "motion_threshold_px":   config.motion_threshold_px,
            "min_frame_persistence": config.min_frame_persistence,
        },
    }

    if progress_callback:
        progress_callback(100)

    return result