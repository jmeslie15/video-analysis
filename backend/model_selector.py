"""
model_selector.py — Analyse the first frame of a video using a locally hosted
vision LLM via LM Studio's OpenAI-compatible API.

Setup:
  1. Open LM Studio and load your vision model (e.g. qwen/qwen3.5-9b)
  2. Start the local server: LM Studio → Local Server → Start
  3. Default endpoint: http://localhost:1234/v1
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from openai import OpenAI

log = logging.getLogger(__name__)

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL    = "qwen/qwen3-8b"


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    domain: str              = "general"
    scene_description: str   = ""
    confidence: float        = 0.45
    custom_classes: list[str]        = field(default_factory=list)
    person_proxy_classes: list[str]  = field(default_factory=lambda: ["person"])
    class_allowlist: list[str]       = field(default_factory=list)
    class_denylist:  list[str]       = field(default_factory=lambda: [
        "car", "truck", "bus", "train", "airplane", "boat", "motorcycle", "bicycle",
    ])
    motion_threshold_px: float = 6.0
    motion_ratio:        float = 0.25
    motion_window:       int   = 30
    proximity_threshold: float = 0.25
    min_frame_persistence: int = 20
    rationale: str = ""

    def summary(self) -> str:
        return "\n".join([
            f"Domain        : {self.domain}",
            f"Description   : {self.scene_description}",
            f"Confidence    : {self.confidence}",
            f"Custom classes: {self.custom_classes or '(none)'}",
            f"Person proxies: {self.person_proxy_classes}",
            f"Denylist      : {self.class_denylist}",
            f"Motion px thr : {self.motion_threshold_px}",
            f"Motion ratio  : {self.motion_ratio}",
            f"Min persist.  : {self.min_frame_persistence} frames",
            f"Rationale     : {self.rationale}",
        ])


# ── Frame helpers ─────────────────────────────────────────────────────────────

def extract_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    return frame


def frame_to_base64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT = """/no_think

You are a computer-vision pipeline configurator. Analyse this image and respond
with ONLY a valid JSON object. No markdown, no explanation, no preamble.

Return exactly this JSON structure:

{
  "domain": "medical_lab",
  "scene_description": "one sentence",
  "confidence": 0.60,
  "custom_classes": ["person", "cable", "spectrophotometer"],
  "person_proxy_classes": ["person", "lab coat"],
  "class_allowlist": [],
  "class_denylist": ["car", "truck", "bicycle"],
  "motion_threshold_px": 4.0,
  "motion_ratio": 0.15,
  "motion_window": 30,
  "proximity_threshold": 0.25,
  "min_frame_persistence": 30,
  "rationale": "one sentence"
}

Rules:
- custom_classes: every specific object visible or expected. Be precise:
  "ethernet cable" not "cable", "spectrophotometer" not "machine".
- person_proxy_classes: subset of custom_classes that are human or worn by human.
  Always include at least "person".
- class_denylist: objects impossible in this scene.
- confidence: 0.55-0.70 for clear lab scenes.
- motion_threshold_px: 3-5 for fine hand manipulation.
- min_frame_persistence: 25-35 for lab scenes to suppress flicker.
- Output ONLY the JSON object. No other text whatsoever."""


# ── JSON extraction helpers ───────────────────────────────────────────────────

KNOWN_FIELDS = {
    "domain", "scene_description", "confidence", "custom_classes",
    "person_proxy_classes", "class_allowlist", "class_denylist",
    "motion_threshold_px", "motion_ratio", "motion_window",
    "proximity_threshold", "min_frame_persistence", "rationale",
}


def _extract_json_from_text(text: str) -> str:
    """
    Try multiple strategies to extract a JSON object from raw text.

    Strategy 1 — find a complete {...} block containing "domain"
    Strategy 2 — reconstruct from bullet/field lines in reasoning text
    Strategy 3 — give up and return empty string
    """
    if not text:
        return ""

    # Strategy 1: find outermost {...} block with "domain" key
    # Use a stack to handle nested braces correctly
    for start in range(len(text)):
        if text[start] != '{':
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    if '"domain"' in candidate:
                        try:
                            json.loads(candidate)
                            return candidate          # valid JSON found
                        except json.JSONDecodeError:
                            pass                      # keep searching
                    break

    # Strategy 2: reconstruct from key: value lines (reasoning text)
    field_re = re.compile(
        r'[`*"\']*(' + '|'.join(KNOWN_FIELDS) + r')[`*"\']*\s*:\s*(.+)',
        re.MULTILINE,
    )
    extracted: dict = {}
    for m in field_re.finditer(text):
        key, raw_val = m.group(1), m.group(2).strip().rstrip(".,")
        if key in extracted:
            continue
        try:
            extracted[key] = json.loads(raw_val)
        except json.JSONDecodeError:
            extracted[key] = raw_val.strip('"`\'')

    if "domain" in extracted:
        log.info("Reconstructed config from %d reasoning fields.", len(extracted))
        return json.dumps(extracted)

    return ""


def _parse_response(msg) -> dict:
    """
    Extract and parse JSON from an LLM message object.
    Checks content first, then reasoning_content as fallback.
    """
    sources = [
        ("content",           (msg.content or "").strip()),
        ("reasoning_content", (getattr(msg, "reasoning_content", "") or "").strip()),
    ]

    for source_name, raw in sources:
        if not raw:
            continue

        # Strip <think>...</think> wrapper
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown fences
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$",        "", cleaned).strip()

        json_str = _extract_json_from_text(cleaned)
        if not json_str:
            log.warning("No JSON found in %s.", source_name)
            continue

        try:
            result = json.loads(json_str)
            log.info("Parsed config from '%s'.", source_name)
            return result
        except json.JSONDecodeError as exc:
            log.warning("JSON parse failed in %s: %s", source_name, exc)

    raise RuntimeError("Could not extract valid JSON from any part of the model response.")


# ── LM Studio API call ────────────────────────────────────────────────────────

def call_lm_studio_vision(frame: np.ndarray, base_url: str, model: str) -> dict:
    client    = OpenAI(base_url=base_url, api_key="lm-studio")
    image_b64 = frame_to_base64_jpeg(frame)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=4096,    # large enough for reasoning + full JSON output
    )

    finish = response.choices[0].finish_reason
    if finish == "length":
        log.warning("Model hit token limit (finish_reason=length). Output may be truncated.")

    return _parse_response(response.choices[0].message)


# ── Config builder ────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def build_config_from_response(resp: dict) -> PipelineConfig:
    custom  = [str(c) for c in resp.get("custom_classes", [])]
    proxies = [str(c) for c in resp.get("person_proxy_classes", ["person"])]
    if "person" not in proxies:
        proxies.insert(0, "person")

    return PipelineConfig(
        domain               = str(resp.get("domain", "general")),
        scene_description    = str(resp.get("scene_description", "")),
        confidence = _clamp(float(resp.get("confidence", 0.10)), 0.05, 0.15),
        custom_classes       = custom,
        person_proxy_classes = proxies,
        class_allowlist      = [str(c) for c in resp.get("class_allowlist", [])],
        class_denylist       = [str(c) for c in resp.get("class_denylist", [])],
        motion_threshold_px  = _clamp(float(resp.get("motion_threshold_px", 6.0)), 1.0, 30.0),
        motion_ratio         = _clamp(float(resp.get("motion_ratio", 0.25)), 0.05, 0.60),
        motion_window        = _clamp(int(resp.get("motion_window", 30)), 10, 90), # type: ignore
        proximity_threshold  = _clamp(float(resp.get("proximity_threshold", 0.25)), 0.10, 0.60),
        min_frame_persistence= _clamp(int(resp.get("min_frame_persistence", 10)), 3, 20), # type: ignore
        rationale            = str(resp.get("rationale", "")),
    )


# ── Public entry point ────────────────────────────────────────────────────────

def select_config(
    video_path: str,
    api_key: str | None = None,
    base_url: str = LM_STUDIO_BASE_URL,
    model: str    = LM_STUDIO_MODEL,
) -> PipelineConfig:
    """
    Analyse the first frame and return a PipelineConfig.
    Falls back to sensible defaults if LM Studio is unreachable or parsing fails.
    """
    log.info("Extracting first frame for scene analysis...")
    try:
        frame = extract_first_frame(video_path)
    except Exception as exc:
        log.warning("Could not extract frame (%s) — using default config.", exc)
        return PipelineConfig()

    log.info("Sending frame to LM Studio (%s @ %s)...", model, base_url)
    try:
        resp   = call_lm_studio_vision(frame, base_url, model)
        config = build_config_from_response(resp)
        log.info("Scene analysis complete:\n%s", config.summary())
        return config
    except Exception as exc:
        log.warning("Scene analysis failed (%s) — using default config.", exc)
        return PipelineConfig()