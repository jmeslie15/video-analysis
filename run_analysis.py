import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from processor import process_video  # type: ignore

VIDEO_PATH  = "tests/Sample Installation Video.mp4"
OUTPUT_PATH = "output.json"

# LM Studio — must be running with vision model loaded
LM_STUDIO_URL   = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "qwen/qwen3-8b"   # match model name shown in LM Studio

print(f"Analysing  : {VIDEO_PATH}")
print(f"Detector   : OWLv2 (google/owlv2-large-patch14-ensemble)")
print(f"LM Studio  : {LM_STUDIO_URL}  [{LM_STUDIO_MODEL}]")
print()

result = process_video(VIDEO_PATH)

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

sa = result.get("sceneAnalysis", {})
an = result.get("actionNarrative", {})

print(f"Scene      : {sa.get('domain', '—')} — {sa.get('description', '')}")
print(f"Summary    : {an.get('scene_summary', '—')}")
print()
print(f"Objects detected : {len(result['objectsDetected'])}")
print(f"Duration         : {result['videoMetadata']['duration_seconds']}s")
print(f"Frames           : {result['videoMetadata']['total_frames']}")
print()
for obj in result["objectsDetected"]:
    interacted = "✓ interacted" if obj["interactions"] else "  no interaction"
    first = obj["motion_history"][0]["frame_range"][0] if obj["motion_history"] else 0
    last  = obj["motion_history"][-1]["frame_range"][1] if obj["motion_history"] else 0
    print(f"  [{obj['object_id']}] {obj['class']:<30} {interacted}  (frames {first}–{last})")

print(f"\nSaved to {OUTPUT_PATH}")