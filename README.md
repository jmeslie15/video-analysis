# Video Analysis Prototype

## Overview

This prototype evaluates a fully local AI-powered video analysis pipeline that combines **OWLv2** for open-vocabulary object detection with a **locally hosted LLM** running through **LM Studio**.

The system automatically analyzes a video, generates a scene-specific detection configuration, detects and tracks objects, classifies object motion, identifies person-object interactions, and produces a structured narrative describing activity within the scene.

Unlike traditional object detection systems that rely on predefined class lists, the pipeline uses an LLM to dynamically generate relevant detection vocabulary and processing parameters based on the video's context. OWLv2 then performs detection using those natural-language queries, allowing the system to adapt to different domains without manual configuration.

The resulting output includes:

- Video metadata
- Scene analysis and classification
- Detected and tracked objects
- Motion history
- Person-object interaction timelines
- Structured action narrative
- Optional keyframes

The project serves as a proof of concept for combining computer vision and local language models to create an adaptable, fully local video-understanding workflow without external AI services.

---

## Architecture

```text
Video
  ↓
LLM Scene Analysis (LM Studio)
  ↓
Dynamic Detection Configuration
  ↓
OWLv2 Object Detection
  ↓
IoU-Based Tracking
  ↓
Motion Classification
  ↓
Interaction Detection
  ↓
Action Narrative Generation
  ↓
Structured JSON Output
```

## Project Structure

```text
video-analysis/
├── backend/
│   ├── processor.py
│   ├── model_selector.py
│   ├── math_helpers.py
│   ├── database.py
│   ├── schemas.py
│   └── main.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_math_helpers.py
│
├── run_analysis.py
├── output.json
└── requirements.txt
```

## Running the Prototype

### Prerequisites

- Python 3.10+
- LM Studio
- A compatible model loaded in LM Studio

Current configuration:

```text
LM Studio Endpoint:
http://localhost:1234/v1

Model:
qwen/qwen3-8b
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Analysis

```bash
python run_analysis.py
```

The pipeline will process the input video and save the resulting analysis to `output.json`.

## Testing

The project includes both unit and integration tests.

Run the test suite:

```bash
pytest tests -v
```
