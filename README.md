# Video Analysis Prototype

## Overview

This prototype combines OWLv2 and a locally hosted LLM to perform automated video analysis. The system detects and tracks objects, identifies person-object interactions, classifies motion, and generates a structured narrative describing activity within the video.

The goal is to evaluate a fully local video understanding pipeline without relying on external AI APIs.

---

## Architecture

text Video   ↓ LLM Scene Analysis (LM Studio)   ↓ Dynamic Detection Configuration   ↓ OWLv2 Object Detection   ↓ IoU-Based Tracking   ↓ Motion Classification   ↓ Interaction Detection   ↓ Action Narrative Generation   ↓ Structured JSON Output 

---

## Key Components

### Scene Analysis

The first stage uses a local LLM (via LM Studio) to analyze the video and generate:

- Domain classification
- Scene description
- Detection vocabulary
- Person proxy classes
- Processing thresholds

This allows the detector to adapt to different environments without hardcoded object lists.

### Object Detection & Tracking

OWLv2 (google/owlv2-base-patch16) performs open-vocabulary detection using the dynamically generated class list.

Detections are tracked across frames using a lightweight IoU-based tracker that assigns persistent object IDs.

### Motion & Interaction Analysis

For each tracked object:

- Motion history is classified as moving or stationary
- Person-object interactions are detected using proximity scoring
- Optional keyframes are generated on motion-state transitions

### Action Narrative

Tracking and interaction events are converted into a structured narrative that summarizes:

- Objects present
- Objects moved
- Interaction timelines
- Overall scene activity

---

## Project Structure

text video-analysis/ ├── backend/ │   ├── processor.py │   ├── model_selector.py │   ├── math_helpers.py │   ├── database.py │   ├── schemas.py │   └── main.py │ ├── tests/ │   ├── conftest.py │   ├── test_api.py │   └── test_math_helpers.py │ ├── run_analysis.py ├── output.json └── requirements.txt 

---

## Running the Prototype

Prerequisites:

- Python 3.10+
- LM Studio running locally
- qwen/qwen3-8b (or compatible model) loaded

Install dependencies:

bash pip install -r requirements.txt 

Run:

bash python run_analysis.py 

The pipeline will process the input video and generate a structured analysis in output.json.

---

## Testing

The project includes:

- Unit tests for geometry, motion classification, and interaction logic (test_math_helpers.py)
- API and integration tests (test_api.py)

Run:

bash pytest tests -v 

---

## Prototype Objective

Evaluate the feasibility of combining:

- Open-vocabulary object detection (OWLv2)
- Local LLM-driven scene understanding
- Motion and interaction analysis
- Structured video summarization

within a lightweight, fully local processing pipeline.
