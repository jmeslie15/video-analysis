"""
tests/test_math_helpers.py — Unit tests for all math / geometry helpers.

Run with:
    cd video-analysis && python -m pytest tests/ -v
"""
import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from math_helpers import (
    bbox_center,
    bbox_area,
    bbox_iou,
    euclidean_distance,
    displacement_series,
    is_moving,
    classify_motion_history,
    proximity_score,
    run_length_encode_interactions,
)


# ── bbox_center ───────────────────────────────────────────────────────────────

class TestBboxCenter:
    def test_unit_square(self):
        assert bbox_center(0, 0, 2, 2) == (1.0, 1.0)

    def test_non_square(self):
        cx, cy = bbox_center(10, 20, 50, 60)
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(40.0)

    def test_zero_area(self):
        assert bbox_center(5, 5, 5, 5) == (5.0, 5.0)


# ── bbox_area ─────────────────────────────────────────────────────────────────

class TestBboxArea:
    def test_normal(self):
        assert bbox_area(0, 0, 4, 5) == pytest.approx(20.0)

    def test_degenerate_width(self):
        assert bbox_area(3, 0, 3, 10) == pytest.approx(0.0)

    def test_inverted_coords(self):
        # x2 < x1 → clamped to 0
        assert bbox_area(10, 0, 5, 5) == pytest.approx(0.0)


# ── bbox_iou ──────────────────────────────────────────────────────────────────

class TestBboxIou:
    def test_identical(self):
        assert bbox_iou([0,0,4,4], [0,0,4,4]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert bbox_iou([0,0,2,2], [3,3,5,5]) == pytest.approx(0.0)

    def test_half_overlap(self):
        # Box A: (0,0)-(4,4)=16; Box B: (2,0)-(6,4)=16; inter=(2,0)-(4,4)=8
        iou = bbox_iou([0,0,4,4], [2,0,6,4])
        # union = 16+16-8 = 24; iou = 8/24 ≈ 0.333
        assert iou == pytest.approx(8/24, rel=1e-4)

    def test_contained(self):
        # small box fully inside large box
        iou = bbox_iou([0,0,10,10], [2,2,4,4])
        # inter=4, union=100+4-4=100; iou=4/100
        assert iou == pytest.approx(4/100, rel=1e-4)


# ── euclidean_distance ────────────────────────────────────────────────────────

class TestEuclideanDistance:
    def test_same_point(self):
        assert euclidean_distance((3, 4), (3, 4)) == 0.0

    def test_3_4_5_triangle(self):
        assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_negative_coords(self):
        assert euclidean_distance((-1, -1), (2, 3)) == pytest.approx(5.0)


# ── displacement_series ───────────────────────────────────────────────────────

class TestDisplacementSeries:
    def test_static(self):
        cs = [(1.0, 1.0)] * 5
        assert all(d == pytest.approx(0.0) for d in displacement_series(cs))

    def test_moving(self):
        cs = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
        disps = displacement_series(cs)
        assert len(disps) == 2
        assert disps[0] == pytest.approx(5.0)
        assert disps[1] == pytest.approx(5.0)

    def test_empty(self):
        assert displacement_series([]) == []

    def test_single(self):
        assert displacement_series([(0, 0)]) == []


# ── is_moving ─────────────────────────────────────────────────────────────────

class TestIsMoving:
    def test_stationary(self):
        centers = [(100.0, 100.0)] * 30
        assert is_moving(centers, threshold_px=4.0, ratio=0.3) is False

    def test_clearly_moving(self):
        centers = [(float(i * 10), float(i * 10)) for i in range(20)]
        assert is_moving(centers, threshold_px=4.0, ratio=0.3) is True

    def test_below_ratio(self):
        # 1 displaced frame out of 10 (= 9 intervals, 1 > threshold)
        centers = [(0.0, 0.0)] * 5 + [(100.0, 100.0)] + [(100.0, 100.0)] * 4
        result = is_moving(centers, threshold_px=4.0, ratio=0.3)
        # only 1/9 intervals > threshold → should be stationary
        assert result is False

    def test_single_center(self):
        assert is_moving([(50.0, 50.0)]) is False


# ── classify_motion_history ───────────────────────────────────────────────────

class TestClassifyMotionHistory:
    def test_empty(self):
        assert classify_motion_history({}) == []

    def test_all_stationary(self):
        centers = {i: (50.0, 50.0) for i in range(60)}
        history = classify_motion_history(centers, window=30)
        assert all(h["state"] == "stationary" for h in history)

    def test_all_moving(self):
        centers = {i: (float(i * 10), float(i * 10)) for i in range(60)}
        history = classify_motion_history(centers, window=30)
        assert all(h["state"] == "moving" for h in history)

    def test_frame_ranges_cover_all(self):
        centers = {i: (50.0, 50.0) for i in range(90)}
        history = classify_motion_history(centers, window=30)
        # All frames 0–89 should be covered
        assert history[0]["frame_range"][0] == 0
        assert history[-1]["frame_range"][1] == 89

    def test_merge_consecutive(self):
        # Two stationary windows should merge into one
        centers = {i: (50.0, 50.0) for i in range(60)}
        history = classify_motion_history(centers, window=30)
        assert len(history) == 1
        assert history[0]["frame_range"] == [0, 59]

    def test_transition_detected(self):
        # First 30 frames: stationary; next 30 frames: moving
        stationary = {i: (50.0, 50.0) for i in range(30)}
        moving = {i + 30: (float(i * 15), float(i * 15)) for i in range(30)}
        centers = {**stationary, **moving}
        history = classify_motion_history(centers, window=30)
        states = [h["state"] for h in history]
        assert "stationary" in states
        assert "moving" in states


# ── proximity_score ───────────────────────────────────────────────────────────

class TestProximityScore:
    def test_overlapping(self):
        # Boxes overlap → should be 1.0
        score = proximity_score([0,0,10,10], [5,5,15,15], frame_diagonal=1000)
        assert score == pytest.approx(1.0)

    def test_adjacent_no_overlap(self):
        # Boxes touch but don't overlap
        score = proximity_score([0,0,10,10], [10,0,20,10], frame_diagonal=1000)
        # IoU = 0, centre distance = 10, max_dist = 250 → score = 1 - 10/250
        assert 0 < score < 1

    def test_far_apart(self):
        # Distance = diagonal/4 → score = 0
        score = proximity_score([0,0,2,2], [1000,1000,1002,1002], frame_diagonal=100)
        assert score == pytest.approx(0.0)

    def test_score_range(self):
        score = proximity_score([0,0,5,5], [50,0,55,5], frame_diagonal=200)
        assert 0.0 <= score <= 1.0


# ── run_length_encode_interactions ────────────────────────────────────────────

class TestRunLengthEncodeInteractions:
    def test_empty(self):
        assert run_length_encode_interactions({}) == []

    def test_all_false(self):
        flags = {i: False for i in range(10)}
        assert run_length_encode_interactions(flags) == []

    def test_all_true(self):
        flags = {i: True for i in range(10)}
        result = run_length_encode_interactions(flags)
        assert result == [(0, 9)]

    def test_single_interval(self):
        flags = {0: False, 1: True, 2: True, 3: True, 4: False}
        assert run_length_encode_interactions(flags) == [(1, 3)]

    def test_multiple_intervals(self):
        flags = {0: True, 1: True, 2: False, 3: False, 4: True, 5: True}
        result = run_length_encode_interactions(flags)
        assert result == [(0, 1), (4, 5)]

    def test_alternating(self):
        flags = {i: (i % 2 == 0) for i in range(6)}  # 0,2,4 True
        result = run_length_encode_interactions(flags)
        assert result == [(0, 0), (2, 2), (4, 4)]
