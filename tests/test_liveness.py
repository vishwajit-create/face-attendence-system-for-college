import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import math
import numpy as np

from app.liveness import eye_aspect_ratio, is_blink, LivenessTracker


def test_eye_aspect_ratio_standard_open_eye():
    """
    Test EAR calculation on synthetic open eye landmarks.
    Expected EAR should be around 0.30.
    """
    landmarks = np.array([
        [10.0, 30.0],  # p1 (outer corner)
        [15.0, 27.0],  # p2 (upper lid 1)
        [25.0, 27.0],  # p3 (upper lid 2)
        [30.0, 30.0],  # p4 (inner corner)
        [25.0, 33.0],  # p5 (lower lid 2)
        [15.0, 33.0],  # p6 (lower lid 1)
    ], dtype=np.float32)

    ear = eye_aspect_ratio(landmarks)
    assert abs(ear - 0.30) < 1e-3, f"Expected EAR 0.30, got {ear}"


def test_eye_aspect_ratio_closed_eye():
    """
    Test EAR calculation on synthetic closed eye landmarks.
    """
    landmarks = np.array([
        [10.0, 30.0],  # p1
        [15.0, 30.5],  # p2
        [25.0, 30.5],  # p3
        [30.0, 30.0],  # p4
        [25.0, 29.5],  # p5
        [15.0, 29.5],  # p6
    ], dtype=np.float32)

    ear = eye_aspect_ratio(landmarks)
    assert ear < 0.15, f"Expected closed EAR < 0.15, got {ear}"


def test_is_blink_pattern():
    """
    Test blink pattern detection: open -> closed (< threshold) -> open.
    """
    ear_sequence_valid = [0.32, 0.31, 0.18, 0.17, 0.30, 0.32]
    assert is_blink(ear_sequence_valid, threshold=0.21, min_consecutive=1) is True

    ear_sequence_static_open = [0.31, 0.31, 0.30, 0.32, 0.31, 0.31]
    assert is_blink(ear_sequence_static_open, threshold=0.21, min_consecutive=1) is False

    ear_sequence_closed = [0.15, 0.14, 0.15, 0.16, 0.15]
    assert is_blink(ear_sequence_closed, threshold=0.21, min_consecutive=1) is False


def test_liveness_tracker_lifecycle():
    """
    Test LivenessTracker maintains state and activates live flag upon blink.
    """
    tracker = LivenessTracker(ear_threshold=0.21, window_seconds=3.0)
    face_id = 42

    assert tracker.is_live(face_id, current_time=100.0) is False

    tracker.update(face_id, ear=0.32, current_time=100.0)
    tracker.update(face_id, ear=0.31, current_time=100.5)
    assert tracker.is_live(face_id, current_time=100.5) is False

    tracker.update(face_id, ear=0.16, current_time=101.0)

    live_status = tracker.update(face_id, ear=0.31, current_time=101.5)
    assert live_status is True
    assert tracker.is_live(face_id, current_time=101.5) is True


if __name__ == "__main__":
    test_eye_aspect_ratio_standard_open_eye()
    test_eye_aspect_ratio_closed_eye()
    test_is_blink_pattern()
    test_liveness_tracker_lifecycle()
    print("All liveness tests passed successfully!")
