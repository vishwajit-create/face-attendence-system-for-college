import os
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import (
    LIVENESS_EAR_THRESHOLD,
    LIVENESS_CONSECUTIVE_FRAMES,
    LIVENESS_WINDOW_SECONDS,
    DLIB_PREDICTOR_PATH,
)

logger = logging.getLogger(__name__)

# Indices for 68-point facial landmarks (dlib / standard)
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

_dlib_detector = None
_dlib_predictor = None
_dlib_attempted = False


def load_dlib_predictor(predictor_path: str = DLIB_PREDICTOR_PATH):
    """
    Attempts to load dlib's 68-point landmark predictor if installed.
    """
    global _dlib_detector, _dlib_predictor, _dlib_attempted
    if _dlib_attempted:
        return _dlib_predictor

    _dlib_attempted = True
    try:
        import dlib
        _dlib_detector = dlib.get_frontal_face_detector()
        if os.path.exists(predictor_path):
            _dlib_predictor = dlib.shape_predictor(predictor_path)
            logger.info(f"Loaded dlib 68-point shape predictor from {predictor_path}")
        else:
            logger.warning(
                f"dlib shape predictor not found at {predictor_path}. "
                "Will use YuNet / geometric aspect ratio fallback."
            )
    except ImportError:
        logger.info("dlib not installed. Using geometric aspect ratio fallback.")
    except Exception as e:
        logger.warning(f"Error initializing dlib: {e}")

    return _dlib_predictor


def eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """
    Computes the Eye Aspect Ratio (EAR) according to Soukupova & Cech (Section 4.4):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    where landmarks is an array of shape (6, 2) representing the 6 eye points in order:
        [p1, p2, p3, p4, p5, p6]
    """
    if landmarks is None or len(landmarks) < 6:
        return 0.30  # Default neutral EAR

    p1, p2, p3, p4, p5, p6 = landmarks[:6]

    # Vertical distances between eyelids
    dist_a = float(np.linalg.norm(p2 - p6))
    dist_b = float(np.linalg.norm(p3 - p5))

    # Horizontal distance between eye corners
    dist_c = float(np.linalg.norm(p1 - p4))

    if dist_c <= 1e-6:
        return 0.30

    ear = (dist_a + dist_b) / (2.0 * dist_c)
    return float(ear)


def compute_frame_ear(frame: np.ndarray, face_box: Tuple[int, int, int, int], yunet_landmarks: Optional[np.ndarray] = None) -> float:
    """
    Computes the average EAR for both eyes from a frame.
    Uses dlib 68-point model if available; otherwise uses YuNet landmark geometry.
    """
    predictor = load_dlib_predictor()
    x, y, w, h = face_box

    if predictor is not None and frame is not None:
        try:
            import cv2
            import dlib
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, dlib_rect)
            coords = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)

            left_eye = coords[LEFT_EYE_INDICES]
            right_eye = coords[RIGHT_EYE_INDICES]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            return (left_ear + right_ear) / 2.0
        except Exception as e:
            logger.debug(f"Dlib landmark detection failed: {e}")

    # Fallback using YuNet 5-point landmarks or face crop eye-region heuristic
    if yunet_landmarks is not None and len(yunet_landmarks) >= 2:
        # yunet_landmarks[0] is right eye, [1] is left eye
        # Approximate eye region aspect ratio using eye distance vs height
        eye_dist = np.linalg.norm(yunet_landmarks[0] - yunet_landmarks[1])
        if w > 0:
            ratio = (eye_dist / float(w)) * 0.5
            return float(np.clip(ratio, 0.15, 0.35))

    return 0.28


def is_blink(ear_sequence: List[float], threshold: float = LIVENESS_EAR_THRESHOLD, min_consecutive: int = 1) -> bool:
    """
    Detects if an EAR sequence contains a valid blink pattern:
    Eyes open (EAR > threshold) -> Eyes closed (EAR < threshold) -> Eyes open again (EAR > threshold).
    """
    if len(ear_sequence) < 3:
        return False

    has_open_before = False
    closed_count = 0
    has_reopened = False

    for ear in ear_sequence:
        if not has_open_before:
            if ear > threshold + 0.03:
                has_open_before = True
        elif not has_reopened:
            if ear <= threshold:
                closed_count += 1
            elif closed_count >= min_consecutive:
                if ear > threshold + 0.03:
                    has_reopened = True
                    break

    return has_reopened


class LivenessTracker:
    """
    Maintains per-face rolling buffer of EAR values keyed by face ID.
    Enforces liveness via blink verification within a sliding time window.
    """

    def __init__(
        self,
        ear_threshold: float = LIVENESS_EAR_THRESHOLD,
        window_seconds: float = LIVENESS_WINDOW_SECONDS,
        min_consecutive_frames: int = 1,
    ):
        self.ear_threshold = ear_threshold
        self.window_seconds = window_seconds
        self.min_consecutive_frames = min_consecutive_frames

        # Map face_id -> deque of (timestamp, ear_value)
        self.history: Dict[int, deque] = {}
        # Map face_id -> timestamp when verified live
        self.verified_live_until: Dict[int, float] = {}

    def update(self, face_id: int, ear: float, current_time: Optional[float] = None) -> bool:
        """
        Records an EAR reading for a face. Returns True if face is currently confirmed live.
        """
        now = current_time or time.time()

        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=60)

        self.history[face_id].append((now, ear))

        # Check if already verified and within the verification grace window (e.g. 15 seconds)
        if now < self.verified_live_until.get(face_id, 0.0):
            return True

        # Prune old records outside the rolling window
        while self.history[face_id] and (now - self.history[face_id][0][0] > self.window_seconds):
            self.history[face_id].popleft()

        ear_vals = [val for _, val in self.history[face_id]]
        if is_blink(ear_vals, threshold=self.ear_threshold, min_consecutive=self.min_consecutive_frames):
            # Mark face as verified live for the next 30 seconds
            self.verified_live_until[face_id] = now + 30.0
            logger.info(f"Face #{face_id} passed liveness check (blink detected).")
            return True

        return False

    def is_live(self, face_id: int, current_time: Optional[float] = None) -> bool:
        """
        Queries whether the face has passed liveness verification.
        """
        now = current_time or time.time()
        return now < self.verified_live_until.get(face_id, 0.0)

    def cleanup_old_faces(self, active_face_ids: List[int]):
        """
        Removes tracking history for faces no longer visible.
        """
        active_set = set(active_face_ids)
        stale_ids = [fid for fid in self.history if fid not in active_set]
        for fid in stale_ids:
            self.history.pop(fid, None)
            self.verified_live_until.pop(fid, None)
