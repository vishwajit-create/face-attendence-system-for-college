import os
import logging
from typing import Optional
import numpy as np
import cv2

from app.config import SFACE_MODEL_PATH, YUNET_MODEL_PATH

logger = logging.getLogger(__name__)

SFACE_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

_sface_recognizer = None
_yunet_detector = None


def ensure_sface_model(model_path: str = str(SFACE_MODEL_PATH)) -> str:
    """Ensures SFace ONNX weights exist locally, downloading from OpenCV Zoo if needed."""
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logger.info(f"SFace model not found at {model_path}. Downloading from {SFACE_DOWNLOAD_URL}...")
    import urllib.request
    urllib.request.urlretrieve(SFACE_DOWNLOAD_URL, model_path)
    logger.info(f"SFace model downloaded successfully ({os.path.getsize(model_path)} bytes).")
    return model_path


def load_sface_recognizer() -> Optional[cv2.FaceRecognizerSF]:
    """Loads OpenCV FaceRecognizerSF (SFace 128-dim deep face embedding model)."""
    global _sface_recognizer
    if _sface_recognizer is not None:
        return _sface_recognizer

    try:
        model_file = ensure_sface_model()
        _sface_recognizer = cv2.FaceRecognizerSF.create(model=str(model_file), config="")
        logger.info(f"OpenCV FaceRecognizerSF loaded from {model_file}")
    except Exception as e:
        logger.error(f"Failed to load FaceRecognizerSF: {e}")
        _sface_recognizer = None

    return _sface_recognizer


def _get_helper_detector():
    """Lightweight detector used to align arbitrary face crops if landmarks aren't provided."""
    global _yunet_detector
    if _yunet_detector is not None:
        return _yunet_detector
    try:
        if os.path.exists(YUNET_MODEL_PATH):
            _yunet_detector = cv2.FaceDetectorYN.create(
                model=str(YUNET_MODEL_PATH),
                config="",
                input_size=(160, 160),
                score_threshold=0.6,
            )
    except Exception as e:
        logger.debug(f"Helper detector init failed: {e}")
    return _yunet_detector


def encode(
    face_crop: np.ndarray,
    raw_face: Optional[np.ndarray] = None,
    full_frame: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Extracts a 128-dimensional L2-normalized deep feature embedding using OpenCV SFace.
    - If `raw_face` and `full_frame` are provided, performs landmark alignment before feature extraction.
    - Otherwise, detects landmarks directly from `face_crop` or normalizes the crop.
    """
    if face_crop is None or (full_frame is None and face_crop.size == 0):
        return None

    recognizer = load_sface_recognizer()
    if recognizer is None:
        logger.error("SFace recognizer is not initialized.")
        return None

    try:
        # Case 1: Landmark-guided alignment from original frame
        if full_frame is not None and raw_face is not None:
            aligned = recognizer.alignCrop(full_frame, raw_face)
            feat = recognizer.feature(aligned).flatten()

        else:
            # Case 2: Face crop provided
            h, w = face_crop.shape[:2]
            det = _get_helper_detector()
            aligned = None

            if det is not None and h >= 30 and w >= 30:
                try:
                    det.setInputSize((w, h))
                    _, detected_faces = det.detect(face_crop)
                    if detected_faces is not None and len(detected_faces) > 0:
                        aligned = recognizer.alignCrop(face_crop, detected_faces[0])
                except Exception:
                    aligned = None

            if aligned is None:
                aligned = cv2.resize(face_crop, (112, 112))

            feat = recognizer.feature(aligned).flatten()

        # L2-normalization for cosine distance
        feat = feat.astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-9:
            feat = feat / norm
        return feat

    except Exception as e:
        logger.error(f"SFace encoding failed: {e}")
        return None


# Backward compatibility aliases
encode_face = encode
load_encoder = load_sface_recognizer
