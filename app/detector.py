import os
import logging
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.config import YUNET_MODEL_PATH

logger = logging.getLogger(__name__)

YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 keypoints: [right_eye, left_eye, nose, right_mouth, left_mouth]

    @property
    def box_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def centroid(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def ensure_yunet_model(model_path: str = YUNET_MODEL_PATH) -> str:
    """
    Ensures that the YuNet ONNX weights file exists locally, downloading from OpenCV Zoo if needed.
    """
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logger.info(f"YuNet model not found locally. Downloading from {YUNET_DOWNLOAD_URL}...")
    try:
        urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, model_path)
        logger.info(f"YuNet model successfully saved to {model_path}")
        return model_path
    except Exception as e:
        logger.warning(f"Could not download YuNet model ({e}). Fallback detector will be used if needed.")
        return model_path


def load_detector(
    model_path: str = YUNET_MODEL_PATH,
    input_size: Tuple[int, int] = (320, 320),
    score_threshold: float = 0.85,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> Optional[cv2.FaceDetectorYN]:
    """
    Loads OpenCV's YuNet ONNX face detector.
    """
    model_file = ensure_yunet_model(model_path)
    if not os.path.exists(model_file):
        logger.error(f"YuNet model file missing at {model_file}")
        return None

    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_file,
            config="",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        logger.info(f"YuNet FaceDetectorYN loaded from {model_file}")
        return detector
    except Exception as e:
        logger.error(f"Failed to initialize FaceDetectorYN: {e}")
        return None


def detect_faces(
    frame: np.ndarray,
    detector: Optional[cv2.FaceDetectorYN] = None,
    conf_threshold: float = 0.85,
) -> List[BoundingBox]:
    """
    Detects faces in the given frame using YuNet (with Haar Cascade fallback).
    Returns a list of BoundingBox objects.
    """
    if frame is None or frame.size == 0:
        return []

    height, width = frame.shape[:2]
    boxes: List[BoundingBox] = []

    if detector is not None:
        try:
            # Update input size to match the incoming frame dimensions
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)

            if faces is not None:
                for face in faces:
                    score = float(face[-1])
                    if score < conf_threshold:
                        continue

                    # Bounding box coordinates
                    x = max(0, int(face[0]))
                    y = max(0, int(face[1]))
                    w = min(width - x, int(face[2]))
                    h = min(height - y, int(face[3]))

                    # 5 facial landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
                    landmarks = np.array(
                        [
                            [face[4], face[5]],
                            [face[6], face[7]],
                            [face[8], face[9]],
                            [face[10], face[11]],
                            [face[12], face[13]],
                        ],
                        dtype=np.float32,
                    )

                    boxes.append(BoundingBox(x=x, y=y, w=w, h=h, confidence=score, landmarks=landmarks))
            return boxes
        except Exception as e:
            logger.warning(f"YuNet detection failed: {e}. Falling back to Haar Cascade.")

    # Fallback to OpenCV Haar cascade if YuNet is unavailable
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            haar_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in haar_faces:
                boxes.append(BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h), confidence=0.90))
        except Exception as e:
            logger.debug(f"Haar cascade detection failed: {e}")

    return boxes


class FaceDetector:
    """Wrapper class providing an object-oriented interface."""
    def __init__(self, conf_threshold: float = 0.70):
        self.detector = load_detector(score_threshold=conf_threshold)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray):
        results = detect_faces(frame, detector=self.detector, conf_threshold=self.conf_threshold)
        return [[b.x, b.y, b.w, b.h] for b in results]
