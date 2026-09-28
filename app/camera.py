import time
import logging
from typing import Generator, Union
import cv2
import numpy as np

from app.config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Manages OpenCV VideoCapture with automatic reconnection, retry loops, and frame resizing.
    """

    def __init__(self, source: Union[int, str] = CAMERA_SOURCE, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.source = source
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture = None
        self.is_running = True

    def open(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """
        Attempts to open the video capture stream with retries.
        """
        for attempt in range(1, max_retries + 1):
            logger.info(f"Opening camera source '{self.source}' (Attempt {attempt}/{max_retries})...")
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info(f"Camera opened successfully ({self.width}x{self.height}).")
                return True

            logger.warning(f"Could not open camera '{self.source}'. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

        logger.error(f"Failed to open camera source '{self.source}' after {max_retries} attempts.")
        return False

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yields BGR frames continuously. Automatically attempts reconnection if camera feed drops.
        """
        consecutive_failures = 0
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                success = self.open(max_retries=3, retry_delay=1.0)
                if not success:
                    # Yield a synthetic standby frame to prevent consumer crash if camera is offline
                    standby = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(
                        standby,
                        "Camera Offline - Reconnecting...",
                        (30, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    yield standby
                    time.sleep(1.0)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame from camera ({consecutive_failures} failures).")
                if consecutive_failures > 10:
                    logger.error("Camera disconnected or empty feed. Resetting capture device...")
                    self.release()
                    consecutive_failures = 0
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            yield frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera capture released.")

    def stop(self):
        self.is_running = False
        self.release()


def frames(source: Union[int, str] = CAMERA_SOURCE) -> Generator[np.ndarray, None, None]:
    """
    Functional wrapper yielding frames from CameraStream.
    """
    stream = CameraStream(source=source)
    try:
        for frame in stream.frames():
            yield frame
    finally:
        stream.stop()
