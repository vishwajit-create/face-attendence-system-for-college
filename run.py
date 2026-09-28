"""
run.py — Entry point: starts Flask app (with optional CV pipeline thread).
"""
import threading
import logging
from app import create_app
from app.config import FLASK_HOST, FLASK_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def start_cv_pipeline():
    """Optional: runs the server-side camera pipeline in a background thread."""
    try:
        from app.camera import CameraStream
        from app.detector import FaceDetector
        import time
        cam = CameraStream()
        cam.start()
        detector = FaceDetector()
        logger.info("CV pipeline started (server-side camera).")
        while True:
            frame = cam.read()
            if frame is not None:
                detector.detect(frame)   # keep warm
            time.sleep(0.1)
    except Exception as e:
        logger.warning(f"CV pipeline could not start: {e}. Browser webcam mode will still work.")


if __name__ == "__main__":
    # Start CV pipeline in daemon thread (optional — browser webcam works without it)
    cv_thread = threading.Thread(target=start_cv_pipeline, daemon=True)
    cv_thread.start()

    app = create_app()
    logger.info(f"🚀 Server running at http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
