import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# ── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/attendance_db")

# ── Camera ────────────────────────────────────────────────────
_raw_source = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(_raw_source) if _raw_source.isdigit() else _raw_source
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", 640))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 480))

# ── Face Recognition ──────────────────────────────────────────
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.363))

# ── Temporal Debounce ─────────────────────────────────────────
DEBOUNCE_FRAMES = int(os.getenv("DEBOUNCE_FRAMES", 5))
DEBOUNCE_WINDOW = int(os.getenv("DEBOUNCE_WINDOW", 7))

# ── Liveness (EAR) ────────────────────────────────────────────
LIVENESS_EAR_THRESHOLD = float(os.getenv("LIVENESS_EAR_THRESHOLD", 0.21))
LIVENESS_CONSECUTIVE_FRAMES = int(os.getenv("LIVENESS_CONSECUTIVE_FRAMES", 1))
LIVENESS_WINDOW_SECONDS = float(os.getenv("LIVENESS_WINDOW_SECONDS", 3.0))

# ── Session / Application ─────────────────────────────────────
CURRENT_SESSION_ID = os.getenv("CURRENT_SESSION_ID", "DEFAULT-SESSION")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "face-attendance-college-super-secret-2026")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

# ── File Paths ────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
YUNET_MODEL_PATH = os.getenv("YUNET_MODEL_PATH", str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"))
SFACE_MODEL_PATH = os.getenv("SFACE_MODEL_PATH", str(MODELS_DIR / "face_recognition_sface_2021dec.onnx"))
DLIB_PREDICTOR_PATH = os.getenv("DLIB_PREDICTOR_PATH", str(MODELS_DIR / "shape_predictor_68_face_landmarks.dat"))

UNKNOWN_FACE_DIR = BASE_DIR / "static" / "unknown_faces"
ENROLLED_FACES_DIR = BASE_DIR / "static" / "enrolled_faces"
REPORTS_DIR = BASE_DIR / "static" / "reports"

for d in [UNKNOWN_FACE_DIR, ENROLLED_FACES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
