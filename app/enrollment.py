"""
enrollment.py — Process face enrollment from uploaded FileStorage objects or file paths.
"""
import os
import logging
from typing import List, Union

import cv2
import numpy as np

from app.config import ENROLLED_FACES_DIR
from app.db import run_write, run_query, get_connection, is_postgres_active

logger = logging.getLogger(__name__)


def process_enrollment(person_id: int, image_sources) -> int:
    """
    Given a person_id (already inserted into people table) and a list of
    FileStorage objects (from request.files) or file path strings,
    detect the face in each, compute embeddings, and store them.

    Returns the count of successfully enrolled face embeddings.
    """
    from app.detector import FaceDetector
    from app.encoder import encode_face

    detector = FaceDetector()
    valid = 0

    for idx, item in enumerate(image_sources):
        frame = None
        img_path = None

        # Handle Flask FileStorage
        if hasattr(item, "read"):
            if item.filename == "":
                continue
            data = item.read()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Save original for reference
            save_path = ENROLLED_FACES_DIR / f"{person_id}_{idx}.jpg"
            if frame is not None:
                cv2.imwrite(str(save_path), frame)
                img_path = str(save_path)
        elif isinstance(item, str) and os.path.exists(item):
            frame = cv2.imread(item)
            img_path = item
        elif isinstance(item, np.ndarray):
            frame = item
            save_path = ENROLLED_FACES_DIR / f"{person_id}_{idx}.jpg"
            cv2.imwrite(str(save_path), frame)
            img_path = str(save_path)

        if frame is None or frame.size == 0:
            logger.warning(f"Invalid image at index {idx} for person {person_id}")
            continue

        from app.detector import detect_faces
        boxes = detect_faces(frame, conf_threshold=0.6)
        if len(boxes) == 0:
            logger.warning(f"No face detected in image #{idx + 1} for person {person_id}")
            continue

        # Pick largest face
        box = max(boxes, key=lambda b: b.w * b.h)
        x, y, w, h = box.x, box.y, box.w, box.h
        pad = int(min(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]

        embedding = encode_face(face_crop, raw_face=box.raw_face, full_frame=frame)
        if embedding is None:
            logger.warning(f"Failed to generate embedding for image #{idx + 1}")
            continue

        # Store embedding
        if is_postgres_active():
            run_write(
                "INSERT INTO enrolled_faces (person_id, embedding, source_image_path) VALUES (%s, %s::vector, %s);",
                (person_id, str(embedding.tolist()), img_path),
            )
        else:
            run_write(
                "INSERT INTO enrolled_faces (person_id, embedding, source_image_path) VALUES (?,?,?);",
                (person_id, embedding.astype(np.float32).tobytes(), img_path),
            )
        valid += 1

    logger.info(f"Enrolled person {person_id}: {valid} face embeddings stored.")
    return valid
