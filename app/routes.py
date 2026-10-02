"""
routes.py — Legacy / misc routes: video_feed, enroll, kept for backward compat.
Now registered as 'main' blueprint.
"""
import io
import logging
import time
from typing import Optional

from flask import (
    Blueprint, Response, jsonify, render_template,
    request, redirect, url_for, flash, send_file
)
from flask_login import login_required, current_user

from app.db import run_query, run_write
from app.config import ENROLLED_FACES_DIR

logger = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)


# ── MJPEG Video Feed ──────────────────────────────────────────────────────────

def _gen_frames():
    from app.camera import CameraStream
    import cv2
    cam = CameraStream()
    cam.start()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
    finally:
        cam.stop()


@main_bp.route("/video_feed")
@login_required
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Enroll ────────────────────────────────────────────────────────────────────

@main_bp.route("/enroll", methods=["GET", "POST"])
@login_required
def enroll():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip() or None
        email = request.form.get("email", "").strip() or None
        class_ids = request.form.getlist("class_ids")

        if not name:
            flash("Name is required.", "danger")
            return redirect(url_for("main.enroll"))

        # Collect images from live camera snapshots and/or uploaded files
        all_images = []
        import base64
        import cv2
        import numpy as np

        webcam_data = request.form.getlist("webcam_snapshots[]") or request.form.getlist("webcam_snapshots")
        for b64 in webcam_data:
            if b64 and b64.startswith("data:image"):
                try:
                    header, encoded = b64.split(",", 1)
                    raw = base64.b64decode(encoded)
                    arr = np.frombuffer(raw, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        all_images.append(img)
                except Exception as e:
                    logger.warning(f"Error decoding webcam snapshot: {e}")

        uploaded = request.files.getlist("images") or request.files.getlist("images[]")
        for f in uploaded:
            if f and f.filename:
                all_images.append(f)

        if not all_images:
            flash("Please capture photos with camera or upload at least one face photo.", "warning")
            return redirect(url_for("main.enroll"))

        # Insert or update person
        from app.db import run_write_returning
        person_id = None
        if roll_no:
            existing = run_query("SELECT id FROM people WHERE roll_no = %s;", (roll_no,))
            if existing:
                person_id = existing[0]["id"]
                run_write("UPDATE people SET name=%s, email=%s WHERE id=%s;", (name, email, person_id))

        if not person_id:
            person_id = run_write_returning(
                "INSERT INTO people (name, roll_no, email, role) VALUES (%s,%s,%s,'student') RETURNING id;",
                (name, roll_no or None, email or None),
            )
        # Assign to classes
        for cid in class_ids:
            run_write(
                "INSERT INTO class_students (class_id, person_id) VALUES (%s,%s) ON CONFLICT DO NOTHING;",
                (int(cid), person_id),
            )

        from app.enrollment import process_enrollment
        count = process_enrollment(person_id, all_images)
        flash(f"Enrolled '{name}' with {count} face image(s).", "success")
        return redirect(url_for("main.enroll"))

    classes = run_query("SELECT * FROM classes ORDER BY code;")
    return render_template("enroll.html", classes=classes)


# ── API: scan_frame (legacy, no class context) ─────────────────────────────

@main_bp.route("/api/scan_frame", methods=["POST"])
@login_required
def api_scan_frame_legacy():
    """Kept for the old dashboard — delegates to teacher route logic."""
    import numpy as np
    import cv2
    from app.detector import FaceDetector
    from app.encoder import encode_face
    from app.matcher import find_best_match

    img_file = request.files.get("image")
    if not img_file:
        return jsonify({"faces": []})
    nparr = np.frombuffer(img_file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"faces": []})
    from app.detector import detect_faces
    boxes = detect_faces(frame, conf_threshold=0.6)
    results = []
    for box in boxes:
        x, y, w, h = box.x, box.y, box.w, box.h
        pad = int(min(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        embedding = encode_face(face_crop, raw_face=box.raw_face, full_frame=frame)
        if embedding is None:
            continue
        match = find_best_match(embedding)
        if match:
            results.append({"name": match["name"], "similarity": round(match["similarity"], 3), "box": [int(x), int(y), int(w), int(h)]})
        else:
            results.append({"name": "Unknown", "box": [int(x), int(y), int(w), int(h)]})
    return jsonify({"faces": results})
