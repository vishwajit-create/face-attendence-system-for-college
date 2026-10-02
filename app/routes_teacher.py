"""
routes_teacher.py — Teacher panel routes

Features:
  - View assigned classes
  - Start / Stop attendance session for a class
  - Live webcam attendance (browser-based scanner)
  - View students in each class + attendance %
  - Edit student names / roll numbers
  - Export attendance to Excel
"""
import uuid
from datetime import datetime
from typing import Optional

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, jsonify, Response, stream_with_context,
)
from flask_login import login_required, current_user

from app.db import run_query, run_write, run_write_returning

teacher_bp = Blueprint("teacher", __name__, url_prefix="/teacher")


def _get_teacher_classes(teacher_id: int):
    return run_query("""
        SELECT c.*, d.name AS dept_name
        FROM classes c
        JOIN class_teachers ct ON ct.class_id = c.id
        LEFT JOIN departments d ON d.id = c.department_id
        WHERE ct.teacher_id = %s
        ORDER BY c.code;
    """, (teacher_id,))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@teacher_bp.route("/")
@login_required
def dashboard():
    classes = _get_teacher_classes(int(current_user.id))
    active_sessions = run_query("""
        SELECT s.*, c.name AS class_name, c.code AS class_code
        FROM sessions s
        JOIN classes c ON c.id = s.class_id
        WHERE s.teacher_id = %s AND s.is_active = 1
        ORDER BY s.started_at DESC;
    """, (int(current_user.id),))
    return render_template("teacher/dashboard.html", classes=classes, active_sessions=active_sessions)


def _format_date_parts(dt_val):
    if not dt_val:
        now = datetime.now()
        return now.strftime("%d %b %Y"), now.strftime("%a"), now.strftime("%Y-%m-%d")
    if isinstance(dt_val, str):
        try:
            clean = dt_val.split(".")[0].replace("Z", "").replace("T", " ")
            dt_obj = datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                dt_obj = datetime.strptime(clean[:10], "%Y-%m-%d")
            except Exception:
                dt_obj = datetime.now()
    else:
        dt_obj = dt_val
    return dt_obj.strftime("%d %b %Y"), dt_obj.strftime("%a"), dt_obj.strftime("%Y-%m-%d")


# ── Class Detail ──────────────────────────────────────────────────────────────

@teacher_bp.route("/classes/<int:class_id>")
@login_required
def class_detail(class_id):
    cls = run_query("SELECT * FROM classes WHERE id=%s;", (class_id,))
    if not cls:
        flash("Class not found.", "danger")
        return redirect(url_for("teacher.dashboard"))
    cls_data = cls[0]

    # Total sessions conducted
    total_sessions = run_query(
        "SELECT COUNT(*) AS c FROM sessions WHERE class_id=%s;", (class_id,)
    )[0]["c"]

    # Total class days (from class record or sessions count)
    total_class_days = cls_data.get("total_working_days") or total_sessions or 0
    if total_class_days == 0 and total_sessions > 0:
        total_class_days = total_sessions

    # All students enrolled in this class
    students_raw = run_query("""
        SELECT p.id, p.name, p.roll_no, p.email,
               COUNT(DISTINCT al.session_id) AS present_days
        FROM class_students cs
        JOIN people p ON p.id = cs.person_id
        LEFT JOIN attendance_log al ON al.person_id = p.id AND al.class_id = %s
        WHERE cs.class_id = %s
        GROUP BY p.id ORDER BY p.name;
    """, (class_id, class_id))

    total_enrolled = len(students_raw)

    students = []
    for s in students_raw:
        p_days = s["present_days"] or 0
        denom = max(total_class_days, p_days, 1)
        a_days = max(0, denom - p_days)
        pct = round((p_days / denom * 100), 2)
        if pct >= 75:
            status = "Good"
            badge_class = "bg-emerald-950/60 text-emerald-400 border border-emerald-700/60"
        elif pct >= 60:
            status = "Average"
            badge_class = "bg-amber-950/60 text-amber-400 border border-amber-700/60"
        else:
            status = "Critical"
            badge_class = "bg-rose-950/60 text-rose-400 border border-rose-700/60"

        students.append({
            "id": s["id"],
            "name": s["name"],
            "roll_no": s["roll_no"] or "—",
            "present_days": p_days,
            "absent_days": a_days,
            "percentage": f"{pct:.2f}%",
            "status": status,
            "badge_class": badge_class,
        })

    # Attendance by Date (right column)
    sessions_raw = run_query("""
        SELECT s.id, s.label, s.started_at, s.is_active,
               COUNT(DISTINCT al.person_id) AS present_count
        FROM sessions s
        LEFT JOIN attendance_log al ON al.session_id = s.id
        WHERE s.class_id = %s
        GROUP BY s.id
        ORDER BY s.started_at DESC LIMIT 30;
    """, (class_id,))

    attendance_by_date = []
    for sess in sessions_raw:
        date_str, day_name, ymd = _format_date_parts(sess["started_at"])
        p_cnt = sess["present_count"] or 0
        is_completed = (p_cnt >= total_enrolled and total_enrolled > 0)
        attendance_by_date.append({
            "session_id": sess["id"],
            "date": date_str,
            "day": day_name,
            "ymd": ymd,
            "present_total": f"{p_cnt} / {total_enrolled}",
            "status": "Completed" if is_completed else "Partial",
            "status_class": "bg-emerald-950/60 text-emerald-400 border border-emerald-700/60" if is_completed else "bg-amber-950/60 text-amber-400 border border-amber-700/60",
            "is_active": sess["is_active"],
        })

    # Top Metric Cards: Present, Absent, Leave for selected or latest date
    selected_date_display = datetime.now().strftime("%d %b %Y")
    metric_present = 0
    metric_absent = 0
    metric_leave = 0

    if attendance_by_date:
        latest = sessions_raw[0]
        metric_present = latest["present_count"] or 0
        metric_absent = max(0, total_enrolled - metric_present)
        date_str, _, _ = _format_date_parts(latest["started_at"])
        selected_date_display = date_str
    elif total_enrolled > 0:
        metric_absent = total_enrolled

    denom_metric = max(total_enrolled, 1)
    present_pct = round((metric_present / denom_metric * 100), 1)
    absent_pct = round((metric_absent / denom_metric * 100), 1)
    leave_pct = round((metric_leave / denom_metric * 100), 1)

    metrics = {
        "present": metric_present,
        "present_pct": present_pct,
        "absent": metric_absent,
        "absent_pct": absent_pct,
        "leave": metric_leave,
        "leave_pct": leave_pct,
        "selected_date": selected_date_display,
    }

    active_session = run_query(
        "SELECT * FROM sessions WHERE class_id=%s AND teacher_id=%s AND is_active=1 ORDER BY started_at DESC LIMIT 1;",
        (class_id, int(current_user.id)),
    )
    active_session = active_session[0] if active_session else None

    return render_template(
        "teacher/class_detail.html",
        cls=cls_data,
        total_class_days=total_class_days,
        students=students,
        metrics=metrics,
        attendance_by_date=attendance_by_date,
        active_session=active_session,
    )


# ── Session Management ────────────────────────────────────────────────────────

@teacher_bp.route("/classes/<int:class_id>/session/start", methods=["GET", "POST"])
@login_required
def start_session(class_id):
    cls = run_query("SELECT * FROM classes WHERE id=%s;", (class_id,))
    if not cls:
        flash("Class not found.", "danger")
        return redirect(url_for("teacher.dashboard"))

    active = run_query(
        "SELECT id FROM sessions WHERE class_id=%s AND teacher_id=%s AND is_active=1 ORDER BY started_at DESC LIMIT 1;",
        (class_id, int(current_user.id)),
    )
    if active:
        session_id = active[0]["id"]
        return redirect(url_for("teacher.take_attendance", class_id=class_id, session_id=session_id))

    session_id = f"{cls[0]['code']}-{datetime.now().strftime('%Y-%m-%d-%I%p')}-{uuid.uuid4().hex[:6]}"
    label = f"{cls[0]['name']} — {datetime.now().strftime('%d %b %Y %I:%M %p')}"
    run_write(
        "INSERT INTO sessions (id, class_id, teacher_id, label, is_active) VALUES (%s,%s,%s,%s,1);",
        (session_id, class_id, int(current_user.id), label),
    )

    # Update total_working_days counter
    run_write(
        "UPDATE classes SET total_working_days = total_working_days + 1 WHERE id=%s;",
        (class_id,),
    )

    flash(f"Attendance session started: {label}", "success")
    return redirect(url_for("teacher.take_attendance", class_id=class_id, session_id=session_id))


@teacher_bp.route("/classes/<int:class_id>/session/<session_id>/stop", methods=["POST"])
@login_required
def stop_session(class_id, session_id):
    run_write(
        "UPDATE sessions SET is_active=0, ended_at=CURRENT_TIMESTAMP WHERE id=%s AND teacher_id=%s;",
        (session_id, int(current_user.id)),
    )
    flash("Session ended successfully.", "success")
    return redirect(url_for("teacher.class_detail", class_id=class_id))


# ── Live Attendance Taking ─────────────────────────────────────────────────────

@teacher_bp.route("/classes/<int:class_id>/session/<session_id>/attend")
@login_required
def take_attendance(class_id, session_id):
    cls = run_query("SELECT * FROM classes WHERE id=%s;", (class_id,))
    session = run_query("SELECT * FROM sessions WHERE id=%s;", (session_id,))
    if not cls or not session:
        flash("Invalid class or session.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # Enrolled students in this class
    enrolled = run_query("""
        SELECT p.id, p.name, p.roll_no FROM class_students cs
        JOIN people p ON p.id = cs.person_id WHERE cs.class_id=%s ORDER BY p.name;
    """, (class_id,))

    # Already marked present this session
    present_ids = {
        row["person_id"]
        for row in run_query("SELECT person_id FROM attendance_log WHERE session_id=%s;", (session_id,))
    }

    return render_template(
        "teacher/take_attendance.html",
        cls=cls[0],
        session=session[0],
        enrolled=enrolled,
        present_ids=present_ids,
    )


@teacher_bp.route("/api/mark-present", methods=["POST"])
@login_required
def api_mark_present():
    """Mark a person as present manually."""
    data = request.get_json()
    person_id = data.get("person_id")
    session_id = data.get("session_id")
    class_id = data.get("class_id")
    if not all([person_id, session_id, class_id]):
        return jsonify({"error": "Missing fields"}), 400
    try:
        run_write(
            "INSERT OR IGNORE INTO attendance_log (person_id, session_id, class_id, marked_by) VALUES (%s,%s,%s,'manual');",
            (person_id, session_id, class_id),
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/mark-absent", methods=["POST"])
@login_required
def api_mark_absent():
    """Remove a person from present list (mark absent)."""
    data = request.get_json()
    person_id = data.get("person_id")
    session_id = data.get("session_id")
    if not all([person_id, session_id]):
        return jsonify({"error": "Missing fields"}), 400
    run_write(
        "DELETE FROM attendance_log WHERE person_id=%s AND session_id=%s;",
        (person_id, session_id),
    )
    return jsonify({"success": True})


@teacher_bp.route("/api/scan_frame", methods=["POST"])
@login_required
def api_scan_frame():
    """
    Browser webcam frame → face recognition → auto-mark present.
    Accepts: multipart/form-data with 'image' file + 'session_id' + 'class_id'
    Returns: JSON list of recognized people
    """
    import numpy as np
    import cv2
    from app.detector import FaceDetector
    from app.encoder import encode_face
    from app.matcher import find_best_match

    img_file = request.files.get("image")
    session_id = request.form.get("session_id")
    class_id = request.form.get("class_id", type=int)
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
            # Auto-mark present if enrolled in this class
            enrolled = run_query(
                "SELECT 1 FROM class_students WHERE class_id=%s AND person_id=%s;",
                (class_id, match["person_id"]),
            )
            if enrolled:
                run_write(
                    "INSERT OR IGNORE INTO attendance_log (person_id, session_id, class_id, confidence) VALUES (%s,%s,%s,%s);",
                    (match["person_id"], session_id, class_id, float(match["similarity"])),
                )
            results.append({
                "person_id": match["person_id"],
                "name": match["name"],
                "similarity": round(match["similarity"], 3),
                "box": [int(x), int(y), int(w), int(h)],
                "in_class": bool(enrolled),
            })
        else:
            results.append({"name": "Unknown", "box": [int(x), int(y), int(w), int(h)]})

    return jsonify({"faces": results})


# ── Student Edit (by teacher, for their own class) ────────────────────────────

@teacher_bp.route("/students/<int:person_id>/edit", methods=["GET", "POST"])
@login_required
def edit_student(person_id):
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip() or None
        email = request.form.get("email", "").strip() or None
        run_write(
            "UPDATE people SET name=%s, roll_no=%s, email=%s WHERE id=%s;",
            (name, roll_no, email, person_id),
        )
        flash("Student details updated.", "success")
        ref = request.form.get("ref_class_id")
        if ref:
            return redirect(url_for("teacher.class_detail", class_id=int(ref)))
        return redirect(url_for("teacher.dashboard"))

    person = run_query("SELECT * FROM people WHERE id=%s;", (person_id,))
    if not person:
        flash("Student not found.", "danger")
        return redirect(url_for("teacher.dashboard"))
    ref_class_id = request.args.get("class_id")
    return render_template("teacher/edit_student.html", person=person[0], ref_class_id=ref_class_id)


# ── Attendance Export ─────────────────────────────────────────────────────────

@teacher_bp.route("/classes/<int:class_id>/export")
@login_required
def export_attendance(class_id):
    from app.reports import generate_class_report
    from flask import send_file
    cls = run_query("SELECT * FROM classes WHERE id=%s;", (class_id,))
    if not cls:
        flash("Class not found.", "danger")
        return redirect(url_for("teacher.dashboard"))
    path = generate_class_report(class_id, cls[0])
    return send_file(path, as_attachment=True)
