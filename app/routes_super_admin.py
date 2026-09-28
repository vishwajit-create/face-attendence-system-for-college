"""
routes_super_admin.py — Super Admin panel routes

Features:
  - Manage departments
  - Manage classes (add / edit / delete)
  - Manage teachers (add / edit / deactivate)
  - Assign teachers to classes
  - View all enrolled students + edit their name / roll_no
  - View attendance % for every student across all classes
  - System-wide reports
"""
import uuid
from datetime import datetime
from typing import Optional

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from functools import wraps

from app.db import run_query, run_write, run_write_returning
from app.auth import hash_password

super_admin_bp = Blueprint("super_admin", __name__, url_prefix="/super-admin")


def super_admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_super_admin:
            flash("Super Admin access required.", "danger")
            return redirect(url_for("teacher.dashboard"))
        return f(*args, **kwargs)
    return decorated


# ── Dashboard ─────────────────────────────────────────────────────────────────

@super_admin_bp.route("/")
@super_admin_required
def dashboard():
    stats = {
        "classes": run_query("SELECT COUNT(*) AS c FROM classes;")[0]["c"],
        "teachers": run_query("SELECT COUNT(*) AS c FROM admin_users WHERE role='teacher';")[0]["c"],
        "students": run_query("SELECT COUNT(*) AS c FROM people WHERE role='student';")[0]["c"],
        "sessions_today": run_query(
            "SELECT COUNT(*) AS c FROM sessions WHERE DATE(started_at)=DATE('now');"
        )[0]["c"],
    }
    recent_sessions = run_query("""
        SELECT s.id, s.label, s.started_at, s.is_active,
               c.name AS class_name, c.code AS class_code,
               u.full_name AS teacher_name
        FROM sessions s
        LEFT JOIN classes c ON s.class_id = c.id
        LEFT JOIN admin_users u ON s.teacher_id = u.id
        ORDER BY s.started_at DESC LIMIT 10;
    """)
    return render_template("super_admin/dashboard.html", stats=stats, recent_sessions=recent_sessions)


# ── Departments ───────────────────────────────────────────────────────────────

@super_admin_bp.route("/departments")
@super_admin_required
def departments():
    depts = run_query("SELECT * FROM departments ORDER BY name;")
    return render_template("super_admin/departments.html", departments=depts)


@super_admin_bp.route("/departments/add", methods=["POST"])
@super_admin_required
def add_department():
    name = request.form.get("name", "").strip()
    code = request.form.get("code", "").strip().upper()
    if not name:
        flash("Department name is required.", "danger")
        return redirect(url_for("super_admin.departments"))
    try:
        run_write("INSERT INTO departments (name, code) VALUES (%s, %s);", (name, code or None))
        flash(f"Department '{name}' added.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("super_admin.departments"))


@super_admin_bp.route("/departments/<int:dept_id>/delete", methods=["POST"])
@super_admin_required
def delete_department(dept_id):
    run_write("DELETE FROM departments WHERE id = %s;", (dept_id,))
    flash("Department deleted.", "success")
    return redirect(url_for("super_admin.departments"))


# ── Classes ───────────────────────────────────────────────────────────────────

@super_admin_bp.route("/classes")
@super_admin_required
def classes():
    all_classes = run_query("""
        SELECT c.*, d.name AS dept_name,
               GROUP_CONCAT(u.full_name, ', ') AS teachers
        FROM classes c
        LEFT JOIN departments d ON c.department_id = d.id
        LEFT JOIN class_teachers ct ON ct.class_id = c.id
        LEFT JOIN admin_users u ON u.id = ct.teacher_id
        GROUP BY c.id ORDER BY c.code;
    """)
    depts = run_query("SELECT * FROM departments ORDER BY name;")
    return render_template("super_admin/classes.html", classes=all_classes, departments=depts)


@super_admin_bp.route("/classes/add", methods=["POST"])
@super_admin_required
def add_class():
    name = request.form.get("name", "").strip()
    code = request.form.get("code", "").strip().upper()
    dept_id = request.form.get("department_id") or None
    semester = request.form.get("semester") or None
    if not name or not code:
        flash("Class name and code are required.", "danger")
        return redirect(url_for("super_admin.classes"))
    try:
        run_write(
            "INSERT INTO classes (name, code, department_id, semester) VALUES (%s,%s,%s,%s);",
            (name, code, dept_id, semester),
        )
        flash(f"Class '{name}' added.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("super_admin.classes"))


@super_admin_bp.route("/classes/<int:class_id>/edit", methods=["GET", "POST"])
@super_admin_required
def edit_class(class_id):
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        code = request.form.get("code", "").strip().upper()
        dept_id = request.form.get("department_id") or None
        semester = request.form.get("semester") or None
        run_write(
            "UPDATE classes SET name=%s, code=%s, department_id=%s, semester=%s WHERE id=%s;",
            (name, code, dept_id, semester, class_id),
        )
        flash("Class updated.", "success")
        return redirect(url_for("super_admin.classes"))
    cls = run_query("SELECT * FROM classes WHERE id=%s;", (class_id,))
    depts = run_query("SELECT * FROM departments ORDER BY name;")
    if not cls:
        flash("Class not found.", "danger")
        return redirect(url_for("super_admin.classes"))
    return render_template("super_admin/edit_class.html", cls=cls[0], departments=depts)


@super_admin_bp.route("/classes/<int:class_id>/delete", methods=["POST"])
@super_admin_required
def delete_class(class_id):
    run_write("DELETE FROM classes WHERE id=%s;", (class_id,))
    flash("Class deleted.", "success")
    return redirect(url_for("super_admin.classes"))


# ── Teachers ──────────────────────────────────────────────────────────────────

@super_admin_bp.route("/teachers")
@super_admin_required
def teachers():
    all_teachers = run_query("""
        SELECT u.*, GROUP_CONCAT(c.name || ' (' || c.code || ')', ', ') AS assigned_classes
        FROM admin_users u
        LEFT JOIN class_teachers ct ON ct.teacher_id = u.id
        LEFT JOIN classes c ON c.id = ct.class_id
        WHERE u.role = 'teacher'
        GROUP BY u.id ORDER BY u.full_name;
    """)
    all_classes = run_query("SELECT * FROM classes ORDER BY code;")
    return render_template("super_admin/teachers.html", teachers=all_teachers, classes=all_classes)


@super_admin_bp.route("/teachers/add", methods=["POST"])
@super_admin_required
def add_teacher():
    full_name = request.form.get("full_name", "").strip()
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip() or None
    password = request.form.get("password", "").strip()
    class_ids = request.form.getlist("class_ids")

    if not full_name or not username or not password:
        flash("Name, username and password are required.", "danger")
        return redirect(url_for("super_admin.teachers"))
    try:
        pw_hash = hash_password(password)
        teacher_id = run_write_returning(
            "INSERT INTO admin_users (username, full_name, email, password_hash, role) VALUES (%s,%s,%s,%s,'teacher') RETURNING id;",
            (username, full_name, email, pw_hash),
        )
        for cid in class_ids:
            run_write(
                "INSERT OR IGNORE INTO class_teachers (class_id, teacher_id) VALUES (%s,%s);",
                (int(cid), teacher_id),
            )
        flash(f"Teacher '{full_name}' added.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("super_admin.teachers"))


@super_admin_bp.route("/teachers/<int:teacher_id>/edit", methods=["GET", "POST"])
@super_admin_required
def edit_teacher(teacher_id):
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip() or None
        password = request.form.get("password", "").strip()
        class_ids = request.form.getlist("class_ids")

        run_write(
            "UPDATE admin_users SET full_name=%s, email=%s WHERE id=%s;",
            (full_name, email, teacher_id),
        )
        if password:
            run_write(
                "UPDATE admin_users SET password_hash=%s WHERE id=%s;",
                (hash_password(password), teacher_id),
            )
        # Re-assign classes
        run_write("DELETE FROM class_teachers WHERE teacher_id=%s;", (teacher_id,))
        for cid in class_ids:
            run_write(
                "INSERT OR IGNORE INTO class_teachers (class_id, teacher_id) VALUES (%s,%s);",
                (int(cid), teacher_id),
            )
        flash("Teacher updated.", "success")
        return redirect(url_for("super_admin.teachers"))

    teacher = run_query("SELECT * FROM admin_users WHERE id=%s;", (teacher_id,))
    all_classes = run_query("SELECT * FROM classes ORDER BY code;")
    assigned = run_query("SELECT class_id FROM class_teachers WHERE teacher_id=%s;", (teacher_id,))
    assigned_ids = {row["class_id"] for row in assigned}
    if not teacher:
        flash("Teacher not found.", "danger")
        return redirect(url_for("super_admin.teachers"))
    return render_template(
        "super_admin/edit_teacher.html",
        teacher=teacher[0],
        classes=all_classes,
        assigned_ids=assigned_ids,
    )


@super_admin_bp.route("/teachers/<int:teacher_id>/toggle", methods=["POST"])
@super_admin_required
def toggle_teacher(teacher_id):
    rows = run_query("SELECT is_active FROM admin_users WHERE id=%s;", (teacher_id,))
    if rows:
        new_val = 0 if rows[0]["is_active"] else 1
        run_write("UPDATE admin_users SET is_active=%s WHERE id=%s;", (new_val, teacher_id))
        flash("Teacher account " + ("activated." if new_val else "deactivated."), "success")
    return redirect(url_for("super_admin.teachers"))


# ── Students ──────────────────────────────────────────────────────────────────

@super_admin_bp.route("/students")
@super_admin_required
def students():
    class_id = request.args.get("class_id", type=int)
    search = request.args.get("q", "").strip()

    all_classes = run_query("SELECT * FROM classes ORDER BY code;")

    base_sql = """
        SELECT p.*, d.name AS dept_name,
               GROUP_CONCAT(c.name || ' (' || c.code || ')', ', ') AS enrolled_classes,
               COUNT(DISTINCT ef.id) AS face_count
        FROM people p
        LEFT JOIN departments d ON d.id = p.department_id
        LEFT JOIN class_students cs ON cs.person_id = p.id
        LEFT JOIN classes c ON c.id = cs.class_id
        LEFT JOIN enrolled_faces ef ON ef.person_id = p.id
    """
    conditions = ["p.role = 'student'"]
    params = []
    if class_id:
        conditions.append("cs.class_id = %s")
        params.append(class_id)
    if search:
        conditions.append("(p.name LIKE %s OR p.roll_no LIKE %s OR p.email LIKE %s)")
        params += [f"%{search}%", f"%{search}%", f"%{search}%"]

    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    query = base_sql + where + " GROUP BY p.id ORDER BY p.name;"
    students_list = run_query(query, tuple(params))

    return render_template(
        "super_admin/students.html",
        students=students_list,
        classes=all_classes,
        selected_class=class_id,
        search=search,
    )


@super_admin_bp.route("/students/<int:person_id>/edit", methods=["GET", "POST"])
@super_admin_required
def edit_student(person_id):
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip() or None
        email = request.form.get("email", "").strip() or None
        phone = request.form.get("phone", "").strip() or None
        dept_id = request.form.get("department_id") or None
        semester = request.form.get("semester") or None
        class_ids = request.form.getlist("class_ids")

        run_write(
            "UPDATE people SET name=%s, roll_no=%s, email=%s, phone=%s, department_id=%s, semester=%s WHERE id=%s;",
            (name, roll_no, email, phone, dept_id, semester, person_id),
        )
        # Update class enrollment
        run_write("DELETE FROM class_students WHERE person_id=%s;", (person_id,))
        for cid in class_ids:
            run_write(
                "INSERT OR IGNORE INTO class_students (class_id, person_id) VALUES (%s,%s);",
                (int(cid), person_id),
            )
        flash("Student updated.", "success")
        return redirect(url_for("super_admin.students"))

    person = run_query("SELECT * FROM people WHERE id=%s;", (person_id,))
    all_classes = run_query("SELECT * FROM classes ORDER BY code;")
    depts = run_query("SELECT * FROM departments ORDER BY name;")
    enrolled = run_query("SELECT class_id FROM class_students WHERE person_id=%s;", (person_id,))
    enrolled_ids = {row["class_id"] for row in enrolled}
    if not person:
        flash("Student not found.", "danger")
        return redirect(url_for("super_admin.students"))
    return render_template(
        "super_admin/edit_student.html",
        person=person[0],
        classes=all_classes,
        departments=depts,
        enrolled_ids=enrolled_ids,
    )


@super_admin_bp.route("/students/<int:person_id>/delete", methods=["POST"])
@super_admin_required
def delete_student(person_id):
    person = run_query("SELECT name FROM people WHERE id=%s;", (person_id,))
    name = person[0]["name"] if person else f"Student #{person_id}"

    # 1. Clean up face photos from static directory
    faces = run_query("SELECT source_image_path FROM enrolled_faces WHERE person_id=%s;", (person_id,))
    import os
    for f in faces:
        path = f.get("source_image_path")
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    # 2. Remove all related database records
    run_write("DELETE FROM enrolled_faces WHERE person_id=%s;", (person_id,))
    run_write("DELETE FROM class_students WHERE person_id=%s;", (person_id,))
    run_write("DELETE FROM attendance_log WHERE person_id=%s;", (person_id,))
    run_write("DELETE FROM people WHERE id=%s;", (person_id,))

    flash(f"Student '{name}' and all associated face enrollments have been deleted.", "success")
    return redirect(url_for("super_admin.students"))


@super_admin_bp.route("/students/<int:person_id>/clear-faces", methods=["POST"])
@super_admin_required
def clear_student_faces(person_id):
    person = run_query("SELECT name FROM people WHERE id=%s;", (person_id,))
    name = person[0]["name"] if person else f"Student #{person_id}"

    faces = run_query("SELECT source_image_path FROM enrolled_faces WHERE person_id=%s;", (person_id,))
    import os
    for f in faces:
        path = f.get("source_image_path")
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    run_write("DELETE FROM enrolled_faces WHERE person_id=%s;", (person_id,))
    flash(f"Face enrollment data cleared for '{name}'. You can now re-enroll their face.", "success")
    return redirect(url_for("super_admin.edit_student", person_id=person_id))


# ── Attendance Reports ─────────────────────────────────────────────────────────

@super_admin_bp.route("/attendance")
@super_admin_required
def attendance():
    class_id = request.args.get("class_id", type=int)
    all_classes = run_query("SELECT * FROM classes ORDER BY code;")
    report = []

    if class_id:
        # Total sessions for this class
        total_sessions = run_query(
            "SELECT COUNT(*) AS c FROM sessions WHERE class_id=%s;", (class_id,)
        )[0]["c"]

        # All enrolled students in this class
        students_in_class = run_query("""
            SELECT p.id, p.name, p.roll_no
            FROM class_students cs
            JOIN people p ON p.id = cs.person_id
            WHERE cs.class_id = %s
            ORDER BY p.name;
        """, (class_id,))

        for student in students_in_class:
            attended = run_query("""
                SELECT COUNT(*) AS c FROM attendance_log
                WHERE person_id=%s AND class_id=%s;
            """, (student["id"], class_id))[0]["c"]

            pct = round((attended / total_sessions * 100) if total_sessions > 0 else 0, 1)
            report.append({
                **student,
                "attended": attended,
                "total": total_sessions,
                "percentage": pct,
                "status": "✅ Good" if pct >= 75 else ("⚠️ Low" if pct >= 50 else "❌ Critical"),
            })

    return render_template(
        "super_admin/attendance_report.html",
        classes=all_classes,
        selected_class=class_id,
        report=report,
    )
