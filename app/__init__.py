"""
__init__.py — Flask application factory
"""
from flask import Flask, redirect, url_for
from flask_login import LoginManager

from app.config import FLASK_SECRET_KEY
from app.db import init_db


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = FLASK_SECRET_KEY

    # ── Init DB ───────────────────────────────────────────────
    init_db()

    # ── Flask-Login ───────────────────────────────────────────
    login_manager = LoginManager(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to continue."
    login_manager.login_message_category = "warning"

    @login_manager.user_loader
    def load_user(user_id):
        from app.auth import get_user_by_id
        return get_user_by_id(int(user_id))

    @app.context_processor
    def inject_csrf():
        return dict(csrf_token=lambda: "")

    # ── Blueprints ────────────────────────────────────────────
    from app.routes_auth import auth_bp
    from app.routes_super_admin import super_admin_bp
    from app.routes_teacher import teacher_bp
    from app.routes import main_bp  # legacy routes (video_feed, enroll)

    app.register_blueprint(auth_bp)
    app.register_blueprint(super_admin_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(main_bp)

    @app.route("/")
    @app.route("/dashboard")
    def index():
        from flask_login import current_user
        if current_user.is_authenticated:
            if current_user.role == "super_admin":
                return redirect(url_for("super_admin.dashboard"))
            return redirect(url_for("teacher.dashboard"))
        return redirect(url_for("auth.login"))

    return app
