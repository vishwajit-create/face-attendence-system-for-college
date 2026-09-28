"""
auth.py — Authentication helpers: password hashing, user loading for Flask-Login
"""
import bcrypt
import logging
from typing import Optional

from flask_login import UserMixin

from app.db import run_query, run_write

logger = logging.getLogger(__name__)


class AdminUser(UserMixin):
    """Flask-Login compatible user object."""

    def __init__(self, data: dict):
        self.id = str(data["id"])
        self.username = data["username"]
        self.full_name = data["full_name"]
        self.email = data.get("email", "")
        self.role = data["role"]   # 'super_admin' | 'teacher'
        self.is_active_flag = data.get("is_active", True)

    def get_id(self):
        return self.id

    @property
    def is_super_admin(self):
        return self.role == "super_admin"

    @property
    def is_teacher(self):
        return self.role == "teacher"

    # Flask-Login requires this
    @property
    def is_active(self):
        return bool(self.is_active_flag)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(12)).decode()


def check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def get_user_by_id(user_id: int) -> Optional[AdminUser]:
    rows = run_query(
        "SELECT * FROM admin_users WHERE id = %s AND is_active = TRUE;",
        (int(user_id),),
    )
    return AdminUser(rows[0]) if rows else None


def get_user_by_username(username: str) -> Optional[AdminUser]:
    rows = run_query(
        "SELECT * FROM admin_users WHERE username = %s AND is_active = TRUE;",
        (username,),
    )
    return AdminUser(rows[0]) if rows else None


def authenticate(username: str, password: str) -> Optional[AdminUser]:
    """Returns AdminUser if credentials are valid, otherwise None."""
    rows = run_query(
        "SELECT * FROM admin_users WHERE username = %s AND is_active = TRUE;",
        (username,),
    )
    if not rows:
        return None
    user_data = rows[0]
    if check_password(password, user_data["password_hash"]):
        run_write(
            "UPDATE admin_users SET last_login = now() WHERE id = %s;",
            (user_data["id"],),
        )
        return AdminUser(user_data)
    return None
