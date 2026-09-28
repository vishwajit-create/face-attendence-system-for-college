"""
db.py — PostgreSQL + pgvector connection pool with SQLite fallback.
"""
import sqlite3
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from pathlib import Path

from app.config import DATABASE_URL, BASE_DIR

logger = logging.getLogger(__name__)

is_postgres = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")
pg_pool = None
_pg_pool = None

if is_postgres:
    try:
        import psycopg2
        from psycopg2 import pool as pg_pool_module
        from psycopg2.extras import RealDictCursor
        from pgvector.psycopg2 import register_vector
    except ImportError:
        logger.warning("psycopg2 or pgvector not installed. Falling back to SQLite.")
        is_postgres = False


_pg_tested = False
_pg_available = False


def _get_pg_pool():
    global _pg_pool, _pg_tested, _pg_available
    if not is_postgres:
        return None
    if _pg_tested:
        return _pg_pool if _pg_available else None

    _pg_tested = True
    try:
        import psycopg2
        from psycopg2 import pool as pg_pool_module
        _pg_pool = pg_pool_module.SimpleConnectionPool(1, 10, dsn=DATABASE_URL, connect_timeout=1)
        _pg_available = True
        logger.info("PostgreSQL connection pool initialized.")
    except Exception as e:
        _pg_available = False
        logger.info("PostgreSQL not reachable on localhost:5432. Using local SQLite.")
        return None
    return _pg_pool


SQLITE_DB_PATH = BASE_DIR / "attendance_local.db"


def is_postgres_active() -> bool:
    return _get_pg_pool() is not None


@contextmanager
def get_connection():
    p_pool = _get_pg_pool()
    if p_pool:
        conn = p_pool.getconn()
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
            yield conn
        finally:
            p_pool.putconn(conn)
    else:
        conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def _to_sqlite_sql(sql: str) -> str:
    s = sql.replace("%s", "?")
    s = s.replace("now()", "CURRENT_TIMESTAMP")
    s = s.replace("::vector", "")
    return s


def run_query(sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        if is_postgres_active():
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        else:
            sqlite_sql = _to_sqlite_sql(sql)
            cur = conn.cursor()
            cur.execute(sqlite_sql, params or ())
            rows = cur.fetchall()
            return [dict(row) for row in rows]


def run_write(sql: str, params: Optional[tuple] = None) -> None:
    with get_connection() as conn:
        if is_postgres_active():
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
            conn.commit()
        else:
            sqlite_sql = _to_sqlite_sql(sql)
            cur = conn.cursor()
            cur.execute(sqlite_sql, params or ())
            conn.commit()


def run_write_returning(sql: str, params: Optional[tuple] = None) -> Optional[Any]:
    """Execute INSERT ... RETURNING id and return the value."""
    with get_connection() as conn:
        if is_postgres_active():
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
        else:
            sqlite_sql = _to_sqlite_sql(sql).replace(" RETURNING id", "").replace(" returning id", "")
            cur = conn.cursor()
            cur.execute(sqlite_sql, params or ())
            conn.commit()
            return cur.lastrowid


def init_db(schema_path=None):
    """Initialize database schema."""
    p_pool = _get_pg_pool()
    if p_pool:
        target_schema = schema_path or (BASE_DIR / "schema.sql")
        import os
        if os.path.exists(target_schema):
            with open(target_schema, "r", encoding="utf-8") as f:
                ddl = f.read()
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
                conn.commit()
            logger.info("PostgreSQL schema initialized.")
    else:
        # SQLite schema with all new tables
        ddl = """
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'teacher',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            code TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            semester INTEGER,
            year INTEGER,
            total_working_days INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS class_teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
            teacher_id INTEGER NOT NULL REFERENCES admin_users(id) ON DELETE CASCADE,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(class_id, teacher_id)
        );
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE,
            email TEXT,
            phone TEXT,
            department_id INTEGER REFERENCES departments(id),
            semester INTEGER,
            role TEXT DEFAULT 'student',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS class_students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
            person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(class_id, person_id)
        );
        CREATE TABLE IF NOT EXISTS enrolled_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
            embedding BLOB,
            source_image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            class_id INTEGER REFERENCES classes(id),
            teacher_id INTEGER REFERENCES admin_users(id),
            label TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            is_active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER REFERENCES people(id),
            session_id TEXT REFERENCES sessions(id),
            class_id INTEGER REFERENCES classes(id),
            marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            marked_by TEXT DEFAULT 'face_recognition',
            UNIQUE(person_id, session_id)
        );
        CREATE TABLE IF NOT EXISTS unknown_face_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES sessions(id),
            class_id INTEGER REFERENCES classes(id),
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT
        );
        """
        with get_connection() as conn:
            conn.executescript(ddl)
            conn.commit()
        # Insert default super admin if not exists
        _insert_default_admin()
        logger.info(f"SQLite schema initialized at {SQLITE_DB_PATH}.")


def _insert_default_admin():
    """Creates default superadmin account. Password: Admin@123"""
    rows = run_query("SELECT id FROM admin_users WHERE username = 'superadmin';")
    if not rows:
        try:
            import bcrypt
            pw_hash = bcrypt.hashpw(b"Admin@123", bcrypt.gensalt(12)).decode()
        except ImportError:
            pw_hash = "plain:Admin@123"
        run_write(
            "INSERT OR IGNORE INTO admin_users (username, full_name, email, password_hash, role) VALUES (?,?,?,?,?);",
            ("superadmin", "Super Administrator", "admin@college.edu", pw_hash, "super_admin"),
        )
