import os
import time
import logging
from collections import deque, Counter
from typing import Dict, Optional
import cv2
import numpy as np

from app.config import (
    DEBOUNCE_FRAMES,
    DEBOUNCE_WINDOW,
    CURRENT_SESSION_ID,
    UNKNOWN_FACE_DIR,
)
from app.db import get_connection, run_query, is_postgres

logger = logging.getLogger(__name__)


class DebounceBuffer:
    """
    Temporal smoothing buffer (Section 4.5).
    Maintains a rolling window of recent matches per tracked face_id.
    Requires at least 'required_votes' matching the same person_id within 'window_size'
    frames before triggering an attendance log.
    """

    def __init__(self, required_votes: int = DEBOUNCE_FRAMES, window_size: int = DEBOUNCE_WINDOW):
        self.required_votes = required_votes
        self.window_size = window_size
        # Map face_id -> deque of recent person_ids (or None for unknown)
        self.buffers: Dict[int, deque] = {}
        # Set of person_ids already marked in the current session to prevent redundant writes
        self.logged_persons: set = set()

    def add_observation(self, face_id: int, person_id: Optional[int]) -> bool:
        """
        Records an observation for face_id.
        Returns True if the person_id reached majority consensus and has not yet been logged.
        """
        if face_id not in self.buffers:
            self.buffers[face_id] = deque(maxlen=self.window_size)

        self.buffers[face_id].append(person_id)

        if person_id is None:
            return False

        if person_id in self.logged_persons:
            return False

        # Count occurrences of person_id in the window
        votes = sum(1 for pid in self.buffers[face_id] if pid == person_id)
        if votes >= self.required_votes:
            self.logged_persons.add(person_id)
            logger.info(f"Majority vote consensus reached for person_id {person_id} ({votes}/{self.window_size} frames).")
            return True

        return False

    def should_log(self, face_id: int, person_id: int) -> bool:
        """
        Compatibility method to check if majority vote rule is satisfied.
        """
        return self.add_observation(face_id, person_id)

    def reset_session(self):
        """
        Clears the logged persons cache for a new session.
        """
        self.logged_persons.clear()
        self.buffers.clear()


def mark_present(person_id: int, session_id: str = CURRENT_SESSION_ID, confidence: float = 1.0) -> bool:
    """
    Records attendance in attendance_log table with DB-level duplicate prevention:
    INSERT ... ON CONFLICT (person_id, session_id) DO NOTHING.
    Returns True if a new row was inserted, False if attendance was already recorded.
    """
    # Ensure session exists
    ensure_session_exists(session_id)

    with get_connection() as conn:
        if is_postgres:
            sql = """
            INSERT INTO attendance_log (person_id, session_id, marked_at, confidence)
            VALUES (%s, %s, now(), %s)
            ON CONFLICT (person_id, session_id) DO NOTHING
            RETURNING id;
            """
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (person_id, session_id, confidence))
                    inserted = cur.fetchone() is not None
                conn.commit()
                if inserted:
                    logger.info(f"Attendance recorded: person_id={person_id}, session={session_id}")
                return inserted
            except Exception as e:
                conn.rollback()
                logger.error(f"Error marking attendance (Postgres): {e}")
                return False
        else:
            # SQLite ON CONFLICT DO NOTHING
            sql = """
            INSERT OR IGNORE INTO attendance_log (person_id, session_id, marked_at, confidence)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?);
            """
            try:
                cur = conn.cursor()
                cur.execute(sql, (person_id, session_id, confidence))
                inserted = cur.rowcount > 0
                conn.commit()
                if inserted:
                    logger.info(f"Attendance recorded: person_id={person_id}, session={session_id}")
                return inserted
            except Exception as e:
                logger.error(f"Error marking attendance (SQLite): {e}")
                return False


def ensure_session_exists(session_id: str, label: Optional[str] = None):
    """
    Creates the session record if it doesn't already exist.
    """
    label_val = label or f"Session {session_id}"
    with get_connection() as conn:
        if is_postgres:
            sql = """
            INSERT INTO sessions (id, label, started_at)
            VALUES (%s, %s, now())
            ON CONFLICT (id) DO NOTHING;
            """
            with conn.cursor() as cur:
                cur.execute(sql, (session_id, label_val))
            conn.commit()
        else:
            sql = """
            INSERT OR IGNORE INTO sessions (id, label, started_at)
            VALUES (?, ?, CURRENT_TIMESTAMP);
            """
            cur = conn.cursor()
            cur.execute(sql, (session_id, label_val))
            conn.commit()


def log_unknown(session_id: str, frame: np.ndarray) -> Optional[str]:
    """
    Saves an unknown face crop for audit/review and logs an entry in unknown_face_log.
    """
    if frame is None or frame.size == 0:
        return None

    ensure_session_exists(session_id)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{session_id}_{timestamp_str}_{int(time.time() * 1000) % 10000}.jpg"
    filepath = UNKNOWN_FACE_DIR / filename

    try:
        cv2.imwrite(str(filepath), frame)
        rel_path = f"/static/unknown_faces/{filename}"

        with get_connection() as conn:
            if is_postgres:
                sql = """
                INSERT INTO unknown_face_log (session_id, captured_at, image_path)
                VALUES (%s, now(), %s);
                """
                with conn.cursor() as cur:
                    cur.execute(sql, (session_id, rel_path))
                conn.commit()
            else:
                sql = """
                INSERT INTO unknown_face_log (session_id, captured_at, image_path)
                VALUES (?, CURRENT_TIMESTAMP, ?);
                """
                cur = conn.cursor()
                cur.execute(sql, (session_id, rel_path))
                conn.commit()

        logger.info(f"Unknown face logged: {rel_path}")
        return rel_path
    except Exception as e:
        logger.error(f"Failed to log unknown face: {e}")
        return None
