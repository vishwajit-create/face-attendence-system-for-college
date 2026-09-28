import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.attendance import DebounceBuffer, mark_present
from app.db import init_db, get_connection, run_query, run_write


def reset_test_db():
    init_db()
    with get_connection() as conn:
        try:
            conn.execute("DELETE FROM attendance_log;")
            conn.execute("DELETE FROM enrolled_faces;")
            conn.execute("DELETE FROM people;")
            conn.execute("DELETE FROM sessions;")
            conn.commit()
        except Exception:
            pass


def test_debounce_buffer_majority_voting():
    """
    Verifies that DebounceBuffer does not trigger logging until majority vote
    threshold (e.g. 5 out of 7) is satisfied.
    """
    debounce = DebounceBuffer(required_votes=5, window_size=7)
    face_id = 1
    person_id = 101

    # 1st to 4th frames: consensus not yet reached
    assert debounce.add_observation(face_id, person_id) is False
    assert debounce.add_observation(face_id, person_id) is False
    assert debounce.add_observation(face_id, person_id) is False
    assert debounce.add_observation(face_id, None) is False  # brief flicker/miss
    assert debounce.add_observation(face_id, person_id) is False  # 4 votes total

    # 5th vote arrives: majority consensus reached!
    assert debounce.add_observation(face_id, person_id) is True

    # Subsequent frames with same person: already logged, should not trigger again
    assert debounce.add_observation(face_id, person_id) is False
    assert debounce.add_observation(face_id, person_id) is False


def test_db_duplicate_prevention_guarantee():
    """
    Verifies that the database UNIQUE (person_id, session_id) constraint
    prevents duplicate attendance entries even if application logic attempts duplicate insert.
    """
    reset_test_db()

    # Create test person
    run_write("INSERT INTO people (id, name, roll_no) VALUES (1, 'Alice Test', 'ALICE001');")
    session_id = "TEST-SESSION-001"

    # First insert succeeds
    first_attempt = mark_present(person_id=1, session_id=session_id, confidence=0.95)
    assert first_attempt is True

    # Check exactly 1 row exists
    rows = run_query("SELECT COUNT(*) AS count FROM attendance_log WHERE person_id = 1 AND session_id = %s;", (session_id,))
    assert rows[0]["count"] == 1

    # Second insert with identical (person_id, session_id) should be safely ignored
    second_attempt = mark_present(person_id=1, session_id=session_id, confidence=0.98)
    assert second_attempt is False

    # Verify still exactly 1 row
    rows_after = run_query("SELECT COUNT(*) AS count FROM attendance_log WHERE person_id = 1 AND session_id = %s;", (session_id,))
    assert rows_after[0]["count"] == 1


if __name__ == "__main__":
    reset_test_db()
    test_debounce_buffer_majority_voting()
    test_db_duplicate_prevention_guarantee()
    print("All attendance tests passed successfully!")
