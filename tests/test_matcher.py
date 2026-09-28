import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from app.matcher import find_closest, MatchResult
from app.db import init_db, get_connection, run_write, is_postgres_active


def reset_test_matcher_db():
    init_db()
    with get_connection() as conn:
        try:
            conn.execute("DELETE FROM enrolled_faces;")
            conn.execute("DELETE FROM people;")
            conn.commit()
        except Exception:
            pass


def test_matcher_known_embeddings_match_and_rejection():
    """
    Verifies that find_closest correctly matches vectors above threshold
    and rejects vectors below threshold.
    """
    reset_test_matcher_db()
    np.random.seed(123)

    # Base embedding for subject 1 (512 dimensions, normalized)
    vec1 = np.random.randn(512).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    # Base embedding for subject 2 (orthogonal / distant)
    vec2 = np.random.randn(512).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)

    # Enroll Subject 1 (Alice)
    run_write("INSERT INTO people (id, name, roll_no) VALUES (1, 'Alice Smith', 'ROLL-01');")
    with get_connection() as conn:
        if is_postgres_active():
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO enrolled_faces (person_id, embedding) VALUES (%s, %s::vector);",
                    (1, vec1.tolist()),
                )
            conn.commit()
        else:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO enrolled_faces (person_id, embedding) VALUES (?, ?);",
                (1, vec1.astype(np.float32).tobytes()),
            )
            conn.commit()

    # Query 1: Very close vector to Alice (high cosine similarity ~ 0.95 in 512-dim)
    query_close = vec1 + np.random.randn(512).astype(np.float32) * 0.01
    query_close /= np.linalg.norm(query_close)

    match = find_closest(query_close, threshold=0.62)
    assert match is not None, "Expected match for Alice, got None"
    assert match.person_id == 1, f"Expected person_id 1, got {match.person_id}"
    assert match.name == "Alice Smith"
    assert match.similarity >= 0.62

    # Query 2: Distant vector (different person, expected similarity < 0.3)
    query_distant = vec2.copy()
    no_match = find_closest(query_distant, threshold=0.62)
    assert no_match is None, f"Expected no match for distant vector, got {no_match}"


def test_matcher_picks_highest_similarity():
    """
    Verifies that when multiple enrolled faces exist, the closest one is chosen.
    """
    reset_test_matcher_db()
    np.random.seed(999)

    vec_bob = np.random.randn(512).astype(np.float32)
    vec_bob /= np.linalg.norm(vec_bob)

    vec_charlie = np.random.randn(512).astype(np.float32)
    vec_charlie /= np.linalg.norm(vec_charlie)

    # Enroll Bob (ID 2) and Charlie (ID 3)
    run_write("INSERT INTO people (id, name, roll_no) VALUES (2, 'Bob Jones', 'ROLL-02');")
    run_write("INSERT INTO people (id, name, roll_no) VALUES (3, 'Charlie Brown', 'ROLL-03');")

    with get_connection() as conn:
        if is_postgres_active():
            with conn.cursor() as cur:
                cur.execute("INSERT INTO enrolled_faces (person_id, embedding) VALUES (%s, %s::vector);", (2, vec_bob.tolist()))
                cur.execute("INSERT INTO enrolled_faces (person_id, embedding) VALUES (%s, %s::vector);", (3, vec_charlie.tolist()))
            conn.commit()
        else:
            cur = conn.cursor()
            cur.execute("INSERT INTO enrolled_faces (person_id, embedding) VALUES (?, ?);", (2, vec_bob.astype(np.float32).tobytes()))
            cur.execute("INSERT INTO enrolled_faces (person_id, embedding) VALUES (?, ?);", (3, vec_charlie.astype(np.float32).tobytes()))
            conn.commit()

    # Query matching Charlie
    query_charlie = vec_charlie + np.random.randn(512).astype(np.float32) * 0.05
    query_charlie /= np.linalg.norm(query_charlie)

    match = find_closest(query_charlie, threshold=0.60)
    assert match is not None, "Expected match for Charlie, got None"
    assert match.person_id == 3, f"Expected person_id 3, got {match.person_id}"
    assert match.name == "Charlie Brown"


if __name__ == "__main__":
    reset_test_matcher_db()
    test_matcher_known_embeddings_match_and_rejection()
    test_matcher_picks_highest_similarity()
    print("All matcher tests passed successfully!")
