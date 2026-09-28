import pickle
import logging
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from app.config import MATCH_THRESHOLD
from app.db import get_connection, run_query, is_postgres_active

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    person_id: int
    name: str
    roll_no: Optional[str]
    similarity: float

    def to_dict(self):
        return {
            "person_id": self.person_id,
            "name": self.name,
            "roll_no": self.roll_no,
            "similarity": round(self.similarity, 4),
        }


def find_closest(
    query_embedding: np.ndarray,
    threshold: float = MATCH_THRESHOLD,
) -> Optional[MatchResult]:
    """
    Performs cosine similarity search against enrolled faces in the database.
    Returns MatchResult if the closest enrolled face has similarity >= threshold; otherwise None.
    """
    if query_embedding is None or len(query_embedding) == 0:
        return None

    # Ensure query embedding is 512-dim float32 and L2-normalized
    query_vec = np.asarray(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 1e-9:
        query_vec = query_vec / norm

    with get_connection() as conn:
        if is_postgres_active():
            try:
                # PostgreSQL with pgvector cosine distance operator (<=>)
                # Cosine similarity = 1 - cosine_distance
                sql = """
                SELECT 
                    ef.person_id,
                    p.name,
                    p.roll_no,
                    (1 - (ef.embedding <=> %s::vector)) AS similarity
                FROM enrolled_faces ef
                JOIN people p ON ef.person_id = p.id
                ORDER BY ef.embedding <=> %s::vector ASC
                LIMIT 1;
                """
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Pass the numpy vector as list
                    vec_param = query_vec.tolist()
                    cur.execute(sql, (vec_param, vec_param))
                    row = cur.fetchone()
                    if row:
                        similarity = float(row["similarity"])
                        if similarity >= threshold:
                            return MatchResult(
                                person_id=int(row["person_id"]),
                                name=str(row["name"]),
                                roll_no=row["roll_no"],
                                similarity=similarity,
                            )
                return None
            except Exception as e:
                logger.warning(f"PostgreSQL pgvector query failed ({e}). Attempting in-memory fallback.")

        # SQLite / In-memory fallback
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT ef.person_id, p.name, p.roll_no, ef.embedding
                FROM enrolled_faces ef
                JOIN people p ON ef.person_id = p.id
            """)
            rows = cur.fetchall()

            best_match: Optional[MatchResult] = None
            highest_sim = -1.0

            for row in rows:
                person_id = row[0]
                name = row[1]
                roll_no = row[2]
                raw_emb = row[3]

                if raw_emb is None:
                    continue

                if isinstance(raw_emb, bytes):
                    emb = np.frombuffer(raw_emb, dtype=np.float32)
                elif isinstance(raw_emb, (list, tuple)):
                    emb = np.array(raw_emb, dtype=np.float32)
                else:
                    emb = np.array(raw_emb, dtype=np.float32)

                # Ensure enrolled vector is normalized
                enrolled_norm = np.linalg.norm(emb)
                if enrolled_norm > 1e-9:
                    emb = emb / enrolled_norm

                # Compute cosine similarity (dot product of unit vectors)
                similarity = float(np.dot(query_vec, emb))

                if similarity > highest_sim:
                    highest_sim = similarity
                    best_match = MatchResult(
                        person_id=int(person_id),
                        name=str(name),
                        roll_no=roll_no,
                        similarity=similarity,
                    )

            if best_match and highest_sim >= threshold:
                return best_match
            return None
        except Exception as e:
            logger.error(f"Fallback matching failed: {e}")
            return None


def find_best_match(query_embedding: np.ndarray, threshold: float = MATCH_THRESHOLD):
    """Alias for find_closest that returns a dict (or None)."""
    result = find_closest(query_embedding, threshold)
    if result is None:
        return None
    return {
        "person_id": result.person_id,
        "name": result.name,
        "roll_no": result.roll_no,
        "similarity": result.similarity,
    }
