import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.detector import BoundingBox


class CentroidTracker:
    """
    Lightweight tracking-by-detection using centroid Euclidean distance matching (Section 4.7).
    Assigns stable IDs to detected faces across frames.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 60.0):
        # Next unique ID to assign
        self.next_object_id = 1

        # Dict mapping object_id -> BoundingBox
        self.objects: OrderedDict[int, BoundingBox] = OrderedDict()

        # Dict mapping object_id -> number of consecutive frames it has been missing
        self.disappeared: OrderedDict[int, int] = OrderedDict()

        # Cache for embeddings and match results to avoid re-embedding every frame
        self.cached_embeddings: Dict[int, np.ndarray] = {}
        self.cached_matches: Dict[int, Optional[dict]] = {}
        self.frames_since_last_embed: Dict[int, int] = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, box: BoundingBox) -> int:
        object_id = self.next_object_id
        self.objects[object_id] = box
        self.disappeared[object_id] = 0
        self.frames_since_last_embed[object_id] = 0
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.cached_embeddings.pop(object_id, None)
        self.cached_matches.pop(object_id, None)
        self.frames_since_last_embed.pop(object_id, None)

    def update(self, detected_boxes: List[BoundingBox]) -> Dict[int, BoundingBox]:
        """
        Updates tracked objects given bounding boxes in the current frame.
        Returns a dictionary mapping object_id -> BoundingBox.
        """
        # If no boxes detected, mark all existing objects as disappeared
        if len(detected_boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # If currently tracking no objects, register all detections
        if len(self.objects) == 0:
            for box in detected_boxes:
                self.register(box)
            return self.objects

        # Otherwise, match existing centroids to input centroids
        object_ids = list(self.objects.keys())
        existing_centroids = np.array([self.objects[fid].centroid for fid in object_ids])
        input_centroids = np.array([box.centroid for box in detected_boxes])

        # Compute Euclidean distance matrix between existing and input centroids
        dist_matrix = np.linalg.norm(existing_centroids[:, np.newaxis] - input_centroids, axis=2)

        # Sort matches by minimum distance
        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # If distance exceeds threshold, don't associate
            if dist_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = detected_boxes[col]
            self.disappeared[object_id] = 0
            if object_id in self.frames_since_last_embed:
                self.frames_since_last_embed[object_id] += 1

            used_rows.add(row)
            used_cols.add(col)

        # Handle rows (existing objects) that were not matched
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Handle cols (new detections) that were not matched
        unused_cols = set(range(len(detected_boxes))) - used_cols
        for col in unused_cols:
            self.register(detected_boxes[col])

        return self.objects

    def should_reencode(self, face_id: int, reencode_every_n: int = 5) -> bool:
        """
        Determines whether the face embedding needs recalculation or can use the cached result.
        """
        if face_id not in self.cached_embeddings:
            return True
        return self.frames_since_last_embed.get(face_id, 0) >= reencode_every_n
