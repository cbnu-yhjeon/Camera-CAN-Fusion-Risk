"""IoU 기반 다중 객체 추적 (SORT 축소판).

v2 는 LK 광류로 박스를 따라갔지만, 매 프레임 검출이 있는 상황에서는 검출–트랙 연관만
하면 된다. 트랙 박스를 등속(픽셀) 예측한 뒤 1−IoU 비용으로 헝가리안 매칭한다.
차량(car·truck·bus)끼리는 클래스가 달라도 연관한다 — 검출기가 SUV 를 car/truck 사이에서
프레임마다 바꾸는 일이 흔하다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle",
         "pedestrian": "pedestrian", "bicycle": "cycle", "motorcycle": "cycle"}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) × (M,4) xyxy → (N, M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass
class _Track:
    tid: int
    box: np.ndarray
    group: str
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    misses: int = 0
    classes: Counter = field(default_factory=Counter)

    def predicted(self) -> np.ndarray:
        return self.box + self.vel * (1 + self.misses)


class IoUTracker:
    def __init__(self, iou_match: float = 0.3, max_age: int = 3):
        self.iou_match, self.max_age = iou_match, max_age
        self.tracks: list[_Track] = []
        self._next = 0

    def update(self, boxes: np.ndarray, classes: list[str]) -> np.ndarray:
        """한 프레임의 검출 → 검출별 트랙 id."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        ids = np.full(len(boxes), -1, dtype=np.int64)
        groups = [GROUP[c] for c in classes]
        if self.tracks and len(boxes):
            pred = np.stack([t.predicted() for t in self.tracks])
            iou = iou_matrix(pred, boxes)
            same = np.array([[t.group == g for g in groups] for t in self.tracks])
            cost = np.where(same & (iou >= self.iou_match), 1 - iou, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= 1e6:
                    continue
                t = self.tracks[r]
                gap = 1 + t.misses
                t.vel = 0.5 * t.vel + 0.5 * (boxes[c] - t.box) / gap
                t.box, t.hits, t.misses = boxes[c], t.hits + 1, 0
                t.classes[classes[c]] += 1
                ids[c] = t.tid
        matched = set(ids[ids >= 0])
        for t in self.tracks:
            if t.tid not in matched:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        for c in np.where(ids < 0)[0]:
            t = _Track(self._next, boxes[c], groups[c], classes=Counter([classes[c]]))
            self._next += 1
            self.tracks.append(t)
            ids[c] = t.tid
        return ids
