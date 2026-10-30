"""v2 위험도 공식 이식 (monocular-forward-collision-warning / Final_exam/src/risk.py).

    base  = 0.45·size + 0.45·center + 0.10·feature
    score = clip(base · vis_w · ttc_w · lane_w, 0, 1)
    level = DANGER (≥0.6) | CAUTION (≥0.35) | SAFE

    size   = clip(1 − (d − 5)/45, 0, 1)           핀홀 거리 d [m]
    center = 1 − ‖(cx, cy) − 화면중심‖ / √2        (정규화 좌표)
    vis_w  = 1 + 0.3·(1 − visibility)
    ttc_w  = clip(1 + 0.4·max(0, 1 − TTC/3), 1, 1.4)
    lane_w = 1.1 (차선 안) | 0.85

이 프로젝트에서 바꾼 것은 ttc_w 에 들어가는 TTC 소스뿐이다 (B0 면적 TTC → F1 융합 TTC).
그 밖의 차이:
  - feature (Harris/SIFT/HoG 점수, 10%)는 계산하지 않고 0 으로 둔다. 모든 소스에 동일하게 적용된다.
  - 차선 검출(Hough) 대신 '차선 안'을 |횡방향 거리| < lane_half_width 로 근사한다.
"""

from __future__ import annotations

import numpy as np

SAFE, CAUTION, DANGER = 0, 1, 2
LEVELS = ["SAFE", "CAUTION", "DANGER"]


def visibility_score(gray: np.ndarray) -> float:
    """v2 preprocessing.compute_visibility_score 그대로."""
    brightness = 1.0 - abs(gray.mean() - 128) / 128.0
    contrast = min(gray.std() / 64.0, 1.0)
    return float(np.clip(0.5 * brightness + 0.5 * contrast, 0.0, 1.0))


def v2_score(distance_m, cx, cy, img_w, img_h, visibility, ttc, in_lane, feature_score=0.0):
    distance_m, ttc = np.asarray(distance_m, float), np.asarray(ttc, float)
    size = np.clip(1.0 - (distance_m - 5.0) / 45.0, 0.0, 1.0)
    size = np.where(np.isfinite(distance_m), size, 0.0)
    dx = (np.asarray(cx) - img_w / 2) / (img_w / 2)
    dy = (np.asarray(cy) - img_h / 2) / (img_h / 2)
    center = 1.0 - np.clip(np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2), 0.0, 1.0)
    vis_w = 1.0 + 0.3 * (1.0 - np.asarray(visibility, float))
    ttc_w = np.where(np.isfinite(ttc), np.clip(1.0 + 0.4 * np.maximum(0.0, 1.0 - ttc / 3.0), 1.0, 1.4), 1.0)
    lane_w = np.where(in_lane, 1.1, 0.85)
    base = 0.45 * size + 0.45 * center + 0.10 * feature_score
    return np.clip(base * vis_w * ttc_w * lane_w, 0.0, 1.0)


def v2_level(score) -> np.ndarray:
    score = np.asarray(score)
    return np.where(score >= 0.6, DANGER, np.where(score >= 0.35, CAUTION, SAFE))
