"""좌표 변환 유틸 — 쿼터니언, 카메라 투영, 3D 박스 꼭짓점.

nuScenes 규약: 쿼터니언 [w, x, y, z], 카메라 좌표는 x 오른쪽 · y 아래 · z 앞(광축).
devkit 없이 numpy만 쓰므로 단위 테스트가 가볍다.
"""

from __future__ import annotations

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_yaw(q) -> float:
    """z축 회전각(heading) [rad]."""
    R = quat_to_rot(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def box_corners(center, size, rotation) -> np.ndarray:
    """3D 박스 꼭짓점 (8, 3). size = [w, l, h] (nuScenes 순서), 박스 x축이 길이 방향."""
    w, l, h = size
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2
    pts = np.stack([x, y, z], axis=1)
    return pts @ quat_to_rot(rotation).T + np.asarray(center, dtype=np.float64)


def global_to_cam(points, ego_t, ego_q, cam_t, cam_q) -> np.ndarray:
    """global → ego → camera. points (N, 3)."""
    p = (np.asarray(points, dtype=np.float64) - ego_t) @ quat_to_rot(ego_q)
    return (p - cam_t) @ quat_to_rot(cam_q)


def project(points_cam, K) -> np.ndarray:
    """카메라 좌표 (N, 3) → 픽셀 (N, 2). z > 0 가정."""
    p = np.asarray(points_cam, dtype=np.float64) @ np.asarray(K).T
    return p[:, :2] / p[:, 2:3]
