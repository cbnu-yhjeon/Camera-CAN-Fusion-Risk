"""단안 거리 추정 — 두 가지 단서.

1. 핀홀 높이 (v2 방식):  Z = f_y · H_real / h_px
   H_real 은 클래스 평균 높이라 차종마다 ±10% 정도 편향이 생긴다. 다만 편향이 객체마다
   **일정한 배율** s 로 들어오므로 (Z_meas = s·Z) TTC = Z / (−Ż) 에서는 상쇄된다.
2. 지면 접촉:  박스 하단 중앙 픽셀의 광선을 ego 좌표 지면(z = 0)과 교차시킨다.
   객체 높이를 몰라도 되지만 노면 경사·피치에 민감하고, 하단이 잘린 근거리 박스에는 못 쓴다.

두 값 모두 카메라 깊이(광축 방향 z)를 돌려준다 — GT 의 z_near 와 같은 양.
"""

from __future__ import annotations

import numpy as np

from src.data.geometry import quat_to_rot


class Camera:
    def __init__(self, K, translation, rotation, width: int, height: int):
        self.K = np.asarray(K, dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.t = np.asarray(translation, dtype=np.float64)       # ego 좌표의 카메라 위치
        self.R = quat_to_rot(rotation)                           # camera → ego
        self.W, self.H = width, height

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    def height_distance(self, h_px, h_real):
        h_px = np.asarray(h_px, dtype=np.float64)
        return np.where(h_px > 1, self.fy * np.asarray(h_real) / np.maximum(h_px, 1), np.nan)

    def ground_distance(self, u, v, max_z: float = 80.0):
        """픽셀 (u, v) 광선과 ego 지면 z=0 의 교점 → 카메라 깊이. 지평선 위·너무 멀면 NaN."""
        uv1 = np.stack([np.asarray(u, float), np.asarray(v, float), np.ones(np.shape(u))], axis=-1)
        d_cam = uv1 @ self.Kinv.T
        d_ego = d_cam @ self.R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -self.t[2] / d_ego[..., 2]                       # 광선 매개변수 (d_cam z 성분 = 1)
        z = np.where((d_ego[..., 2] < -1e-6) & (s > 0), s * d_cam[..., 2], np.nan)
        return np.where(z <= max_z, z, np.nan)

    def lateral(self, u, z):
        """카메라 x (오른쪽 +) [m]."""
        return (np.asarray(u, float) - self.K[0, 2]) * np.asarray(z, float) / self.fx

    def truncation(self, x1, y1, x2, y2, border: float = 4):
        """(높이 측정 불가, 하단 측정 불가) — 박스가 위/아래 경계에 닿으면 높이가 잘린 것."""
        top = np.asarray(y1) <= border
        bottom = np.asarray(y2) >= self.H - border
        side = (np.asarray(x1) <= border) | (np.asarray(x2) >= self.W - border)
        return top | bottom, bottom | side
