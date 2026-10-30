"""GT 3D 박스로 만든 정답 — TTC 와 위험 등급.

    TTC_gt = z_near / (−Ż_gt)   (−Ż_gt > eps 일 때, 아니면 ∞)
    위험 등급: 진행 경로(|x_lat_gt| < corridor) 안에서
        TTC_gt < danger_ttc  → DANGER
        TTC_gt < caution_ttc → CAUTION
        그 밖, 또는 경로 밖   → SAFE

규칙 기반 위험도(v2 공식)로 라벨을 만들면 규칙을 규칙으로 평가하는 순환이 생긴다.
라벨은 추정에 쓰지 않는 GT 위치·속도로만 만든다.
"""

from __future__ import annotations

import numpy as np

from src.risk.rule_based import CAUTION, DANGER, SAFE


def gt_ttc(z_near, zdot_gt, eps: float = 0.1) -> np.ndarray:
    z, zd = np.asarray(z_near, float), np.asarray(zdot_gt, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(-zd > eps, z / -zd, np.inf)
    return np.where(np.isfinite(zd), t, np.nan)


def risk_label(ttc, x_lat, corridor: float, danger_ttc: float, caution_ttc: float) -> np.ndarray:
    ttc, x = np.asarray(ttc, float), np.asarray(x_lat, float)
    in_path = np.abs(x) < corridor
    lab = np.where(ttc < danger_ttc, DANGER, np.where(ttc < caution_ttc, CAUTION, SAFE))
    return np.where(in_path, lab, SAFE)
