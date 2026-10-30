"""객체 상태 분류 — 정지 / 동행 / 접근 / 이탈.

    v_obj ≈ v_ego + Ż       (진행 방향 성분, v_ego 는 CAN)
    |v_obj| < stationary_v   → 정지        (주차 차량, 신호 대기 중인 앞차)
    |Ż|     < following_zdot → 동행        (같은 속도로 달리는 앞차)
    Ż < 0                    → 접근
    그 외                     → 이탈

영상만으로는 Ż 밖에 모른다. 자차 속도 10 m/s 에서 Ż = −10 이면 '정지 장애물'과
'마주 오는 차와 내가 멈춘 상황'을 구분할 수 없다 — CAN 의 v_ego 가 그 구분을 만든다.
판정 순서상 정지가 동행보다 먼저다: 자차가 정지해 있으면 정지 객체는 Ż≈0 이기도 하다.
"""

from __future__ import annotations

import numpy as np

STATES = ["stationary", "following", "approaching", "receding"]


def classify(v_obj, zdot, stationary_v: float = 1.0, following_zdot: float = 0.5) -> np.ndarray:
    v_obj, zdot = np.asarray(v_obj, float), np.asarray(zdot, float)
    out = np.where(zdot < 0, "approaching", "receding").astype(object)
    out = np.where(np.abs(zdot) < following_zdot, "following", out)
    out = np.where(np.abs(v_obj) < stationary_v, "stationary", out)
    return np.where(np.isfinite(v_obj) & np.isfinite(zdot), out, None)
