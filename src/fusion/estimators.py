"""트랙 하나의 프레임 시계열 → TTC 추정. 실험 표의 TTC 소스별 구현.

B0  박스 면적 변화율 (v2 그대로):  TTC = Δt · A_prev / (A − A_prev),  하한 0.5 s.
    v2 는 이 값을 '프레임 수'로 계산하고 초로 읽었다. 여기서는 Δt 를 곱해 단위만 고친다.
    면적은 1/Z² 에 비례하므로 A_prev/ΔA ≈ TTC/2 — v2 공식의 체계적 과소추정도 그대로 남긴다.
B1  영상 거리 차분 (CAN 없음):  Ż = (Z_t − Z_{t−w}) / Δt,  창 w 초.
F0  칼만 필터, CAN 입력 없음.
F1a 칼만 필터 + CAN 자차 가속도 입력 (README 초안 그대로).
F1  IMM (등속 + 정지 가설) — CAN 자차 가속도 입력 + 자차 속도로 정지 가설 Ż = −v_ego.
"""

from __future__ import annotations

import math

import numpy as np

from src.fusion.kalman import DepthKF, IMMDepth, ttc_from_state


def combine_measurements(z_h, s_h, z_g, s_g):
    """두 거리 단서의 역분산 가중 평균 (B0·B1 과 초기화용). 둘 다 NaN 이면 NaN."""
    w_h = np.where(np.isfinite(z_h), 1 / np.square(s_h), 0.0)
    w_g = np.where(np.isfinite(z_g), 1 / np.square(s_g), 0.0)
    w = w_h + w_g
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (np.nan_to_num(z_h) * w_h + np.nan_to_num(z_g) * w_g) / w
        s = np.sqrt(1 / w)
    return np.where(w > 0, z, np.nan), np.where(w > 0, s, np.nan)


def ttc_area(t_s: np.ndarray, area: np.ndarray, b0_min: float = 0.5, max_ttc: float = 20.0) -> np.ndarray:
    out = np.full(len(area), np.inf)
    for i in range(1, len(area)):
        dt = t_s[i] - t_s[i - 1]
        da = area[i] - area[i - 1]
        if da > 1e-3 and area[i - 1] > 0 and dt > 0:
            ttc = max(dt * area[i - 1] / da, b0_min)
            out[i] = ttc if ttc < max_ttc else np.inf
    return out


def zdot_difference(t_s: np.ndarray, z: np.ndarray, window_s: float) -> np.ndarray:
    """창 window_s 이상 떨어진 가장 가까운 과거 측정과의 차분. 인과적."""
    out = np.full(len(z), np.nan)
    j = 0
    for i in range(len(z)):
        if not np.isfinite(z[i]):
            continue
        while j < i and t_s[i] - t_s[j + 1] >= window_s:
            j += 1
        if j < i and t_s[i] - t_s[j] >= window_s * 0.8 and np.isfinite(z[j]):
            out[i] = (z[i] - z[j]) / (t_s[i] - t_s[j])
    return out


def run_kf(t_s, z_h, s_h, z_g, s_g, a_ego, sigma_a: float, use_can: bool,
           sigma_v0: float = 5.0, gate: float = 9.0, sigma_a_no_can_extra: float = 1.5,
           max_coast_s: float = 0.5):
    """트랙 하나에 칼만 필터. (Z, Ż, σ_Ż) 배열.

    측정이 아직 없거나 마지막 측정 후 max_coast_s 가 지난 프레임은 NaN — 잘린 박스가
    측정 없이 예측만 이어 가면 Z 가 0 아래로 내려가 TTC 0 을 만든다.
    """
    n = len(t_s)
    Z, V, SV = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    sa = sigma_a if use_can else math.hypot(sigma_a, sigma_a_no_can_extra)
    kf, t_prev = None, None
    z0, s0 = combine_measurements(z_h, s_h, z_g, s_g)
    for i in range(n):
        if kf is None:
            if np.isfinite(z0[i]):
                kf = DepthKF(float(z0[i]), float(s0[i]), sigma_v0, gate=gate)
                t_prev = t_meas = t_s[i]
                Z[i], V[i], SV[i] = kf.z, kf.v, kf.sigma_v
            continue
        u = float(a_ego[i]) if use_can and np.isfinite(a_ego[i]) else 0.0
        kf.predict(t_s[i] - t_prev, u, sa)
        t_prev = t_s[i]
        ok = False
        if np.isfinite(z_h[i]):
            ok |= kf.update(float(z_h[i]), float(s_h[i]))
        if np.isfinite(z_g[i]):
            ok |= kf.update(float(z_g[i]), float(s_g[i]))
        if ok:
            t_meas = t_s[i]
        if t_s[i] - t_meas <= max_coast_s:
            Z[i], V[i], SV[i] = kf.z, kf.v, kf.sigma_v
    return Z, V, SV


def run_imm(t_s, z_h, s_h, z_g, s_g, a_ego, v_ego, sigma_a: float, imm: dict,
            sigma_v0: float = 5.0, gate: float = 9.0, max_coast_s: float = 0.5):
    """트랙 하나에 IMM. (Z, Ż, σ_Ż, P(정지)) 배열. 측정 공백 처리는 run_kf 와 같다."""
    n = len(t_s)
    Z, V, SV, PS = (np.full(n, np.nan) for _ in range(4))
    f, t_prev = None, None
    z0, s0 = combine_measurements(z_h, s_h, z_g, s_g)
    for i in range(n):
        ve = float(v_ego[i]) if np.isfinite(v_ego[i]) else float("nan")
        if f is None:
            if np.isfinite(z0[i]):
                f = IMMDepth(float(z0[i]), float(s0[i]), ve if ve == ve else 0.0, sigma_v0,
                             imm["p_stay"], imm["mu_stationary0"], imm["pseudo_rel"],
                             imm["pseudo_floor"], imm["sigma_a_stationary"], gate)
                t_prev = t_meas = t_s[i]
            else:
                continue
        else:
            meas = [(float(z_h[i]), float(s_h[i]))] if np.isfinite(z_h[i]) else []
            if np.isfinite(z_g[i]):
                meas.append((float(z_g[i]), float(s_g[i])))
            u = float(a_ego[i]) if np.isfinite(a_ego[i]) else 0.0
            if f.step(max(t_s[i] - t_prev, 1e-3), u, ve, sigma_a, meas):
                t_meas = t_s[i]
            t_prev = t_s[i]
            if t_s[i] - t_meas > max_coast_s:
                continue
        x = f.estimate
        Z[i], V[i] = x
        SV[i] = math.sqrt(max(f.P_comb[1, 1], 0.0))
        PS[i] = f.p_stationary
    return Z, V, SV, PS


def ttc_series(z: np.ndarray, zdot: np.ndarray, eps: float = 0.1, max_ttc: float = 20.0) -> np.ndarray:
    return np.array([ttc_from_state(a, b, eps, max_ttc) if np.isfinite(b) else np.inf
                     for a, b in zip(z, zdot)])
