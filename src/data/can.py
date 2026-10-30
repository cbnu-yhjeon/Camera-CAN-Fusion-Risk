"""CAN bus 시계열 로드와 카메라 프레임 시간 정렬.

메시지마다 주기가 다르다 (nuScenes CAN bus expansion):
    pose               50 Hz   vel[0] 자차 종방향 속도, rotation_rate[2] yaw rate
    steeranglefeedback 100 Hz  조향각 [rad]
    vehicle_monitor    2 Hz    brake(0–126), brake_switch
    ms_imu             100 Hz  (정렬 검증용 — 여기서는 쓰지 않는다)

정렬 규칙
---------
- 연속 신호(속도·yaw rate·조향)는 카메라 타임스탬프에서 선형 보간한다.
- 브레이크는 2 Hz 이산 신호라 보간하지 않고 **직전 메시지 값을 유지**한다(인과적).
  가장 가까운 메시지를 쓰면 최대 0.25 s 미래 값을 가져와 near-miss 판정이 거저 좋아진다.
- 가속도는 pose.accel 대신 속도를 미분해 평활한다. accel[0] 에는 노면 경사로 인한
  중력 성분이 섞여 오르막에서 '감속'으로 보인다.
- CAN 시간 범위 밖(±0.2 s 초과)인 프레임은 can_ok=False 로 두고 값은 NaN.
- pose 계열(속도·가속도·yaw rate)은 카메라보다 늦게 찍힌다 — trainval 주행 scene 723개에서
  ego_pose 속도 대비 지연 중앙값 +0.08 s (scripts/01b_align_check.py). pose_lag_s 만큼 당겨 읽는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MARGIN_US = 200_000


@dataclass
class CanSeries:
    t_pose: np.ndarray        # utime [us]
    v: np.ndarray             # m/s
    a: np.ndarray             # m/s², 속도 미분 + 평활
    yaw_rate: np.ndarray      # rad/s
    t_steer: np.ndarray
    steer: np.ndarray         # rad
    t_mon: np.ndarray
    brake: np.ndarray         # 0–126


def _read(dataroot: str, scene: str, msg: str) -> list[dict]:
    path = os.path.join(dataroot, "can_bus", f"{scene}_{msg}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


def smooth_derivative(t_s: np.ndarray, x: np.ndarray, window_s: float) -> np.ndarray:
    """이동 평균으로 평활한 뒤 미분. 비균일 샘플이므로 시간축 기준 창을 쓴다."""
    if len(x) < 3:
        return np.zeros_like(x)
    dt = np.median(np.diff(t_s))
    k = max(1, int(round(window_s / dt)))
    kernel = np.ones(k) / k
    pad = k // 2
    xs = np.convolve(np.pad(x, (pad, k - 1 - pad), mode="edge"), kernel, mode="valid")
    return np.gradient(xs, t_s)


def load_can(dataroot: str, scene: str, accel_smooth_s: float = 0.5) -> CanSeries | None:
    pose = _read(dataroot, scene, "pose")
    mon = _read(dataroot, scene, "vehicle_monitor")
    if len(pose) < 10:
        return None
    steer = _read(dataroot, scene, "steeranglefeedback")
    t = np.array([m["utime"] for m in pose], dtype=np.int64)
    order = np.argsort(t)
    t = t[order]
    v = np.array([m["vel"][0] for m in pose])[order]
    yaw = np.array([m["rotation_rate"][2] for m in pose])[order]
    a = smooth_derivative(t / 1e6, v, accel_smooth_s)
    ts = np.array([m["utime"] for m in steer], dtype=np.int64)
    sv = np.array([m["value"] for m in steer])
    tm = np.array([m["utime"] for m in mon], dtype=np.int64)
    br = np.array([m["brake"] for m in mon], dtype=float)
    return CanSeries(t, v, a, yaw, ts, sv, tm, br)


def _interp(tq, t, x):
    if len(t) == 0:
        return np.full(len(tq), np.nan)
    y = np.interp(tq, t, x)
    y[(tq < t[0] - MARGIN_US) | (tq > t[-1] + MARGIN_US)] = np.nan
    return y


def _hold(tq, t, x):
    """직전 메시지 값 유지. 첫 메시지 이전은 NaN."""
    if len(t) == 0:
        return np.full(len(tq), np.nan)
    idx = np.searchsorted(t, tq, side="right") - 1
    y = np.where(idx >= 0, x[np.clip(idx, 0, None)], np.nan)
    y[tq > t[-1] + 1_000_000] = np.nan         # 마지막 메시지 뒤 1 s 넘으면 결측
    return y


def align(can: CanSeries | None, utimes: np.ndarray, brake_threshold: float = 0,
          pose_lag_s: float = 0.0) -> pd.DataFrame:
    """카메라 프레임 타임스탬프에 CAN 신호를 맞춘다. pose 계열은 pose_lag_s 늦게 찍힌 것으로 본다."""
    utimes = np.asarray(utimes, dtype=np.int64)
    tp = utimes + int(round(pose_lag_s * 1e6))
    if can is None:
        nan = np.full(len(utimes), np.nan)
        return pd.DataFrame({"v_ego": nan, "a_ego": nan, "yaw_rate": nan, "steer": nan,
                             "brake": nan, "brake_on": False, "can_ok": False})
    v = _interp(tp, can.t_pose, can.v)
    brake = _hold(utimes, can.t_mon, can.brake)
    return pd.DataFrame({
        "v_ego": v,
        "a_ego": _interp(tp, can.t_pose, can.a),
        "yaw_rate": _interp(tp, can.t_pose, can.yaw_rate),
        "steer": _interp(utimes, can.t_steer, can.steer),
        "brake": brake,
        "brake_on": np.nan_to_num(brake, nan=0) > brake_threshold,
        "can_ok": np.isfinite(v),
    })


def estimate_lag(t_ref_s, v_ref, t_can_s, v_can, max_lag_s: float = 1.0, step_s: float = 0.01) -> tuple[float, float]:
    """CAN 속도를 lag 만큼 밀었을 때 기준 속도와의 RMSE 가 최소인 lag [s] 와 그 RMSE.

    lag > 0 이면 CAN 이 늦게 찍힌다(같은 사건의 CAN 타임스탬프가 더 크다).
    """
    best = (0.0, np.inf)
    for lag in np.arange(-max_lag_s, max_lag_s + 1e-9, step_s):
        y = np.interp(t_ref_s + lag, t_can_s, v_can, left=np.nan, right=np.nan)
        m = np.isfinite(y)
        if m.sum() < 5:
            continue
        rmse = float(np.sqrt(np.mean((y[m] - v_ref[m]) ** 2)))
        if rmse < best[1]:
            best = (float(lag), rmse)
    return best
