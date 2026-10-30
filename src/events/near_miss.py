"""near-miss 이벤트 추출 — '아찔했던 순간' = 짧은 TTC + 운전자의 급제동 반응.

정의 (README 초안을 데이터에 맞게 다듬음):
    프레임 t 가 후보  ⇔  진행 경로 안 객체의 min TTC(t) < ttc_thresh
                         그리고 [t − w, t + w] 안에 제동 트리거가 있다
    제동 트리거       ⇔  브레이크 ON 으로 바뀐 순간(onset) & v_ego > 1 m/s
                         또는 a_ego < decel_thresh
    후보 프레임 사이 간격이 merge_gap 이하이면 한 이벤트로 묶는다.

'브레이크 ON' 대신 onset 을 쓰는 이유: nuScenes 차량은 정차 중 브레이크를 계속 밟고 있어
(mini scene-0553 은 98% 프레임이 ON) 신호 대기만으로 트리거가 걸린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brake_trigger(f: pd.DataFrame, decel_thresh: float, min_speed: float = 1.0) -> np.ndarray:
    """한 scene 프레임 표(시간순) → 프레임별 제동 트리거."""
    on = f["brake_on"].to_numpy(bool)
    onset = on & ~np.r_[False, on[:-1]]
    moving = np.nan_to_num(f["v_ego"].to_numpy(float), nan=0.0) > min_speed
    hard = np.nan_to_num(f["a_ego"].to_numpy(float), nan=0.0) < decel_thresh
    return (onset & moving) | hard


def near_trigger(t_s: np.ndarray, trig_t: np.ndarray, window_s: float) -> np.ndarray:
    """t_s 각 시각의 ±window 안에 트리거가 있는가."""
    if len(trig_t) == 0:
        return np.zeros(len(t_s), dtype=bool)
    idx = np.searchsorted(trig_t, t_s)
    lo = np.abs(t_s - trig_t[np.clip(idx - 1, 0, len(trig_t) - 1)])
    hi = np.abs(t_s - trig_t[np.clip(idx, 0, len(trig_t) - 1)])
    return np.minimum(lo, hi) <= window_s


def to_events(t_s: np.ndarray, flags: np.ndarray, merge_gap_s: float) -> list[tuple[float, float]]:
    ts = t_s[flags]
    if len(ts) == 0:
        return []
    events, start, prev = [], ts[0], ts[0]
    for t in ts[1:]:
        if t - prev > merge_gap_s:
            events.append((start, prev))
            start = t
        prev = t
    events.append((start, prev))
    return events


def match_events(cand: list, ref: list, tol_s: float) -> tuple[np.ndarray, np.ndarray]:
    """구간이 tol 안에서 겹치면 매칭. (후보별 매칭 여부, 기준별 매칭 여부)."""
    c_hit = np.zeros(len(cand), dtype=bool)
    r_hit = np.zeros(len(ref), dtype=bool)
    for i, (a0, a1) in enumerate(cand):
        for j, (b0, b1) in enumerate(ref):
            if a0 - tol_s <= b1 and b0 <= a1 + tol_s:
                c_hit[i] = r_hit[j] = True
    return c_hit, r_hit


def min_ttc_in_path(rows: pd.DataFrame, ttc_col: str, x_col: str, corridor: float,
                    min_age: int = 2) -> pd.DataFrame:
    """(frame_idx 별) 진행 경로 안 객체의 min TTC 와 그 트랙 id. 좌우 경계에 잘린 박스는 뺀다."""
    keep = (rows[x_col].abs() < corridor) & (rows["track_age"] >= min_age) & np.isfinite(rows[ttc_col])
    if "side_trunc" in rows:
        keep &= ~rows["side_trunc"].astype(bool)
    r = rows[keep]
    if r.empty:
        return pd.DataFrame(columns=["frame_idx", "ttc_min", "tid"])
    i = r.groupby("frame_idx")[ttc_col].idxmin()
    return pd.DataFrame({"frame_idx": r.loc[i, "frame_idx"].to_numpy(),
                         "ttc_min": r.loc[i, ttc_col].to_numpy(), "tid": r.loc[i, "tid"].to_numpy()})
