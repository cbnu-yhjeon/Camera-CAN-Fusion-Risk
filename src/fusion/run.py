"""트랙 표 → TTC 소스별 추정 (B0 · B1 · F0 · F1a · F1) + 객체 상태.

입력 행은 (scene, tid, frame_idx) 순으로 정렬돼 있어야 한다 (03_tracks.py 출력).
출력 열 (소스 m ∈ {b0, b1, f0, f1a, f1}):
    z_<m>, zdot_<m>, ttc_<m>   — B0 는 TTC 만
    x_lat                      — F1 거리로 계산한 횡방향 위치
    v_obj_<m>, state_<m>       — CAN v_ego 를 쓰는 b1, f1a, f1 만
    pstat_f1                   — IMM 정지 모델 확률
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fusion.estimators import (combine_measurements, run_imm, run_kf, ttc_area, ttc_series,
                                   zdot_difference)
from src.fusion.motion_state import classify
from src.vision.distance import Camera
from src.vision.tracker import GROUP


def apply_noise(df: pd.DataFrame, noise: dict, floor: float) -> pd.DataFrame:
    """bias 보정 + σ_Z 열 추가."""
    df = df.copy()
    for cue in ["z_h", "z_g"]:
        per = noise[cue]
        bias = df["cls"].map(lambda c: per.get(c, per["_all"])["bias"]).to_numpy(float)
        sig = df["cls"].map(lambda c: per.get(c, per["_all"])["sig_rel"]).to_numpy(float)
        df[cue] = df[cue] / bias
        df["s" + cue[1:]] = np.maximum(sig * df[cue], floor)
    return df


def fuse_scene(df: pd.DataFrame, calib: dict, noise: dict, cfg: dict) -> pd.DataFrame:
    kc, tc, mc = cfg["kalman"], cfg["ttc"], cfg["motion_state"]
    df = apply_noise(df, noise, cfg["distance"]["sigma_floor"])
    method = cfg["distance"]["method"]
    if method == "height":
        df["z_g"] = np.nan
    elif method == "ground":
        df["z_h"] = np.nan
    cam = Camera(calib["K"], calib["translation"], calib["rotation"], calib["width"], calib["height"])
    cols = {k: np.full(len(df), np.nan) for k in
            ["z_raw", "zdot_b1", "ttc_b0", "ttc_b1", "z_f0", "zdot_f0", "ttc_f0",
             "z_f1a", "zdot_f1a", "ttc_f1a", "z_f1", "zdot_f1", "szdot_f1", "pstat_f1", "ttc_f1"]}
    starts = np.r_[0, np.flatnonzero(np.diff(df["tid"].to_numpy())) + 1, len(df)]
    t_all = df["utime"].to_numpy() / 1e6
    zh, sh, zg, sg = (df[c].to_numpy(float) for c in ["z_h", "s_h", "z_g", "s_g"])
    a_ego = df["a_ego"].to_numpy(float)
    v_ego = df["v_ego"].to_numpy(float)
    area = df["area"].to_numpy(float)
    grp = df["cls"].map(GROUP).to_numpy()
    for a, b in zip(starts[:-1], starts[1:]):
        sl = slice(a, b)
        t = t_all[sl]
        z_raw, _ = combine_measurements(zh[sl], sh[sl], zg[sl], sg[sl])
        cols["z_raw"][sl] = z_raw
        cols["ttc_b0"][sl] = ttc_area(t, area[sl], tc["b0_min"], tc["max"])
        zd = zdot_difference(t, z_raw, tc["b1_window_s"])
        cols["zdot_b1"][sl] = zd
        cols["ttc_b1"][sl] = ttc_series(z_raw, zd, tc["eps"], tc["max"])
        sa = kc["sigma_a"]["pedestrian" if grp[a] == "pedestrian" else "vehicle"]
        for tag, use_can in [("f0", False), ("f1a", True)]:
            Z, V, _ = run_kf(t, zh[sl], sh[sl], zg[sl], sg[sl], a_ego[sl], sa, use_can,
                             kc["init_sigma_v"], kc["gate_chi2"], kc["sigma_a_no_can_extra"])
            cols[f"z_{tag}"][sl], cols[f"zdot_{tag}"][sl] = Z, V
            cols[f"ttc_{tag}"][sl] = ttc_series(Z, V, tc["eps"], tc["max"])
        Z, V, SV, PS = run_imm(t, zh[sl], sh[sl], zg[sl], sg[sl], a_ego[sl], v_ego[sl], sa,
                               cfg["imm"], kc["init_sigma_v"], kc["gate_chi2"])
        cols["z_f1"][sl], cols["zdot_f1"][sl], cols["szdot_f1"][sl], cols["pstat_f1"][sl] = Z, V, SV, PS
        cols["ttc_f1"][sl] = ttc_series(Z, V, tc["eps"], tc["max"])
    for k, v in cols.items():
        df[k] = v
    df["x_lat"] = cam.lateral(df["u_c"], df["z_f1"])
    for tag in ["b1", "f1a", "f1"]:
        df[f"v_obj_{tag}"] = v_ego + df[f"zdot_{tag}"]
        df[f"state_{tag}"] = classify(df[f"v_obj_{tag}"], df[f"zdot_{tag}"], mc["stationary_v"], mc["following_zdot"])
    df["track_age"] = df.groupby("tid").cumcount()
    # 좌우 경계에 걸친 박스 — 시야 밖으로 빠져나가는 옆 차로 객체. 중심 u 가 안쪽으로 치우쳐 x_lat 이 작게 나온다
    b = cfg["tracker"]["border_px"]
    df["side_trunc"] = (df.x1 <= b) | (df.x2 >= calib["width"] - b)
    return df
