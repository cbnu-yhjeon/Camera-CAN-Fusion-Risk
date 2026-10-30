"""scene 하나: 검출 → 추적 → 거리 측정 → (keyframe) GT 매칭.

결과는 (프레임, 검출) 한 행. 이후 융합 단계는 (scene, tid) 로 묶어 시계열로 처리한다.
GT 매칭은 평가용이다 — 추정에는 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.vision.distance import Camera
from src.vision.tracker import GROUP, IoUTracker, iou_matrix

CAN_COLS = ["v_ego", "a_ego", "yaw_rate", "steer", "brake", "brake_on", "can_ok"]


def match_gt(det: pd.DataFrame, gt: pd.DataFrame, min_iou: float) -> np.ndarray:
    """검출 ↔ GT 투영 박스 헝가리안 매칭 (같은 그룹끼리). 검출별 GT 행 위치, 없으면 −1."""
    out = np.full(len(det), -1, dtype=np.int64)
    if len(det) == 0 or len(gt) == 0:
        return out
    iou = iou_matrix(det[["x1", "y1", "x2", "y2"]].to_numpy(), gt[["x1", "y1", "x2", "y2"]].to_numpy())
    dg = det["cls"].map(GROUP).to_numpy()
    gg = gt["cls"].map(GROUP).to_numpy()
    iou = np.where(dg[:, None] == gg[None, :], iou, 0.0)
    rows, cols = linear_sum_assignment(-iou)
    for r, c in zip(rows, cols):
        if iou[r, c] >= min_iou:
            out[r] = c
    return out


def process_scene(frames: pd.DataFrame, det: pd.DataFrame, gt: pd.DataFrame, calib: dict,
                  heights: dict, cfg: dict) -> pd.DataFrame:
    tc, dc = cfg["tracker"], cfg["distance"]
    cam = Camera(calib["K"], calib["translation"], calib["rotation"], calib["width"], calib["height"])
    tracker = IoUTracker(tc["iou_match"], tc["max_age"])
    det = det.merge(frames[["sd_token", "frame_idx"]], on="sd_token")
    by_frame = {k: g for k, g in det.groupby("frame_idx")}
    gt_by_frame = {k: g.reset_index(drop=True) for k, g in gt.groupby("frame_idx")}
    parts = []
    for _, fr in frames.sort_values("frame_idx").iterrows():
        d = by_frame.get(fr["frame_idx"])
        if d is None:
            tracker.update(np.zeros((0, 4)), [])
            continue
        d = d.reset_index(drop=True).copy()
        d["tid"] = tracker.update(d[["x1", "y1", "x2", "y2"]].to_numpy(), d["cls"].tolist())
        d["ann_token"] = None
        d["gt_iou_ok"] = False
        if fr["is_key"] and fr["frame_idx"] in gt_by_frame:
            g = gt_by_frame[fr["frame_idx"]]
            m = match_gt(d, g, cfg["eval"]["match_iou"])
            d.loc[m >= 0, "ann_token"] = g["ann_token"].to_numpy()[m[m >= 0]]
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(frames[["frame_idx", "utime", "is_key"] + CAN_COLS], on="frame_idx")

    # 거리 단서 두 가지 — 잘린 박스는 해당 단서를 쓰지 않는다
    no_h, no_g = cam.truncation(out.x1, out.y1, out.x2, out.y2, tc["border_px"])
    h_real = out["cls"].map(heights).to_numpy(dtype=float)
    z_h = cam.height_distance(out.y2 - out.y1, h_real)
    z_g = cam.ground_distance((out.x1 + out.x2) / 2, out.y2, dc["max_z"])
    out["z_h"] = np.where(no_h | (z_h > dc["max_z"]), np.nan, z_h)
    out["z_g"] = np.where(no_g, np.nan, z_g)
    out["u_c"] = (out.x1 + out.x2) / 2
    out["area"] = (out.x2 - out.x1) * (out.y2 - out.y1)
    out["scene"] = frames["scene"].iloc[0]
    out = out.drop(columns=["gt_iou_ok"])
    return out.sort_values(["tid", "frame_idx"]).reset_index(drop=True)
