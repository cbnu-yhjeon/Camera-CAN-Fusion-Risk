"""평가 공통 — 추정 표와 GT 를 잇고, 실험 소스 이름을 한곳에 둔다."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.fusion.motion_state import classify
from src.risk.labels import gt_ttc

METHODS = {
    "b0": "B0 box-area (v2)",
    "b1": "B1 distance diff",
    "f0": "F0 KF, no CAN",
    "f1a": "F1a KF + CAN accel",
    "f1": "F1 IMM + CAN speed/accel",
}
TTC_CAP = 10.0     # MAE 계산 시 ∞ 추정을 이 값으로 자른다 (GT < 5 s 구간이라 오차 상한 역할)


def load(cache: str, est_name: str = "est.parquet"):
    est = pd.read_parquet(os.path.join(cache, est_name))
    gt = pd.read_parquet(os.path.join(cache, "gt.parquet"))
    frames = pd.read_parquet(os.path.join(cache, "frames.parquet"))
    split = frames.drop_duplicates("scene").set_index("scene")["split"]
    est["split"] = est["scene"].map(split)
    gt["split"] = gt["scene"].map(split)
    return est, gt, frames


def matched(est: pd.DataFrame, gt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """GT 와 매칭된 keyframe 검출 + GT 열(접미사 _gt)."""
    g = gt[["ann_token", "instance_token", "z_near", "x_lat", "zdot", "v_long", "visibility", "attribute"]]
    g = g.rename(columns={"x_lat": "x_lat_gt", "zdot": "zdot_gt", "z_near": "z_gt", "v_long": "v_long_gt"})
    m = est.merge(g, on="ann_token")
    m = m[(m.visibility >= cfg["eval"]["min_visibility"]) & m.zdot_gt.notna()].copy()
    m["ttc_gt"] = gt_ttc(m.z_gt, m.zdot_gt, cfg["ttc"]["eps"])
    mc = cfg["motion_state"]
    m["state_gt"] = classify(m.v_long_gt, m.zdot_gt, mc["stationary_v"], mc["following_zdot"])
    return m


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=conv)
