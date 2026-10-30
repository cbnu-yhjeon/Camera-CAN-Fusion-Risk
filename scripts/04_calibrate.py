"""4단계: 거리 측정 잡음 보정 (train split 만) → cache/<version>/noise.json.

GT 와 매칭된 train keyframe 검출에서 클래스·단서별로
    bias    = median(Z_meas / Z_gt)              → 측정값을 bias 로 나눠 보정
    sig_rel = 1.4826 · MAD(log(Z_meas / Z_gt))   → σ_Z = sig_rel · Z
를 구한다. 칼만 필터의 R 과 B1 의 측정이 이 값을 쓴다. val/test 는 보지 않는다.

    python scripts/04_calibrate.py --version v1.0-mini
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import cli  # noqa: E402

MIN_N = 30


def robust(r: pd.Series) -> tuple[float, float, int]:
    r = np.log(r[np.isfinite(r) & (r > 0)])
    med = float(np.median(r))
    return float(np.exp(med)), float(1.4826 * np.median(np.abs(r - med))), int(len(r))


def main():
    args, cfg = cli(__doc__)
    cache = cfg["data"]["cache_dir"]
    t = pd.read_parquet(os.path.join(cache, "tracks.parquet"))
    gt = pd.read_parquet(os.path.join(cache, "gt.parquet"))[["ann_token", "z_near", "visibility"]]
    frames = pd.read_parquet(os.path.join(cache, "frames.parquet"), columns=["scene", "split"])
    train = set(frames.loc[frames.split == "train", "scene"])
    if not train:                                  # mini 의 train 이 비는 경우는 없지만 방어
        train = set(frames.scene)
    m = t[t.scene.isin(train)].merge(gt, on="ann_token")
    m = m[m.visibility >= cfg["eval"]["min_visibility"]]

    out = {}
    for cue in ["z_h", "z_g"]:
        b_all, s_all, n_all = robust(m[cue] / m.z_near)
        per = {"_all": {"bias": b_all, "sig_rel": s_all, "n": n_all}}
        for cls, g in m.groupby("cls"):
            b, s, n = robust(g[cue] / g.z_near)
            per[cls] = {"bias": b, "sig_rel": s, "n": n} if n >= MIN_N else per["_all"] | {"n": n}
        out[cue] = per
    with open(os.path.join(cache, "noise.json"), "w") as f:
        json.dump(out, f, indent=2)
    for cue, per in out.items():
        print(cue, {k: (round(v["bias"], 3), round(v["sig_rel"], 3), v["n"]) for k, v in per.items()})


if __name__ == "__main__":
    main()
