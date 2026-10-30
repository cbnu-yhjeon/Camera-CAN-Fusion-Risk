"""6b단계: F1(IMM) 하이퍼파라미터 민감도 — val split 만 → results/sensitivity_<version>.md.

기본값은 사전에 정했고 test 로는 고르지 않는다. 이 표는 '기본값 근처에서 결과가
얼마나 흔들리는가'를 보여 주기 위한 것이다 (한 번에 하나씩 바꿈).

    python scripts/06b_sensitivity.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.common import TTC_CAP, matched  # noqa: E402
from src.fusion.run import fuse_scene  # noqa: E402
from src.utils.config import ROOT, cli, deep_update  # noqa: E402

GRID = ([{}]
        + [{"imm": {k: v}} for k, vs in {"p_stay": [0.9, 0.99], "mu_stationary0": [0.4, 0.8],
                                          "pseudo_rel": [0.02, 0.1], "sigma_a_stationary": [0.1, 1.0]}.items()
           for v in vs]
        + [{"kalman": {"init_sigma_v": v}} for v in [3.0, 8.0]]
        + [{"kalman": {"sigma_a": {"vehicle": v, "pedestrian": 1.0}}} for v in [1.0, 3.0]])


def main():
    args, cfg0 = cli(__doc__)
    cache = cfg0["data"]["cache_dir"]
    t = pd.read_parquet(os.path.join(cache, "tracks.parquet"))
    fr = pd.read_parquet(os.path.join(cache, "frames.parquet"), columns=["scene", "split"]).drop_duplicates()
    t = t[t.scene.isin(set(fr[fr.split == "val"].scene))]
    with open(os.path.join(cache, "calib.json")) as f:
        calib = json.load(f)
    with open(os.path.join(cache, "noise.json")) as f:
        noise = json.load(f)
    gt = pd.read_parquet(os.path.join(cache, "gt.parquet"))
    groups = list(t.groupby("scene"))
    L = [f"# F1 (IMM) 민감도 — {cfg0['data']['version']} val split", "",
         "| 변경 | TTC MAE [s] | 객체 상태 정확도 | 오경보율 |", "|---|---:|---:|---:|"]
    for g in GRID:
        cfg = deep_update(cfg0, g)
        with ProcessPoolExecutor(10) as ex:
            parts = list(ex.map(fuse_scene, [d for _, d in groups], [calib[s] for s, _ in groups],
                                [noise] * len(groups), [cfg] * len(groups)))
        m = matched(pd.concat(parts), gt, cfg)
        near, far = m[m.ttc_gt < 5], m[~(m.ttc_gt < 10)]
        mae = np.abs(np.minimum(near.ttc_f1, TTC_CAP) - near.ttc_gt).mean()
        acc = (m.state_f1 == m.state_gt).mean()
        fa = (far.ttc_f1 < 2).mean()
        L.append(f"| {json.dumps(g) if g else '기본값'} | {mae:.3f} | {acc:.1%} | {fa:.1%} |")
        print(L[-1], flush=True)
    (ROOT / "results" / f"sensitivity_{cfg0['data']['version']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
