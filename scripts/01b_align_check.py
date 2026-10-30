"""1b단계: CAN ↔ 카메라 시간 정렬 검증 → results/align_<version>.{json,md}, results/figures/align_*.png.

CAN pose.vel[0] (50 Hz)과 ego_pose 차분 속도(카메라 프레임, 12 Hz)를 비교해
scene 별 시간 지연(lag)과 정렬 후 RMSE 를 구한다. lag 이 0 근처에 모여 있으면
타임스탬프를 그대로 맞춰도 된다.

    python scripts/01b_align_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.can import estimate_lag, load_can  # noqa: E402
from src.eval.common import save_json  # noqa: E402
from src.utils.config import ROOT, cli  # noqa: E402


def main():
    args, cfg = cli(__doc__)
    d = cfg["data"]
    frames = pd.read_parquet(os.path.join(d["cache_dir"], "frames.parquet"))
    rows = []
    for name, f in tqdm(frames.groupby("scene"), desc="align"):
        if not f.can_ok.any():
            rows.append({"scene": name, "can": False})
            continue
        can = load_can(d["dataroot"], name)
        t_ref = f.utime.to_numpy() / 1e6
        v_ref = f.ego_speed_gt.to_numpy()
        lag, rmse = estimate_lag(t_ref, v_ref, can.t_pose / 1e6, can.v)
        rmse0 = float(np.sqrt(np.nanmean((f.v_ego - f.ego_speed_gt) ** 2)))
        rows.append({"scene": name, "can": True, "lag_s": lag, "rmse_aligned": rmse, "rmse_raw": rmse0,
                     "coverage": float(f.can_ok.mean()), "mean_speed": float(np.nanmean(v_ref))})
    r = pd.DataFrame(rows)
    ok = r[r.can & (r.mean_speed > 1.0)]               # 정지 scene 은 lag 이 정의되지 않는다
    res = {"n_scenes": int(len(r)), "n_no_can": int((~r.can).sum()), "n_moving": int(len(ok)),
           "lag_median_s": float(ok.lag_s.median()), "lag_iqr_s": ok.lag_s.quantile([.25, .75]).tolist(),
           "abs_lag_gt_0.1s": float((ok.lag_s.abs() > 0.1).mean()),
           "rmse_raw_median": float(ok.rmse_raw.median()), "rmse_aligned_median": float(ok.rmse_aligned.median()),
           "partial_coverage_scenes": r.loc[r.can & (r.coverage < 0.99), "scene"].tolist(),
           "no_can_scenes": r.loc[~r.can, "scene"].tolist()}
    tag = d["version"]
    save_json(res, str(ROOT / "results" / f"align_{tag}.json"))
    r.to_csv(ROOT / "results" / f"align_{tag}_per_scene.csv", index=False)

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ok.lag_s, bins=np.arange(-1, 1.01, 0.02))
    axes[0].set(xlabel="CAN lag vs camera ego speed [s]", ylabel="scenes", title=f"lag per scene (n={len(ok)})")
    ex = ok.sort_values("mean_speed").iloc[len(ok) // 2].scene if len(ok) else r.scene.iloc[0]
    f = frames[frames.scene == ex]
    can = load_can(d["dataroot"], ex)
    t0 = f.utime.min()
    axes[1].plot((can.t_pose - t0) / 1e6, can.v, lw=1, label="CAN pose.vel (50 Hz)")
    axes[1].plot((f.utime - t0) / 1e6, f.ego_speed_gt, ".", ms=3, label="ego_pose diff (camera 12 Hz)")
    axes[1].set(xlabel="time [s]", ylabel="speed [m/s]", title=ex)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"align_{tag}.png", dpi=130)

    md = (f"# CAN ↔ 카메라 시간 정렬 ({tag})\n\n"
          f"- scene {res['n_scenes']}개 중 CAN 없음 {res['n_no_can']}개 (devkit blacklist + 파일 결손)\n"
          f"- 주행 중 scene {res['n_moving']}개의 lag 중앙값 {res['lag_median_s']:+.2f} s, "
          f"IQR [{res['lag_iqr_s'][0]:+.2f}, {res['lag_iqr_s'][1]:+.2f}] s, |lag| > 0.1 s 비율 {res['abs_lag_gt_0.1s']:.1%}\n"
          f"- 속도 RMSE (보간 그대로 / lag 보정 후) 중앙값: {res['rmse_raw_median']:.3f} / {res['rmse_aligned_median']:.3f} m/s\n"
          f"- CAN 이 scene 일부만 덮는 scene: {len(res['partial_coverage_scenes'])}개\n")
    (ROOT / "results" / f"align_{tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
