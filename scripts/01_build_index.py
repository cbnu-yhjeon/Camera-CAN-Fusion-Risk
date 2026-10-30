"""1단계: nuScenes 색인 + CAN 시간 정렬 → cache/<version>/{frames,gt}.parquet, calib.json.

    python scripts/01_build_index.py --version v1.0-mini
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.can import align, load_can  # noqa: E402
from src.data.nusc_index import build_index  # noqa: E402
from src.data.splits import load_split, scene_to_split  # noqa: E402
from src.utils.config import cli  # noqa: E402


def main():
    args, cfg = cli(__doc__)
    d = cfg["data"]
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=d["version"], dataroot=d["dataroot"], verbose=False)
    split = scene_to_split(load_split(d["split_file"]))
    available = {s["name"] for s in nusc.scene}
    names = sorted(n for n in split if n in available)
    if args.limit:
        names = names[: args.limit]
    frames, gt, calib = build_index(nusc, names, d["camera"], split)

    blacklist = set(NuScenesCanBus(dataroot=d["dataroot"]).can_blacklist)
    parts = []
    for name, f in tqdm(frames.groupby("scene", sort=False), desc="can"):
        can = None if int(name.split("-")[1]) in blacklist else load_can(
            d["dataroot"], name, cfg["can"]["accel_smooth_s"])
        a = align(can, f["utime"].to_numpy(), cfg["can"]["brake_threshold"], cfg["can"]["pose_lag_s"])
        a.index = f.index
        parts.append(a)
    frames = pd.concat([frames, pd.concat(parts)], axis=1)

    out = d["cache_dir"]
    frames.to_parquet(os.path.join(out, "frames.parquet"), index=False)
    gt.to_parquet(os.path.join(out, "gt.parquet"), index=False)
    with open(os.path.join(out, "calib.json"), "w") as f:
        json.dump(calib, f)

    # 클래스 평균 높이 — train split, 40% 이상 보이는 객체만 (테스트 누수 없음)
    tr = gt[gt["scene"].map(split).eq("train") & (gt["visibility"] >= 2)]
    heights = tr.groupby("cls")["size_h"].mean().round(3).to_dict()
    with open(os.path.join(out, "class_heights.json"), "w") as f:
        json.dump(heights, f, indent=2)

    can_scenes = frames.groupby("scene")["can_ok"].mean()
    print(f"scenes={len(names)} frames={len(frames)} keyframes={int(frames.is_key.sum())} gt={len(gt)}")
    print(f"CAN 결손 scene: {int((can_scenes < 0.5).sum())} / {len(can_scenes)}  (blacklist {len(blacklist)})")
    print("class heights (train):", heights)


if __name__ == "__main__":
    main()
