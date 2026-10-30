"""3단계: 추적 + 거리 측정 + GT 매칭 → cache/<version>/tracks.parquet.

    python scripts/03_tracks.py --version v1.0-mini
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.utils.config import cli  # noqa: E402
from src.vision.build_tracks import process_scene  # noqa: E402


def _job(args):
    name, frames, gt, calib, heights, cfg = args
    det_path = os.path.join(cfg["data"]["cache_dir"], "det", f"{name}.parquet")
    if not os.path.exists(det_path):
        return None
    return process_scene(frames, pd.read_parquet(det_path), gt, calib, heights, cfg)


def main():
    args, cfg = cli(__doc__)
    cache = cfg["data"]["cache_dir"]
    frames = pd.read_parquet(os.path.join(cache, "frames.parquet"))
    gt = pd.read_parquet(os.path.join(cache, "gt.parquet"))
    with open(os.path.join(cache, "calib.json")) as f:
        calib = json.load(f)
    with open(os.path.join(cache, "class_heights.json")) as f:
        heights = json.load(f)
    scenes = list(frames["scene"].unique())[: args.limit]
    fg, gg = dict(tuple(frames.groupby("scene"))), dict(tuple(gt.groupby("scene")))
    jobs = [(s, fg[s], gg.get(s, gt.iloc[:0]), calib[s], heights, cfg) for s in scenes]
    with ProcessPoolExecutor(10) as ex:
        parts = [p for p in tqdm(ex.map(_job, jobs, chunksize=4), total=len(jobs), desc="tracks")
                 if p is not None and len(p)]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(cache, "tracks.parquet"), index=False)
    n_missing = len(scenes) - len(parts)
    print(f"rows={len(out)} tracks={out.groupby(['scene', 'tid']).ngroups} "
          f"gt_matched_keyrows={int(out.ann_token.notna().sum())} scenes_without_det={n_missing}")


if __name__ == "__main__":
    main()
