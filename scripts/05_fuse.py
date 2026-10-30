"""5단계: TTC 추정 (B0 · B1 · F0 · F1) + 객체 상태 → cache/<version>/est.parquet.

    python scripts/05_fuse.py --version v1.0-mini
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

from src.fusion.run import fuse_scene  # noqa: E402
from src.utils.config import cli  # noqa: E402


def _job(a):
    return fuse_scene(*a)


def main():
    args, cfg = cli(__doc__, lambda ap: ap.add_argument("--out", default="est.parquet"))
    cache = cfg["data"]["cache_dir"]
    t = pd.read_parquet(os.path.join(cache, "tracks.parquet"))
    with open(os.path.join(cache, "calib.json")) as f:
        calib = json.load(f)
    with open(os.path.join(cache, "noise.json")) as f:
        noise = json.load(f)
    groups = list(t.groupby("scene", sort=False))[: args.limit]
    jobs = [(g, calib[s], noise, cfg) for s, g in groups]
    with ProcessPoolExecutor(10) as ex:
        parts = list(tqdm(ex.map(_job, jobs, chunksize=4), total=len(jobs), desc="fuse"))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(cache, args.out), index=False)
    print(f"rows={len(out)} → {args.out}")


if __name__ == "__main__":
    main()
