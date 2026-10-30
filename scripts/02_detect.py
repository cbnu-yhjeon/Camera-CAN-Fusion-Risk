"""2단계: CAM_FRONT 전 프레임(12 Hz) 검출 → cache/<version>/det/<scene>.parquet.

이미 있는 scene 은 건너뛰므로 중단 후 다시 실행하면 이어서 돈다. GPU 가 남으면
--reverse 로 하나 더 띄워 뒤에서부터 처리하게 할 수 있다 (임시 파일은 프로세스별).

    python scripts/02_detect.py --version v1.0-mini
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.utils.config import ROOT, cli  # noqa: E402
from src.vision.detector import Detector  # noqa: E402


def main():
    args, cfg = cli(__doc__, lambda ap: ap.add_argument("--reverse", action="store_true"))
    d, dc = cfg["data"], cfg["detector"]
    frames = pd.read_parquet(os.path.join(d["cache_dir"], "frames.parquet"))
    out_dir = os.path.join(d["cache_dir"], "det")
    os.makedirs(out_dir, exist_ok=True)
    weights_dir = ROOT / "cache" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_dir)                      # ultralytics 는 가중치를 cwd 에 내려받는다
    det = Detector(dc["weights"], dc["classes"], dc["imgsz"], dc["conf"])

    scenes = list(frames["scene"].unique())[: args.limit]
    if args.reverse:
        scenes = scenes[::-1]
    pool = ThreadPoolExecutor(10)
    t0, n_done = time.time(), 0
    for i, name in enumerate(scenes):
        path = os.path.join(out_dir, f"{name}.parquet")
        if os.path.exists(path):
            continue
        f = frames[frames["scene"] == name]
        paths = [os.path.join(d["dataroot"], p) for p in f["filename"]]
        df = det.run(paths, f["sd_token"].tolist(), dc["batch"], pool)
        tmp = f"{path}.{os.getpid()}.tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        n_done += len(f)
        rate = n_done / (time.time() - t0)
        print(f"[{i + 1}/{len(scenes)}] {name} frames={len(f)} dets={len(df)} {rate:.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
