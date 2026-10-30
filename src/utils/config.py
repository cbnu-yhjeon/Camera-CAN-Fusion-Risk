"""YAML 설정 로드. `--config a.yaml --config b.yaml` 처럼 여러 개를 주면 뒤가 앞을 덮어쓴다."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths: list[str] | None = None, version: str | None = None) -> dict:
    cfg: dict = {}
    for p in paths or [str(ROOT / "configs" / "default.yaml")]:
        with open(p, encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if version:
        cfg["data"]["version"] = version
        if version == "v1.0-mini":
            cfg["data"]["split_file"] = "splits/scene_split_mini.json"
    cfg["data"]["dataroot"] = os.path.expanduser(cfg["data"]["dataroot"])
    cache = Path(cfg["data"]["cache_dir"])
    if not cache.is_absolute():
        cache = ROOT / cache
    # 버전별 캐시 분리 — mini 실행이 trainval 캐시를 덮어쓰지 않게
    cfg["data"]["cache_dir"] = str(cache / cfg["data"]["version"])
    return cfg


def cli(description: str, extra=None) -> tuple[argparse.Namespace, dict]:
    """공통 CLI: --config(반복 가능), --version, --limit."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개 scene만 (디버그)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    paths = [str(ROOT / "configs" / "default.yaml")] + (args.config or [])
    cfg = load_config(paths, args.version)
    os.makedirs(cfg["data"]["cache_dir"], exist_ok=True)
    return args, cfg
