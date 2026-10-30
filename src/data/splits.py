"""scene 단위 분할 — 연구 저장소(Adaptive-Camera-View-Selection-System)의 분할 파일을 그대로 쓴다.

같은 scene이 train과 test에 동시에 들어가면 같은 도로·같은 차량이 양쪽에 나타나 누수가 생긴다.
분할 파일은 splits/에 복사해 두고 변경하지 않는다.
"""

from __future__ import annotations

import json

from src.utils.config import ROOT


def load_split(path: str) -> dict[str, list[str]]:
    p = ROOT / path
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {k: list(d[k]) for k in ("train", "val", "test")}


def scene_to_split(split: dict[str, list[str]]) -> dict[str, str]:
    return {name: k for k, names in split.items() for name in names}
