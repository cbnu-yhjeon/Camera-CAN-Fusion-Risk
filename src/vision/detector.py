"""2D 객체 검출 — COCO 사전학습 YOLO.

v2 의 Canny·컨투어 검출은 박스가 프레임마다 크게 흔들려 TTC 비교 자체를 무의미하게 만든다.
이 프로젝트의 질문은 'TTC 를 어떻게 추정하느냐'이므로, 모든 실험(B0·B1·F0·F1)이
같은 검출 결과를 공유하도록 검출을 한 번만 돌려 캐시한다.

nuScenes CAM_FRONT 는 모두 1600×900 이라 ultralytics 의 이미지별 letterbox 대신
고정 비율 축소 + 하단 패딩을 스레드에서 하고 한 배치 텐서로 넣는다 (약 5배 빠르다).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch


class Detector:
    def __init__(self, weights: str, classes: dict[int, str], imgsz: int = 960,
                 conf: float = 0.25, iou: float = 0.7, device: str = "cuda"):
        from ultralytics import YOLO
        torch.backends.cudnn.benchmark = True
        self.model = YOLO(weights).model.to(device).half().eval()
        self.classes = {int(k): v for k, v in classes.items()}
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self.batch = 1

    def _load(self, path: str) -> np.ndarray:
        im = cv2.imread(path)
        h, w = im.shape[:2]
        self.scale = self.imgsz / w
        nh = int(round(h * self.scale))
        im = cv2.resize(im, (self.imgsz, nh), interpolation=cv2.INTER_AREA)
        pad = (-nh) % 32                          # stride 32 배수로 하단 패딩
        return cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """전처리된 BGR 이미지들 → 이미지별 (N, 6) [x1, y1, x2, y2, conf, coco_cls] (원본 픽셀)."""
        from ultralytics.utils.nms import non_max_suppression
        n = len(images)
        if n < self.batch:                        # 고정 배치 크기 — cudnn 재튜닝 방지
            images = images + [images[-1]] * (self.batch - n)
        x = torch.from_numpy(np.stack(images)[..., ::-1].copy()).to(self.device)
        x = x.permute(0, 3, 1, 2).half().div_(255)
        out = non_max_suppression(self.model(x), self.conf, self.iou, classes=list(self.classes))
        res = []
        for o in out[:n]:
            o = o.float().cpu().numpy()
            o[:, :4] /= self.scale
            res.append(o)
        return res

    def run(self, paths: list[str], tokens: list[str], batch: int = 32,
            pool: ThreadPoolExecutor | None = None) -> pd.DataFrame:
        pool = pool or ThreadPoolExecutor(8)
        self.batch = batch
        rows = []
        starts = list(range(0, len(paths), batch))

        def load_batch(s):
            return list(pool.map(self._load, paths[s:s + batch]))

        nxt = pool.submit(load_batch, starts[0]) if starts else None
        for j, i in enumerate(starts):
            imgs = nxt.result()
            if j + 1 < len(starts):               # 다음 배치 디코딩을 추론과 겹친다
                nxt = pool.submit(load_batch, starts[j + 1])
            for tok, det in zip(tokens[i:i + batch], self(imgs)):
                for x1, y1, x2, y2, c, k in det:
                    rows.append((tok, x1, y1, x2, y2, c, self.classes[int(k)]))
        return pd.DataFrame(rows, columns=["sd_token", "x1", "y1", "x2", "y2", "conf", "cls"])
