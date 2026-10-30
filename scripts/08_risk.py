"""8단계: 위험도 — 규칙(v2 공식) vs 학습(XGBoost + SHAP) → results/risk_<version>.{json,md}.

표본: GT 와 매칭된 keyframe 검출. 라벨: GT TTC + 진행 경로 (src/risk/labels.py).
학습 train, 조기 종료 val, 보고 test.

  R-B0  v2 규칙, TTC = 박스 면적 (B0)            — 기준선
  R-B1  v2 규칙, TTC = 거리 차분 (B1)
  R-F1  v2 규칙, TTC = F1 융합                    — ttc_weight 만 교체
  F2    XGBoost, F1 특징 + CAN                    — G4
  A1    XGBoost, F1 특징, CAN 특징 제거            — CAN 기여도
  A2    XGBoost, F0 특징(CAN 없는 필터), CAN 특징 제거 — 영상만으로 학습했을 때

마지막에 모든 keyframe 검출(매칭 여부 무관)에 F2 를 적용해 위험도 벡터를 내보낸다
(cache/<version>/risk_vectors.parquet — 연구 저장소 입력용).

    python scripts/08_risk.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402

from src.eval.common import load, matched, save_json  # noqa: E402
from src.risk.labels import risk_label  # noqa: E402
from src.risk.learned import make_features, shap_values, train_xgb  # noqa: E402
from src.risk.rule_based import DANGER, LEVELS, v2_level, v2_score, visibility_score  # noqa: E402
from src.utils.config import ROOT, cli  # noqa: E402


def keyframe_visibility(frames: pd.DataFrame, dataroot: str, cache: str) -> pd.Series:
    path = os.path.join(cache, "visibility.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path).set_index("sd_token")["visibility_score"]
    kf = frames[frames.is_key]

    def one(fn):
        g = cv2.imread(os.path.join(dataroot, fn), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        return visibility_score(g)

    with ThreadPoolExecutor(12) as ex:
        v = list(ex.map(one, kf.filename))
    s = pd.DataFrame({"sd_token": kf.sd_token.to_numpy(), "visibility_score": v})
    s.to_parquet(path, index=False)
    return s.set_index("sd_token")["visibility_score"]


def metrics(y, p) -> dict:
    return {"macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
            "danger_recall": float(recall_score(y == DANGER, p == DANGER, zero_division=0)),
            "danger_precision": float(precision_score(y == DANGER, p == DANGER, zero_division=0)),
            "per_class_f1": f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(), "n": int(len(y))}


def rule_scores(df: pd.DataFrame, src: str, calib: dict, lane_half: float) -> np.ndarray:
    z = df["z_f1"] if src == "f1" else df["z_raw"]
    x = df["x_lat"] if src == "f1" else df["x_lat"] * df["z_raw"] / df["z_f1"]
    W = df.scene.map(lambda s: calib[s]["width"])
    H = df.scene.map(lambda s: calib[s]["height"])
    return v2_score(z, (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2, W, H, df.visibility_score,
                    df[f"ttc_{src}"], np.abs(x) < lane_half)


def main():
    args, cfg = cli(__doc__)
    cache = cfg["data"]["cache_dir"]
    rc = cfg["risk"]
    est, gt, frames = load(cache)
    with open(os.path.join(cache, "calib.json")) as f:
        calib = json.load(f)
    vis = keyframe_visibility(frames, cfg["data"]["dataroot"], cache)
    est["visibility_score"] = est.sd_token.map(vis)
    m = matched(est[est.is_key], gt, cfg)
    m["y"] = risk_label(m.ttc_gt, m.x_lat_gt, rc["corridor_half_width"], rc["danger_ttc"], rc["caution_ttc"])
    tr, va, te = (m[m.split == s] for s in ["train", "val", "test"])
    if len(te) == 0:                                        # mini: test 가 비면 val 로 보고
        te = va
    res = {"label_dist": {sp: np.bincount(d.y, minlength=3).tolist() for sp, d in
                          [("train", tr), ("val", va), ("test", te)]}}

    lane_half = rc["lane_half_width"]
    for src in ["b0", "b1", "f1"]:
        p = v2_level(rule_scores(te, src, calib, lane_half))
        res[f"R-{src.upper()}"] = metrics(te.y.to_numpy(), p)

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, src, can in [("F2", "f1", True), ("A1", "f1", False), ("A2", "f0", False)]:
        Xtr, Xva, Xte = (make_features(d, src, can) for d in (tr, va, te))
        model = train_xgb(Xtr, tr.y.to_numpy(), Xva, va.y.to_numpy(), rc["xgb"])
        res[name] = metrics(te.y.to_numpy(), model.predict(Xte)) | {"best_iteration": int(model.best_iteration)}
        models[name] = (model, src, can)

    # SHAP (F2, test, DANGER 클래스)
    model, src, can = models["F2"]
    Xte = make_features(te, src, can)
    Xs = Xte.sample(min(5000, len(Xte)), random_state=0)
    sv = shap_values(model, Xs)                       # (n, f, 3)
    imp = np.abs(sv[:, :, DANGER]).mean(0)
    res["shap_mean_abs_danger"] = dict(zip(Xs.columns, imp.tolist()))
    import shap
    plt.figure()
    shap.summary_plot(sv[:, :, DANGER], Xs, show=False, plot_size=(7, 4))
    plt.title("SHAP — DANGER class (F2, test)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_danger_{cfg['data']['version']}.png", dpi=130)
    plt.close("all")
    model.save_model(os.path.join(cache, "xgb_f2.json"))

    # 위험도 벡터 내보내기 — 모든 keyframe 검출
    kf = est[est.is_key].copy()
    P = model.predict_proba(make_features(kf, src, can))
    rv = kf[["scene", "sd_token", "frame_idx", "tid", "cls", "x1", "y1", "x2", "y2", "z_f1", "x_lat",
             "zdot_f1", "ttc_f1", "state_f1", "pstat_f1", "ann_token"]].copy()
    rv = rv.merge(frames[["sd_token", "sample_token"]], on="sd_token")
    rv[["p_safe", "p_caution", "p_danger"]] = P
    rv.to_parquet(os.path.join(cache, "risk_vectors.parquet"), index=False)

    save_json(res, str(ROOT / "results" / f"risk_{cfg['data']['version']}.json"))
    md = to_md(res, cfg["data"]["version"])
    (ROOT / "results" / f"risk_{cfg['data']['version']}.md").write_text(md, encoding="utf-8")
    print(md)


def to_md(res: dict, version: str) -> str:
    names = {"R-B0": "R-B0 v2 규칙 + 박스 면적 TTC", "R-B1": "R-B1 v2 규칙 + 거리 차분 TTC",
             "R-F1": "R-F1 v2 규칙 + F1 융합 TTC", "F2": "F2 XGBoost (F1 특징 + CAN)",
             "A1": "A1 XGBoost (CAN 특징 제거)", "A2": "A2 XGBoost (CAN 없는 필터 + CAN 특징 제거)"}
    d = res["label_dist"]["test"]
    L = [f"# 위험도 평가 ({version}, test split)", "",
         f"test 라벨 분포 (SAFE / CAUTION / DANGER): {d[0]:,} / {d[1]:,} / {d[2]:,}", "",
         "| 모델 | Macro-F1 | DANGER 재현율 | DANGER 정밀도 | F1 SAFE | F1 CAUTION | F1 DANGER |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, name in names.items():
        x = res[k]
        pc = x["per_class_f1"]
        L.append(f"| {name} | {x['macro_f1']:.3f} | {x['danger_recall']:.1%} | {x['danger_precision']:.1%} | "
                 f"{pc[0]:.3f} | {pc[1]:.3f} | {pc[2]:.3f} |")
    L += ["", "## SHAP (F2, DANGER 클래스, mean |SHAP|)", ""]
    for f, v in sorted(res["shap_mean_abs_danger"].items(), key=lambda kv: -kv[1]):
        L.append(f"- `{f}` {v:.3f}")
    L += ["", "## 혼동 행렬 (행 = GT, 열 = 예측, 순서 " + " / ".join(LEVELS) + ")", ""]
    for k in names:
        L.append(f"- {k}: `{res[k]['confusion']}`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
