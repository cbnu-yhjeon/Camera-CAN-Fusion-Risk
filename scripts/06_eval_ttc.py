"""6단계: TTC · 객체 상태 평가 → results/ttc_<split>.{json,md}, results/figures/.

지표
  TTC MAE        GT TTC < 5 s 인 매칭 객체에서 |추정 − GT| 평균 (∞ 추정은 10 s 로 자름)
  TTC 흔들림      GT TTC < 5 s keyframe 앞뒤 ±0.5 s(12 Hz) 구간에서 프레임 간 ΔTTC 의 표준편차
  오경보율        GT TTC ≥ 10 s 또는 ∞ 인데 추정 TTC < 2 s 인 비율
  Ż MAE          모든 매칭 객체
  객체 상태       GT 속도로 만든 정지/동행/접근/이탈 대비 정확도 · macro-F1

    python scripts/06_eval_ttc.py --split test
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
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src.eval.common import METHODS, TTC_CAP, load, matched, save_json  # noqa: E402
from src.fusion.motion_state import STATES, classify  # noqa: E402
from src.utils.config import ROOT, cli  # noqa: E402

JITTER_HALF = 6     # 12 Hz 에서 ±0.5 s


def ttc_metrics(m: pd.DataFrame, est: pd.DataFrame) -> dict:
    near = m[m.ttc_gt < 5]
    far = m[~(m.ttc_gt < 10)]
    # 흔들림: GT<5 keyframe 주변 프레임 집합
    keys = near[["scene", "tid", "frame_idx"]].drop_duplicates()
    off = pd.DataFrame({"d": np.arange(-JITTER_HALF, JITTER_HALF + 1)})
    win = keys.merge(off, how="cross")
    win["frame_idx"] = win["frame_idx"] + win.pop("d")
    win = win.drop_duplicates()
    e = est.merge(win, on=["scene", "tid", "frame_idx"]).sort_values(["scene", "tid", "frame_idx"])
    consec = (e.groupby(["scene", "tid"])["frame_idx"].diff() == 1).to_numpy()
    out = {"n_near": int(len(near)), "n_far": int(len(far)), "n_all": int(len(m))}
    for k in METHODS:
        est_near = np.minimum(near[f"ttc_{k}"].to_numpy(), TTC_CAP)
        err = np.abs(est_near - near.ttc_gt.to_numpy())
        cap = np.minimum(e[f"ttc_{k}"].to_numpy(), TTC_CAP)
        dj = np.diff(cap, prepend=np.nan)[consec]
        r = {"ttc_mae": float(err.mean()), "ttc_medae": float(np.median(err)),
             "ttc_within_1s": float((err < 1.0).mean()),
             "detect_rate": float((near[f"ttc_{k}"] < 5).mean()),
             "false_alarm": float((far[f"ttc_{k}"] < 2).mean()),
             "jitter_std": float(np.nanstd(dj))}
        if k != "b0":
            dz = (m[f"zdot_{k}"] - m.zdot_gt).abs()
            r["zdot_mae"] = float(dz.mean())
            r["zdot_medae"] = float(dz.median())
        out[k] = r
    # 거리
    rel = lambda c: float(((m[c] - m.z_gt).abs() / m.z_gt).median())  # noqa: E731
    out["distance_rel_medae"] = {c: rel(c) for c in ["z_h", "z_g", "z_raw", "z_f1"]}
    out["distance_coverage"] = {c: float(m[c].notna().mean()) for c in ["z_h", "z_g", "z_raw"]}
    # GT 상태별 F1 vs B1 TTC MAE
    by = {}
    for st, g in near.groupby("state_gt"):
        by[st] = {k: float(np.abs(np.minimum(g[f"ttc_{k}"], TTC_CAP) - g.ttc_gt).mean()) for k in METHODS} | {"n": int(len(g))}
    out["ttc_mae_by_gt_state"] = by
    return out


def state_metrics(m: pd.DataFrame, cfg: dict) -> dict:
    mc = cfg["motion_state"]
    g = m[m.state_gt.notna()]
    y = g.state_gt.astype(str).to_numpy()
    preds = {k: g[f"state_{k}"] for k in ["b1", "f1a", "f1"]}
    # 영상만: v_ego 를 모르므로 '정지'를 판정할 수 없다 (v_obj 를 아주 큰 값으로 두어 정지 분기를 끈다)
    preds["f0_video_only"] = pd.Series(classify(np.full(len(g), 1e9), g.zdot_f0, mc["stationary_v"], mc["following_zdot"]), index=g.index)
    out = {"n": int(len(g)), "gt_dist": {s: float((y == s).mean()) for s in STATES}}
    for k, p in preds.items():
        p = p.fillna("none").astype(str).to_numpy()
        out[k] = {"accuracy": float((p == y).mean()),
                  "macro_f1": float(f1_score(y, p, labels=STATES, average="macro", zero_division=0)),
                  "recall": {s: float((p[y == s] == s).mean()) if (y == s).any() else None for s in STATES},
                  "confusion": confusion_matrix(y, p, labels=STATES).tolist()}
    return out


def figures(m: pd.DataFrame, est: pd.DataFrame, fig_dir: str, tag: str):
    os.makedirs(fig_dir, exist_ok=True)
    near = m[m.ttc_gt < 5]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, k in zip(axes, ["b0", "b1", "f1"]):
        ax.scatter(near.ttc_gt, np.minimum(near[f"ttc_{k}"], TTC_CAP), s=3, alpha=0.25)
        ax.plot([0, 5], [0, 5], "k--", lw=1)
        ax.set(title=METHODS[k], xlabel="GT TTC [s]", xlim=(0, 5), ylim=(0, TTC_CAP))
    axes[0].set_ylabel(f"estimated TTC [s] (inf -> {TTC_CAP:.0f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"ttc_scatter_{tag}.png"), dpi=130)
    plt.close(fig)

    bins = [-1, 2, 5, 11, 23, 47, 1000]
    lab = ["0-2", "3-5", "6-11", "12-23", "24-47", "48+"]
    near = near.assign(age=pd.cut(near.track_age, bins, labels=lab))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in ["b0", "b1", "f0", "f1a", "f1"]:
        s = near.groupby("age", observed=True).apply(
            lambda g: np.abs(np.minimum(g[f"ttc_{k}"], TTC_CAP) - g.ttc_gt).mean(), include_groups=False)
        ax.plot(s.index.astype(str), s.values, marker="o", label=METHODS[k])
    ax.set(xlabel="track age [frames @12 Hz]", ylabel="TTC MAE [s] (GT TTC < 5 s)", title="TTC error vs track age")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"ttc_vs_age_{tag}.png"), dpi=130)
    plt.close(fig)

    # 예시 트랙: GT<5 keyframe 이 가장 많은 트랙
    best = near.groupby(["scene", "tid"]).size().idxmax()
    e = est[(est.scene == best[0]) & (est.tid == best[1])].sort_values("frame_idx")
    g = m[(m.scene == best[0]) & (m.tid == best[1])]
    t0 = e.utime.min()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k in ["b0", "b1", "f1a", "f1"]:
        axes[0].plot((e.utime - t0) / 1e6, np.minimum(e[f"ttc_{k}"], TTC_CAP), label=METHODS[k], lw=1)
    axes[0].scatter((g.utime - t0) / 1e6, np.minimum(g.ttc_gt, TTC_CAP), c="k", zorder=5, label="GT (keyframe)")
    axes[0].set(ylabel="TTC [s]", title=f"{best[0]} track {best[1]}", ylim=(0, TTC_CAP))
    axes[0].legend(fontsize=7)
    axes[1].plot((e.utime - t0) / 1e6, e.z_h, ".", ms=3, label="Z height-pinhole")
    axes[1].plot((e.utime - t0) / 1e6, e.z_g, ".", ms=3, label="Z ground-contact")
    axes[1].plot((e.utime - t0) / 1e6, e.z_f1, "-", label="Z F1")
    axes[1].scatter((g.utime - t0) / 1e6, g.z_gt, c="k", zorder=5, label="GT")
    axes[1].set(xlabel="time [s]", ylabel="Z [m]")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"ttc_example_{tag}.png"), dpi=130)
    plt.close(fig)


def to_md(r: dict, s: dict, split: str) -> str:
    L = [f"# TTC · 객체 상태 평가 ({split} split)", "",
         f"매칭 객체 {r['n_all']:,}개, 그중 GT TTC < 5 s: {r['n_near']:,}개. "
         f"40 m 이내 GT(40% 이상 보임) 중 검출과 매칭된 비율 {r['gt_match_rate_40m']:.1%}.", "",
         "| 소스 | TTC MAE [s] | 중앙값 [s] | 1 s 이내 | 흔들림 σ(ΔTTC) [s] | 검출률 | 오경보율 | Ż MAE [m/s] |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, name in METHODS.items():
        x = r[k]
        zd = f"{x['zdot_mae']:.2f}" if "zdot_mae" in x else "–"
        L.append(f"| {name} | {x['ttc_mae']:.2f} | {x['ttc_medae']:.2f} | {x['ttc_within_1s']:.1%} | "
                 f"{x['jitter_std']:.2f} | {x['detect_rate']:.1%} | {x['false_alarm']:.1%} | {zd} |")
    L += ["", "검출률 = GT TTC < 5 s 일 때 추정도 < 5 s. 오경보율 = GT TTC ≥ 10 s(또는 ∞)인데 추정 < 2 s.", "",
          "## GT 상태별 TTC MAE (GT TTC < 5 s)", "", "| GT 상태 | n | " + " | ".join(METHODS) + " |",
          "|---|---:|" + "---:|" * len(METHODS)]
    for st, x in r["ttc_mae_by_gt_state"].items():
        L.append(f"| {st} | {x['n']:,} | " + " | ".join(f"{x[k]:.2f}" for k in METHODS) + " |")
    d = r["distance_rel_medae"]
    L += ["", "## 거리 (상대 오차 중앙값)", "",
          f"핀홀 높이 {d['z_h']:.1%} · 지면 접촉 {d['z_g']:.1%} · 두 단서 결합 {d['z_raw']:.1%} · F1 필터 {d['z_f1']:.1%}",
          "", "## 객체 상태 분류", "",
          f"n = {s['n']:,}. GT 분포: " + ", ".join(f"{k} {v:.1%}" for k, v in s["gt_dist"].items()), "",
          "| 방법 | 정확도 | Macro-F1 | 정지 재현율 | 동행 | 접근 | 이탈 |", "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"f0_video_only": "영상만 (F0, v_ego 없음)", "b1": "B1 차분 + CAN v_ego",
             "f1a": "F1a KF + CAN", "f1": "F1 IMM + CAN"}
    for k in ["f0_video_only", "b1", "f1a", "f1"]:
        x = s[k]
        rc = [x["recall"][st] for st in STATES]
        L.append(f"| {names[k]} | {x['accuracy']:.1%} | {x['macro_f1']:.3f} | " +
                 " | ".join("–" if v is None else f"{v:.1%}" for v in rc) + " |")
    return "\n".join(L) + "\n"


def main():
    args, cfg = cli(__doc__, lambda ap: ap.add_argument("--split", default="test"))
    est, gt, _ = load(cfg["data"]["cache_dir"])
    if args.split != "all":
        est = est[est.split == args.split]
    m = matched(est, gt, cfg)
    r, s = ttc_metrics(m, est), state_metrics(m, cfg)
    # TTC 는 검출과 매칭된 GT 에서만 잴 수 있다 — 분모를 함께 남긴다
    g = gt[(gt.visibility >= cfg["eval"]["min_visibility"]) & (gt.z_near < 40)]
    if args.split != "all":
        g = g[g.split == args.split]
    r["gt_match_rate_40m"] = float(g.ann_token.isin(set(est.ann_token.dropna())).mean())
    tag = f"{cfg['data']['version']}_{args.split}"
    out = ROOT / "results"
    save_json({"ttc": r, "motion_state": s}, str(out / f"ttc_{tag}.json"))
    md = to_md(r, s, f"{cfg['data']['version']} / {args.split}")
    (out / f"ttc_{tag}.md").write_text(md, encoding="utf-8")
    figures(m, est, str(out / "figures"), tag)
    print(md)


if __name__ == "__main__":
    main()
