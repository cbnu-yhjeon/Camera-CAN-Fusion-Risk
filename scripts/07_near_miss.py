"""7단계: near-miss 이벤트 후보 추출 + 평가 → results/near_miss/.

기준(reference) 이벤트는 두 가지다.
  auto  같은 규칙을 GT 3D 박스 TTC(keyframe, 2 Hz)에 적용한 이벤트 — 추정 TTC 의 품질만 본다.
  gold  사람이 검수 페이지(review.html)에서 고른 이벤트 — --gold goldset.json 으로 준다.

    python scripts/07_near_miss.py                      # 후보 추출 + auto 평가 + 검수 페이지
    python scripts/07_near_miss.py --gold results/near_miss/goldset.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.eval.common import load, save_json  # noqa: E402
from src.events.near_miss import brake_trigger, match_events, min_ttc_in_path, near_trigger, to_events  # noqa: E402
from src.risk.labels import gt_ttc  # noqa: E402
from src.utils.config import ROOT, cli  # noqa: E402

SOURCES = ["b0", "b1", "f1"]


def scene_events(f, e, g, cfg):
    nc = cfg["near_miss"]
    f = f.sort_values("frame_idx")
    t_frame = f.utime.to_numpy() / 1e6
    trig = brake_trigger(f, nc["decel_thresh"])
    trig_t = t_frame[trig]
    t_of = dict(zip(f.frame_idx, t_frame))
    out = {}
    for k in SOURCES:
        xcol = "x_lat" if k == "f1" else "x_raw"
        s = min_ttc_in_path(e, f"ttc_{k}", xcol, nc["corridor_half_width"])
        t = s.frame_idx.map(t_of).to_numpy(float)
        low = s.ttc_min.to_numpy() < nc["ttc_thresh"]
        flags = low & near_trigger(t, trig_t, nc["window_s"])
        evs = to_events(t, flags, nc["merge_gap_s"])
        out[k] = {"events": evs, "ttc_only": len(to_events(t, low, nc["merge_gap_s"])), "series": s.assign(t=t)}
    # GT 기준: keyframe 의 모든 GT 객체 (매칭 여부 무관)
    gg = g[(g.x_lat.abs() < nc["corridor_half_width"]) & (g.visibility >= 1)]
    if len(gg):
        gt_t = gg.assign(ttc=gt_ttc(gg.z_near, gg.zdot, cfg["ttc"]["eps"])).groupby("frame_idx").ttc.min()
        t = gt_t.index.map(t_of).to_numpy(float)
        low = gt_t.to_numpy() < nc["ttc_thresh"]
        out["gt"] = {"events": to_events(t, low & near_trigger(t, trig_t, nc["window_s"]), nc["merge_gap_s"]),
                     "ttc_only": len(to_events(t, low, nc["merge_gap_s"]))}
    else:
        out["gt"] = {"events": [], "ttc_only": 0}
    out["n_trigger"] = len(to_events(trig_t, np.ones(len(trig_t), bool), nc["merge_gap_s"]))
    return out, f, trig


def describe(ev, k, scene, f, e, trig, series):
    t0, t1 = ev
    s = series[(series.t >= t0) & (series.t <= t1)]
    peak = s.loc[s.ttc_min.idxmin()]
    fr = f[f.frame_idx == peak.frame_idx].iloc[0]
    row = e[(e.frame_idx == peak.frame_idx) & (e.tid == peak.tid)].iloc[0]
    tt = f.utime.to_numpy() / 1e6
    near = np.abs(tt - fr.utime / 1e6) <= 1.0
    return {"event_id": f"{scene}_{k}_{t0:.2f}", "scene": scene, "source": k, "t_start": t0, "t_end": t1,
            "peak_frame": int(peak.frame_idx), "sd_token": fr.sd_token, "filename": fr.filename,
            "min_ttc": float(peak.ttc_min), "tid": int(peak.tid), "cls": row.cls, "z": float(row["z_f1" if k == "f1" else "z_raw"]),
            "x_lat": float(row.x_lat), "v_ego": float(fr.v_ego), "min_a_ego": float(np.nanmin(f.a_ego.to_numpy()[near])),
            "brake_onset_near": bool(trig[near].any()), "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)]}


def thumbnail(ev, dataroot, out_dir):
    img = cv2.imread(os.path.join(dataroot, ev["filename"]))
    x1, y1, x2, y2 = map(int, ev["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    txt = f"TTC {ev['min_ttc']:.2f}s  Z {ev['z']:.1f}m  v_ego {ev['v_ego']:.1f}m/s  a_min {ev['min_a_ego']:.1f}"
    cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    img = cv2.resize(img, (800, 450), interpolation=cv2.INTER_AREA)
    name = f"{ev['event_id']}.jpg"
    cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return name


REVIEW_HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Near-miss 검수</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--card:#fff;--muted:#6b6b66;--line:#ddd;--yes:#17803d;--no:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--card:#222220;--muted:#a3a39c;--line:#3a3a37}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{max-width:860px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:0 0 16px;overflow:hidden}
.card img{width:100%;display:block}.meta{padding:10px 14px;color:var(--muted);font-size:13px}
.btns{display:flex;gap:8px;padding:0 14px 12px}button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.card[data-v="1"]{border-color:var(--yes);box-shadow:0 0 0 2px var(--yes)}.card[data-v="0"]{border-color:var(--no);box-shadow:0 0 0 2px var(--no)}
</style></head><body>
<header><b>Near-miss 후보 검수</b><span id="cnt"></span><button id="exp">goldset.json 내보내기</button></header>
<main id="list"></main>
<script>
const EV=__EVENTS__;const KEY="nearmiss-gold-v1";let lab={};try{lab=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(lab))}catch(e){}count()}
function count(){const n=Object.keys(lab).length;document.getElementById("cnt").textContent=`${n} / ${EV.length} 검수`}
const L=document.getElementById("list");
for(const e of EV){const c=document.createElement("div");c.className="card";c.dataset.v=lab[e.event_id]??"";
c.innerHTML=`<img loading="lazy" src="img/${e.img}" alt=""><div class="meta">${e.scene} · ${e.t_start.toFixed(1)}–${e.t_end.toFixed(1)} s · ${e.cls} · 최소 TTC ${e.min_ttc.toFixed(2)} s · 자동 기준 일치: ${e.matched_auto?"예":"아니오"}</div><div class="btns"><button data-a="1">위험했다</button><button data-a="0">아니다</button></div>`;
c.querySelectorAll("button").forEach(b=>b.onclick=()=>{lab[e.event_id]=+b.dataset.a;c.dataset.v=b.dataset.a;save()});L.appendChild(c)}
document.getElementById("exp").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(lab,null,1)],{type:"application/json"}));a.download="goldset.json";a.click()};
count();
</script></body></html>"""


def main():
    args, cfg = cli(__doc__, lambda ap: ap.add_argument("--gold", default=None))
    est, gt, frames = load(cfg["data"]["cache_dir"])
    est["x_raw"] = est.x_lat * est.z_raw / est.z_f1
    out_dir = ROOT / "results" / "near_miss"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    eg, gg = dict(tuple(est.groupby("scene"))), dict(tuple(gt.groupby("scene")))
    tol = cfg["near_miss"]["window_s"]
    stats = {k: {"cand": 0, "hit_c": 0, "ref": 0, "hit_r": 0, "ttc_only": 0} for k in SOURCES}
    per_split = {}
    rows, n_ref, n_trig, gt_ttc_only = [], 0, 0, 0
    for scene, f in tqdm(frames.groupby("scene"), desc="near-miss"):
        if scene not in eg:
            continue
        ev, f, trig = scene_events(f, eg[scene], gg.get(scene, gt.iloc[:0]), cfg)
        ref = ev["gt"]["events"]
        n_ref += len(ref)
        n_trig += ev["n_trigger"]
        gt_ttc_only += ev["gt"]["ttc_only"]
        sp = f.split.iloc[0]
        for k in SOURCES:
            c_hit, r_hit = match_events(ev[k]["events"], ref, tol)
            for d in (stats[k], per_split.setdefault(sp, {}).setdefault(k, {"cand": 0, "hit_c": 0, "ref": 0, "hit_r": 0})):
                d["cand"] += len(c_hit)
                d["hit_c"] += int(c_hit.sum())
                d["ref"] += len(r_hit)
                d["hit_r"] += int(r_hit.sum())
            stats[k]["ttc_only"] += ev[k]["ttc_only"]
            for e_, h in zip(ev[k]["events"], c_hit):
                d = describe(e_, k, scene, f, eg[scene], trig, ev[k]["series"])
                d["matched_auto"], d["split"] = bool(h), sp
                rows.append(d)
    cand = pd.DataFrame(rows)
    cand.to_csv(out_dir / "candidates.csv", index=False)

    def pr(d):
        return {"precision": d["hit_c"] / d["cand"] if d["cand"] else None,
                "recall": d["hit_r"] / d["ref"] if d["ref"] else None, **d}

    res = {"n_reference_auto": n_ref, "n_gt_ttc_only_episodes": gt_ttc_only, "n_brake_trigger_episodes": n_trig,
           "all": {k: pr(v) for k, v in stats.items()},
           "by_split": {sp: {k: pr(v) for k, v in d.items()} for sp, d in per_split.items()}}

    # 검수 페이지 — F1 후보 (+ F1 이 놓친 자동 기준 이벤트는 B1 후보로 보완하지 않는다: 검수 대상은 시스템 출력)
    f1 = cand[cand.source == "f1"].copy()
    f1["img"] = [thumbnail(r, cfg["data"]["dataroot"], str(out_dir / "img")) for r in tqdm(f1.to_dict("records"), desc="thumbs")]
    keep = ["event_id", "scene", "t_start", "t_end", "cls", "min_ttc", "matched_auto", "img"]
    html = REVIEW_HTML.replace("__EVENTS__", json.dumps(f1[keep].to_dict("records"), ensure_ascii=False))
    (out_dir / "review.html").write_text(html, encoding="utf-8")

    if args.gold:
        with open(args.gold) as fh:
            gold = json.load(fh)
        lab = f1.event_id.map(gold)
        reviewed = lab.notna()
        res["gold"] = {"n_reviewed": int(reviewed.sum()), "n_positive": int((lab == 1).sum()),
                       "precision_f1": float((lab[reviewed] == 1).mean()) if reviewed.any() else None}
    save_json(res, str(out_dir / f"near_miss_{cfg['data']['version']}.json"))
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in res["all"].items()}, indent=1))
    print(f"auto reference events={n_ref}  GT TTC-only episodes={gt_ttc_only}  brake-trigger episodes={n_trig}")


if __name__ == "__main__":
    main()
