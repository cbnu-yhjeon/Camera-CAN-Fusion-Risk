"""nuScenes → 평평한 표(parquet) 변환. 이후 단계는 devkit 없이 이 표만 읽는다.

만드는 것
---------
frames : CAM_FRONT 의 모든 프레임(keyframe 2 Hz + sweep, 합쳐 약 12 Hz).
         ego pose 와 ego_pose 차분 속도(GT 자차 속도)를 함께 둔다.
gt     : keyframe 의 GT 3D 박스를 카메라 좌표로 옮긴 값.
           z_near  = 박스 8개 꼭짓점 중 가장 가까운 깊이 — 카메라가 보는 면까지의 거리.
                     핀홀 거리(박스 높이)와 지면 접촉 거리가 모두 이 면을 잰다.
           x_lat   = 박스 중심의 카메라 x (횡방향, 오른쪽 +)
           zdot    = (v_obj − v_ego) · 광축      — 다가오면 음수
           v_long  = v_obj · 광축                 — 객체 절대 속도의 진행 방향 성분
         카메라 회전(자차 yaw rate)으로 생기는 Ż 항(ω·X)은 넣지 않는다. TTC 는 병진 접근을 묻는 값이다.
calib  : scene 별 CAM_FRONT intrinsic / extrinsic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.geometry import box_corners, global_to_cam, project, quat_to_rot, quat_yaw

CATEGORY_MAP = [
    ("vehicle.car", "car"), ("vehicle.emergency", "car"),
    ("vehicle.truck", "truck"), ("vehicle.trailer", "truck"), ("vehicle.construction", "truck"),
    ("vehicle.bus", "bus"),
    ("vehicle.bicycle", "bicycle"), ("vehicle.motorcycle", "motorcycle"),
    ("human.pedestrian", "pedestrian"),
]


def map_category(name: str) -> str | None:
    for prefix, cls in CATEGORY_MAP:
        if name.startswith(prefix):
            return cls
    return None


def _scene_frames(nusc, scene, camera: str) -> list[dict]:
    first = nusc.get("sample", scene["first_sample_token"])
    tok = first["data"][camera]
    rows = []
    while tok:
        sd = nusc.get("sample_data", tok)
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        rows.append({
            "sd_token": tok, "utime": sd["timestamp"], "filename": sd["filename"],
            "is_key": bool(sd["is_key_frame"]), "sample_token": sd["sample_token"],
            "ego_t": np.array(ego["translation"]), "ego_q": np.array(ego["rotation"]),
            "cs_token": sd["calibrated_sensor_token"], "width": sd["width"], "height": sd["height"],
        })
        tok = sd["next"]
    # 자차 GT 속도: 이웃 프레임 ego_pose 중앙 차분
    t = np.array([r["utime"] for r in rows]) / 1e6
    p = np.stack([r["ego_t"][:2] for r in rows])
    v = np.gradient(p, t, axis=0) if len(rows) > 1 else np.zeros_like(p)
    for r, vi in zip(rows, v):
        r["ego_v"] = vi
    return rows


def _gt_rows(nusc, frame: dict, cs: dict) -> list[dict]:
    sample = nusc.get("sample", frame["sample_token"])
    K = np.array(cs["camera_intrinsic"])
    cam_t, cam_q = np.array(cs["translation"]), np.array(cs["rotation"])
    W, H = frame["width"], frame["height"]
    axis_g = quat_to_rot(frame["ego_q"]) @ quat_to_rot(cam_q) @ np.array([0.0, 0.0, 1.0])
    axis_g = axis_g[:2] / np.linalg.norm(axis_g[:2])
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls = map_category(ann["category_name"])
        if cls is None:
            continue
        corners = global_to_cam(box_corners(ann["translation"], ann["size"], ann["rotation"]),
                                frame["ego_t"], frame["ego_q"], cam_t, cam_q)
        if np.any(corners[:, 2] < 0.5):          # 일부가 카메라 뒤 — 투영 불가
            continue
        uv = project(corners, K)
        x1, y1 = np.clip(uv.min(0), [0, 0], [W, H])
        x2, y2 = np.clip(uv.max(0), [0, 0], [W, H])
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        center = global_to_cam(np.array([ann["translation"]]), frame["ego_t"], frame["ego_q"], cam_t, cam_q)[0]
        v_obj = np.asarray(nusc.box_velocity(ann_token)[:2])  # NaN 가능 (앞뒤 주석 없음)
        out.append({
            "ann_token": ann_token, "instance_token": ann["instance_token"],
            "category": ann["category_name"], "cls": cls,
            "attribute": nusc.get("attribute", ann["attribute_tokens"][0])["name"] if ann["attribute_tokens"] else "",
            "visibility": int(ann["visibility_token"] or 0),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "z_near": float(corners[:, 2].min()), "z_center": float(center[2]), "x_lat": float(center[0]),
            "size_w": ann["size"][0], "size_l": ann["size"][1], "size_h": ann["size"][2],
            "v_long": float(v_obj @ axis_g) if np.all(np.isfinite(v_obj)) else np.nan,
            "zdot": float((v_obj - frame["ego_v"]) @ axis_g) if np.all(np.isfinite(v_obj)) else np.nan,
            "num_lidar_pts": ann["num_lidar_pts"],
        })
    return out


def build_index(nusc, scene_names: list[str], camera: str, scene_split: dict[str, str]):
    frames, gts, calib = [], [], {}
    by_name = {s["name"]: s for s in nusc.scene}
    for name in tqdm(scene_names, desc="index"):
        scene = by_name[name]
        rows = _scene_frames(nusc, scene, camera)
        cs = nusc.get("calibrated_sensor", rows[0]["cs_token"])
        calib[name] = {"K": cs["camera_intrinsic"], "translation": cs["translation"],
                       "rotation": cs["rotation"], "width": rows[0]["width"], "height": rows[0]["height"]}
        for i, r in enumerate(rows):
            frames.append({
                "scene": name, "split": scene_split.get(name, ""), "frame_idx": i,
                "sd_token": r["sd_token"], "utime": r["utime"], "filename": r["filename"],
                "is_key": r["is_key"], "sample_token": r["sample_token"] if r["is_key"] else "",
                "ego_x": r["ego_t"][0], "ego_y": r["ego_t"][1], "ego_yaw": quat_yaw(r["ego_q"]),
                "ego_speed_gt": float(np.linalg.norm(r["ego_v"])),
            })
            if r["is_key"]:
                for g in _gt_rows(nusc, r, cs):
                    gts.append({"scene": name, "sd_token": r["sd_token"], "utime": r["utime"],
                                "frame_idx": i, **g})
    return pd.DataFrame(frames), pd.DataFrame(gts), calib
