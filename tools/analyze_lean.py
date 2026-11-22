"""2단계 인식: 전체 프레임에서 라이더를 추적한 뒤, 확대 크롭에서 자세·자전거를 다시 인식한다.

사용법:
    uv run python tools/analyze_lean.py sessions/2026-11-21/segments.json [--runs 1 3]

출력 (<feature>_lean/):
    crops/run<N>_f<frame>.jpg   : 오버레이 없는 원본 크롭 (기준선 표시·검증용)
    overlay/run<N>_f<frame>.jpg : 인식 결과 오버레이
    run<N>.csv                  : 프레임별 측정값

각도: 화면 수직 기준, +는 화면 왼쪽(카메라가 뒤에 있으면 라이더 왼쪽).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

import cv2
import numpy as np
from ultralytics import YOLO

PAD = 0.4
IMGSZ = 1280
CROP = 768          # 크롭 해상도 (확대)
CROP_SCALE = 2.0    # 라이더 박스 높이 대비 크롭 크기
KP = dict(nose=0, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
          l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def lean_deg(top, bottom):
    dx, dy = top[0] - bottom[0], top[1] - bottom[1]
    return math.degrees(math.atan2(-dx, -dy))


def joint_angle(a, b, c):
    """b에서의 관절 각도(도). 180=완전히 폄."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def bike_axis(mask, max_tilt=60):
    """마스크 아래쪽 55%(뒷바퀴 위주) 주축. 수직 ±max_tilt 밖이면(옆모습 등) None."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 150:
        return None
    y0, y1 = ys.min(), ys.max()
    sel = ys > y0 + (y1 - y0) * 0.45
    pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    if len(pts) < 100:
        return None
    c = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    if s[0] < 1.8 * s[1]:
        return None
    v = vt[0] if vt[0][1] < 0 else -vt[0]
    half = (y1 - y0) * 0.55
    top, bottom = c + v * half, c - v * half
    ang = lean_deg(top, bottom)
    if abs(ang) > max_tilt:
        return None
    return ang, top, bottom


def track_rider(cap, fps, s, e, pose_m):
    f0, f1 = max(0, int((s - PAD) * fps)), int((e + PAD) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    pose_m.predictor = None
    frames = []
    for fi in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        pr = pose_m.track(img, persist=True, imgsz=IMGSZ, classes=[0], verbose=False, tracker="bytetrack.yaml")[0]
        boxes = {}
        if pr.boxes.id is not None:
            boxes = dict(zip(pr.boxes.id.int().tolist(), pr.boxes.xyxy.tolist()))
        frames.append((fi, img, boxes))
    area = {}
    for fi, _, boxes in frames:
        if s * fps <= fi <= e * fps:
            for tid, b in boxes.items():
                area[tid] = area.get(tid, 0) + (b[2] - b[0]) * (b[3] - b[1])
    rider = max(area, key=area.get) if area else None
    return frames, rider


def crop_affine(box, prev):
    bh = box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 0.15 * bh
    sz = max(bh, box[2] - box[0]) * CROP_SCALE
    if prev is not None:
        a = 0.4
        cx, cy, sz = (prev[0] + a * (cx - prev[0]), prev[1] + a * (cy - prev[1]), prev[2] + 0.2 * (sz - prev[2]))
    k = CROP / sz
    M = np.float32([[k, 0, -(cx - sz / 2) * k], [0, k, -(cy - sz / 2) * k]])
    return M, (cx, cy, sz)


def pick_center_person(pr):
    if pr.keypoints is None or len(pr.boxes) == 0:
        return None
    c = np.array([CROP / 2, CROP / 2])
    best, bd = None, 1e9
    for i, b in enumerate(pr.boxes.xyxy.cpu().numpy()):
        d = np.linalg.norm(((b[:2] + b[2:]) / 2) - c)
        if d < bd:
            bd, best = d, i
    return best


def pick_bike_mask(sr, rider_box):
    if sr.masks is None:
        return None, 0.0
    x0, y0, x1, y1 = rider_box
    best, bov, bconf = None, 0, 0.0
    for b, m, cf in zip(sr.boxes.xyxy.tolist(), sr.masks.data, sr.boxes.conf.tolist()):
        ov = max(0, min(x1, b[2]) - max(x0, b[0])) * max(0, min(y1, b[3]) - max(y0, b[1]))
        if ov > bov:
            bov, best, bconf = ov, m, cf
    if best is None:
        return None, 0.0
    m = cv2.resize(best.cpu().numpy().astype(np.uint8), (CROP, CROP), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool), bconf


def analyze_run(r, vdir, out, pose_m, seg_m):
    cap = cv2.VideoCapture(str(vdir / r["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    s, e = r["start"], r["end"]
    frames, rider = track_rider(cap, fps, s, e, pose_m)
    cap.release()
    if rider is None:
        print(f"  run {r['run']}: 라이더 없음")
        return
    rows, prev = [], None
    for fi, img, boxes in frames:
        t = fi / fps
        if not (s <= t <= e) or rider not in boxes:
            continue
        M, prev = crop_affine(boxes[rider], prev)
        crop = cv2.warpAffine(img, M, (CROP, CROP), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
        name = f"run{r['run']}_f{fi:04d}"
        cv2.imwrite(str(out / "crops" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        row = dict(run=r["run"], frame=fi, t=round(t, 3), progress=round((t - s) / (e - s), 4))

        pr = pose_m.predict(crop, imgsz=CROP, classes=[0], verbose=False)[0]
        pi = pick_center_person(pr)
        vis = crop.copy()
        if pi is not None:
            kxy = pr.keypoints.xy[pi].cpu().numpy()
            kcf = pr.keypoints.conf[pi].cpu().numpy()
            rbox = pr.boxes.xyxy[pi].tolist()
            P = lambda n: kxy[KP[n]]
            C = lambda *ns: min(kcf[KP[n]] for n in ns)
            sh, hp = (P("l_sh") + P("r_sh")) / 2, (P("l_hip") + P("r_hip")) / 2
            row["torso_conf"] = round(float(C("l_sh", "r_sh", "l_hip", "r_hip")), 3)
            row["body_lean"] = round(lean_deg(sh, hp), 2)
            torso = float(np.linalg.norm(sh - hp))
            row["view_ratio"] = round(float(np.linalg.norm(P("l_sh") - P("r_sh"))) / max(torso, 1), 3)
            row["limb_conf"] = round(float(C("l_wr", "r_wr", "l_an", "r_an")), 3)
            wr, an = (P("l_wr") + P("r_wr")) / 2, (P("l_an") + P("r_an")) / 2
            row["handfoot_lean"] = round(lean_deg(wr, an), 2)
            # 관절 각도 (좌우 평균) — 시점 영향을 덜 받는 웅크림 지표
            for side in ("l", "r"):
                if C(f"{side}_hip", f"{side}_kn", f"{side}_an") > 0.3:
                    row[f"knee_{side}"] = round(joint_angle(P(f"{side}_hip"), P(f"{side}_kn"), P(f"{side}_an")), 1)
                if C(f"{side}_sh", f"{side}_hip", f"{side}_kn") > 0.3:
                    row[f"hip_{side}"] = round(joint_angle(P(f"{side}_sh"), P(f"{side}_hip"), P(f"{side}_kn")), 1)
                if C(f"{side}_sh", f"{side}_el", f"{side}_wr") > 0.3:
                    row[f"elbow_{side}"] = round(joint_angle(P(f"{side}_sh"), P(f"{side}_el"), P(f"{side}_wr")), 1)
            for a, b in SKELETON:
                if kcf[a] > 0.3 and kcf[b] > 0.3:
                    cv2.line(vis, tuple(map(int, kxy[a])), tuple(map(int, kxy[b])), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(vis, tuple(map(int, hp)), tuple(map(int, sh + (sh - hp) * 0.5)), (60, 220, 60), 3, cv2.LINE_AA)

            sr = seg_m.predict(crop, imgsz=CROP, classes=[1], conf=0.05, verbose=False, retina_masks=True)[0]
            bh = rbox[3] - rbox[1]
            mask, bconf = pick_bike_mask(sr, (rbox[0] - 0.3 * bh, rbox[1] + 0.3 * bh, rbox[2] + 0.3 * bh, rbox[3] + 0.4 * bh))
            if mask is not None:
                row["bike_conf"] = round(bconf, 3)
                vis[mask] = (vis[mask] * 0.5 + np.array([0, 140, 255]) * 0.5).astype(np.uint8)
                ax = bike_axis(mask)
                if ax is not None:
                    row["bike_lean"] = round(ax[0], 2)
                    row["separation"] = round(ax[0] - row["body_lean"], 2)
                    cv2.line(vis, tuple(map(int, ax[2])), tuple(map(int, ax[1])), (0, 60, 255), 3, cv2.LINE_AA)
        txt = " ".join(f"{k}={row[k]:+.0f}" for k in ("bike_lean", "body_lean") if k in row)
        cv2.putText(vis, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "overlay" / f"{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(row)

    keys = ["run", "frame", "t", "progress", "view_ratio", "torso_conf", "body_lean", "bike_conf", "bike_lean",
            "separation", "limb_conf", "handfoot_lean", "knee_l", "knee_r", "hip_l", "hip_r", "elbow_l", "elbow_r"]
    with open(out / f"run{r['run']}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    nb = sum(1 for x in rows if "bike_lean" in x)
    nm = sum(1 for x in rows if "bike_conf" in x)
    nrear = sum(1 for x in rows if x.get("view_ratio", 0) > 0.75)
    nrb = sum(1 for x in rows if x.get("view_ratio", 0) > 0.75 and "bike_lean" in x)
    print(f"  run {r['run']}: {len(rows)}프레임, 자전거 마스크 {nm}, 자전거 축 {nb}, 뒷모습 {nrear} 중 축 {nrb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--runs", type=int, nargs="*")
    a = ap.parse_args()
    seg = Path(a.segments)
    cfg = json.loads(seg.read_text(encoding="utf-8"))
    out = PATHS.feat_dir(seg, cfg, "lean")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(exist_ok=True)
    models = PATHS.MODELS
    pose_m = YOLO(str(models / "yolo11x-pose.pt"))
    seg_m = YOLO(str(models / "yolo11x-seg.pt"))
    for r in cfg["runs"]:
        if a.runs and r["run"] not in a.runs:
            continue
        analyze_run(r, PATHS.video_dir(cfg), out, pose_m, seg_m)
    print("done:", out)


if __name__ == "__main__":
    main()
