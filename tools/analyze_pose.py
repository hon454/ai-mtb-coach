"""피처 구간(segments.json)마다 라이더를 추적하고 자세·자전거 기울기를 계산한다.

사용법:
    uv run python tools/analyze_pose.py sessions/2026-11-21/segments.json [--runs 1 3] [--debug]

출력 (segments.json 옆 <feature>_pose/ 폴더):
    run<N>.csv           : 프레임별 지표 (진행률, 자전거/몸 기울기, 분리각, 높이)
    run<N>_crop.mp4      : 라이더 중심 크롭 + 뼈대/기울기선 오버레이
    <feature>_crop_sync.mp4 : 7번 크롭 영상을 진행률로 맞춘 그리드
    <feature>_metrics.png   : 진행률별 지표 그래프 (런별 선 + 평균±표준편차)

각도 규칙 (카메라가 라이더 뒤쪽에 있다고 가정):
    화면 수직 기준, +는 화면 왼쪽(=라이더 왼쪽, 좌회전 안쪽)으로 기운 것.
    카메라 자체가 기울어진(roll) 만큼 오차가 있다.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

import cv2
import numpy as np
from ultralytics import YOLO

PAD = 0.4          # 구간 앞뒤로 추적 워밍업용 여유(초)
CROP = 512         # 크롭 출력 크기(px)
IMGSZ = 1280       # 작은 라이더 검출용 입력 크기
LIMB_CONF = 0.15   # 손목·발목·무릎은 뒷모습에서 가려지기 쉬워 낮은 신뢰도도 허용 (모델이 추정한 위치)
KP = dict(nose=0, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
          l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def lean_deg(top, bottom):
    """bottom→top 벡터의 화면 수직 대비 기울기. +면 화면 왼쪽."""
    dx, dy = top[0] - bottom[0], top[1] - bottom[1]
    return math.degrees(math.atan2(-dx, -dy))


def bike_axis(mask):
    """자전거 마스크 아래쪽 55%(뒷바퀴 위주)의 주축. 반환: (lean, top, bottom) 또는 None."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 80:
        return None
    y0, y1 = ys.min(), ys.max()
    sel = ys > y0 + (y1 - y0) * 0.45
    pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    if len(pts) < 50:
        return None
    c = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    if s[0] < 1.6 * s[1]:          # 충분히 길쭉하지 않으면(옆모습 등) 신뢰 안 함
        return None
    v = vt[0]
    if v[1] > 0:
        v = -v                       # 위쪽을 향하게
    half = (y1 - y0) * 0.5
    top, bottom = c + v * half, c - v * half
    return lean_deg(top, bottom), top, bottom


def pick_bike(seg_res, rider_box, shape):
    """라이더 박스와 가장 많이 겹치는 bicycle 마스크."""
    if seg_res.masks is None:
        return None
    best, best_ov = None, 0
    x0, y0, x1, y1 = rider_box
    for cls, box, m in zip(seg_res.boxes.cls.tolist(), seg_res.boxes.xyxy.tolist(), seg_res.masks.data):
        if int(cls) != 1:            # COCO 1 = bicycle
            continue
        bx0, by0, bx1, by1 = box
        ov = max(0, min(x1, bx1) - max(x0, bx0)) * max(0, min(y1, by1) - max(y0, by0))
        if ov > best_ov:
            best_ov, best = ov, m
    if best is None:
        return None
    m = cv2.resize(best.cpu().numpy().astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def analyze_run(r, vdir, out_dir, pose_m, seg_m):
    cap = cv2.VideoCapture(str(vdir / r["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    s, e = r["start"], r["end"]
    f0 = max(0, int((s - PAD) * fps))
    f1 = int((e + PAD) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    pose_m.predictor = None          # 런마다 추적 상태 초기화
    frames = []
    for fi in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        pr = pose_m.track(img, persist=True, imgsz=IMGSZ, classes=[0], verbose=False, tracker="bytetrack.yaml")[0]
        frames.append((fi, img, pr))
    cap.release()

    # 라이더 = 구간 안에서 박스 면적 합이 가장 큰 트랙
    area = {}
    for fi, img, pr in frames:
        if not (s * fps <= fi <= e * fps) or pr.boxes.id is None:
            continue
        for tid, b in zip(pr.boxes.id.int().tolist(), pr.boxes.xyxy.tolist()):
            area[tid] = area.get(tid, 0) + (b[2] - b[0]) * (b[3] - b[1])
    if not area:
        print(f"  run {r['run']}: 라이더를 찾지 못함")
        return None
    rider = max(area, key=area.get)

    rows, crops = [], []
    cx_s = cy_s = sz_s = None
    for fi, img, pr in frames:
        t = fi / fps
        if not (s <= t <= e):
            continue
        prog = (t - s) / (e - s)
        row = dict(run=r["run"], frame=fi, t=round(t, 3), progress=round(prog, 4))
        det = None
        if pr.boxes.id is not None:
            ids = pr.boxes.id.int().tolist()
            if rider in ids:
                det = ids.index(rider)
        if det is None:
            rows.append(row)
            crops.append(np.zeros((CROP, CROP, 3), np.uint8))
            continue
        box = pr.boxes.xyxy[det].tolist()
        kxy = pr.keypoints.xy[det].cpu().numpy()
        kcf = pr.keypoints.conf[det].cpu().numpy()

        bh = box[3] - box[1]

        def ok(*ix, th=0.4):
            return all(kcf[i] > th for i in ix)

        def mid(a, b):
            return (kxy[KP[a]] + kxy[KP[b]]) / 2

        bax = None
        if ok(KP["l_sh"], KP["r_sh"], KP["l_hip"], KP["r_hip"]):
            sh, hp = mid("l_sh", "r_sh"), mid("l_hip", "r_hip")
            torso = float(np.linalg.norm(sh - hp))
            row["body_lean"] = round(lean_deg(sh, hp), 2)
            row["torso_px"] = round(torso, 1)
            # 어깨 폭 / 몸통 길이: 뒷모습(정면)일수록 크고, 옆모습일수록 작다 → 시점 표시용
            row["view_ratio"] = round(float(np.linalg.norm(kxy[KP["l_sh"]] - kxy[KP["r_sh"]])) / max(torso, 1), 3)
        # 자전거 축 근사: 발(페달) 중간 → 손(핸들) 중간
        if ok(KP["l_wr"], KP["r_wr"], KP["l_an"], KP["r_an"], th=LIMB_CONF):
            wr, an = mid("l_wr", "r_wr"), mid("l_an", "r_an")
            bax = (lean_deg(wr, an), wr, an)
            row["bike_lean"] = round(bax[0], 2)
            if "body_lean" in row:
                row["separation"] = round(row["bike_lean"] - row["body_lean"], 2)
                if ok(KP["l_kn"], KP["r_kn"], th=LIMB_CONF):
                    kn = mid("l_kn", "r_kn")
                    legs = row["torso_px"] + np.linalg.norm(hp - kn) + np.linalg.norm(kn - an)
                    # 자세 높이: 발→어깨 수직거리 / 관절 길이 합 (1=완전히 선 자세, 작을수록 웅크림)
                    row["stance_h"] = round(float((an[1] - sh[1]) / max(legs, 1)), 3)
        rows.append(row)

        # --- 오버레이 그리기 ---
        vis = img.copy()
        for a, b in SKELETON:
            if kcf[a] > LIMB_CONF and kcf[b] > LIMB_CONF:
                cv2.line(vis, tuple(map(int, kxy[a])), tuple(map(int, kxy[b])), (255, 255, 0), 2, cv2.LINE_AA)
        for i in range(17):
            if kcf[i] > LIMB_CONF:
                cv2.circle(vis, tuple(map(int, kxy[i])), 3, (0, 255, 255), -1, cv2.LINE_AA)
        if "body_lean" in row:
            cv2.line(vis, tuple(map(int, hp)), tuple(map(int, sh + (sh - hp) * 0.6)), (60, 220, 60), 3, cv2.LINE_AA)
        if bax is not None:
            cv2.line(vis, tuple(map(int, bax[2])), tuple(map(int, bax[1])), (0, 80, 255), 3, cv2.LINE_AA)

        # --- 라이더 중심 크롭 (부드럽게 따라감) ---
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 0.15 * bh
        sz = max(bh * 2.0, (box[2] - box[0]) * 2.0, 160)
        if cx_s is None:
            cx_s, cy_s, sz_s = cx, cy, sz
        else:
            a = 0.35
            cx_s, cy_s, sz_s = cx_s + a * (cx - cx_s), cy_s + a * (cy - cy_s), sz_s + 0.15 * (sz - sz_s)
        half = sz_s / 2
        M = np.float32([[CROP / sz_s, 0, -(cx_s - half) * CROP / sz_s],
                        [0, CROP / sz_s, -(cy_s - half) * CROP / sz_s]])
        crop = cv2.warpAffine(vis, M, (CROP, CROP), borderValue=(0, 0, 0))
        y = 26
        for k, col, name in (("bike_lean", (0, 80, 255), "bike"), ("body_lean", (60, 220, 60), "body"),
                             ("separation", (255, 255, 255), "sep")):
            if k in row:
                txt = f"{name} {row[k]:+.0f}"
                cv2.putText(crop, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(crop, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                y += 26
        crops.append(crop)

    with open(out_dir / f"run{r['run']}.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["run", "frame", "t", "progress", "bike_lean", "body_lean", "separation", "stance_h", "view_ratio", "torso_px"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_mp4(out_dir / f"run{r['run']}_crop.mp4", crops, fps)
    valid = sum(1 for x in rows if "bike_lean" in x and "body_lean" in x)
    print(f"  run {r['run']}: rider track {rider}, {len(rows)} frames, 지표 계산 {valid}프레임")
    return rows, crops


def write_mp4(path, frames, fps):
    tmp = path.with_suffix(".raw.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    tmp.unlink()


def sync_grid(results, out, n_out=120, fps=30):
    """런별 크롭을 진행률로 맞춰 4열 그리드로."""
    runs = sorted(results)
    cols = 4
    nrows = -(-len(runs) // cols)
    cell = 384
    frames = []
    for k in range(n_out):
        p = k / (n_out - 1)
        canvas = np.zeros((nrows * cell, cols * cell, 3), np.uint8)
        for j, rn in enumerate(runs):
            crops = results[rn]
            img = cv2.resize(crops[round(p * (len(crops) - 1))], (cell, cell))
            cv2.putText(img, f"Run {rn}", (8, cell - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, f"Run {rn}", (8, cell - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            r_, c_ = divmod(j, cols)
            canvas[r_ * cell:(r_ + 1) * cell, c_ * cell:(c_ + 1) * cell] = img
        if len(runs) < nrows * cols:
            r_, c_ = divmod(len(runs), cols)
            x0, y0 = c_ * cell + 16, r_ * cell + 40
            cv2.putText(canvas, f"progress {p*100:5.1f}%", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            for i, (col, name) in enumerate((((0, 80, 255), "orange: feet->hands"), ((60, 220, 60), "green: torso"),
                                             ((255, 255, 0), "cyan: skeleton"))):
                cv2.putText(canvas, name, (x0, y0 + 50 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.putText(canvas, "+ = lean to rider's left", (x0, y0 + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        frames.append(canvas)
    write_mp4(out, frames, fps)


def plot(all_rows, out, feature):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    metrics = [("bike_lean", "자전거 축 기울기 (발→손 근사, °, +=왼쪽)"), ("body_lean", "상체 기울기 (°, +=왼쪽)"),
               ("separation", "분리각 = 자전거 - 상체 (°)"), ("stance_h", "자세 높이 (1=완전히 선 자세, 낮을수록 웅크림)"),
               ("view_ratio", "시점 (어깨폭/몸통, 클수록 뒷모습 · 작을수록 옆모습)")]
    grid = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)), sharex=True)
    cmap = plt.get_cmap("viridis")
    runs = sorted(all_rows)
    for ax, (k, title) in zip(axes, metrics):
        stack = []
        for i, rn in enumerate(runs):
            pts = [(x["progress"], x[k]) for x in all_rows[rn] if k in x]
            if len(pts) < 5:
                continue
            p, v = map(np.array, zip(*pts))
            v = np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # 5프레임 이동평균
            ax.plot(p * 100, v, color=cmap(i / max(len(runs) - 1, 1)), lw=1.3, alpha=0.85, label=f"Run {rn}")
            stack.append(np.interp(grid, p, v))
        if stack:
            st = np.array(stack)
            ax.fill_between(grid * 100, st.mean(0) - st.std(0), st.mean(0) + st.std(0), color="gray", alpha=0.2, label="평균±표준편차")
            ax.plot(grid * 100, st.mean(0), color="black", lw=2, ls="--", label="평균")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=5, fontsize=8, loc="upper left")
    axes[-1].set_xlabel(f"{feature} 진행률 (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--runs", type=int, nargs="*")
    a = ap.parse_args()
    seg = Path(a.segments)
    cfg = json.loads(seg.read_text(encoding="utf-8"))
    feat = cfg.get("feature", "feature")
    out_dir = PATHS.feat_dir(seg, cfg, "pose")
    out_dir.mkdir(exist_ok=True)
    models = Path(__file__).resolve().parent.parent / "models"
    models.mkdir(exist_ok=True)
    pose_m = YOLO(str(models / "yolo11x-pose.pt"))
    seg_m = None  # 자전거 분할은 기운 뒷모습에서 검출이 약해 현재 미사용
    vdir = PATHS.video_dir(cfg)
    rows, crops = {}, {}
    for r in cfg["runs"]:
        if a.runs and r["run"] not in a.runs:
            continue
        res = analyze_run(r, vdir, out_dir, pose_m, seg_m)
        if res:
            rows[r["run"]], crops[r["run"]] = res
    if len(crops) > 1:
        sync_grid(crops, out_dir / f"{feat}_crop_sync.mp4")
    plot(rows, out_dir / f"{feat}_metrics.png", feat)
    print("done:", out_dir)


if __name__ == "__main__":
    main()
