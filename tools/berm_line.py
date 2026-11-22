"""좌회전 뱅크(berm) 벽면 인식 + 라이더 라인(벽 위 어디를 타는가) 측정.

아이디어: 코치 폰은 거의 같은 자리에서 '회전(팬)'만 하므로 프레임 간 변환 ≈ 호모그래피.
  1) track : 라이더 추적(YOLO pose+ByteTrack) → SAM2 비디오로 라이더+자전거 마스크 → 바퀴 최저점 = 접지점.
             전체 프레임을 구간 중간 프레임에 정합(SIFT+RANSAC, 라이더 영역 제외)하고
             라이더를 뺀 중앙값 합성 = '배경판(plate)'. 뱅크는 정지 물체라 판 위에서 한 번만 분할하면 된다.
  2) seg   : 판 위에서 SAM2(이미지)로 뱅크 벽면 + 앞쪽 둔덕(가림막) 분할. 프롬프트는 sessions/<세션>/berm_prompts.json(판 좌표).
             같은 자리에서 찍은 런(1–5, 7)은 판끼리 다시 정합해 run1 판 = '공통 좌표계'로 옮긴다.
             프레임별 측정값 CSV.
  3) report: 런별 몽타주, 장면 비교 보드, 진행률 그래프, 공통 판 위 궤적 지도.

사용법:
    uv run python tools/berm_line.py sessions/2026-11-21/segments.json --stage track [--runs 1]
    uv run python tools/berm_line.py sessions/2026-11-21/segments.json --stage seg
    uv run python tools/berm_line.py sessions/2026-11-21/segments.json --stage report

측정값 (측정 좌표 = 같은 자리 런은 런1 배경판 공통 좌표, 런6·코치는 자기 판; 접지점 열 x에서):
    contact    : SAM2 라이더+자전거 마스크 최저점(두 바퀴 높이가 비슷하면 중점). 실제 타이어-지면 접점의 근사
    lip_y      : 뱅크 마스크의 위쪽 경계(립) — 열별 최상단 픽셀, 윗포락선 후 평활화
    bot_y      : 보이는 아래쪽 경계. 앞 둔덕/능선이 바로 아래면 occl_bot=1 (= 진짜 벽 바닥이 아니라 '보이는 하단선')
    depth_px   : 접지점이 립보다 몇 px 아래인가 (공통 좌표에서 런 1–5·7끼리 직접 비교 가능)
    rel_h      : (bot_y - y) / (bot_y - lip_y), 0=보이는 하단선 1=립
    occl_wheel : 접지점 바로 아래가 가림막(앞 둔덕·능선) 마스크 = 최저점이 '보이는 하단선'에 붙음.
                 확인된 사례(런1 f185, 런4 f222, 코치 전반): 바퀴 아래쪽이 능선 뒤로 가려져 실제 접지는 그 아래.
                 → 이 프레임의 rel_h는 상한, depth_px는 하한. (라이더가 그 앞 둔덕 위를 달리는 경우와 2D로는 구분 불가)
    gx, gy     : 측정 좌표의 접지점. 공통 좌표 x로 '뱅크를 따라 어디까지(나무 쪽으로) 들어갔나'를 비교 — 가려짐 영향 적음
시점 의존: 모두 한 대의 폰에서 본 2D 값. 벽을 정면에 가깝게 볼 때(옆모습 구간)만 '벽 위 높이'에 가깝고,
뒷모습 구간에선 같은 열에 깊이가 다른 면이 겹쳐 '높이'가 아니라 '하단선 대비 위/아래' 정도만 의미가 있다.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402
import analyze_lean as AL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
PLATE_N = 25          # 중앙값 합성에 쓸 최대 프레임 수
MARGIN = 0.9          # 판 캔버스 최대 크기 = 기준 프레임의 (1+2*MARGIN)배 폭 제한


# ---------------------------------------------------------------- 공통 유틸
def out_dir(seg_path):
    cfg = json.loads(Path(seg_path).read_text(encoding="utf-8"))
    return cfg, PATHS.feat_dir(seg_path, cfg, "berm")


def berm_cfg(out, name):
    """수동 입력 파일(프롬프트·캡션)은 git이 추적하는 sessions/<세션>/berm_<name>에 둔다. 없으면 산출물 폴더."""
    p = PATHS.ROOT / "sessions" / out.parent.name / f"berm_{name}"
    return p if p.exists() else out / name


def sift_match(a, b, mask_a=None, mask_b=None, ratio=0.75):
    """a의 점 → b의 점 호모그래피 (a→b). 반환 (H, 인라이어 수)."""
    sift = cv2.SIFT_create(nfeatures=6000)
    ka, da = sift.detectAndCompute(a, mask_a)
    kb, db = sift.detectAndCompute(b, mask_b)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
    if len(good) < 20:
        return None, 0
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None:
        return None, 0
    return H, int(inl.sum())


def warp_pt(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


# ---------------------------------------------------------------- stage: track
def rider_box_full(img, box, seg_m):
    """사람 박스 + 겹치는 자전거 박스(YOLO-seg) 합집합. 자전거가 없으면 아래로 늘린 박스."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    sr = seg_m.predict(img, imgsz=1280, classes=[1], conf=0.15, verbose=False)[0]
    best, bov = None, 0
    for b in sr.boxes.xyxy.tolist():
        ov = max(0, min(x1, b[2]) - max(x0, b[0])) * max(0, min(y1 + 0.3 * h, b[3]) - max(y0, b[1]))
        if ov > bov:
            bov, best = ov, b
    if best is not None:
        return [min(x0, best[0]), min(y0, best[1]), max(x1, best[2]), max(y1, best[3])], True
    return [x0 - 0.15 * h, y0, x1 + 0.15 * h, y1 + 0.25 * h], False


def sam_rider(frames_bgr, box, pos_pt, tmp):
    """SAM2 비디오로 첫 프레임 박스(+몸통 양성점) 프롬프트 → 프레임별 라이더+자전거 마스크."""
    import torch
    from ultralytics.models.sam import SAM2VideoPredictor
    tmp.mkdir(parents=True, exist_ok=True)
    for j, im in enumerate(frames_bgr):
        cv2.imwrite(str(tmp / f"{j:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    vid = tmp / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(tmp / "%04d.jpg"),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(vid)], check=True)
    pred = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                                             model=str(MODELS / "sam2.1_b.pt"), save=False, verbose=False))
    H, W = frames_bgr[0].shape[:2]
    masks = []
    for res in pred(source=str(vid), bboxes=[box], points=[[pos_pt]], labels=[[1]], stream=True):
        if res.masks is None or len(res.masks.data) == 0:
            masks.append(np.zeros((H, W), bool))
            continue
        m = res.masks.data[0].cpu().numpy().astype(np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(m.astype(bool))
    del pred
    torch.cuda.empty_cache()
    shutil.rmtree(tmp, ignore_errors=True)
    return masks


def contact_from_mask(m):
    """마스크 최대 연결요소에서 바퀴 접지점 근사.
    하단 8% 띠를 x 간격으로 묶어 바퀴별 최저점을 구하고, 두 바퀴 높이가 비슷하면(둘 다 접지) 중점,
    아니면 더 낮은 바퀴. (옆모습에서 앞/뒤 바퀴 사이를 오가며 튀는 것을 막음)"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    if len(ys) < 200:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    sel = ys >= y1 - max(4, 0.08 * h)
    bx, by = xs[sel], ys[sel]
    order = np.argsort(bx)
    bx, by = bx[order], by[order]
    gaps = np.diff(bx)
    groups = [(bx, by)]
    if len(gaps) and gaps.max() > 0.12 * w:
        g = int(np.argmax(gaps)) + 1
        groups = [(bx[:g], by[:g]), (bx[g:], by[g:])]
    wheels = []
    for gx, gy in groups:
        yb = gy.max()
        wheels.append((float(np.median(gx[gy >= yb - 3])), float(yb)))
    if len(wheels) == 2 and abs(wheels[0][1] - wheels[1][1]) < 0.06 * h:
        cx, cy = (wheels[0][0] + wheels[1][0]) / 2, (wheels[0][1] + wheels[1][1]) / 2
    else:
        cx, cy = max(wheels, key=lambda t: t[1])
    return cx, cy, (int(x0), int(y0), int(x1), int(y1))


def recontact(out, run):
    """저장된 라이더 마스크에서 접지점을 다시 계산해 track CSV 갱신 (SAM 재실행 없이)."""
    c = np.load(out / "_cache" / f"run{run}.npz")
    H = {int(f): h for f, h in zip(c["frames"], c["H"])}
    p = out / "_cache" / f"track_run{run}.csv"
    rows = read_rows(p)
    for x in rows:
        fi = int(x["frame"])
        m = cv2.imread(str(out / "_cache" / f"run{run}" / f"f{fi:04d}.png"), 0)
        r = contact_from_mask(m > 0) if m is not None else None
        for k in ("cx", "cy", "mx0", "my0", "mx1", "my1", "px", "py"):
            x[k] = ""
        if r is not None:
            cx, cy, bb = r
            px, py = warp_pt(H[fi], cx, cy)
            x.update(cx=round(cx, 1), cy=round(cy, 1), mx0=bb[0], my0=bb[1], mx1=bb[2], my1=bb[3],
                     px=round(px, 1), py=round(py, 1))
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def stage_track(cfg, out, runs):
    from ultralytics import YOLO
    pose_m = YOLO(str(MODELS / "yolo11x-pose.pt"))
    seg_m = YOLO(str(MODELS / "yolo11x-seg.pt"))
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for r in cfg["runs"]:
        if runs and r["run"] not in runs:
            continue
        run = r["run"]
        cap = cv2.VideoCapture(str(PATHS.video_dir(cfg) / r["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, rider = AL.track_rider(cap, fps, r["start"], r["end"], pose_m)
        cap.release()
        seg = [(fi, img, b) for fi, img, b in frames if r["start"] * fps <= fi <= r["end"] * fps]
        # 첫 프레임: 라이더 박스가 있는 첫 구간 프레임
        k0 = next(i for i, (_, _, b) in enumerate(seg) if rider in b)
        seg = seg[k0:]
        fi0, img0, b0 = seg[0]
        box, had_bike = rider_box_full(img0, b0[rider], seg_m)
        pb = b0[rider]
        pos = [(pb[0] + pb[2]) / 2, pb[1] + 0.45 * (pb[3] - pb[1])]
        masks = sam_rider([x[1] for x in seg], box, pos, cache / f"_tmp{run}")

        # 호모그래피: 모든 프레임(앞뒤 패딩 포함) → 기준(구간 중간) 프레임
        ref_i = len(frames) // 2
        mdict = {fi: m for (fi, _, _), m in zip(seg, masks)}

        def feat_mask(fi, boxes):
            g = np.full(frames[0][1].shape[:2], 255, np.uint8)
            if fi in mdict:
                g[cv2.dilate(mdict[fi].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0] = 0
            elif rider in boxes:
                x0, y0, x1, y1 = map(int, boxes[rider])
                h = y1 - y0
                g[max(0, y0 - h // 4):y1 + h // 2, max(0, x0 - h // 3):x1 + h // 3] = 0
            return cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        grays = [cv2.cvtColor(cv2.resize(im, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY) for _, im, _ in frames]
        fm = [feat_mask(fi, b) for fi, _, b in frames]
        S = np.diag([0.5, 0.5, 1.0])
        Hs = [None] * len(frames)
        Hs[ref_i] = np.eye(3)
        ninl = [0] * len(frames)
        for step in (1, -1):
            i = ref_i + step
            while 0 <= i < len(frames):
                H, n = sift_match(grays[i], grays[ref_i], fm[i], fm[ref_i])
                if H is None or n < 60:   # 직접 정합 실패 → 이웃 프레임 경유(연쇄)
                    Hn, n2 = sift_match(grays[i], grays[i - step], fm[i], fm[i - step])
                    if Hn is not None:
                        Hfull = Hs[i - step] @ (np.linalg.inv(S) @ Hn @ S)
                        Hs[i], ninl[i] = Hfull, -n2
                    else:
                        Hs[i], ninl[i] = Hs[i - step], 0
                else:
                    Hs[i], ninl[i] = np.linalg.inv(S) @ H @ S, n
                i += step
        # 캔버스: 모든 프레임 모서리를 덮는 범위 (너무 크면 제한)
        Hh, Ww = frames[0][1].shape[:2]
        corners = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]]).reshape(-1, 1, 2)
        allc = np.concatenate([cv2.perspectiveTransform(corners, H) for H in Hs])
        x_min, y_min = np.maximum(allc.min((0, 1)), [-MARGIN * Ww, -0.3 * Hh])
        x_max, y_max = np.minimum(allc.max((0, 1)), [(1 + MARGIN) * Ww, 1.3 * Hh])
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], float)
        cw, ch = int(x_max - x_min), int(y_max - y_min)
        Hp = [T @ H for H in Hs]  # 프레임 → 판
        # 중앙값 합성 (라이더 제외)
        idx = np.linspace(0, len(frames) - 1, min(PLATE_N, len(frames))).astype(int)
        stack = np.full((len(idx), ch, cw, 3), np.nan, np.float32)
        for j, i in enumerate(idx):
            fi, im, b = frames[i]
            valid = (fm[i] > 0).astype(np.uint8)
            valid = cv2.resize(valid, (Ww, Hh), interpolation=cv2.INTER_NEAREST)
            w = cv2.warpPerspective(im, Hp[i], (cw, ch))
            v = cv2.warpPerspective(valid, Hp[i], (cw, ch), flags=cv2.INTER_NEAREST)
            stack[j][v > 0] = w[v > 0]
        plate = np.nanmedian(stack, 0)
        del stack
        cnt = np.isfinite(plate[..., 0])
        plate = np.nan_to_num(plate).astype(np.uint8)
        cv2.imwrite(str(out / f"plate_run{run}.jpg"), plate, [cv2.IMWRITE_JPEG_QUALITY, 93])

        # 접지점 (프레임 좌표 + 판 좌표)
        rows = []
        mdir = cache / f"run{run}"
        mdir.mkdir(exist_ok=True)
        fidx = {fi: i for i, (fi, _, _) in enumerate(frames)}
        for (fi, im, b), m in zip(seg, masks):
            t = fi / fps
            row = dict(run=run, frame=fi, t=round(t, 3), progress=round((t - r["start"]) / (r["end"] - r["start"]), 4))
            cv2.imwrite(str(mdir / f"f{fi:04d}.png"), m.astype(np.uint8) * 255)
            c = contact_from_mask(m)
            i = fidx[fi]
            row["h_inliers"] = ninl[i]
            if c is not None:
                cx, cy, bb = c
                row.update(cx=round(cx, 1), cy=round(cy, 1), mx0=bb[0], my0=bb[1], mx1=bb[2], my1=bb[3])
                px, py = warp_pt(Hp[i], cx, cy)
                row.update(px=round(px, 1), py=round(py, 1))
            rows.append(row)
        np.savez(cache / f"run{run}.npz", frames=np.array([fi for fi, _, _ in frames]),
                 H=np.stack(Hp), canvas=np.array([cw, ch]), ninl=np.array(ninl), cover=cnt)
        keys = ["run", "frame", "t", "progress", "cx", "cy", "mx0", "my0", "mx1", "my1", "px", "py", "h_inliers"]
        with open(cache / f"track_run{run}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        bad = sum(1 for n in ninl if n <= 0)
        print(f"  run {run}: {len(seg)} seg frames, bike box={had_bike}, contacts {sum('cx' in x for x in rows)}, "
              f"H weak/chained {bad}/{len(frames)}, canvas {cw}x{ch}")


# ---------------------------------------------------------------- stage: seg
def sam_image(sam, img, box, pos, neg):
    pts = list(pos) + list(neg)
    lab = [1] * len(pos) + [0] * len(neg)
    kw = dict(points=[pts], labels=[lab], verbose=False)
    if box is not None:
        kw["bboxes"] = [box]
    res = sam(img, **kw)[0]
    m = res.masks.data[0].cpu().numpy().astype(np.uint8)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def keep_main(m, seeds):
    """양성점이 닿은 연결요소만 남김 (떨어진 조각 제거)."""
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    keep = {lab[int(y), int(x)] for x, y in seeds if 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1]}
    keep.discard(0)
    return np.isin(lab, list(keep)) if keep else m


def fill_holes(m):
    """그림자 줄무늬로 생긴 내부 구멍 메움 (바깥과 연결 안 된 배경 = 구멍)."""
    m8 = m.astype(np.uint8)
    pad = np.pad(m8, 1)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = (ff == 0)[1:-1, 1:-1]
    return m | holes


def edges(berm, occ, x_rng):
    """열별 립(위 경계)·바닥(아래 경계)·바닥 가림 여부. 립은 이동 중앙값으로 평활화."""
    W = berm.shape[1]
    lip = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    occl = np.zeros(W, bool)
    for x in range(max(0, x_rng[0]), min(W, x_rng[1])):
        ys = np.nonzero(berm[:, x])[0]
        if len(ys) < 5:
            continue
        lip[x], bot[x] = ys.min(), ys.max()
        b = int(ys.max())
        occl[x] = occ[b + 1:b + 12, x].any()
    ok = np.isfinite(lip)
    if ok.sum() < 10:
        return lip, bot, occl
    xs = np.arange(W)
    for arr, hw in ((lip, 15), (bot, 45)):   # 바닥 경계는 더 모호 → 넓게 평활화
        f = np.interp(xs, xs[ok], arr[ok])
        if arr is lip:   # 나무 줄기·그림자로 생긴 립의 좁은 홈을 메움 (±30px 윗포락선)
            f = np.array([f[max(0, i - 30):i + 31].min() for i in range(W)])
        sm = np.array([np.median(f[max(0, i - hw):i + hw + 1]) for i in range(W)])
        arr[:] = np.where((xs >= xs[ok].min()) & (xs <= xs[ok].max()), sm, np.nan)
    return lip, bot, occl


def plate_register(src_plate, dst_plate):
    a = cv2.cvtColor(cv2.resize(src_plate, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(dst_plate, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    ma = (a > 0).astype(np.uint8) * 255
    mb = (b > 0).astype(np.uint8) * 255
    H, n = sift_match(a, b, ma, mb)
    S = np.diag([0.5, 0.5, 1.0])
    return (np.linalg.inv(S) @ H @ S if H is not None else None), n


def read_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def lean_view(seg_path, run):
    p = PATHS.out_base(seg_path) / "left_bank_lean" / f"run{run}.csv"
    if not p.exists():
        return {}
    return {int(r["frame"]): float(r["view_ratio"]) for r in read_rows(p) if r.get("view_ratio")}


def stage_seg(seg_path, cfg, out, runs):
    from ultralytics import SAM
    P = json.loads(berm_cfg(out, "prompts.json").read_text(encoding="utf-8"))
    sam = SAM(str(MODELS / "sam2.1_b.pt"))
    common = P.get("common_run")
    cplate = cv2.imread(str(out / f"plate_run{common}.jpg")) if common is not None else None
    reg = {}
    for r in cfg["runs"]:
        run = r["run"]
        if runs and run not in runs:
            continue
        plate = cv2.imread(str(out / f"plate_run{run}.jpg"))
        key = str(run) if str(run) in P["runs"] else None
        if key is not None:          # 이 런 자체 프롬프트
            pr, Hc = P["runs"][key], None
            if common is not None and run != common and pr.get("in_common", False):
                Hc, n = plate_register(plate, cplate)
        else:                        # 공통 판 프롬프트를 이 판으로 옮김
            Hc, n = plate_register(plate, cplate)
            print(f"  run {run}: plate→common inliers {n}")
            pr = P["runs"][str(common)]
        Hinv = np.linalg.inv(Hc) if Hc is not None and key is None else None

        def tp(pts):
            return [list(warp_pt(Hinv, *p)) for p in pts] if Hinv is not None else pts

        def tb(b):
            if Hinv is None or b is None:
                return b
            c = np.array([warp_pt(Hinv, b[0], b[1]), warp_pt(Hinv, b[2], b[3])])
            return [c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]

        bp, bn = tp(pr["berm_pos"]), tp(pr["berm_neg"])
        berm = keep_main(sam_image(sam, plate, tb(pr.get("berm_box")), bp, bn), bp)
        op, on = tp(pr["occ_pos"]), tp(pr.get("occ_neg", []))
        occ = keep_main(sam_image(sam, plate, tb(pr.get("occ_box")), op, on), op) if op else np.zeros_like(berm)
        if pr.get("occ_pos2"):   # 두 번째 가림막(예: 코치 영상의 앞 능선) — 합집합
            op2 = tp(pr["occ_pos2"])
            occ |= keep_main(sam_image(sam, plate, None, op2, tp(pr.get("occ_neg2", []))), op2)
            op = op + op2
        berm &= ~occ
        berm = fill_holes(keep_main(berm, bp)) & ~occ
        cv2.imwrite(str(out / "_cache" / f"berm_run{run}.png"), berm.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "_cache" / f"occ_run{run}.png"), occ.astype(np.uint8) * 255)
        xr = [int(v) for v in tb([pr["x_range"][0], 0, pr["x_range"][1], 1])[::2]] if "x_range" in pr else [0, berm.shape[1]]
        np.savez(out / "_cache" / f"segmeta_run{run}.npz", Hc=Hc if Hc is not None else np.eye(3),
                 has_common=(Hc is not None) or (run == common), xr=np.array(xr))
        vis = plate.copy()
        vis[berm] = (vis[berm] * 0.5 + np.array([255, 120, 0]) * 0.5).astype(np.uint8)
        vis[occ] = (vis[occ] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)
        for p in bp:
            cv2.circle(vis, tuple(map(int, p)), 9, (0, 255, 0), 3)
        for p in bn:
            cv2.circle(vis, tuple(map(int, p)), 9, (0, 0, 255), 3)
        for p in op:
            cv2.circle(vis, tuple(map(int, p)), 9, (255, 0, 255), 3)
        cv2.imwrite(str(out / f"plate_seg_run{run}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  run {run}: berm px {berm.sum()}, occ px {occ.sum()}")
    stage_measure(seg_path, cfg, out, P)


def stage_measure(seg_path, cfg, out, P):
    """같은 자리 런들은 런별 마스크를 공통 판으로 옮겨 다수결(합의) 마스크 하나로 측정 → 마스크 흔들림이 라인 차이로 둔갑하지 않게."""
    common = P.get("common_run")
    meta = {}
    for r in cfg["runs"]:
        f = out / "_cache" / f"segmeta_run{r['run']}.npz"
        if f.exists():
            meta[r["run"]] = np.load(f)
    cons = None
    if common is not None and common in meta:
        cplate = cv2.imread(str(out / f"plate_run{common}.jpg"))
        ch, cw = cplate.shape[:2]
        acc_b = np.zeros((ch, cw), np.float32)
        acc_o = np.zeros((ch, cw), np.float32)
        n = 0
        for run, m in meta.items():
            if not bool(m["has_common"]):
                continue
            b = cv2.imread(str(out / "_cache" / f"berm_run{run}.png"), 0)
            o = cv2.imread(str(out / "_cache" / f"occ_run{run}.png"), 0)
            acc_b += cv2.warpPerspective(b, m["Hc"], (cw, ch), flags=cv2.INTER_NEAREST) > 0
            acc_o += cv2.warpPerspective(o, m["Hc"], (cw, ch), flags=cv2.INTER_NEAREST) > 0
            n += 1
        cocc = acc_o >= n / 2
        seeds = [tuple(p) for p in P["runs"][str(common)]["berm_pos"]]
        cberm = fill_holes(keep_main((acc_b >= n / 2) & ~cocc, seeds)) & ~cocc
        cv2.imwrite(str(out / "_cache" / "berm_common.png"), cberm.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "_cache" / "occ_common.png"), cocc.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "_cache" / "berm_votes.png"), (acc_b / max(n, 1) * 255).astype(np.uint8))
        cons = (cberm, cocc, n)
        print(f"  consensus mask from {n} runs")
    for run, m in meta.items():
        recontact(out, run)
        if cons is not None and bool(m["has_common"]):
            berm, occ, geom, Hg = cons[0], cons[1], "common", m["Hc"]
            xr = meta[common]["xr"]
        else:
            berm = cv2.imread(str(out / "_cache" / f"berm_run{run}.png"), 0) > 0
            occ = cv2.imread(str(out / "_cache" / f"occ_run{run}.png"), 0) > 0
            geom, Hg, xr = "own", np.eye(3), m["xr"]
        lip, bot, occl = edges(berm, occ, [int(xr[0]), int(xr[1])])
        np.savez(out / "_cache" / f"geom_run{run}.npz", lip=lip, bot=bot, occl=occl, Hg=Hg, geom=geom)
        cv2.imwrite(str(out / "_cache" / f"geom_berm_run{run}.png"), berm.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "_cache" / f"geom_occ_run{run}.png"), occ.astype(np.uint8) * 255)
        view = lean_view(seg_path, run)
        rows = read_rows(out / "_cache" / f"track_run{run}.csv")
        for x in rows:
            vr = view.get(int(x["frame"]))
            x["view_ratio"] = vr if vr is not None else ""
            x["geom"] = geom
            if not x.get("px"):
                continue
            gx, gy = warp_pt(Hg, float(x["px"]), float(x["py"]))
            x["gx"], x["gy"] = round(gx, 1), round(gy, 1)
            c = int(round(gx))
            if not (0 <= c < len(lip)) or not np.isfinite(lip[c]):
                x["note"] = "off_berm_cols"
                continue
            x["lip_y"], x["bot_y"], x["occl_bot"] = round(lip[c], 1), round(bot[c], 1), int(occl[c])
            x["depth_px"] = round(gy - lip[c], 1)
            hh = float(x["my1"]) - float(x["my0"])
            x["depth_rider"] = round((gy - lip[c]) / max(hh, 1), 3)   # 라이더+자전거 마스크 높이 단위(프레임 px 기준, 근사)
            if bot[c] - lip[c] > 10:   # 바닥이 가려진 열이면 occl_bot=1 → '보이는 바닥' 기준 값(상한 성격)
                x["rel_h"] = round((bot[c] - gy) / (bot[c] - lip[c]), 3)
            yy = int(gy)
            x["occl_wheel"] = int(occ[yy + 2:yy + 14, max(0, c - 4):c + 5].any())
        keys = ["run", "frame", "t", "progress", "view_ratio", "geom", "cx", "cy", "px", "py", "gx", "gy", "lip_y", "bot_y",
                "occl_bot", "depth_px", "depth_rider", "rel_h", "occl_wheel", "h_inliers", "note", "mx0", "my0", "mx1", "my1"]
        with open(out / f"run{run}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  run {run}: geom={geom}, measured {sum('depth_px' in x for x in rows)}/{len(rows)}, "
              f"rel_h {sum('rel_h' in x for x in rows)}, wheel occluded {sum(x.get('occl_wheel') == 1 for x in rows)}")


# ---------------------------------------------------------------- stage: report
BLUE = (214, 120, 42)      # BGR — #2a78d6
GRAY = (150, 150, 150)


class RunCtx:
    """한 런의 판·마스크·변환·측정값 묶음."""

    def __init__(self, seg_path, run):
        cfg, out = out_dir(seg_path)
        self.cfg, self.out, self.run = cfg, out, run
        self.r = next(x for x in cfg["runs"] if x["run"] == run)
        self.label = self.r.get("label", f"run{run}")
        c = np.load(out / "_cache" / f"run{run}.npz")
        self.H = {int(f): h for f, h in zip(c["frames"], c["H"])}
        e = np.load(out / "_cache" / f"geom_run{run}.npz")
        self.lip, self.bot, self.occl, Hg = e["lip"], e["bot"], e["occl"], e["Hg"]
        self.geom = str(e["geom"])
        self.H = {f: Hg @ h for f, h in self.H.items()}          # 프레임 → 측정 좌표(공통 판 또는 자기 판)
        self.berm = cv2.imread(str(out / "_cache" / f"geom_berm_run{run}.png"), 0) > 0
        self.occ = cv2.imread(str(out / "_cache" / f"geom_occ_run{run}.png"), 0) > 0
        P = json.loads(berm_cfg(out, "prompts.json").read_text(encoding="utf-8"))
        prun = P.get("common_run") if self.geom == "common" else run
        self.plate = cv2.imread(str(out / f"plate_run{prun}.jpg"))
        self.rows = read_rows(out / f"run{run}.csv")
        self.cap = cv2.VideoCapture(str(PATHS.video_dir(self.cfg) / self.r["file"]))

    def frame(self, fi):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        return self.cap.read()[1]

    def f(self, row, k):
        v = row.get(k, "")
        return float(v) if v not in ("", None) else np.nan

    def at(self, p):
        return min(self.rows, key=lambda x: abs(float(x["progress"]) - p))

    def apex(self):
        """뱅크 깊숙이 들어간 순간 = 판 위 접지점 x가 최대(진입 방향 끝)인 프레임 (15–85% 구간, 5프레임 평활)."""
        px = np.array([self.f(x, "gx") for x in self.rows])
        pr = np.array([float(x["progress"]) for x in self.rows])
        sm = np.array([np.nanmedian(px[max(0, i - 2):i + 3]) for i in range(len(px))])
        sm[(pr < 0.15) | (pr > 0.85)] = -1e9
        return float(pr[int(np.nanargmax(sm))])

    def polyline(self, arr, Hinv, step=6):
        xs = np.arange(len(arr))[np.isfinite(arr)][::step]
        if len(xs) < 2:
            return None, xs
        pts = np.float32([[x, arr[x]] for x in xs]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2), xs

    def cell(self, row, w=420, h=300, text=True):
        fi = int(row["frame"])
        img = self.frame(fi)
        Hh, Ww = img.shape[:2]
        Hinv = np.linalg.inv(self.H[fi])
        bm = cv2.warpPerspective(self.berm.astype(np.uint8), Hinv, (Ww, Hh), flags=cv2.INTER_NEAREST) > 0
        om = cv2.warpPerspective(self.occ.astype(np.uint8), Hinv, (Ww, Hh), flags=cv2.INTER_NEAREST) > 0
        rm = cv2.imread(str(self.out / "_cache" / f"run{self.run}" / f"f{fi:04d}.png"), 0)
        if rm is not None:
            bm &= rm == 0
        vis = img.copy()
        vis[bm] = (vis[bm] * 0.55 + np.array([255, 140, 30]) * 0.45).astype(np.uint8)
        vis[om] = (vis[om] * 0.7 + np.array([60, 60, 200]) * 0.3).astype(np.uint8)
        lp, _ = self.polyline(self.lip, Hinv)
        if lp is not None:
            cv2.polylines(vis, [lp.astype(np.int32)], False, (0, 240, 255), 3, cv2.LINE_AA)
        bp, xs = self.polyline(self.bot, Hinv)
        if bp is not None:
            for k in range(len(bp) - 1):
                oc = self.occl[xs[k]]
                if oc and k % 2:
                    continue   # 가려진 바닥 = 점선(빨강)
                cv2.line(vis, tuple(bp[k].astype(int)), tuple(bp[k + 1].astype(int)),
                         (60, 60, 230) if oc else (80, 220, 80), 2, cv2.LINE_AA)
        cx, cy = self.f(row, "cx"), self.f(row, "cy")
        hh = self.f(row, "my1") - self.f(row, "my0")
        if not np.isfinite(cx):
            cx, cy, hh = Ww / 2, Hh / 2, 300
        hollow = row.get("occl_wheel") == "1"
        cv2.circle(vis, (int(cx), int(cy)), 11, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(vis, (int(cx), int(cy)), 9, (255, 0, 255), 2 if hollow else -1, cv2.LINE_AA)
        # 수직 기준: 접지점 열에서 립까지
        if row.get("lip_y"):
            lx, ly = warp_pt(Hinv, float(row["gx"]), float(row["lip_y"]))
            cv2.line(vis, (int(cx), int(cy)), (int(lx), int(ly)), (255, 0, 255), 1, cv2.LINE_AA)
        # 크롭
        cw = max(3.0 * hh, 520)
        ch = cw * h / w
        x0 = int(np.clip(cx - cw / 2, 0, Ww - cw))
        y0 = int(np.clip(cy - 0.62 * ch, 0, Hh - ch))
        crop = cv2.resize(vis[y0:y0 + int(ch), x0:x0 + int(cw)], (w, h), interpolation=cv2.INTER_AREA)
        if text and self.label == "coach":
            s = f"p={float(row['progress']):.2f} (높이 측정 불가: 바퀴 가려짐)" if hollow else f"p={float(row['progress']):.2f}"
            cv2.rectangle(crop, (0, 0), (w, 24), (0, 0, 0), -1)
            crop = put_kr(crop, [s], (6, 2), 15)
        elif text:
            rh = row.get("rel_h") or "-"
            dr = row.get("depth_rider") or "-"
            s = f"p={float(row['progress']):.2f} rel={rh} d={dr}" + (" pin" if hollow else "")
            cv2.rectangle(crop, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(crop, s, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return crop

    def plate_view(self, x_pad=120, y_pad=90, width=None, others=()):
        """판 위 뱅크 영역 + 마스크 + 립/바닥 + 궤적(진행률 색)."""
        vis = self.plate.copy()
        b = self.berm
        vis[b] = (vis[b] * 0.6 + np.array([255, 140, 30]) * 0.4).astype(np.uint8)
        for x in range(len(self.lip)):
            if np.isfinite(self.lip[x]):
                cv2.circle(vis, (x, int(self.lip[x])), 1, (0, 240, 255), -1)
                if x % 6 < 3 or not self.occl[x]:
                    cv2.circle(vis, (x, int(self.bot[x])), 1, (60, 60, 230) if self.occl[x] else (80, 220, 80), -1)
        pts = [(self.f(x, "gx"), self.f(x, "gy"), float(x["progress"]), x.get("occl_wheel") == "1") for x in self.rows]
        pts = [p for p in pts if np.isfinite(p[0])]
        for sg in segments_of(pts):
            for (x1, y1, _, _), (x2, y2, p, _) in zip(sg[:-1], sg[1:]):
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), seq_color(p), 2, cv2.LINE_AA)
        for x, y, p, oc in pts:
            cv2.circle(vis, (int(x), int(y)), 5, seq_color(p), 1 if oc else -1, cv2.LINE_AA)
        ys, xs = np.nonzero(b)
        allx = np.r_[xs, [p[0] for p in pts]]
        ally = np.r_[ys, [p[1] for p in pts]]
        X0, X1 = int(max(0, allx.min() - x_pad)), int(min(vis.shape[1], allx.max() + x_pad))
        Y0, Y1 = int(max(0, ally.min() - y_pad)), int(min(vis.shape[0], ally.max() + y_pad))
        crop = vis[Y0:Y1, X0:X1]
        if width:
            crop = cv2.resize(crop, (width, int(crop.shape[0] * width / crop.shape[1])), interpolation=cv2.INTER_AREA)
        return crop


def seq_color(p):
    """진행률 색 (단일 계열 밝음→어두움 대신 가독성 위해 노랑→자홍). BGR."""
    a = np.array([0, 220, 255]), np.array([200, 0, 200])
    return tuple(int(v) for v in a[0] * (1 - p) + a[1] * p)


def put_kr(img, lines, xy, size=22, color=(255, 255, 255), bold=False, bg=None):
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    ft = ImageFont.truetype(FONT_B if bold else FONT, size)
    x, y = xy
    for ln in lines:
        if bg is not None:
            bb = d.textbbox((x, y), ln, font=ft)
            d.rectangle((bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2), fill=bg)
        d.text((x, y), ln, font=ft, fill=color[::-1])
        y += int(size * 1.35)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def montage(ctx, n=8):
    cells = [ctx.cell(ctx.at(p)) for p in np.linspace(0, 1, n)]
    grid = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, n, 4)])
    pv = ctx.plate_view(width=grid.shape[1])
    head = np.zeros((70, grid.shape[1], 3), np.uint8)
    head = put_kr(head, [f"{ctx.cfg['session']} {ctx.label} — 좌회전 뱅크 벽면(파랑)·립(노랑)·보이는 하단선(초록; 앞 능선 윗선=빨강 점선)·접지점(자홍; 속빈 원=하단선에 붙음→실제 접지는 그 아래)",
                         "위: 배경판(파노라마) 위 접지점 궤적 (노랑=진입 → 자홍=탈출).  아래: 진행률 0→1 프레임. rel=높이비(0=보이는 하단선, 1=립), d=립 아래 깊이/라이더 마스크 높이(참고), pin=하단선에 붙음(가려짐)"],
                  (10, 8), 18)
    return np.vstack([head, pv, grid])


def stage_report(seg_path, ref_path, out_ref):
    cfg, out = out_dir(seg_path)
    runs = [r["run"] for r in cfg["runs"]]
    ctxs = {r: RunCtx(seg_path, r) for r in runs}
    coach = RunCtx(ref_path, 0) if ref_path else None
    for r, c in list(ctxs.items()) + ([("coach", coach)] if coach else []):
        img = montage(c)
        dst = (out_ref if r == "coach" else out) / f"montage_{c.label}.jpg"
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    summary = summarize(ctxs, coach)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    board(ctxs, coach, summary, out)
    plots(ctxs, coach, summary, out)
    common_map(ctxs, out, cfg)
    print("report:", out)


def anchor_from_file(c):
    """compare_reference.py가 만든 view_ratio 앵커(코치처럼 접지점이 가려져 판 x가 고정된 경우에 사용)."""
    f = c.out.parent / f"{c.cfg.get('feature', 'feature')}_vs_ref_anchors.txt"
    if not f.exists():
        return None
    for ln in f.read_text(encoding="utf-8").splitlines():
        if ln.strip().startswith("coach:") and c.label == "coach":
            return float(ln.split("anchor progress")[1].split()[0])
    return None


def moments(c):
    a = anchor_from_file(c) if c.label == "coach" else None
    a = a if a is not None else c.apex()
    return {"entry": a / 2, "apex": a, "exit": a + (1 - a) / 2, "end": 0.97}


MOMENT_KR = {"entry": "진입", "apex": "정점(가장 깊이 들어간 순간)", "exit": "탈출", "end": "출구"}
X_END = 2760   # 공통 판(런1)에서 뱅크 벽 오른쪽 끝(오렌지 나무 쪽) x


def summarize(ctxs, coach):
    """순간별 중앙값. pinned = 접지점이 '보이는 벽 하단선(앞 능선 윗선)'에 붙은 비율 → 실제 접지는 그 아래(가려짐)."""
    S = {}
    for key, c in list(ctxs.items()) + ([("coach", coach)] if coach else []):
        m = moments(c)
        d = {"moments": {k: round(v, 3) for k, v in m.items()}, "geom": c.geom}
        for name, p in m.items():
            near = [x for x in c.rows if abs(float(x["progress"]) - p) <= 0.06]
            for k in ("rel_h", "depth_px", "gx"):
                v = [c.f(x, k) for x in near]
                v = [x for x in v if np.isfinite(x)]
                d[f"{name}_{k}"] = round(float(np.median(v)), 3) if v else None
            d[f"{name}_pinned"] = round(float(np.mean([x.get("occl_wheel") == "1" for x in near])), 2) if near else None
        if c.geom == "common" and d.get("apex_gx") is not None:
            # 정점 = 5프레임 평활 최대 x 시점, 그 ±6% 창의 중앙값 x (한 프레임 튐 배제)
            d["apex_from_end_px"] = round(X_END - d["apex_gx"], 1)   # 나무 쪽 벽 끝에서 정점까지 (공통 판 px)
        if c.label == "coach":   # 바퀴가 능선에 가려 높이·깊이 값은 의미 없음 → 공개하지 않음
            for k in list(d):
                if k.endswith(("_rel_h", "_depth_px")):
                    d[k] = None
        d["n_frames"] = len(c.rows)
        d["n_pinned"] = sum(1 for x in c.rows if x.get("occl_wheel") == "1")
        S[str(key)] = d
    return S


def board(ctxs, coach, S, out):
    cols = [r for r in ctxs] + (["coach"] if coach else [])
    cw, chh = 300, 214
    capw = 330
    capfile = berm_cfg(out, "board_captions.json")
    caps = json.loads(capfile.read_text(encoding="utf-8")) if capfile.exists() else {}
    rows_img = []
    head = np.full((44, capw + cw * len(cols), 3), 30, np.uint8)
    for j, k in enumerate(cols):
        lab = "코치(여름·낮은 카메라)" if k == "coach" else (f"런 {k}" + (" (다른 자리)" if k == 6 else ""))
        head = put_kr(head, [lab], (capw + j * cw + 10, 8), 20, bold=True)
    rows_img.append(head)
    for name in ("entry", "apex", "exit", "end"):
        cells = []
        for k in cols:
            c = coach if k == "coach" else ctxs[k]
            p = S[str(k)]["moments"][name]
            cells.append(c.cell(c.at(p), cw, chh))
        txt = caps.get(name)
        if txt is None:
            st = [S[str(k)].get(f"{name}_rel_h") for k in ctxs if k != 6]
            st = [x for x in st if x is not None]
            co = S["coach"].get(f"{name}_rel_h") if coach else None
            txt = [MOMENT_KR[name], f"학생 rel_h 중앙 {np.median(st):.2f}" if st else "학생 rel_h -",
                   f"코치 {co:.2f}" if co is not None else "코치 -"]
        cap = np.full((chh, capw, 3), 245, np.uint8)
        cap = put_kr(cap, [txt[0]], (12, 10), 21, color=(20, 20, 20), bold=True)
        cap = put_kr(cap, txt[1:], (12, 44), 17, color=(40, 40, 40))
        rows_img.append(np.hstack([cap] + cells))
    foot = np.full((64, rows_img[0].shape[1], 3), 30, np.uint8)
    foot = put_kr(foot, ["파랑=뱅크 벽면, 노랑=립, 초록/빨강 점선=보이는 벽 하단선(빨강=앞 둔덕·능선 윗선), 자홍 점=타이어 최저점(속빈 원·pin=하단선에 붙음 → 실제 접지는 그 아래로 가려짐).",
                         "rel=접지점 열에서 (하단선-접지)/(하단선-립): 0=보이는 하단선, 1=립. 카메라 위치 의존 2D 값 — 같은 자리 런(1-5·7)끼리만 비교, 런6·코치는 참고."],
                  (10, 8), 16)
    cv2.imwrite(str(out / "berm_line_board.jpg"), np.vstack(rows_img + [foot]), [cv2.IMWRITE_JPEG_QUALITY, 88])


P_MIN = 0.12      # 착지 직후(바퀴가 번갈아 최저점이 되는 공중/착지 프레임) 제외
JUMP = 80         # 이 px 이상 튀면 선을 끊음


def segments_of(pts):
    """[(x,y,...)] → 큰 점프에서 끊은 연속 구간 목록."""
    segs, cur = [], []
    for q in pts:
        if cur and np.hypot(q[0] - cur[-1][0], q[1] - cur[-1][1]) > JUMP:
            segs.append(cur)
            cur = []
        cur.append(q)
    if cur:
        segs.append(cur)
    return segs


def plots(ctxs, coach, S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    keys = list(ctxs)
    comp = [k for k in keys if ctxs[k].geom == "common"]

    def series(c, k):
        rows = [x for x in c.rows if float(x["progress"]) >= P_MIN]
        pr = np.array([float(x["progress"]) for x in rows])
        v = np.array([c.f(x, k) for x in rows])
        oc = np.array([x.get("occl_wheel") == "1" for x in rows])
        return pr, v, oc

    def panel(ax, k, key, others, ylim=None):
        for kk in others:
            if kk != k:
                pr, v, _ = series(ctxs[kk], key)
                ax.plot(pr, v, color="#c9c8c2", lw=1, zorder=1)
        pr, v, oc = series(ctxs[k], key)
        ax.plot(pr, v, color="#2a78d6", lw=2, zorder=3)
        ax.scatter(pr[~oc], v[~oc], s=14, color="#2a78d6", zorder=4)
        ax.scatter(pr[oc], v[oc], s=24, facecolors="white", edgecolors="#2a78d6", zorder=4)
        ax.axvline(S[str(k)]["moments"]["apex"], color="#52514e", lw=0.8, ls=":")
        ax.grid(alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)

    # 1) 높이비 (보이는 하단선 기준)
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True, sharey=True)
    for ax, k in zip(axs.flat, keys + ["coach"]):
        if k == "coach":
            ax.set_title("코치 (참고)", fontsize=12, loc="left")
            ax.text(0.5, 0.5, "높이 측정 불가\n카메라가 낮아 정점 전까지\n바퀴가 앞 능선에 가려짐", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#52514e")
            ax.grid(alpha=0.25)
            continue
        panel(ax, k, "rel_h", comp, ylim=(-0.4, 1.1))
        ax.axhline(0, color="#9a9992", lw=0.8)
        ax.axhline(1, color="#9a9992", lw=0.8)
        ax.set_title(f"런 {k}" + (" (다른 촬영 위치 — 자체 기준, 참고)" if ctxs[k].geom != "common" else ""),
                     fontsize=12, loc="left")
    for ax in axs[1]:
        ax.set_xlabel("구간 진행률 (착지 직후 12%는 제외)")
    for ax in axs[:, 0]:
        ax.set_ylabel("높이비 (0=보이는 벽 하단선, 1=립)")
    fig.suptitle("타이어 접지점의 벽 위 높이비 — 파랑=해당 런, 회색=같은 자리 다른 런, 세로 점선=정점.  "
                 "속빈 점 = 접지점이 '보이는 하단선(앞 능선)'에 붙음 → 실제 접지는 그 아래로 가려짐(값은 상한)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "berm_line_rel_h.png", dpi=110)
    plt.close(fig)

    # 2) 뱅크를 따라 어디까지 들어갔나 (공통 좌표, 나무 쪽 끝에서의 거리)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    for ax, k in zip(axs.flat, comp):
        for kk in comp:
            if kk != k:
                pr, v, _ = series(ctxs[kk], "gx")
                ax.plot(pr, X_END - v, color="#c9c8c2", lw=1)
        pr, v, oc = series(ctxs[k], "gx")
        ax.plot(pr, X_END - v, color="#2a78d6", lw=2)
        ax.scatter(pr, X_END - v, s=10, color="#2a78d6")
        a = S[str(k)]
        ax.axvline(a["moments"]["apex"], color="#52514e", lw=0.8, ls=":")
        ax.set_title(f"런 {k} — 정점: 나무 쪽 끝에서 {a['apex_from_end_px']:.0f}px", loc="left")
        ax.grid(alpha=0.25)
        ax.set_ylim(800, 0)
    for ax in axs[1]:
        ax.set_xlabel("구간 진행률")
    for ax in axs[:, 0]:
        ax.set_ylabel("벽 오른쪽 끝(오렌지 나무)에서 거리 px\n(공통 좌표, 위=나무 쪽 깊숙이)")
    fig.suptitle("뱅크를 따라 얼마나 깊이(나무 쪽으로) 들어갔다 나오나 — 같은 자리 촬영 런 1-5·7 (가려짐과 무관한 측정)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "berm_line_along.png", dpi=110)
    plt.close(fig)

    # 3) 립 아래 깊이 (공통 판 px)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    for ax, k in zip(axs.flat, comp):
        panel(ax, k, "depth_px", comp, ylim=(170, 0))
        ax.set_title(f"런 {k}", loc="left")
    for ax in axs[1]:
        ax.set_xlabel("구간 진행률")
    for ax in axs[:, 0]:
        ax.set_ylabel("립 아래 깊이 (px, 공통 좌표, 위=립 가까움)")
    fig.suptitle("접지점이 립보다 몇 px 아래인가 (런1 배경판 공통 좌표). 속빈 점=하단선에 붙음(실제는 더 깊음). "
                 "정점 부근 보이는 벽면 높이 ≈ 100-140px", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "berm_line_depth_common.png", dpi=110)
    plt.close(fig)


def common_map(ctxs, out, cfg):
    """공통 판 위에 런별 궤적(작은 배수). 같은 자리 런만. 착지 직후 제외, 큰 점프에서 선 끊음."""
    comp = [k for k in ctxs if ctxs[k].geom == "common"]
    base = ctxs[comp[0]]
    pts_all = {}
    for k in comp:
        c = ctxs[k]
        q = [(c.f(x, "gx"), c.f(x, "gy"), float(x["progress"]), x.get("occl_wheel") == "1") for x in c.rows
             if float(x["progress"]) >= P_MIN]
        pts_all[k] = [p for p in q if np.isfinite(p[0])]
    ys, xs = np.nonzero(base.berm)
    X0, X1 = xs.min() - 80, xs.max() + 60
    Y0, Y1 = ys.min() - 60, ys.max() + 90
    tiles = []
    for k in comp:
        vis = base.plate.copy()
        b = base.berm
        vis[b] = (vis[b] * 0.65 + np.array([255, 140, 30]) * 0.35).astype(np.uint8)
        for x in range(len(base.lip)):
            if np.isfinite(base.lip[x]):
                cv2.circle(vis, (x, int(base.lip[x])), 1, (0, 240, 255), -1)
                if x % 6 < 3:
                    cv2.circle(vis, (x, int(base.bot[x])), 1, (60, 60, 230), -1)
        for kk in comp:
            if kk == k:
                continue
            for sg in segments_of(pts_all[kk]):
                for a, bb in zip(sg[:-1], sg[1:]):
                    cv2.line(vis, (int(a[0]), int(a[1])), (int(bb[0]), int(bb[1])), (215, 215, 215), 1, cv2.LINE_AA)
        for sg in segments_of(pts_all[k]):
            for a, bb in zip(sg[:-1], sg[1:]):
                cv2.line(vis, (int(a[0]), int(a[1])), (int(bb[0]), int(bb[1])), seq_color(bb[2]), 3, cv2.LINE_AA)
        for x, y, pp, oc in pts_all[k]:
            cv2.circle(vis, (int(x), int(y)), 5, seq_color(pp), 1 if oc else -1, cv2.LINE_AA)
        a = max(pts_all[k], key=lambda t: t[0] if 0.15 < t[2] < 0.85 else -1)
        cv2.drawMarker(vis, (int(a[0]), int(a[1])), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
        t = vis[Y0:Y1, X0:X1]
        t = cv2.resize(t, (900, int(t.shape[0] * 900 / t.shape[1])), interpolation=cv2.INTER_AREA)
        t = put_kr(t, [f"런 {k}"], (10, 8), 24, bold=True, bg=(0, 0, 0))
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    head = np.full((70, grid.shape[1], 3), 30, np.uint8)
    head = put_kr(head, ["같은 자리 촬영 런 1-5·7 타이어 접지점 궤적 (런1 배경판 공통 좌표). 굵은 선=해당 런(노랑 진입→자홍 탈출), 회색=다른 런, 흰 X=정점(가장 깊이)",
                         "파랑=뱅크 벽면, 노랑=립, 빨강 점선=보이는 하단선(앞 능선 윗선). 속빈 원=하단선에 붙음 → 실제 접지는 그 아래(가려짐). 착지 직후 12% 제외."],
                  (10, 8), 18)
    cv2.imwrite(str(out / "berm_line_map.jpg"), np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--stage", required=True, choices=["track", "seg", "report"])
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--ref", help="report: 코치 segments.json")
    a = ap.parse_args()
    cfg, out = out_dir(a.segments)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "track":
        stage_track(cfg, out, a.runs)
    elif a.stage == "seg":
        stage_seg(a.segments, cfg, out, a.runs)
    elif a.stage == "report":
        ref_out = None
        if a.ref:
            _, ref_out = out_dir(a.ref)
        stage_report(a.segments, a.ref, ref_out)


if __name__ == "__main__":
    main()
