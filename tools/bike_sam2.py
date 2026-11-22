"""SAM2로 크롭 영상 내내 자전거를 추적 분할하고, 뒷모습 구간의 자전거 축 기울기를 계산한다.

analyze_lean.py 다음에 실행:
    uv run python tools/bike_sam2.py sessions/2026-11-21/segments.json [--runs 1]
    PYTHONPATH=tools uv run python tools/bike_sam2.py sessions/reference-coach/segments.json --runs 0 --kp-prompt         --montage sessions/reference-coach/left_bank_lean/coach_masks_montage.jpg

기본 프롬프트: YOLO-seg가 자전거를 잡은 가장 이른 크롭 프레임의 박스 → 그 프레임부터 끝까지 추적.
키포인트 프롬프트(--kp-prompt, 또는 YOLO-seg가 한 프레임도 못 잡았을 때 자동):
  포즈 키포인트로 박스 + 양성 점(발목 사이 아래 = 타이어/BB, 손목 안쪽 = 핸들바)
  + 음성 점(머리·어깨·허리·허벅지·무릎·종아리·팔꿈치·타이어 양옆 바닥)을 만든다.
  포즈가 좋은 앵커 프레임 여러 개(구간 3등분마다 1개, --kp-anchors로 지정 가능)에서
  각각 앞·뒤로 추적하고, 프레임마다 앵커들끼리 가장 많이 겹치는 마스크(medoid)를 고른다
  (포즈가 믿을 만하면 몸통·머리·접지선 아래 침범을 감점). 작은 조각·납작한 바닥 줄무늬는 지운다.
결과: run<N>.csv 에 bike_lean_sam, separation_sam 열 추가, masks/ 와 overlay_sam/ 에 저장.
--deshadow: 있는 마스크에서 땅에 진 바퀴 그림자를 뺀다(아래 '그림자 제거' 주석). SAM2를 다시 돌렸으면 그 뒤에 다시 실행.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import SAM2VideoPredictor

import analyze_lean as AL


def first_bike_prompt(rows, crops_dir, run, seg_m, pose_m):
    for i, x in enumerate(rows):
        img = cv2.imread(str(crops_dir / f"run{run}_f{int(x['frame']):04d}.jpg"))
        sr = seg_m.predict(img, imgsz=AL.CROP, classes=[1], conf=0.25, verbose=False)[0]
        if len(sr.boxes) == 0:
            continue
        c = np.array([AL.CROP / 2, AL.CROP / 2])
        bxs = sr.boxes.xyxy.cpu().numpy()
        b = bxs[np.argmin([np.linalg.norm((bb[:2] + bb[2:]) / 2 - c) for bb in bxs])]
        pr = pose_m.predict(img, imgsz=AL.CROP, classes=[0], verbose=False)[0]
        pi = AL.pick_center_person(pr)
        neg = None
        if pi is not None:
            k = pr.keypoints.xy[pi].cpu().numpy()
            neg = ((k[5] + k[6]) / 2 * 0.5 + (k[11] + k[12]) / 2 * 0.5).tolist()  # 상체 중앙
        return i, b.tolist(), neg
    return None


# ---------------------------------------------------------------- 공통: 크롭 → 영상 → SAM2 추적

def sam_track(imgs, tmp_base, sam_path, **prompt):
    """imgs(크롭 리스트) 첫 장에 prompt를 주고 끝까지 추적. 프레임별 768px bool 마스크(없으면 None)."""
    tmpdir = tmp_base.parent / (tmp_base.name + "_frames")
    tmpdir.mkdir(exist_ok=True)
    for j, img in enumerate(imgs):
        cv2.imwrite(str(tmpdir / f"{j:04d}.jpg"), img)
    vid = tmp_base.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(tmpdir / "%04d.jpg"),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(vid)], check=True)
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()
    pred = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                                             model=str(sam_path), save=False, verbose=False))
    masks = [None] * len(imgs)
    for j, res in enumerate(pred(source=str(vid), stream=True, **prompt)):
        if res.masks is None or len(res.masks.data) == 0:
            continue
        m = res.masks.data[0].cpu().numpy().astype(np.uint8)
        m = cv2.resize(m, (AL.CROP, AL.CROP), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks[j] = m
    vid.unlink()
    return masks


def save_mask(out, run, x, m):
    """마스크 PNG, overlay_sam, CSV 값(x에 기록). 축을 얻었으면 True."""
    fr = int(x["frame"])
    (out / "masks").mkdir(exist_ok=True)
    (out / "overlay_sam").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "masks" / f"run{run}_f{fr:04d}.png"), m.astype(np.uint8) * 255)
    img = cv2.imread(str(out / "overlay" / f"run{run}_f{fr:04d}.jpg"))
    img[m] = (img[m] * 0.45 + np.array([255, 0, 200]) * 0.55).astype(np.uint8)
    ax = AL.bike_axis(m)
    ok = False
    if ax is not None:
        x["bike_lean_sam"] = round(ax[0], 2)
        if x.get("body_lean"):
            x["separation_sam"] = round(ax[0] - float(x["body_lean"]), 2)
        cv2.line(img, tuple(map(int, ax[2])), tuple(map(int, ax[1])), (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(img, f"bike={ax[0]:+.0f}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        ok = True
    cv2.imwrite(str(out / "overlay_sam" / f"run{run}_f{fr:04d}.jpg"), img)
    return ok


def write_csv(csv_path, rows):
    keys = list(rows[0].keys())
    for k in ("bike_lean_sam", "separation_sam"):
        if k not in keys:
            keys.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- 키포인트 프롬프트(YOLO-seg 없이)

def pose_kp(img, pose_m):
    pr = pose_m.predict(img, imgsz=AL.CROP, classes=[0], conf=0.1, verbose=False)[0]
    pi = AL.pick_center_person(pr)
    if pi is None:
        return None
    return (pr.keypoints.xy[pi].cpu().numpy(), pr.keypoints.conf[pi].cpu().numpy(), float(pr.boxes.conf[pi]))


def kp_geom(k):
    hip, ank, sh = (k[11] + k[12]) / 2, (k[15] + k[16]) / 2, (k[5] + k[6]) / 2
    L = max(float(np.linalg.norm(hip - ank)), 1.0)
    down = (ank - hip) / L
    rear = np.linalg.norm(k[5] - k[6]) > 0.45 * np.linalg.norm(sh - hip)  # 어깨가 넓게 보이면 뒷모습
    return hip, ank, sh, L, down, rear


def kp_prompt(k, c):
    """포즈 키포인트 → SAM2 프롬프트(박스 + 양성/음성 점). 크롭 좌표."""
    ok = lambda i: c[i] > 0.3
    hip, ank, sh, L, down, rear = kp_geom(k)
    pos = [ank + down * 0.15 * L]  # 발목 사이 아래(옆모습: BB 아래 체인링 쪽)
    neg = [k[0], sh, hip - down * 0.1 * L]  # 머리, 어깨 중앙, 허리
    for s in (0, 1):
        h, kn, a, e = k[11 + s], k[13 + s], k[15 + s], k[7 + s]
        neg += [(h + kn) / 2, kn, (kn + a) / 2, e]  # 허벅지, 무릎, 종아리, 팔꿈치
    side = np.array([-down[1], down[0]])
    if rear:
        # 뒷모습: 보이는 건 다리 사이 뒷타이어뿐. 넓은 박스나 핸들바 점은 라이더·먼지를 끌어들여서
        # 타이어만 좁은 박스(무릎 높이~접지) + 타이어 선 위 양성 점 3개로 잡는다.
        pos = [ank + down * f * L for f in (0.05, 0.15, 0.28)]
        kn = (k[13] + k[14]) / 2
        box = [min(k[15][0], k[16][0]) - 0.2 * L, kn[1], max(k[15][0], k[16][0]) + 0.2 * L, ank[1] + 0.45 * L]
        g = ank + down * 0.42 * L
        neg += [g + side * 0.4 * L, g - side * 0.4 * L]  # 타이어 접지 양옆 바닥
    else:
        pos.append(ank)
        wi = [i for i in (9, 10) if ok(i)] or [9, 10]
        wr = np.mean([k[i] for i in wi], 0)
        fwd = float(np.sign(wr[0] - hip[0]) or 1.0)
        pos.append(wr + np.array([fwd * 0.08 * L, 0.04 * L]))
        x_front, x_back = wr[0] + fwd * 0.75 * L, hip[0] - fwd * 0.65 * L
        box = [min(x_front, x_back), wr[1] - 0.15 * L, max(x_front, x_back), ank[1] + 0.5 * L]
    box = np.clip(box, 0, AL.CROP - 1).tolist()
    pts = np.vstack([pos, neg]).tolist()
    return dict(bboxes=[box], points=[pts], labels=[[1] * len(pos) + [0] * len(neg)])


def kp_ok(p, need_wrists):
    if p is None:
        return False
    k, c, pc = p
    core = all(c[i] >= 0.5 for i in (5, 6, 11, 12, 13, 14, 15, 16))
    wr = (c[9] >= 0.5 and c[10] >= 0.5) if need_wrists else max(c[9], c[10]) >= 0.5
    return pc >= 0.5 and core and wr


def pick_anchors(poses, n_parts=3, min_gap=8):
    """구간을 n_parts로 나눠 각 부분에서 포즈가 가장 확실한 프레임. 앵커끼리 min_gap 프레임 이상 떨어뜨려
    (가까운 앵커 둘은 거의 같은 마스크를 내서 medoid 투표가 2:1로 쏠림) 서로 다른 시점을 보게 한다."""
    n = len(poses)
    out = []
    for s in range(n_parts):
        lo, hi = s * n // n_parts, (s + 1) * n // n_parts
        far = lambda i: all(abs(i - j) >= min_gap for j in out)
        cand = [i for i in range(lo, hi) if far(i) and poses[i] is not None and poses[i][2] >= 0.7
                and kp_ok(poses[i], need_wrists=kp_geom(poses[i][0])[5])]
        if not cand:
            cand = [i for i in range(lo, hi) if far(i) and kp_ok(poses[i], need_wrists=False)]
        if cand:
            out.append(max(cand, key=lambda i: poses[i][2]))
    return out


def clean_mask(m):
    """작은 조각(전체의 2％ 미만)과 납작한 바닥 줄무늬(가장 큰 조각의 15％ 미만, 높이<폭/4) 제거.
    뒷타이어와 핸들바 끝은 원래 떨어져 있으므로 가장 큰 조각만 남기지는 않는다."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    big, tot = areas.max(), areas.sum()
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a, w, h = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        flat = a < 0.15 * big and h < w / 4
        keep[i] = a >= 0.02 * tot and not flat
    return keep[lab]


def mask_penalty(m, p):
    """프롬프트 점과 무관한 침범 점수(낮을수록 좋음): 몸통·머리 비율 + 접지선 아래 바닥 비율."""
    k, c, _ = p
    hip, ank, sh, L, down, rear = kp_geom(k)
    area = m.sum()
    if area == 0:
        return 9.0
    body = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(body, np.int32([k[5], k[6], k[12], k[11]]), 1)
    cv2.circle(body, tuple(map(int, k[0])), int(0.22 * L), 1, -1)
    torso = (m & body.astype(bool)).sum() / area
    ys = np.nonzero(m)[0]
    ground = (ys > ank[1] + 0.6 * L).sum() / area
    return float(torso + ground)


def run_kp(r, out, pose_m, sam_path, anchors=None, debug=False):
    run = r["run"]
    csv_path = out / f"run{run}.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    imgs = [cv2.imread(str(out / "crops" / f"run{run}_f{int(x['frame']):04d}.jpg")) for x in rows]
    poses = [pose_kp(im, pose_m) for im in imgs]
    frames = [int(x["frame"]) for x in rows]
    if anchors:
        idx = [frames.index(f) for f in anchors]
    else:
        idx = pick_anchors(poses)
    idx = [i for i in idx if poses[i] is not None]
    if not idx:
        print(f"  run {run}: 키포인트 앵커를 못 찾음")
        return
    print(f"  run {run}: 키포인트 앵커 프레임 {[frames[i] for i in idx]}")
    n = len(rows)
    per = {}
    for a in idx:
        pr = kp_prompt(*poses[a][:2])
        ms = [None] * n
        fwd = sam_track(imgs[a:], out / "_kp_tmp", sam_path, **pr)
        bwd = sam_track(imgs[a::-1], out / "_kp_tmp", sam_path, **pr)
        for j, m in enumerate(fwd):
            ms[a + j] = clean_mask(m) if m is not None and m.any() else None
        for j, m in enumerate(bwd[1:], 1):
            ms[a - j] = clean_mask(m) if m is not None and m.any() else None
        per[a] = ms
        if debug:
            dd = out / "_kp_debug" / f"anchor{frames[a]}"
            dd.mkdir(parents=True, exist_ok=True)
            for i, m in enumerate(ms):
                img = imgs[i].copy()
                if m is not None:
                    img[m] = (img[m] * 0.5 + np.array([255, 0, 200]) * 0.5).astype(np.uint8)
                pen = mask_penalty(m, poses[i]) if (m is not None and kp_ok(poses[i], False)) else -1
                cv2.putText(img, f"f{frames[i]} a{frames[a]} pen={pen:.2f}", (8, 30), 0, 0.9, (0, 255, 255), 2)
                cv2.imwrite(str(dd / f"f{frames[i]:04d}.jpg"), img)

    # 이전 결과 지우기(빈 마스크 프레임에 옛 값이 남지 않도록)
    for d in ("masks", "overlay_sam"):
        for f in (out / d).glob(f"run{run}_f*.*") if (out / d).exists() else []:
            f.unlink()
    for x in rows:
        x["bike_lean_sam"] = ""
        x["separation_sam"] = ""

    got, chosen = 0, []
    for i, x in enumerate(rows):
        cands = [(a, per[a][i]) for a in idx if per[a][i] is not None]
        if not cands:
            chosen.append(None)
            continue
        # 앵커들끼리 가장 많이 동의하는 마스크(평균 IoU 최대 = medoid)를 고른다. 한 앵커만 바닥·몸으로
        # 새어 나가면 다른 앵커들과 겹침이 낮아져 빠진다. 포즈가 믿을 만하면 몸통·접지선 아래 침범을 뺀다.
        def iou(p, q):
            return (p & q).sum() / max((p | q).sum(), 1)
        scores = []
        for a, m in cands:
            agree = np.mean([iou(m, mm) for b, mm in cands if b != a]) if len(cands) > 1 else 1.0
            pen = mask_penalty(m, poses[i]) if kp_ok(poses[i], need_wrists=False) else 0.0
            scores.append(agree - pen - 0.002 * abs(a - i))  # 동점이면 가까운 앵커
        a, m = cands[int(np.argmax(scores))]
        chosen.append(frames[a])
        got += save_mask(out, run, x, m)
    write_csv(csv_path, rows)
    print(f"  run {run}: 프레임별 앵커 {list(zip(frames, chosen))}")
    rear = [x for x in rows if x.get("view_ratio") and float(x["view_ratio"]) > 0.75]
    print(f"  run {run}: 마스크 {sum(c is not None for c in chosen)}/{n}, SAM 축 {got}프레임, "
          f"뒷모습 {len(rear)} 중 {sum(1 for x in rear if x.get('bike_lean_sam'))}")


def montage(out, run, path, n=12, cols=4, size=384):
    """크롭 위에 마스크(보라)와 윤곽을 그린 프레임 n장(구간 전체에 고르게) 몽타주. 마스크 없는 프레임은 표시."""
    rows = list(csv.DictReader(open(out / f"run{run}.csv", encoding="utf-8")))
    idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int)
    tiles = []
    for i in idx:
        fr = int(rows[i]["frame"])
        img = cv2.imread(str(out / "crops" / f"run{run}_f{fr:04d}.jpg"))
        mp = out / "masks" / f"run{run}_f{fr:04d}.png"
        label = f"f{fr}"
        if mp.exists():
            m = cv2.imread(str(mp), 0) > 0
            img[m] = (img[m] * 0.5 + np.array([255, 0, 200]) * 0.5).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (255, 0, 255), 2)
        else:
            label += " no mask"
        img = cv2.resize(img, (size, size))
        cv2.putText(img, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)]))
    print(f"  run {run}: 몽타주 {path}")


# ---------------------------------------------------------------- 기본 경로(YOLO-seg 박스)

def run_one(r, out, seg_m, pose_m, sam_path, kp_prompt_flag=False, anchors=None, debug=False):
    run = r["run"]
    if kp_prompt_flag:
        return run_kp(r, out, pose_m, sam_path, anchors, debug)
    csv_path = out / f"run{run}.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    p = first_bike_prompt(rows, out / "crops", run, seg_m, pose_m)
    if p is None:
        print(f"  run {run}: 프롬프트용 자전거를 못 찾음 → 키포인트 프롬프트로 대체")
        return run_kp(r, out, pose_m, sam_path, anchors, debug)
    i0, box, neg = p
    # 프롬프트 프레임부터 크롭들을 영상으로 묶음
    seq = rows[i0:]
    imgs = [cv2.imread(str(out / "crops" / f"run{run}_f{int(x['frame']):04d}.jpg")) for x in seq]
    masks = sam_track(imgs, out / f"_run{run}_crops", sam_path, bboxes=[box])
    got = 0
    for x, m in zip(seq, masks):
        if m is None:
            continue
        got += save_mask(out, run, x, m)
    write_csv(csv_path, rows)
    rear = [x for x in rows if x.get("view_ratio") and float(x["view_ratio"]) > 0.75]
    print(f"  run {run}: 프롬프트 프레임 {i0}, SAM 축 {got}프레임, 뒷모습 {len(rear)} 중 {sum(1 for x in rear if x.get('bike_lean_sam'))}")


# ---------------------------------------------------------------- 그림자 제거 (--deshadow)
# 해가 든 코너 출구에서 SAM2 자전거 마스크가 땅에 진 바퀴 그림자(호 모양)까지 감싼다(2026-11-21 7런 모두).
# 마스크 안에서 "땅 색(밖 고리의 색상)과 같은 색상이고 더 어두운" 덩어리 중 아래 조건을 모두 만족하면 그림자로 본다.
#   - 테두리 비율: 덩어리 둘레 중 마스크 바깥과 맞닿은 비율 ≥ SH_EDGE (바퀴 안으로 비친 땅은 타이어에 둘러싸여 낮다)
#   - 밝은 그림자 비율 ≥ SH_BRIGHT (L<SH_LDARK인 짙은 화소는 밝은 그림자 곁 SH_NEAR px 안에서만 덩어리에 붙는다)
#   - 땅 높이 아래 비율 ≥ SH_BELOW. 땅 높이 = 낮은 발목 y + 0.15 m. 흙 묻은 타이어 띠(뒷모습)는 대부분 이보다 위에 있다.
# 발목이 안 잡히면 그 프레임은 건드리지 않는다. 원본은 masks_raw/에 두고 늘 거기서 다시 만든다.
SH_EDGE, SH_BRIGHT, SH_BELOW, SH_NEAR, SH_LDARK = 0.40, 0.35, 0.5, 8, 28


def _lab(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    L[..., 0] *= 100 / 255
    L[..., 1:] -= 128
    return L


def shadow_region(img, m, ground_y):
    """마스크 m 안의 그림자 화소(bool). ground_y: 이 y보다 아래가 땅."""
    Lb = _lab(img)
    ys = np.nonzero(m)[0]
    y0, y1 = ys.min(), ys.max()
    ring = cv2.dilate(m.astype(np.uint8), np.ones((61, 61))) > 0
    ring &= ~(cv2.dilate(m.astype(np.uint8), np.ones((15, 15))) > 0)
    ring[:int(y0 + 0.4 * (y1 - y0))] = False
    g = Lb[ring]
    gL, gh = np.median(g[:, 0]), np.degrees(np.arctan2(np.median(g[:, 2]), np.median(g[:, 1])))
    L, a, b = Lb[..., 0], Lb[..., 1], Lb[..., 2]
    ch, hu = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    cand = (m & (L >= 10) & (L <= gL * 0.92) & (ch >= np.where(L < SH_LDARK, 6.0, 4.5))
            & (np.abs((hu - gh + 180) % 360 - 180) <= 22))
    bright = cand & (L >= SH_LDARK)
    near = cv2.dilate(cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5))),
                      np.ones((2 * SH_NEAR + 1, 2 * SH_NEAR + 1))) > 0
    cand = bright | (cand & (L < SH_LDARK) & near)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5)))
    cand = (cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9))) > 0) & m
    outside = cv2.dilate((~m).astype(np.uint8), np.ones((5, 5))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    rem = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 300 or st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] / 2 < y0 + 0.45 * (y1 - y0):
            continue
        comp = lab == i
        per = comp & ~(cv2.erode(comp.astype(np.uint8), np.ones((3, 3))) > 0)
        if (per & outside).sum() < SH_EDGE * max(per.sum(), 1):
            continue
        if (comp & (L >= SH_LDARK)).sum() < SH_BRIGHT * comp.sum():
            continue
        if np.mean(np.nonzero(comp)[0] > ground_y) < SH_BELOW:
            continue
        rem |= comp
    return rem


def deshadow_mask(img, m, ground_y):
    """그림자를 뺀 마스크. 지운 곳 가장자리 5px와, 그 곁에 남은 작은 부스러기(전체 3％ 미만)도 뺀다.
    핸들바 끝처럼 원래 따로 떨어진 조각은 그대로 둔다."""
    if m.sum() < 500:
        return m
    rem = shadow_region(img, m, ground_y)
    if not rem.any():
        return m
    k = m & ~(cv2.dilate(rem.astype(np.uint8), np.ones((5, 5))) > 0)
    k = cv2.morphologyEx(k.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(k.astype(np.uint8), connectivity=8)
    if n > 2:
        tot = st[1:, cv2.CC_STAT_AREA].sum()
        near = cv2.dilate(rem.astype(np.uint8), np.ones((11, 11))) > 0
        keep = np.ones(n, bool)
        keep[0] = False
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 0.03 * tot and (near & (lab == i)).any():
                keep[i] = False
        k = keep[lab]
    return k


def deshadow_run(r, out, pose_m):
    """이미 만든 masks/run<N>_f*.png에서 그림자를 뺀다. 원본은 masks_raw/에 한 번만 복사해 두고 늘 거기서 다시 만든다."""
    run = r["run"]
    raw, dst = out / "masks_raw", out / "masks"
    raw.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(out / f"run{run}.csv", encoding="utf-8")))
    sc = [float(x["fit_scale"]) * 4 for x in rows if x.get("fit_scale")]   # px/m @768 (bike_fit 결과가 있으면)
    s_run = float(np.median(sc)) if sc else None
    log, changed = {}, []
    for x in rows:
        name = f"run{run}_f{int(x['frame']):04d}"
        mp, rp = dst / f"{name}.png", raw / f"{name}.png"
        if not rp.exists():
            if not mp.exists():
                continue
            rp.write_bytes(mp.read_bytes())
        m = cv2.imread(str(rp), 0) > 127
        img = cv2.imread(str(out / "crops" / f"{name}.jpg"))
        gy = None
        kp = pose_kp(img, pose_m)
        if kp is not None:
            k, c, _ = kp
            ank = [k[i][1] for i in (15, 16) if c[i] >= 0.3]
            if ank:
                leg = float(np.linalg.norm((k[11] + k[12]) / 2 - (k[15] + k[16]) / 2))   # 엉덩이-발목 ≈ 0.9 m
                gy = max(ank) + 0.15 * (s_run if s_run else leg / 0.9)
        km = m if gy is None else deshadow_mask(img, m, gy)
        cv2.imwrite(str(mp), km.astype(np.uint8) * 255)
        frac = 1 - km.sum() / max(m.sum(), 1)
        log[name] = round(float(frac), 3)
        if frac > 0.02:
            changed.append((name, img, m, km, frac))
    (raw / f"deshadow_run{run}.json").write_text(json.dumps(log, indent=0), encoding="utf-8")
    if changed:   # 확인용: 지운 곳 빨강, 남은 마스크 하늘색
        tiles = []
        for name, img, m, km, frac in changed[:40]:
            v = img.copy()
            gone = m & ~km
            v[gone] = (v[gone] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
            cs, _ = cv2.findContours(km.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cs, -1, (255, 255, 0), 2)
            cv2.putText(v, f"{name} -{frac * 100:.0f}%", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            tiles.append(cv2.resize(v, (320, 320)))
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(raw / f"deshadow_check_run{run}.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    v = np.array(list(log.values()) or [0])
    print(f"  run {run}: 그림자 제거 {len(log)}프레임 중 {int((v > 0.02).sum())}프레임 바뀜 "
          f"(10％ 넘게 {int((v > 0.10).sum())}), 최대 {v.max() * 100:.0f}％")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--kp-prompt", action="store_true", help="YOLO-seg 대신 포즈 키포인트 프롬프트(여러 앵커) 사용")
    ap.add_argument("--montage", help="실행한 런의 마스크 몽타주(12장) 저장 경로. 여러 런이면 _run<N> 붙임")
    ap.add_argument("--kp-debug", action="store_true", help="앵커별 추적 결과를 _kp_debug/에 저장")
    ap.add_argument("--kp-anchors", type=int, nargs="*", help="키포인트 앵커 프레임 번호 직접 지정")
    ap.add_argument("--deshadow", action="store_true", help="SAM2를 다시 돌리지 않고, 있는 마스크에서 땅 그림자만 뺌(원본은 masks_raw/)")
    a = ap.parse_args()
    seg = Path(a.segments)
    cfg = json.loads(seg.read_text(encoding="utf-8"))
    out = PATHS.feat_dir(seg, cfg, "lean")
    models = PATHS.MODELS
    pose_m = YOLO(str(models / "yolo11x-pose.pt"))
    if a.deshadow:
        for r in cfg["runs"]:
            if not a.runs or r["run"] in a.runs:
                deshadow_run(r, out, pose_m)
        return
    seg_m = YOLO(str(models / "yolo11x-seg.pt"))
    for r in cfg["runs"]:
        if a.runs and r["run"] not in a.runs:
            continue
        run_one(r, out, seg_m, pose_m, models / "sam2.1_b.pt", a.kp_prompt, a.kp_anchors, a.kp_debug)
        if a.montage:
            mp = Path(a.montage)
            if a.runs is None or len(a.runs) != 1:
                mp = mp.with_name(f"{mp.stem}_run{r['run']}{mp.suffix}")
            montage(out, r["run"], mp)


if __name__ == "__main__":
    main()
