"""린아웃(자전거를 몸보다 더 눕히고 엉덩이를 바깥으로) 확인 보드.

뒷모습 구간 프레임에서:
  자전거 선  = 타이어 접지점(SAM2 마스크 맨 아래) → 핸들 중앙(마스크 맨 위 영역 중앙)   [주황]
  라이더 선  = 같은 접지점 → 엉덩이 중앙(포즈)                                            [초록]
  엉덩이 오프셋 = 엉덩이 높이에서 자전거 선 대비 엉덩이의 가로 위치 / 몸통 길이
                  +면 화면 오른쪽 = (뒤에서 볼 때) 라이더 오른쪽 = 좌회전의 바깥쪽 → 린아웃
  분리각 = 자전거 선 기울기 − 상체(엉덩이→어깨) 기울기, +면 자전거가 더 누움

한계: 카메라가 정확히 뒤가 아니면(yaw) 자전거 선이 실제 자전거 평면과 어긋난다 → view_ratio 높은 프레임만 사용.

사용법:
    uv run python tools/leanout_board.py sessions/2026-11-21/segments.json --runs 1 2 3 4 5 7
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402
import analyze_lean as AL  # noqa: E402
import compare_reference as CR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONT, FONT_B = "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"
CELL, TEXT_W = 300, 470
REF = ROOT / "sessions/reference-coach/segments.json"


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def bike_line(mask, torso):
    """타이어 접지점 → 뒷바퀴 영역 중심. 바퀴 반지름(375mm) ≈ 몸통(어깨~엉덩이, ~500mm)의 0.75배로 가정해
    접지점 위 '바퀴 지름' 높이 띠의 마스크 중심을 허브 근사로 쓴다. (핸들바는 몸에 가려 한쪽만 보여 쓰지 않음)"""
    ys, xs = np.nonzero(mask)
    y1 = ys.max()
    h = ys.max() - ys.min()
    bot = np.array([xs[ys >= y1 - 0.03 * h].mean(), y1], float)
    band = ys >= y1 - 1.5 * torso
    hub = np.array([xs[band].mean(), ys[band].mean()], float)
    return bot, hub


def lean(top, bottom):
    return math.degrees(math.atan2(-(top[0] - bottom[0]), -(top[1] - bottom[1])))


def measure(crop_path, mask_path, pose_m):
    img = cv2.imread(str(crop_path))
    mask = cv2.imread(str(mask_path), 0) > 0
    pr = pose_m.predict(img, imgsz=AL.CROP, classes=[0], verbose=False)[0]
    pi = AL.pick_center_person(pr)
    if pi is None or mask.sum() < 200:
        return None
    k = pr.keypoints.xy[pi].cpu().numpy()
    sh, hp = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2
    torso = np.linalg.norm(sh - hp)
    bot, top = bike_line(mask, torso)
    # 엉덩이 높이에서 자전거 선의 x
    t = (hp[1] - bot[1]) / (top[1] - bot[1])
    bx = bot[0] + t * (top[0] - bot[0])
    m = dict(hip_off=(hp[0] - bx) / torso, bike=lean(top, bot), torso=lean(sh, hp), rider=lean(hp, bot))
    m["sep"] = m["bike"] - m["torso"]
    vis = img.copy()
    vis[mask] = (vis[mask] * 0.6 + np.array([255, 0, 200]) * 0.4).astype(np.uint8)
    for a, b in AL.SKELETON:
        cv2.line(vis, tuple(map(int, k[a])), tuple(map(int, k[b])), (255, 255, 0), 2, cv2.LINE_AA)
    ext = top + (top - bot) * 1.6
    cv2.line(vis, tuple(map(int, bot)), tuple(map(int, ext)), (0, 140, 255), 6, cv2.LINE_AA)
    cv2.line(vis, tuple(map(int, bot)), tuple(map(int, hp)), (60, 230, 60), 5, cv2.LINE_AA)
    cv2.line(vis, (int(bx), int(hp[1])), tuple(map(int, hp)), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(vis, tuple(map(int, hp)), 10, (60, 230, 60), -1, cv2.LINE_AA)
    cv2.circle(vis, tuple(map(int, bot)), 8, (0, 140, 255), -1, cv2.LINE_AA)
    return m, vis


def cell(vis, title, m):
    img = cv2.resize(vis, (CELL, CELL), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    dr.rectangle([0, CELL - 52, CELL, CELL], fill=(0, 0, 0))
    side = "바깥(오른쪽)" if m["hip_off"] > 0.05 else ("안쪽(왼쪽)" if m["hip_off"] < -0.05 else "자전거 위")
    col = (120, 230, 120) if m["hip_off"] > 0.05 else ((255, 120, 120) if m["hip_off"] < -0.05 else (230, 230, 230))
    dr.text((8, CELL - 50), title, font=f(18, True), fill=(255, 255, 255))
    dr.text((8, CELL - 26), f"엉덩이 {side} {m['hip_off']:+.2f} · 분리 {m['sep']:+.0f}°", font=f(16), fill=col)
    return pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3, 4, 5, 7])
    ap.add_argument("--u", type=float, nargs="*", default=[0.46, 0.52, 0.58])
    a = ap.parse_args()
    seg = ROOT / a.segments
    cfg = json.loads(seg.read_text(encoding="utf-8"))
    lean_dir = PATHS.feat_dir(seg, cfg, "lean")
    out_dir = PATHS.out_base(seg) / "insights"
    out_dir.mkdir(exist_ok=True)
    pose_m = YOLO(str(PATHS.MODELS / "yolo11x-pose.pt"))
    runs = [CR.load_run(seg, r) for r in a.runs]
    a_dst = float(np.mean([d["anchor"] for d in runs] + [CR.load_run(REF, 0)["anchor"]]))

    W = TEXT_W + CELL * len(a.u) + 230
    head = 150
    H = head + CELL * len(runs) + 60
    board = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(board)
    dr.text((24, 16), "좌회전 뱅크 · 린아웃 확인 (뒤에서 본 장면)", font=f(36, True), fill=(255, 255, 255))
    dr.text((26, 66), "주황 = 자전거 선(타이어 접지 → 뒷바퀴 중심, 연장)   초록 = 라이더 선(접지 → 엉덩이)   흰 가로선 = 엉덩이가 자전거 선에서 벗어난 거리",
            font=f(18), fill=(200, 200, 200))
    dr.text((26, 96), "린아웃이면: 주황선이 초록선보다 더 눕고, 엉덩이(초록 점)가 주황선 오른쪽(바깥)에 있음.  숫자 = 엉덩이 오프셋(몸통 길이 배수), 분리각(자전거 - 상체)",
            font=f(18), fill=(200, 200, 200))
    summary = []
    for i, d in enumerate(runs):
        y0 = head + i * CELL
        rows = CR.read_csv(lean_dir / f"run{d['run']}.csv")
        ms = []
        for j, u in enumerate(a.u):
            p = CR.warp(u, d["anchor"], a_dst)
            r = min(rows, key=lambda x: abs(float(x["progress"]) - p))
            name = f"run{d['run']}_f{int(r['frame']):04d}"
            res = measure(lean_dir / "crops" / f"{name}.jpg", lean_dir / "masks" / f"{name}.png", pose_m)
            if res is None:
                continue
            m, vis = res
            ms.append(m)
            board.paste(cell(vis, f"Run {d['run']} · {p*100:.0f}%", m), (TEXT_W + j * CELL, y0))
        # 뒷모습 구간 전체 평균(view_ratio > 0.8)
        allm = []
        for r in rows:
            if r.get("view_ratio") and float(r["view_ratio"]) > 0.8:
                name = f"run{d['run']}_f{int(r['frame']):04d}"
                if (lean_dir / "masks" / f"{name}.png").exists():
                    res = measure(lean_dir / "crops" / f"{name}.jpg", lean_dir / "masks" / f"{name}.png", pose_m)
                    if res:
                        allm.append(res[0])
        if allm:
            ho = np.median([m["hip_off"] for m in allm])
            sp = np.median([m["sep"] for m in allm])
            summary.append((d["run"], ho, sp, len(allm)))
            x0 = TEXT_W + CELL * len(a.u) + 16
            dr.text((x0, y0 + 20), f"Run {d['run']} 뒷모습 {len(allm)}프레임", font=f(17, True), fill=(255, 255, 255))
            dr.text((x0, y0 + 50), f"엉덩이 중앙값 {ho:+.2f}", font=f(17), fill=(220, 220, 220))
            dr.text((x0, y0 + 76), f"분리각 중앙값 {sp:+.0f}°", font=f(17), fill=(220, 220, 220))
    # 왼쪽 설명 패널
    dr.rectangle([0, head, TEXT_W - 1, H - 61], fill=(34, 37, 43))
    note = ("린아웃(lean-out)이란\n"
            "자전거는 코너 안쪽(좌회전이면 왼쪽)으로 더 눕히고, 몸은 상대적으로 세워 엉덩이를 바깥(오른쪽)으로 두는 자세. "
            "바깥 발(오른발)에 체중을 실어 타이어를 눌러 접지력을 얻는다.\n\n"
            "뱅크에서는?\n"
            "뱅크 벽이 받쳐주기 때문에 평지 코너만큼 큰 분리는 필요 없고, 자전거와 몸을 뱅크 면에 수직으로 함께 기울여도 된다. "
            "그래도 엉덩이는 BB 위나 약간 바깥에 두고 자전거를 벽에 밀어 넣는 게 기본. "
            "엉덩이가 안쪽(왼쪽)으로 빠지면 앞바퀴 하중이 빠지고 밀리기 쉽다.\n\n"
            "읽는 법: 초록 점이 주황선 오른쪽이면 바깥, 왼쪽이면 안쪽.")
    yy = head + 18
    for para in note.split("\n"):
        line = ""
        for ch in para:
            if dr.textlength(line + ch, font=f(18)) > TEXT_W - 40:
                dr.text((20, yy), line, font=f(18, para.endswith("?") or para.endswith("이란")), fill=(225, 225, 225))
                yy += 27
                line = ch
            else:
                line += ch
        dr.text((20, yy), line, font=f(18, para.endswith("?") or para.endswith("이란")),
                fill=(255, 196, 0) if para.endswith("?") or para.endswith("이란") else (225, 225, 225))
        yy += 27
    dr.text((24, H - 46), "주의: 자전거 선은 SAM2 마스크의 접지점과 뒷바퀴 영역 중심으로 근사. 카메라가 비스듬하면 오차가 큼. 코치님 영상은 자전거 마스크가 없어 이번 보드에서 제외.",
            font=f(16), fill=(160, 160, 160))
    out = out_dir / "left_bank_leanout.png"
    board.save(out)
    print("saved:", out)
    for s in summary:
        print(f"  run {s[0]}: hip_off {s[1]:+.2f}, sep {s[2]:+.1f}, n={s[3]}")


if __name__ == "__main__":
    main()
