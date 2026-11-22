"""뒷모습 구간의 자전거 기울기(3D 맞추기) vs 상체 기울기(2D 포즈) 보드.

선택 기준: view_ratio > 0.75, |fit_yaw| < 25, fit_iou >= 0.55, fit_lowconf가 아닌 프레임 (bike_fit.py 결과 사용).
분리각 = fit_lean_s - body_lean, 런별 중앙값. +면 자전거가 상체보다 더 누움.
대표 프레임 = 분리각이 중앙값에 가장 가까운 프레임의 overlay_fit 이미지.

사용법:
    uv run python tools/separation_board.py sessions/2026-11-21/segments.json --runs 1 2 3 4 5 7
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT, FONT_B = "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"
CELL, BAR_W, ROW = 300, 820, 300
BIKE_C, BODY_C, SEP_C = (255, 196, 0), (60, 230, 60), (120, 200, 255)
GOLD = (255, 196, 0)


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def select(rows, min_iou=0.55):
    out = []
    for x in rows:
        try:
            vr, yaw, iou = float(x["view_ratio"]), float(x["fit_yaw"]), float(x["fit_iou"])
            bl, fl = float(x["body_lean"]), float(x["fit_lean_s"])
        except (ValueError, KeyError):
            continue
        if str(x.get("fit_lowconf", "0")).strip() in ("1", "1.0", "True"):
            continue  # bike_fit이 기울기를 실루엣으로 정할 수 없다고 표시한 프레임
        if vr > 0.75 and abs(yaw) < 25 and iou >= min_iou:
            out.append((int(x["frame"]), fl, bl, fl - bl))
    return out


def lean_dir_of(seg):
    cfg = json.loads(Path(seg).read_text(encoding="utf-8"))
    return PATHS.feat_dir(seg, cfg, "lean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3, 4, 5, 7])
    ap.add_argument("--ref", default=None, help="코치 segments.json (run 0을 첫 줄에 추가)")
    ap.add_argument("--min-iou", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seg = ROOT / a.segments
    out = Path(a.out) if a.out else PATHS.out_base(seg) / "insights" / "left_bank_separation.png"
    items = []
    if a.ref:
        items.append((lean_dir_of(ROOT / a.ref), 0, "코치님", True))
    items += [(lean_dir_of(seg), r, f"Run {r}", False) for r in a.runs]

    head, foot = 170, 190
    W, H = CELL + BAR_W + 40, head + ROW * len(items) + foot
    board = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(board)
    dr.text((24, 16), "좌회전 뱅크 · 자전거 vs 상체 기울기 (뒤에서 본 구간)", font=f(34, True), fill=(255, 255, 255))
    dr.text((26, 66), "노랑 = 자전거 기울기(3D 맞추기)   초록 = 상체 기울기(엉덩이→어깨)   파랑 숫자 = 분리각(자전거 - 상체)",
            font=f(18), fill=(210, 210, 210))
    dr.text((26, 94), "분리각이 +면 자전거를 몸보다 더 눕힌 것(린아웃 방향). 0이면 몸과 자전거가 한 덩어리(인라인).",
            font=f(18), fill=(210, 210, 210))
    dr.text((26, 122), "왼쪽 사진: 각 런에서 분리각이 중앙값에 가장 가까운 프레임 (노랑 바퀴·초록 프레임·빨강 핸들 = 3D 모델)",
            font=f(16), fill=(160, 160, 160))

    scale = (BAR_W - 260) / 50.0  # 50°까지
    x0 = CELL + 40 + 150
    summary = []
    for i, (lean_dir, r, name, is_ref) in enumerate(items):
        y0 = head + i * ROW
        rows = list(csv.DictReader(open(lean_dir / f"run{r}.csv", encoding="utf-8")))
        sel = select(rows, a.min_iou)
        if not sel:
            dr.text((CELL + 40, y0 + 20), f"{name}: 뒷모습 프레임 없음", font=f(20), fill=(200, 200, 200))
            continue
        bike = float(np.median([s[1] for s in sel]))
        body = float(np.median([s[2] for s in sel]))
        sep = float(np.median([s[3] for s in sel]))
        summary.append((name, len(sel), bike, body, sep))
        rep = min(sel, key=lambda s: abs(s[3] - sep))
        img = Image.open(lean_dir / "overlay_fit" / f"run{r}_f{rep[0]:04d}.jpg").convert("RGB")
        img = img.resize((CELL, CELL))
        d2 = ImageDraw.Draw(img)
        d2.rectangle([0, CELL - 30, CELL, CELL], fill=(0, 0, 0))
        d2.text((8, CELL - 28), f"{name} · f{rep[0]} · 자전거 {rep[1]:.0f}° / 상체 {rep[2]:.0f}°", font=f(15), fill=(255, 255, 255))
        if is_ref:
            d2.rectangle([1, 1, CELL - 2, CELL - 2], outline=GOLD, width=5)
        board.paste(img, (0, y0))
        # bars
        dr.text((CELL + 40, y0 + 40), name, font=f(28, True), fill=GOLD if is_ref else (255, 255, 255))
        dr.text((CELL + 40, y0 + 80), f"{len(sel)}프레임", font=f(16), fill=(160, 160, 160))
        for k, (val, col, lab) in enumerate([(bike, BIKE_C, "자전거"), (body, BODY_C, "상체")]):
            yy = y0 + 50 + k * 70
            dr.rectangle([x0, yy, x0 + val * scale, yy + 44], fill=col)
            dr.text((x0 + val * scale + 10, yy + 6), f"{lab} {val:.0f}°", font=f(22, True), fill=col)
        dr.text((x0, y0 + 200), f"분리각 {sep:+.0f}°", font=f(30, True), fill=SEP_C)
        # 10° gridlines
        for g in range(0, 51, 10):
            gx = x0 + g * scale
            dr.line([gx, y0 + 44, gx, y0 + 170], fill=(70, 70, 76), width=1)
            dr.text((gx - 10, y0 + 172), f"{g}°", font=f(13), fill=(120, 120, 120))
        dr.line([0, y0 + ROW - 1, W, y0 + ROW - 1], fill=(50, 52, 58), width=1)

    note = ("측정값(3D 자전거 모델 + 2D 포즈). 프레임별 오차 ±10° 수준이고 모델 수정 때마다 런별 값이 최대 15° 움직임 → 순서만 참고.\n"
            "기울기를 정할 수 없는 프레임(LOW-CONF)은 제외. 2D 상체각은 상체를 앞으로 숙이면 뒤에서 볼 때 기울기가 섞여 들어갈 수 있음. Run 6은 뒷모습 프레임이 1개뿐이라 제외.")
    if a.ref:
        note += ("\n코치님: 자전거 기종 미확인(일반 29\" 근사), 뒷모습 마스크에 뒷바퀴만 있어 기울기는 주로 손 위치로 정해짐.\n"
                 "코치님 뒷모습 프레임은 정렬 기준점보다 앞(뱅크 진입 구간)이라 내 런(뱅크 중간)과 구간이 다름 → 참고용.")
    yy = H - foot + 14
    for para in note.split("\n"):
        line = ""
        for ch in para:  # 폭에 맞춰 자동 줄바꿈
            if dr.textlength(line + ch, font=f(16)) > W - 48:
                dr.text((24, yy), line, font=f(16), fill=(160, 160, 160))
                yy += 24
                line = ch
            else:
                line += ch
        dr.text((24, yy), line, font=f(16), fill=(160, 160, 160))
        yy += 26
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out)
    print("saved:", out)
    for s in summary:
        print(f"  {s[0]}: n={s[1]} bike {s[2]:.1f} body {s[3]:.1f} sep {s[4]:+.1f}")


if __name__ == "__main__":
    main()
