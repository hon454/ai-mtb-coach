"""핸들바 선 그리기 보드: 사람이 그린 핸들바 선을 3D 자전거 맞추기(bike_fit)의 힌트로 쓴다.

뒷모습에서 핸들바는 자전거 면에 수직이라, 바가 기운 각도가 곧 자전거 기울기다(요 ±10°에 1.5° 이내).
바는 대부분 몸에 가려 실루엣만으로는 잡히지 않는다. 그래서 사람이 보이는 손·팔을 보고 바를 그리고, 도구가 그 선을 읽는다.

  make     뒷모습 프레임을 골라 깨끗한 크롭(덧그림 없음) 보드를 만든다. 런마다 1장, 칸 배치는 같은 이름의 .json에 저장
  extract  사람이 선을 그린 보드 이미지를 읽어 칸마다 선을 맞추고, sessions/<세션>/bar_hints.json에 더한다

예)
  uv run python tools/bar_hint_board.py make sessions/2026-11-21/segments.json --runs 1 2 3 4 5 7
  uv run python tools/bar_hint_board.py extract sessions/2026-11-21/segments.json <그린 이미지> --board bar_hint_draw_run4

그리는 법: 칸마다 핸들바 중심선을 한 획으로 긋는다. 몸에 가린 쪽도 추정해서 잇는다. 확실하지 않은 칸은 비워 둔다.
색은 무엇이든 된다(원본 보드와의 차이로 읽는다). 보드를 줄이거나 webp/jpg로 보내도 된다(가로세로 비율만 유지).
bar_hints.json의 형식은 그 파일의 "note"에 있다. 좌표는 768px 크롭 기준.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

PANEL = 420          # 칸 이미지 크기(px)
LABEL = 30           # 칸 위 이름 띠
GAP = 8
COLS = 4
HEAD = 92            # 보드 위 안내문
REGION = (184, 120, 400)   # 크롭에서 보여 줄 영역 (x0, y0, 한 변) — 크롭은 라이더 중심이라 고정 영역으로 충분
FONT = "C:/Windows/Fonts/malgun.ttf"
FONTB = "C:/Windows/Fonts/malgunbd.ttf"
NOTE = ("사용자가 보드에 그린 핸들바 선(사람 판단, 측정값 아님). 좌표는 768px 크롭(outputs/<세션>/left_bank_lean/crops) 기준. "
        "a=왼쪽 끝, b=오른쪽 끝(이미지 기준). angle은 +면 오른쪽이 올라감(°). visible_end는 몸에 가리지 않은 쪽 손"
        "(그 쪽 그립만 위치 제약). 같은 프레임을 여러 번 그리면 lines에 모두 둔다.")


def font(n, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, n)
    except OSError:
        return ImageFont.load_default()


def hints_path(seg):
    return Path(seg).resolve().parent / "bar_hints.json"


def load_hints(seg):
    p = hints_path(seg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"note": NOTE, "hints": []}


def rear_frames(csv_path, step, skip):
    """뒷모습(view_ratio > 0.75, |fit_yaw| < 30) 프레임을 step 간격으로. skip(이미 힌트 있는 프레임) ±1은 뺀다."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    sel = [int(d["frame"]) for d in rows
           if d.get("view_ratio") and float(d["view_ratio"]) > 0.75
           and d.get("fit_yaw") and abs(float(d["fit_yaw"])) < 30]
    out, last = [], -99
    for f in sel:
        if any(abs(f - s) <= 1 for s in skip):
            continue
        if f - last >= step:
            out.append(f)
            last = f
    return out


def make(args):
    cfg = PATHS.load_cfg(args.segments)
    lean = PATHS.feat_dir(args.segments, cfg, "lean")
    outd = PATHS.out_base(args.segments) / "insights"
    outd.mkdir(parents=True, exist_ok=True)
    have = load_hints(args.segments)["hints"]
    runs = args.runs or sorted(int(p.stem[3:]) for p in lean.glob("run*.csv"))
    x0, y0, side = REGION
    k = PANEL / side
    for run in runs:
        skip = [h["frame"] for h in have if h["run"] == run]
        frames = rear_frames(lean / f"run{run}.csv", args.step, skip)[: args.max]
        if not frames:
            print(f"run{run}: 뒷모습 프레임 없음, 건너뜀")
            continue
        rows = math.ceil(len(frames) / COLS)
        W = COLS * PANEL + (COLS + 1) * GAP
        H = HEAD + rows * (LABEL + PANEL + GAP) + GAP
        B = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(B)
        d.text((12, 8), f"핸들바 선 그리기 — {PATHS.session_name(args.segments)} run{run}", font=font(26, True), fill=(255, 255, 255))
        d.text((12, 46), "칸마다 핸들바 중심선을 한 획으로 그려 주세요. 몸에 가린 쪽도 추정해서 잇고, 확실하지 않은 칸은 비워 두세요.",
               font=font(17), fill=(220, 220, 220))
        d.text((12, 68), "색은 무엇이든 됩니다. 칸 위 이름 띠에는 그리지 마세요.", font=font(17), fill=(170, 170, 170))
        panels = []
        for i, f in enumerate(frames):
            r, c = divmod(i, COLS)
            px = GAP + c * (PANEL + GAP)
            py = HEAD + r * (LABEL + PANEL + GAP)
            name = f"run{run}_f{f:04d}"
            im = Image.open(lean / "crops" / f"{name}.jpg").convert("RGB")
            im = im.crop((x0, y0, x0 + side, y0 + side)).resize((PANEL, PANEL), Image.BICUBIC)
            d.rectangle([px, py, px + PANEL, py + LABEL], fill=(28, 28, 28))
            d.text((px + 8, py + 3), f"run{run} f{f}", font=font(18, True), fill=(255, 255, 255))
            B.paste(im, (px, py + LABEL))
            panels.append({"run": run, "frame": f, "x": px, "y": py + LABEL, "size": PANEL,
                           "crop_x0": x0, "crop_y0": y0, "scale": k})
        stem = f"bar_hint_draw_run{run}"
        B.save(outd / f"{stem}.png")
        (outd / f"{stem}.json").write_text(json.dumps({"segments": str(Path(args.segments).as_posix()), "size": [W, H],
                                                       "panels": panels}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"run{run}: {len(frames)}칸 → {outd / (stem + '.png')}")


def fit_line(pts):
    vx, vy, cx, cy = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if vx < 0:
        vx, vy = -vx, -vy
    t = (pts - [cx, cy]) @ [vx, vy]
    a = np.array([cx, cy]) + t.min() * np.array([vx, vy])
    b = np.array([cx, cy]) + t.max() * np.array([vx, vy])
    resid = np.abs((pts - [cx, cy]) @ [-vy, vx])
    return a, b, math.degrees(math.atan2(-vy, vx)), float(np.percentile(resid, 90))


def extract(args):
    outd = PATHS.out_base(args.segments) / "insights"
    lay = json.loads((outd / f"{args.board}.json").read_text(encoding="utf-8"))
    clean = np.asarray(Image.open(outd / f"{args.board}.png").convert("RGB")).astype(int)
    H, W = clean.shape[:2]
    drawn_im = Image.open(args.image).convert("RGB")
    if abs(drawn_im.width / drawn_im.height - W / H) > 0.02:
        sys.exit(f"그린 이미지의 비율({drawn_im.width}x{drawn_im.height})이 보드({W}x{H})와 다르다. 잘리거나 다른 보드인지 확인")
    drawn = np.asarray(drawn_im.resize((W, H), Image.LANCZOS)).astype(int)
    # 그린 이미지가 줄어든 경우 원본도 같은 만큼 줄였다 키워서 압축·보간 차이를 맞춘다
    sm = Image.fromarray(clean.astype(np.uint8)).resize((drawn_im.width, drawn_im.height), Image.LANCZOS).resize((W, H), Image.LANCZOS)
    diff = np.abs(drawn - np.asarray(sm).astype(int)).sum(2)
    stroke = (diff > args.thresh).astype(np.uint8)
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = Path(args.out) if args.out else hints_path(args.segments)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else load_hints(args.segments)
    src = f"{args.board} ({Path(args.image).name})"
    for h in data["hints"]:   # 같은 보드를 다시 읽으면 그 출처의 선은 바꾼다
        h["lines"] = [ln for ln in h["lines"] if ln.get("source") != src]
    data["hints"] = [h for h in data["hints"] if h["lines"]]
    check = Image.fromarray(drawn.astype(np.uint8))
    cd = ImageDraw.Draw(check)
    n_new = 0
    for p in lay["panels"]:
        x, y, s = p["x"], p["y"], p["size"]
        sub = stroke[y + 2:y + s - 2, x + 2:x + s - 2]
        nlab, lab, st, _ = cv2.connectedComponentsWithStats(sub)
        keep = [i for i in range(1, nlab) if st[i, cv2.CC_STAT_AREA] >= args.min_area]
        if not keep:
            continue
        ys, xs = np.nonzero(np.isin(lab, keep))
        pts = np.stack([xs + 2, ys + 2], 1).astype(float) / p["scale"] + [p["crop_x0"], p["crop_y0"]]
        a, b, ang, r90 = fit_line(pts)
        span = float(np.linalg.norm(b - a))
        tag = f"run{p['run']} f{p['frame']}"
        if span < args.min_span:
            print(f"{tag}: 선이 짧음({span:.0f}px < {args.min_span}), 건너뜀")
            continue
        if r90 > args.max_wiggle:
            print(f"{tag}: 한 직선으로 보기 어려움(90% 잔차 {r90:.0f}px), 확인 필요 — 그래도 넣음")
        h = next((h for h in data["hints"] if h["run"] == p["run"] and h["frame"] == p["frame"]), None)
        if h is None:
            h = {"run": p["run"], "frame": p["frame"], "visible_end": args.visible_end, "lines": []}
            data["hints"].append(h)
        h["lines"].append({"a": [round(a[0], 1), round(a[1], 1)], "b": [round(b[0], 1), round(b[1], 1)],
                           "angle": round(ang, 1), "source": src})
        n_new += 1
        q = lambda v: (x + (v[0] - p["crop_x0"]) * p["scale"], y + (v[1] - p["crop_y0"]) * p["scale"])
        cd.line([q(a), q(b)], fill=(0, 255, 255), width=3)
        cd.text((x + 8, y + 6), f"{ang:+.0f}°", font=font(20, True), fill=(0, 255, 255))
        print(f"{tag}: {ang:+.1f}° 길이 {span:.0f}px")
    data["note"] = NOTE
    data["hints"].sort(key=lambda h: (h["run"], h["frame"]))
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    cp = PATHS.feat_dir(args.segments, PATHS.load_cfg(args.segments), "lean") / "bar_hints"
    cp.mkdir(parents=True, exist_ok=True)
    chk = cp / f"{Path(args.image).stem}_extracted.png"
    check.save(chk)
    print(f"{n_new}개 선 → {out}  (확인용: {chk})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("segments")
    m.add_argument("--runs", type=int, nargs="*")
    m.add_argument("--step", type=int, default=4, help="뒷모습 프레임 간격")
    m.add_argument("--max", type=int, default=8, help="런당 최대 칸 수")
    m.set_defaults(fn=make)
    e = sub.add_parser("extract")
    e.add_argument("segments")
    e.add_argument("image", help="선을 그린 보드 이미지")
    e.add_argument("--board", required=True, help="make가 만든 보드 이름 (예: bar_hint_draw_run4)")
    e.add_argument("--visible-end", choices=["right", "left", "none"], default="right",
                   help="몸에 가리지 않은 손 (좌회전 뒷모습은 보통 right)")
    e.add_argument("--thresh", type=int, default=120, help="원본과의 RGB 차 합 임계값")
    e.add_argument("--min-area", type=int, default=30, help="획 조각 최소 픽셀")
    e.add_argument("--min-span", type=float, default=60, help="선 최소 길이(크롭 px)")
    e.add_argument("--max-wiggle", type=float, default=25, help="직선에서 벗어난 정도 경고(크롭 px)")
    e.add_argument("--out", help="세션 bar_hints.json 대신 이 파일에 쓴다(있으면 이어서, 없으면 세션 파일을 바탕으로)")
    e.set_defaults(fn=extract)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
