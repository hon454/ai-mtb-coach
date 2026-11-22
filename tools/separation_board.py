"""자전거 vs 상체 기울기 보드 (좌회전 뱅크).

--bike gvhmr (기본, 2026-11-23부터): GVHMR 3D(중력 기준) 값.
    자전거 = grip_roll(왼손목–오른손목 선 = 핸들바 근사), 상체 = torso_roll(골반→목). 분리각 = grip_roll - torso_roll.
    가로축은 골반 방향이 구간 시작부터 왼쪽으로 돈 각도(body3d_gvhmr.turn_of). 회전 15·30·45°(±5°) 구간의 프레임별 분리각 중앙값을 런끼리 비교.
    고른 이유: 그린 핸들바 선 45프레임과 평균 3.3° 차이로 가장 가깝고, 마스크 변화에 흔들리지 않음(LOG 2026-11-23).
    대표 사진 = 회전 30° 구간에서 분리각이 중앙값에 가까운 프레임의 GVHMR 메시 오버레이.
--bike fit (예전 방식): 뒷모습 구간의 bike_fit 3D 맞추기 lean vs 2D 포즈 상체.
    선택 기준: view_ratio > 0.75, |fit_yaw| < 25, fit_iou >= 0.55, fit_lowconf가 아닌 프레임. 분리각 = fit_lean_s - body_lean.

사용법:
    uv run python tools/separation_board.py sessions/2026-11-21/segments.json --ref sessions/reference-coach/segments.json
    uv run python tools/separation_board.py sessions/2026-11-21/segments.json --bike fit --runs 1 2 3 4 5 7
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


# ---------------------------------------------------------------- GVHMR 모드
TURNS, TURN_WIN, TURN_MAX = (15, 30, 45), 5, 60


def gvhmr_item(seg, run):
    """런 하나의 GVHMR 값: 프레임별 회전·자전거(손목선)·상체 롤과 회전 단계별 중앙값."""
    import body3d_gvhmr as G
    cfg = PATHS.load_cfg(seg)
    od = G.out_dir(seg, cfg)
    rows = G.read_csv(od / f"run{run}.csv")
    z = np.load(od / f"gvhmr_run{run}.npz")
    turn = G.turn_of(z, rows)

    def num(k):
        return np.array([G.fnum(x.get(k)) if G.fnum(x.get(k)) is not None else np.nan for x in rows])

    grip, torso, view = num("grip_roll"), num("torso_roll"), num("cam_view_signed")
    sep = grip - torso
    stats = {}
    for T in TURNS:
        m = (np.abs(turn - T) <= TURN_WIN) & np.isfinite(sep)
        stats[T] = dict(n=int(m.sum()), mask=m,
                        bike=float(np.median(grip[m])) if m.any() else None,
                        body=float(np.median(torso[m])) if m.any() else None,
                        sep=float(np.median(sep[m])) if m.any() else None,
                        view=float(np.median(view[m])) if m.any() else None)
    return dict(G=G, cfg=cfg, od=od, rows=rows, z=z, turn=turn, grip=grip, torso=torso, sep=sep, stats=stats)


def gvhmr_tile(seg, run, d, name, size):
    """회전 30° 구간에서 분리각이 중앙값에 가까운 프레임(메시가 있는 프레임 중)에 메시·손목선을 그린 사진."""
    import cv2
    G, cfg, z = d["G"], d["cfg"], d["z"]
    vf = [int(v) for v in z["verts_frames"]]
    seqs = np.array([int(x["seq_frame"]) for x in d["rows"]])
    has = np.isin(seqs, vf) & np.isfinite(d["sep"])
    st = d["stats"][30]
    cand = np.nonzero(st["mask"] & has)[0]
    if len(cand):
        k = cand[np.argmin(np.abs(d["sep"][cand] - st["sep"]))]
    else:
        hk = np.nonzero(has)[0]
        k = hk[np.argmin(np.abs(d["turn"][hk] - 30))]
    rr = next(x for x in cfg["runs"] if x["run"] == run)
    off = G.seek_offset(PATHS.video_dir(cfg) / rr["file"], rr["start"])
    crow, extra = G.compute_run(d["od"] / f"gvhmr_run{run}.npz", G.lean_rows_of(seg, cfg, run), raw=False, off=off)
    seq = int(seqs[k])
    row = next(x for x in crow if x["seq_frame"] == seq)
    cap = cv2.VideoCapture(str(PATHS.video_dir(cfg) / rr["file"]))
    img = None
    for _ in range(seq + 1):
        ok, img = cap.read()
    cap.release()
    i = extra["idx"][seq]
    head = (f"{name} · 회전 {d['turn'][k]:.0f}° · f{d['rows'][k]['frame']}",
            f"자전거 {d['grip'][k]:+.0f}°  상체 {d['torso'][k]:+.0f}°  분리각 {d['sep'][k]:+.0f}°")
    return G.render_tile(img, z["verts_c"][vf.index(seq)], z["faces"], z["K"], extra["Jc"][i], extra["up_c"][i],
                         row, z["boxes"][i], tile=size, line="grip", header=head)


def board_gvhmr(a, seg, runs, out):
    items = []
    if a.ref:
        items.append((ROOT / a.ref, 0, "코치님", True))
    items += [(seg, r, f"Run {r}", False) for r in runs]
    CW = 900
    head, foot = 170, 200
    W, H = CELL + 40 + CW + 30, head + ROW * len(items) + foot
    board = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(board)
    dr.text((24, 16), "좌회전 뱅크 · 자전거 vs 상체 기울기 (GVHMR 3D, 중력 기준)", font=f(34, True), fill=(255, 255, 255))
    dr.text((26, 66), "노랑 = 자전거 기울기(3D 손목선 = 핸들바 근사)   초록 = 상체 롤(골반→목)   파랑 숫자 = 분리각(자전거 - 상체)",
            font=f(18), fill=(210, 210, 210))
    dr.text((26, 94), "가로축 = 골반 방향이 구간 시작부터 왼쪽으로 돈 각도. 같은 회전 단계(15·30·45° ±5°)끼리 비교. 분리각 +면 자전거를 몸보다 더 눕힘.",
            font=f(18), fill=(210, 210, 210))
    dr.text((26, 122), "왼쪽 사진: 회전 30° 근처 프레임에 GVHMR 메시를 겹침 (초록 화살표 = 중력 위, 빨강 = 상체, 주황 = 손목선)",
            font=f(16), fill=(160, 160, 160))
    cx0, cx1 = CELL + 200, W - 40

    def X(t):
        return cx0 + (cx1 - cx0) * t / TURN_MAX

    summary = []
    for i, (s, r, name, is_ref) in enumerate(items):
        y0 = head + i * ROW
        d = gvhmr_item(s, r)
        tile = gvhmr_tile(s, r, d, name, CELL)
        if is_ref:
            ImageDraw.Draw(tile).rectangle([1, 1, CELL - 2, CELL - 2], outline=GOLD, width=5)
        board.paste(tile, (0, y0))
        dr.text((CELL + 30, y0 + 30), name, font=f(28, True), fill=GOLD if is_ref else (255, 255, 255))
        ns = "/".join(str(d["stats"][T]["n"]) for T in TURNS)
        dr.text((CELL + 30, y0 + 74), f"프레임 {ns}", font=f(15), fill=(160, 160, 160))
        dr.text((CELL + 30, y0 + 96), "(15/30/45°)", font=f(13), fill=(130, 130, 130))
        cy0, cy1 = y0 + 20, y0 + ROW - 92

        def Y(v):
            return cy1 - (cy1 - cy0) * (v + 10) / 60   # -10..50°

        for T in TURNS:
            dr.rectangle([X(T - TURN_WIN), cy0, X(T + TURN_WIN), cy1], fill=(34, 37, 44))
        for g in (0, 20, 40):
            dr.line([cx0, Y(g), cx1, Y(g)], fill=(60, 62, 68), width=1)
            dr.text((cx0 - 34, Y(g) - 9), f"{g}°", font=f(13), fill=(120, 120, 120))
        for t in range(0, TURN_MAX + 1, 15):
            dr.text((X(t) - 10, cy1 + 4), f"{t}°", font=f(13), fill=(120, 120, 120))
        m = d["turn"] <= TURN_MAX
        o = np.argsort(d["turn"][m], kind="stable")
        for vals, col in ((d["torso"], BODY_C), (d["grip"], BIKE_C)):
            pts = [(X(t), Y(v)) for t, v in zip(d["turn"][m][o], vals[m][o]) if np.isfinite(v)]
            if len(pts) > 1:
                dr.line(pts, fill=col, width=2)
            for q in pts:
                dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=col)
        for T in TURNS:
            st = d["stats"][T]
            tx = X(T) - 60
            if st["sep"] is None:
                dr.text((tx, cy1 + 26), "프레임 없음", font=f(16), fill=(150, 150, 150))
                continue
            dr.text((tx, cy1 + 24), f"분리각 {st['sep']:+.0f}°", font=f(22, True), fill=SEP_C)
            dr.text((tx, cy1 + 56), f"자전거 {st['bike']:.0f} 상체 {st['body']:.0f} 시점 {st['view']:+.0f}°",
                    font=f(13), fill=(150, 150, 150))
        summary.append((name, d["stats"]))
        dr.line([0, y0 + ROW - 1, W, y0 + ROW - 1], fill=(50, 52, 58), width=1)
    note = ("측정값(GVHMR 3D). 손목선은 사용자가 그린 핸들바 선 45프레임과 평균 3.3° 차이(최대 16°). 상체 롤은 정답으로 검증하지 못함. "
            "회전 단계마다 프레임이 2~9개뿐이라 런별 값은 ±3~5° 흔들릴 수 있음 → 큰 차이만 본다.\n"
            "뒷모습에서 몸에 가린 손목은 모델이 추정한 위치. 시점(0° = 바로 뒤)이 옆으로 갈수록 롤 추정이 약해짐.")
    if a.ref:
        note += "\n코치님: 촬영 자리가 달라 같은 회전에서 시점이 더 옆쪽이다(시점 숫자 참고). 직접 비교는 참고만."
    if 6 in runs:
        note += " Run 6: 다른 촬영 자리."
    yy = H - foot + 14
    for para in note.split("\n"):
        line = ""
        for ch in para:
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
    for name, st in summary:
        parts = []
        for T in TURNS:
            x = st[T]
            parts.append(f"{T}° n={x['n']} " + (f"bike {x['bike']:.1f} body {x['body']:.1f} sep {x['sep']:+.1f} view {x['view']:+.0f}"
                                                 if x["sep"] is not None else "-"))
        print(f"  {name}: " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--bike", choices=["gvhmr", "fit"], default="gvhmr",
                    help="gvhmr: GVHMR 손목선 vs 3D 상체 롤(기본). fit: bike_fit lean vs 2D 상체(예전 방식)")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="기본: gvhmr 1~7, fit 1 2 3 4 5 7")
    ap.add_argument("--ref", default=None, help="코치 segments.json (run 0을 첫 줄에 추가)")
    ap.add_argument("--min-iou", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seg = ROOT / a.segments
    if a.bike == "gvhmr":
        out = Path(a.out) if a.out else PATHS.out_base(seg) / "insights" / "left_bank_separation.png"
        return board_gvhmr(a, seg, a.runs or [1, 2, 3, 4, 5, 6, 7], out)
    a.runs = a.runs or [1, 2, 3, 4, 5, 7]
    out = Path(a.out) if a.out else PATHS.out_base(seg) / "insights" / "left_bank_separation_fit.png"
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
