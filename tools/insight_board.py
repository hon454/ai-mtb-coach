"""인사이트를 '장면별 비교 보드' 이미지로 만든다 (코치 vs 내 런, 같은 순간끼리).

사용법:
    uv run python tools/insight_board.py sessions/2026-11-21/insights/left_bank_timing.json

spec JSON:
{
  "title": "...", "subtitle": "...",
  "ref": "sessions/reference-coach/segments.json",
  "student": "sessions/2026-11-21/segments.json",
  "runs": [1, 2, 3, 4, 5, 7],
  "anchor_override": {"0": 0.40},          # 선택
  "phases": [{"u": 0.40, "title": "...", "note": "..."}],
  "footer": "...",
  "out": "sessions/2026-11-21/insights/left_bank_timing.png"
}
u = 앵커 정렬 진행률(compare_reference.py와 같은 축, 공통 앵커 = 평균 앵커).
셀 이미지: <feature>_lean/crops 원본 크롭 위에 뼈대·상체선·수직 기준선을 새로 그림.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402
import compare_reference as CR  # noqa: E402
import analyze_lean as AL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
CELL = 300
TEXT_W = 430
GOLD = (255, 196, 0)


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def lean_frames(seg_path, run):
    """lean 폴더의 (진행률, 크롭 경로) 목록."""
    cfg = json.loads(Path(seg_path).read_text(encoding="utf-8"))
    d = PATHS.feat_dir(seg_path, cfg, "lean")
    rows = CR.read_csv(d / f"run{run}.csv")
    return [(float(r["progress"]), d / "crops" / f"run{run}_f{int(r['frame']):04d}.jpg") for r in rows]


def draw_cell(img_path, pose_m, label, sub, is_ref):
    img = cv2.imread(str(img_path))
    pr = pose_m.predict(img, imgsz=AL.CROP, classes=[0], verbose=False)[0]
    pi = AL.pick_center_person(pr)
    if pi is not None:
        k = pr.keypoints.xy[pi].cpu().numpy()
        c = pr.keypoints.conf[pi].cpu().numpy()
        for a, b in AL.SKELETON:
            if c[a] > 0.3 and c[b] > 0.3:
                cv2.line(img, tuple(map(int, k[a])), tuple(map(int, k[b])), (255, 255, 0), 3, cv2.LINE_AA)
        sh, hp = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2
        # 수직 기준선(점선) + 상체선(엉덩이→어깨, 연장)
        for y in range(int(hp[1]) - 260, int(hp[1]), 14):
            cv2.line(img, (int(hp[0]), y), (int(hp[0]), y + 7), (255, 255, 255), 2, cv2.LINE_AA)
        top = sh + (sh - hp) * 0.6
        cv2.line(img, tuple(map(int, hp)), tuple(map(int, top)), (60, 230, 60), 6, cv2.LINE_AA)
    img = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    dr.rectangle([0, CELL - 50, CELL, CELL], fill=(0, 0, 0))
    dr.text((10, CELL - 48), label, font=font(22, True), fill=GOLD if is_ref else (255, 255, 255))
    dr.text((10, CELL - 22), sub, font=font(15), fill=(200, 200, 200))
    if is_ref:
        dr.rectangle([1, 1, CELL - 2, CELL - 2], outline=GOLD, width=5)
    return pil


def wrap_text(dr, text, f, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            if dr.textlength(cur + ch, font=f) > width:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
    return lines


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    pose_m = YOLO(str(ROOT / "models" / "yolo11x-pose.pt"))
    ref_seg, stu_seg = ROOT / spec["ref"], ROOT / spec["student"]
    ref = CR.load_run(ref_seg, 0)
    studs = [CR.load_run(stu_seg, r) for r in spec["runs"]]
    for d in [ref] + studs:
        if str(d["run"]) in spec.get("anchor_override", {}):
            d["anchor"] = spec["anchor_override"][str(d["run"])]
    a_dst = float(np.mean([d["anchor"] for d in [ref] + studs]))
    cols = [(ref, ref_seg, True)] + [(d, stu_seg, False) for d in studs]

    W = TEXT_W + CELL * len(cols)
    head_h, foot_h = 120, 70
    H = head_h + CELL * len(spec["phases"]) + foot_h
    board = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(board)
    dr.text((24, 18), spec["title"], font=font(38, True), fill=(255, 255, 255))
    dr.text((26, 72), spec.get("subtitle", ""), font=font(19), fill=(180, 180, 180))

    for i, ph in enumerate(spec["phases"]):
        y0 = head_h + i * CELL
        dr.rectangle([0, y0, TEXT_W - 1, y0 + CELL - 1], fill=(34, 37, 43))
        dr.text((22, y0 + 18), f"{i + 1}. {ph['title']}", font=font(26, True), fill=GOLD)
        f = font(19)
        yy = y0 + 64
        for line in wrap_text(dr, ph["note"], f, TEXT_W - 44):
            dr.text((22, yy), line, font=f, fill=(225, 225, 225))
            yy += 28
        for j, (d, seg, is_ref) in enumerate(cols):
            p = CR.warp(ph["u"], d["anchor"], a_dst)
            fr = lean_frames(seg, d["run"])
            path = min(fr, key=lambda x: abs(x[0] - p))[1]
            dur = d["end"] - d["start"]
            lab = "코치님" if is_ref else f"Run {d['run']}"
            cell = draw_cell(path, pose_m, lab, f"구간 {p*100:.0f}% · {p*dur:.2f}s", is_ref)
            board.paste(cell, (TEXT_W + j * CELL, y0))
    dr.text((24, H - foot_h + 14), spec.get("footer", ""), font=font(17), fill=(160, 160, 160))
    out = ROOT / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out, quality=92)
    print("saved:", out)


if __name__ == "__main__":
    main()
