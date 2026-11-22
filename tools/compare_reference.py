"""코치(기준) 라이딩 크롭과 학생 런 크롭을 나란히 놓고 진행률로 맞춘 비교 영상을 만든다.

analyze_pose.py(필수)와 analyze_lean.py(선택, 시점 추정 보강)를 양쪽 segments.json에 먼저 실행해 둘 것.

사용법:
    uv run python tools/compare_reference.py sessions/reference-coach/segments.json \
        sessions/2026-11-21/segments.json [--runs 1 2 3 4 5 7] [--anchor 3=0.51 ...]

출력 (기준 segments.json 옆):
    <feature>_vs_ref_progress.mp4 : 구간 시작~끝을 0~100% 진행률로 선형 정렬
    <feature>_vs_ref_anchor.mp4   : view_ratio 피크(뒷모습이 카메라를 정면으로 향한 순간)를
                                    공통 지점에 고정하고 앞뒤를 구간별로 선형 신축(piecewise-linear)
    <feature>_vs_ref_anchors.txt  : 런별 앵커 진행률/시각

왜 앵커 정렬인가: 구간 경계를 눈으로 정해서 진행률 기준 ±12% 정도 어긋날 수 있다.
앵커 = view_ratio(어깨폭/몸통) 곡선의 최대점. pose(전체 프레임)와 lean(확대 재인식) 두 CSV의
평균을 진행률 격자에서 9% 폭 이동평균한 뒤, 경계 영향(착지 직후·탈출 직전 노이즈)을 피하려고
10~90% 범위에서만 찾는다. 카메라 위치가 다르면 같은 지점이라도 시점비가 달라지므로 대략적인 기준이다.
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

GRID = np.linspace(0, 1, 101)
BIG = 720            # 코치 셀 크기
SMALL = 360          # 학생 셀 크기 (3열 x 2행 = 1080 x 720)
COLS = 3
N_OUT = 150          # 출력 프레임 수 (30fps 기준 5초, 실제보다 느리게 재생)
FPS = 30


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def view_curve(rows):
    """진행률 격자 위 view_ratio 곡선 (빈칸 보간 + 9칸 이동평균). 데이터가 없으면 None."""
    pts = [(float(r["progress"]), float(r["view_ratio"])) for r in rows if r.get("view_ratio")]
    if len(pts) < 5:
        return None
    p, v = map(np.array, zip(*pts))
    vs = np.interp(GRID, p, v)
    return np.convolve(np.pad(vs, 4, mode="edge"), np.ones(9) / 9, mode="valid")


def find_anchor(curves):
    cs = [c for c in curves if c is not None]
    m = np.mean(cs, axis=0)
    win = (GRID >= 0.1) & (GRID <= 0.9)
    return float(GRID[win][m[win].argmax()])


def load_run(seg_path, run):
    cfg = json.loads(Path(seg_path).read_text(encoding="utf-8"))
    feat = cfg.get("feature", "feature")
    base = PATHS.out_base(seg_path)
    r = next(x for x in cfg["runs"] if x["run"] == run)
    pose_dir, lean_dir = base / f"{feat}_pose", base / f"{feat}_lean"
    frames = read_frames(pose_dir / f"run{run}_crop.mp4")
    rows = read_csv(pose_dir / f"run{run}.csv")
    if len(frames) != len(rows):
        raise SystemExit(f"run {run}: 크롭 프레임 {len(frames)} != CSV 행 {len(rows)} ({pose_dir})")
    curves = [view_curve(rows)]
    if (lean_dir / f"run{run}.csv").exists():
        curves.append(view_curve(read_csv(lean_dir / f"run{run}.csv")))
    return dict(run=run, label=r.get("label", f"Run {run}"), start=r["start"], end=r["end"],
                frames=frames, prog=np.array([float(x["progress"]) for x in rows]),
                anchor=find_anchor(curves), feature=feat)


def frame_at(d, p):
    """진행률 p에 가장 가까운 크롭 프레임."""
    return d["frames"][int(np.abs(d["prog"] - p).argmin())]


def warp(u, a_src, a_dst):
    """출력 진행률 u → 해당 런의 진행률. u=a_dst가 런의 앵커 a_src로 가도록 구간별 선형."""
    if u <= a_dst:
        return u / a_dst * a_src
    return a_src + (u - a_dst) / (1 - a_dst) * (1 - a_src)


def label(img, lines, scale, y0=None):
    h = img.shape[0]
    y = y0 if y0 is not None else h - 12 - int(30 * scale) * (len(lines) - 1)
    for txt, col in lines:
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        y += int(30 * scale)


def cell(d, p, size, scale, show_anchor):
    img = cv2.resize(frame_at(d, p), (size, size))
    dur = d["end"] - d["start"]
    near = show_anchor and abs(p - d["anchor"]) < 0.02
    lines = [(d["label"], (255, 255, 255)),
             (f"{p*100:4.0f}%  t+{p*dur:.2f}/{dur:.2f}s", (0, 255, 255))]
    label(img, lines, scale)
    # 아래쪽 진행 막대 + 앵커 표시
    bar_y = size - 4
    cv2.line(img, (0, bar_y), (size, bar_y), (80, 80, 80), 4)
    cv2.line(img, (0, bar_y), (int(p * size), bar_y), (0, 255, 255), 4)
    if show_anchor:
        ax = int(d["anchor"] * size)
        cv2.line(img, (ax, bar_y - 10), (ax, bar_y), (255, 0, 255), 3)
    if near:
        cv2.rectangle(img, (1, 1), (size - 2, size - 2), (255, 0, 255), 4)
        cv2.putText(img, "ANCHOR", (size - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    return img


def render(ref, students, out, mode, a_dst):
    frames = []
    rows_n = -(-len(students) // COLS)
    W, H = BIG + COLS * SMALL, max(BIG, rows_n * SMALL) + 50
    for k in range(N_OUT):
        u = k / (N_OUT - 1)
        canvas = np.zeros((H, W, 3), np.uint8)

        def prog(d):
            return u if mode == "progress" else warp(u, d["anchor"], a_dst)

        canvas[50:50 + BIG, :BIG] = cell(ref, prog(ref), BIG, 1.0, mode == "anchor")
        for j, d in enumerate(students):
            r_, c_ = divmod(j, COLS)
            y0, x0 = 50 + r_ * SMALL, BIG + c_ * SMALL
            canvas[y0:y0 + SMALL, x0:x0 + SMALL] = cell(d, prog(d), SMALL, 0.6, mode == "anchor")
        title = (f"progress sync {u*100:5.1f}%" if mode == "progress"
                 else f"anchored on view_ratio peak (common point {a_dst*100:.0f}%)  u={u*100:5.1f}%")
        cv2.putText(canvas, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        frames.append(canvas)
    tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="코치 segments.json (첫 번째 런 사용)")
    ap.add_argument("student", help="학생 segments.json")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3, 4, 5, 7],
                    help="학생 런 (기본: 카메라 위치가 다른 6번 제외)")
    ap.add_argument("--anchor", nargs="*", default=[], help="앵커 수동 지정, 예: 0=0.40 3=0.51")
    a = ap.parse_args()

    ref_cfg = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    ref = load_run(a.reference, ref_cfg["runs"][0]["run"])
    students = [load_run(a.student, rn) for rn in a.runs]
    manual = {int(k): float(v) for k, v in (s.split("=") for s in a.anchor)}
    for d in [ref] + students:
        if d["run"] in manual:      # 코치는 run 0이라 학생 번호와 겹치지 않음
            d["anchor"] = manual[d["run"]]

    out_dir = PATHS.out_base(a.reference)
    feat = ref["feature"]
    a_dst = float(np.mean([d["anchor"] for d in [ref] + students]))
    lines = [f"공통 앵커 위치 (출력 진행률): {a_dst:.2f}"]
    for d in [ref] + students:
        t = d["start"] + d["anchor"] * (d["end"] - d["start"])
        lines.append(f"{d['label']:>8}: anchor progress {d['anchor']:.2f}  t={t:.2f}s  "
                     f"(segment {d['start']:.2f}-{d['end']:.2f}s, {d['end'] - d['start']:.2f}s)")
    print("\n".join(lines))
    (out_dir / f"{feat}_vs_ref_anchors.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    render(ref, students, out_dir / f"{feat}_vs_ref_progress.mp4", "progress", a_dst)
    render(ref, students, out_dir / f"{feat}_vs_ref_anchor.mp4", "anchor", a_dst)
    print("done:", out_dir)


if __name__ == "__main__":
    main()
