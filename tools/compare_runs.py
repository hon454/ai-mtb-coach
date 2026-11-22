"""같은 피처(예: 좌회전 뱅크)를 여러 번 탄 구간을 나란히 비교한다.

사용법:
    python compare_runs.py <segments.json> [--cols 8] [--slow 4.0]

출력 (segments.json 옆에):
    <feature>_sheet.jpg  : 행=런, 열=구간 진행률(0~100%) 프레임 시트
    <feature>_sync.mp4   : 각 런 구간을 같은 길이로 맞춘 동기화 그리드 영상
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

W, H = 640, 360  # 셀 크기


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"ffmpeg 실패:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")


FONT = "fontfile='C\\:/Windows/Fonts/malgun.ttf'"  # 한글 표시용


def label(text, size=28, x="10", y="10"):
    text = text.replace(":", r"\:").replace("%", "％")
    return (f"drawtext={FONT}:text='{text}':x={x}:y={y}:fontsize={size}:"
            "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6")


def make_sheet(cfg, out, cols):
    vdir = PATHS.video_dir(cfg)
    tmp = out.parent / "_tmp_sheet"
    tmp.mkdir(exist_ok=True)
    rows = []
    for r in cfg["runs"]:
        s, e = r["start"], r["end"]
        cells = []
        for i in range(cols):
            p = i / (cols - 1)
            t = s + (e - s) * p
            f = tmp / f"r{r['run']}_{i}.jpg"
            txt = f"Run {r['run']}  {int(p*100)}%" if i == 0 else f"{int(p*100)}%"
            run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(vdir / r["file"]),
                 "-frames:v", "1", "-vf", f"scale={W}:{H},{label(txt, 26)}", str(f)])
            cells.append(f)
        rows.append(cells)
    inputs = []
    for row in rows:
        for f in row:
            inputs += ["-i", str(f)]
    n = len(rows) * cols
    layout = "|".join(f"{c*W}_{r*H}" for r in range(len(rows)) for c in range(cols))
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
         f"{''.join(f'[{i}:v]' for i in range(n))}xstack=inputs={n}:layout={layout}",
         "-q:v", "3", str(out)])
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def make_sync(cfg, out, slow):
    vdir = PATHS.video_dir(cfg)
    runs = cfg["runs"]
    inputs, chains = [], []
    for i, r in enumerate(runs):
        s, e = r["start"], r["end"]
        dur = e - s
        factor = slow / dur  # 모든 런을 slow초 길이로 맞춤
        inputs += ["-ss", f"{s:.3f}", "-t", f"{dur:.3f}", "-i", str(vdir / r["file"])]
        chains.append(
            f"[{i}:v]setpts=(PTS-STARTPTS)*{factor:.4f},fps=30,scale={W}:{H},"
            f"{label(f'Run {r['run']}  ({dur:.2f}s)')},"
            f"drawtext={FONT}:text='%{{eif\\:t/{slow}*100\\:d}}％':x=w-tw-12:y=10:fontsize=28:"
            f"fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=6[v{i}]")
    n = len(runs)
    cols = 4 if n > 4 else n
    nrows = -(-n // cols)
    # 빈 칸은 검은 화면 + 범례로 채움
    pad = cols * nrows - n
    for j in range(pad):
        chains.append(
            f"color=c=black:s={W}x{H}:d={slow}:r=30,"
            f"{label(cfg.get('feature', ''), 30, '20', '20')},"
            f"{label('같은 구간을 진행률 기준으로 맞춤', 22, '20', '80')}[p{j}]")
    labels = [f"[v{i}]" for i in range(n)] + [f"[p{j}]" for j in range(pad)]
    layout = "|".join(f"{(k % cols)*W}_{(k // cols)*H}" for k in range(len(labels)))
    fc = ";".join(chains) + f";{''.join(labels)}xstack=inputs={len(labels)}:layout={layout}[out]"
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[out]",
         "-t", f"{slow}", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--slow", type=float, default=4.0, help="동기화 영상 길이(초)")
    a = ap.parse_args()
    seg = Path(a.segments)
    cfg = json.loads(seg.read_text(encoding="utf-8"))
    feat = cfg.get("feature", "feature")
    base = PATHS.out_base(seg)
    base.mkdir(parents=True, exist_ok=True)
    make_sheet(cfg, base / f"{feat}_sheet.jpg", a.cols)
    make_sync(cfg, base / f"{feat}_sync.mp4", a.slow)
    print("done:", base)


if __name__ == "__main__":
    main()
