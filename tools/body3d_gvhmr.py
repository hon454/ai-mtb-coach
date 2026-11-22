"""GVHMR(중력-시점 3D 사람 복원)로 시점에 무관한 상체·자전거 각도를 잰다.

GVHMR(https://github.com/zju3dv/GVHMR)은 움직이는 카메라 영상에서 SMPL-X 몸을 중력 기준 세계 좌표로 복원한다.
2D `body_lean`은 라이더가 뒷모습→옆모습으로 돌면 앞숙임이 기울기에 섞인다. 여기서는 3D 관절로 각도를 다시 잰다.

세 단계. 단계마다 쓰는 파이썬 환경이 다르다.

1) infer — GVHMR 환경(WSL2 Ubuntu, `$GVHMR_HOME/.venv`)에서 실행한다. 구간 앞뒤 여유를 둔 창을 잘라
   라이더 추적 → ViTPose → HMR2 특징 → SimpleVO(카메라 회전, 이동 카메라 경로) → GVHMR 예측을 하고
   `outputs/<세션>/<feature>_gvhmr/gvhmr_run<N>.npz`로 관절·메시·카메라를 내보낸다.
   `--stage prep`은 SMPL 파일 없이 되는 전처리까지만 한다. 중간 산출물은 `$GVHMR_HOME/outputs/aimtb/`.

       # WSL 안에서, 저장소 폴더에서 (데이터 루트는 /mnt/<드라이브>/<경로> 형식)
       export GVHMR_HOME=~/gvhmr/GVHMR AIMTB_DATA_ROOT=/mnt/<드라이브>/<경로>/ai-mtb-coach
       $GVHMR_HOME/.venv/bin/python tools/body3d_gvhmr.py infer sessions/2026-11-21/segments.json --runs 1 2
       # 초점거리만 바꾼 변형: --f-mm 50 --suffix _f50  (추적·ViTPose·특징은 재사용)

   설치: GVHMR docs/INSTALL.md를 따르되 WSL에서 uv venv(Python 3.10) + requirements(cython_bbox 제외),
   chumpy는 --no-build-isolation, opencv-python 대신 opencv-python-headless. SMPL-X는
   `$GVHMR_HOME/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` (사용자가 등록 후 직접 받음).

2) analyze — 저장소 환경(uv)에서 실행. npz에서 각도를 계산해 `run<N>.csv`와 메시 오버레이 몽타주를 만든다.
   `--raw`는 GVHMR 후처리를 끈 결과, `--suffix`는 infer 변형 결과를 읽는다.

       uv run python tools/body3d_gvhmr.py analyze sessions/2026-11-21/segments.json --ref sessions/reference-coach/segments.json

   prepcheck — infer --stage prep 결과(라이더 박스·ViTPose)를 확인하는 보드. Windows에서 GVHMR_HOME을
   `\\\\wsl.localhost\\<배포판>\\home\\<사용자>\\gvhmr\\GVHMR`처럼 WSL 경로로 준다.

3) board — 2D·실루엣 값과 비교하는 보드 `compare_board.png`.

       uv run python tools/body3d_gvhmr.py board sessions/2026-11-21/segments.json --ref sessions/reference-coach/segments.json

좌표와 각도 정의 (SAM 3D Body 세션과 같은 열 이름)
- 세계 좌표: GVHMR 전역 출력(smpl_params_global). +y = 중력 반대(위). 단위 m.
  SMPL-X 관절: 0 골반, 1 왼엉덩이, 2 오른엉덩이, 7·8 왼·오른발목, 12 목, 15 머리, 20·21 왼·오른손목.
- 진행 방향 f: 골반이 향한 방향(GVHMR 전역 회전 × 몸 +z)의 수평 성분, 단위벡터. heading_src=facing.
  원래 계획은 세계 골반 궤적의 속도 방향이었다. 그런데 GVHMR은 페달 위에 정지 자세로 선 라이더를 거의 움직이지 않는
  사람으로 복원한다(창 5초 동안 수평 이동 0.5~1.2 m, 실제는 20 m 이상). 그래서 궤적 방향을 쓸 수 없다.
  카메라 기준 궤적(VO 회전 × incam 골반 위치)도 깊이 추정이 흔들려 방향 IQR이 30~150°라 쓰지 않는다.
  궤적 방식은 travel_dir(mode="traj")로 남겨 둔다(속도 0.5 m/s 미만이면 facing).
  교차 점검: 손목선(왼손목−오른손목)에 수직인 수평 방향 = 핸들 기준 자전거 방향. heading_bar_diff에 기록한다.
- 라이더 왼쪽 l = up × f (오른손 좌표계, SMPL은 왼엉덩이가 +x). 자기 점검: (왼엉덩이−오른엉덩이)·l > 0.
- torso_roll: 골반→목 벡터 v를 f에 수직인 평면에 투영했을 때 중력 수직과의 각. atan2(v·l, v·up). + = 라이더 왼쪽.
- torso_pitch: v를 (f, up) 평면에서 본 앞숙임. atan2(v·f, v·up). + = 앞으로 숙임.
- bike_lean_body: 손목 중점 W(그립)와 발목 중점 A(페달·BB)를 지나고 f를 품는 평면의 중력 기준 기울기.
  s = W − A를 f에 수직인 평면에 투영한 방향 b에 대해 atan2(s·l, s·up). + = 왼쪽.
- hip_offset_mm: 골반 중심(관절 0)이 그 자전거 중심 평면에서 떨어진 거리. 평면 법선 n = f × b(라이더 오른쪽).
  (골반 − A)·n × 1000. + = 라이더 오른쪽 = 좌회전의 바깥.
- b3d_conf: ViTPose 2D 신뢰도 평균(어깨·엉덩이·손목·발목 COCO 5,6,9,10,11,12,15,16). 0~1.

보조 열
- separation_b3d = bike_lean_body − torso_roll (+ = 자전거를 몸보다 더 눕힘).
- grip_roll: 오른손목→왼손목 선의 수평면 대비 기울기. asin(−(Lw−Rw)·up/|Lw−Rw|). 진행 방향이 필요 없는 교차 점검용.
  + = 왼손이 낮음 = 자전거가 왼쪽으로 기움(바가 자전거 평면에 수직이라고 가정).
- heading_src(facing/traj), travel_vs_facing(진행 방향 대비 골반이 향한 방향의 yaw 차, + = 왼쪽),
  speed_mps(GVHMR 세계 골반 수평 속도 — 실제 속도가 아님, 궤적이 멈춰 있음을 보이는 진단값).
- heading_bar_diff: 골반 방향 → 손목선 기준 방향 yaw 차(+ = 왼쪽). bike_lean_barhead, torso_roll_barhead:
  진행 방향으로 손목선 기준 방향을 썼을 때의 값(진행 방향 정의에 따른 차이를 보는 용도).
- frame: 기존 left_bank_lean CSV의 번호(그 CSV와 join용). seq_frame: 영상을 처음부터 순차 디코드한 번호.
  analyze_lean.py의 OpenCV 시크가 코치님 클립에서 한 프레임 앞에 떨어져 seq_frame = frame − 1 (런 1~7은 같음).
- roll_sens15, bike_sens15: 진행 방향을 ±15° 돌렸을 때 torso_roll, bike_lean_body가 바뀌는 최대 크기(진행 방향 오차 민감도).
- cam_view_deg: 카메라 광축의 수평 성분과 f 사이의 각. 0 = 바로 뒤에서 봄, 90 = 옆, 180 = 앞.
  cam_view_signed: 같은 각에 부호. + = 광축이 라이더 왼쪽으로 돎 = 카메라가 라이더 오른쪽 뒤에 있음.
  카메라가 수평이고 상체 롤이 0이면 앞숙임 p가 화면에 −atan(tan p · sin ψ)만큼 기울기로 보인다(2D 혼입의 크기).
  카메라 회전은 같은 프레임의 incam 관절과 global 관절을 Kabsch로 맞춰 얻는다.
- cam_roll_deg: GVHMR가 본 중력 위쪽 방향이 화면에서 수직과 이루는 각. + = 화면 오른쪽으로 기움.
  2D body_lean은 카메라가 수평이라고 가정하므로, 뒷모습에서 body_lean + cam_roll_deg ≈ 카메라 기울기를 뺀 2D 값.
- limb_conf: 손목·발목 ViTPose 신뢰도 평균. reproj_err: ViTPose(신뢰도>0.5)와 GVHMR COCO 관절 투영의 중앙값 거리 / 박스 높이.

전제·한계: GVHMR은 걷기·서기 데이터로 학습했다. 페달 위의 발, 뱅크에 가린 다리, 작은 라이더에서 틀릴 수 있다.
초점거리는 모름 → `--f-mm`(기본 auto: 배경 SIFT 호모그래피로 추정, 실패 시 26mm). 몽타주로 메시가 몸에 맞는지 확인한다.
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRE, POST = 2.0, 1.5          # 구간 앞뒤 여유(초). GVHMR 시간 모델과 VO에 문맥을 준다.
F_MM_FALLBACK = 26.0          # 폰 기본 카메라 35mm 환산 초점거리 근사
UP = np.array([0.0, 1.0, 0.0])
J_PELVIS, J_LHIP, J_RHIP, J_LANK, J_RANK, J_NECK, J_HEAD, J_LWR, J_RWR = 0, 1, 2, 7, 8, 12, 15, 20, 21
COCO_CONF = [5, 6, 9, 10, 11, 12, 15, 16]
COCO_LIMB = [9, 10, 15, 16]


def out_dir(seg, cfg):
    d = PATHS.feat_dir(seg, cfg, "gvhmr")
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_csv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ============================================================ 1) infer (GVHMR env)

def read_window(video, f0, f1):
    """소스 영상을 처음부터 순차로 읽어 f0..f1 프레임을 RGB로 돌려준다(시크 오차 없음)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, fi = [], 0
    while fi <= f1:
        ok, img = cap.read()
        if not ok:
            break
        if fi >= f0:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fi += 1
    cap.release()
    return frames, fps, n


def track_people(frames, yolo_path):
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    hist = []
    for img in frames:
        r = yolo.track(np.ascontiguousarray(img[..., ::-1]), persist=True, imgsz=1280, conf=0.25, classes=[0], verbose=False,
                       tracker="bytetrack.yaml", device="cuda")[0]
        dets = {}
        if r.boxes.id is not None:
            for tid, b in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.cpu().numpy()):
                dets[int(tid)] = b.astype(np.float32)
        hist.append(dets)
    del yolo
    return hist


def _close(b1, b2):
    c1, c2 = (b1[:2] + b1[2:]) / 2, (b2[:2] + b2[2:]) / 2
    h1, h2 = b1[3] - b1[1], b2[3] - b2[1]
    return np.linalg.norm(c1 - c2) < 0.6 * max(h1, h2) and 0.5 < h1 / max(h2, 1) < 2.0


def pick_rider(hist, lo, hi, force_id=None):
    """구간 [lo,hi] 안에서 박스 면적 합이 가장 큰 트랙(analyze_lean과 같은 기준) + 끊긴 트랙 이어붙이기."""
    tracks = defaultdict(dict)
    for i, dets in enumerate(hist):
        for tid, b in dets.items():
            tracks[tid][i] = b
    if not tracks:
        return None, [], {}
    if force_id is None:
        area = {tid: sum((b[2] - b[0]) * (b[3] - b[1]) for i, b in tr.items() if lo <= i <= hi) for tid, tr in tracks.items()}
        main = max(area, key=area.get)
    else:
        main = force_id
    chain, used = dict(tracks[main]), [main]
    changed = True
    while changed:
        changed = False
        first, last = min(chain), max(chain)
        for tid, tr in tracks.items():
            if tid in used:
                continue
            fs, ls = min(tr), max(tr)
            if last < fs <= last + 15 and _close(chain[last], tr[fs]):
                chain.update(tr); used.append(tid); changed = True; break
            if first - 15 <= ls < first and _close(tr[ls], chain[first]):
                chain.update(tr); used.append(tid); changed = True; break
    return main, used, chain


def fill_track(chain, a, b):
    idx = np.array(sorted(chain))
    boxes = np.stack([chain[i] for i in idx])
    t = np.arange(a, b + 1)
    out = np.stack([np.interp(t, idx, boxes[:, k]) for k in range(4)], 1)
    for _ in range(2):  # GVHMR get_one_track과 같은 이동평균(5) 두 번
        pad = np.pad(out, ((2, 2), (0, 0)), mode="edge")
        out = np.stack([np.convolve(pad[:, k], np.ones(5) / 5, mode="valid") for k in range(4)], 1)
    return out.astype(np.float32)


def estimate_f_mm(frames, boxes, step=6, scale=0.5):
    """패닝 영상의 배경 호모그래피 H ≈ K R K⁻¹ 에서 초점거리를 고른다(라이더 박스는 가림).
    회전이 2° 이상인 프레임 쌍에서 K⁻¹HK가 회전행렬에 가장 가까운 f. 실패하면 None."""
    import cv2
    H_img, W_img = frames[0].shape[:2]
    sift = cv2.SIFT_create(3000)
    bf = cv2.BFMatcher()
    Hs = []
    for i in range(0, len(frames) - step, step):
        g = []
        for j in (i, i + step):
            im = cv2.cvtColor(cv2.resize(frames[j], None, fx=scale, fy=scale), cv2.COLOR_RGB2GRAY)
            m = np.full(im.shape, 255, np.uint8)
            x0, y0, x1, y1 = (boxes[j] * scale).astype(int)
            hgt = y1 - y0
            cv2.rectangle(m, (x0 - hgt // 2, y0 - hgt // 4), (x1 + hgt // 2, y1 + hgt // 2), 0, -1)
            g.append(sift.detectAndCompute(im, m))
        (k1, d1), (k2, d2) = g
        if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
            continue
        good = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
        if len(good) < 60:
            continue
        p1 = np.float32([k1[m.queryIdx].pt for m in good]) / scale
        p2 = np.float32([k2[m.trainIdx].pt for m in good]) / scale
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        if H is not None and inl.sum() > 50:
            Hs.append(H)
    if len(Hs) < 4:
        return None, {}
    diag = math.hypot(W_img, H_img)
    cands = np.arange(14, 160, 1.0)
    errs = []
    for fmm in cands:
        f = diag / math.hypot(24, 36) * fmm
        K = np.array([[f, 0, W_img / 2], [0, f, H_img / 2], [0, 0, 1]])
        Ki = np.linalg.inv(K)
        e = []
        for H in Hs:
            M = Ki @ H @ K
            M /= np.cbrt(np.linalg.det(M))
            ang = math.degrees(math.acos(np.clip((np.trace(M) - 1) / 2, -1, 1)))
            if ang < 2:
                continue
            e.append(np.linalg.norm(M.T @ M - np.eye(3)))
        errs.append(np.median(e) if e else np.inf)
    errs = np.array(errs)
    if not np.isfinite(errs).any():
        return None, {}
    k = int(np.argmin(errs))
    edge = k in (0, len(cands) - 1)
    return (None if edge else float(cands[k])), {"n_pairs": len(Hs), "err": float(errs[k]), "edge": edge}


def cmd_infer(a):
    import torch
    gh = Path(os.environ["GVHMR_HOME"]).expanduser().resolve()
    seg = Path(a.segments).resolve()
    cfg = PATHS.load_cfg(seg)
    vdir = PATHS.video_dir(cfg)
    od = out_dir(seg, cfg)
    work_root = gh / "outputs" / "aimtb" / PATHS.session_name(seg)
    os.chdir(gh)                      # GVHMR은 inputs/checkpoints/... 상대경로를 쓴다
    sys.path.insert(0, str(gh))
    from hmr4d.utils.video_io_utils import get_writer
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, create_camera_sensor

    runs = [r for r in cfg["runs"] if not a.runs or r["run"] in a.runs]
    force = dict(map(int, s.split(":")) for s in (a.track_id or []))
    metas = {}
    # ---- stage A: 창 자르기, 라이더 추적, 초점거리 (런마다)
    for r in runs:
        wd = work_root / f"run{r['run']}"
        wd.mkdir(parents=True, exist_ok=True)
        meta_p = wd / "meta.json"
        if meta_p.exists() and not a.force:
            metas[r["run"]] = json.loads(meta_p.read_text())
            continue
        src = vdir / r["file"]
        import cv2
        cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        s_f, e_f = round(r["start"] * fps), round(r["end"] * fps)
        f0, f1 = max(0, round((r["start"] - a.pre) * fps)), min(n - 1, round((r["end"] + a.post) * fps))
        frames, fps, n = read_window(src, f0, f1)
        f1 = f0 + len(frames) - 1
        hist = track_people(frames, gh / "inputs/checkpoints/yolo/yolov8x.pt")
        main, used, chain = pick_rider(hist, s_f - f0, e_f - f0, force.get(r["run"]))
        if not chain:
            print(f"run {r['run']}: 라이더 트랙 없음"); continue
        ia, ib = min(chain), max(chain)
        cover = sum(1 for i in range(s_f - f0, e_f - f0 + 1) if i in chain) / (e_f - s_f + 1)
        boxes = fill_track(chain, ia, ib)
        frames = frames[ia:ib + 1]
        f0_orig = f0
        f0, f1 = f0 + ia, f0 + ib
        if a.f_mm == "auto":
            fmm, finfo = estimate_f_mm(frames, boxes)
            if fmm is None:
                fmm, finfo["fallback"] = F_MM_FALLBACK, True
        else:
            fmm, finfo = float(a.f_mm), {"manual": True}
        clip = wd / "clip.mp4"
        w = get_writer(str(clip), fps=30, crf=17)
        for img in frames:
            w.write_frame(img)
        w.close()
        H, W = frames[0].shape[:2]
        meta = dict(run=r["run"], src=r["file"], fps=fps, f0=f0, f1=f1, f0_orig=f0_orig, seg_f0=s_f, seg_f1=e_f, W=W, H=H,
                    track_main=main, track_used=used, seg_cover=round(cover, 3), f_mm=fmm, f_info=finfo)
        np.save(wd / "boxes.npy", boxes)
        all_tracks = {str(i): {str(k): v.tolist() for k, v in d.items()} for i, d in enumerate(hist)}
        (wd / "tracks.json").write_text(json.dumps(all_tracks))
        meta_p.write_text(json.dumps(meta, indent=1))
        metas[r["run"]] = meta
        del frames
        print(f"run {r['run']}: 창 f{f0}-{f1} (구간 f{s_f}-{e_f}), 트랙 {main}+{used[1:]}, 구간 커버 {cover:.0%}, f_mm={fmm} {finfo}")
        torch.cuda.empty_cache()

    # ---- stage B: ViTPose (모델 한 번 로드)
    from hmr4d.utils.preproc import VitPoseExtractor, Extractor, SimpleVO

    def bbx_xys_of(wd):
        return get_bbx_xys_from_xyxy(torch.from_numpy(np.load(wd / "boxes.npy")), base_enlarge=1.2).float()

    todo = [r for r in runs if r["run"] in metas and (a.force or not (work_root / f"run{r['run']}" / "vitpose.pt").exists())]
    if todo:
        ex = VitPoseExtractor()
        for r in todo:
            wd = work_root / f"run{r['run']}"
            torch.save(ex.extract(str(wd / "clip.mp4"), bbx_xys_of(wd)), wd / "vitpose.pt")
        del ex; torch.cuda.empty_cache()
    todo = [r for r in runs if r["run"] in metas and (a.force or not (work_root / f"run{r['run']}" / "vit_features.pt").exists())]
    if todo:
        ex = Extractor()
        for r in todo:
            wd = work_root / f"run{r['run']}"
            torch.save(ex.extract_video_features(str(wd / "clip.mp4"), bbx_xys_of(wd)), wd / "vit_features.pt")
        del ex; torch.cuda.empty_cache()
    # 변형 실행(--suffix + 고정 --f-mm): 같은 추적·ViTPose·특징으로 초점거리만 바꿔 VO와 예측을 다시 한다
    variant = bool(a.suffix) and a.f_mm != "auto"
    for r in runs:
        if r["run"] in metas and variant:
            metas[r["run"]] = dict(metas[r["run"]], f_mm=float(a.f_mm))
    for r in runs:
        wd = work_root / f"run{r['run']}"
        slam_p = wd / f"slam{a.suffix if variant else ''}.pt"
        if r["run"] in metas and (a.force or not slam_p.exists()):
            vo = SimpleVO(str(wd / "clip.mp4"), scale=0.5, step=8, method="sift", f_mm=metas[r["run"]]["f_mm"])
            torch.save(vo.compute(), slam_p)
    if a.stage == "prep":
        print("prep 끝 (SMPL 없이 되는 단계)")
        return

    # ---- stage C: GVHMR 예측 + 내보내기 (SMPL-X 필요)
    import hydra
    from hydra import initialize_config_module, compose
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401 (import 시 model/gvhmr/gvhmr_pl_demo 설정이 등록됨)
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        gcfg = compose(config_name="demo", overrides=["video_name=aimtb", "static_cam=False"])
    model = hydra.utils.instantiate(gcfg.model, _recursive_=False)
    model.load_pretrained_model(gcfg.ckpt_path)
    model = model.eval().cuda()
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    J_coco = torch.load("hmr4d/utils/body_model/smpl_coco17_J_regressor.pt").cuda()
    faces = np.load(gh / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)["f"].astype(np.int32)

    for r in runs:
        if r["run"] not in metas:
            continue
        m = metas[r["run"]]
        wd = work_root / f"run{r['run']}"
        L = m["f1"] - m["f0"] + 1
        K = create_camera_sensor(m["W"], m["H"], m["f_mm"])[2]
        T_w2c = torch.load(wd / f"slam{a.suffix if variant else ''}.pt")
        R_w2c = torch.from_numpy(np.asarray(T_w2c)[:, :3, :3]).float()
        data = {"length": torch.tensor(L), "bbx_xys": bbx_xys_of(wd), "kp2d": torch.load(wd / "vitpose.pt"),
                "K_fullimg": K.repeat(L, 1, 1), "cam_angvel": compute_cam_angvel(R_w2c),
                "f_imgseq": torch.load(wd / "vit_features.pt")}
        outs = {}
        with torch.no_grad():
            for tag, pp in (("pp", True), ("raw", False)):
                batch = {"length": data["length"][None], "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
                         "bbx_xys": data["bbx_xys"][None], "K_fullimg": data["K_fullimg"][None],
                         "cam_angvel": data["cam_angvel"][None], "f_imgseq": data["f_imgseq"][None]}
                batch = {k: v.cuda() for k, v in batch.items()}
                o = model.pipeline.forward(batch, train=False, postproc=pp, static_cam=False)
                pg = detach_to_cpu({k: v[0] for k, v in o["pred_smpl_params_global"].items()})
                pc = detach_to_cpu({k: v[0] for k, v in o["pred_smpl_params_incam"].items()})
                og, oc = smplx(**to_cuda(pg)), smplx(**to_cuda(pc))
                sm_c = torch.stack([smplx2smpl @ v for v in oc.vertices])
                coco_c = torch.einsum("jv,lvi->lji", J_coco, sm_c)
                outs[tag] = dict(pg=pg, pc=pc, jg=og.joints[:, :22].cpu().numpy(), jc=oc.joints[:, :22].cpu().numpy(),
                                 coco=coco_c.cpu().numpy(), vc=oc.vertices.cpu().numpy())
            torch.save({"pp": {k: outs["pp"][k] for k in ("pg", "pc")}, "raw": {k: outs["raw"][k] for k in ("pg", "pc")}},
                       wd / f"hmr4d_results{a.suffix}.pt")
        frames = np.arange(m["f0"], m["f1"] + 1)
        sel = (frames >= m["seg_f0"] - 3) & (frames <= m["seg_f1"] + 3)
        P = outs["pp"]
        np.savez_compressed(
            od / f"gvhmr_run{r['run']}{a.suffix}.npz",
            frames=frames, fps=m["fps"], W=m["W"], H=m["H"], K=K.numpy(), f_mm=m["f_mm"],
            seg_f0=m["seg_f0"], seg_f1=m["seg_f1"], track_main=m["track_main"], seg_cover=m["seg_cover"],
            joints_g=P["jg"], joints_c=P["jc"], coco_c=P["coco"],
            joints_g_raw=outs["raw"]["jg"], joints_c_raw=outs["raw"]["jc"], coco_c_raw=outs["raw"]["coco"],
            go_g=P["pg"]["global_orient"].numpy(), go_c=P["pc"]["global_orient"].numpy(),
            transl_g=P["pg"]["transl"].numpy(), betas=P["pg"]["betas"].numpy(),
            vitpose=torch.load(wd / "vitpose.pt").numpy(), boxes=np.load(wd / "boxes.npy"),
            T_w2c=np.asarray(T_w2c, dtype=np.float32),
            verts_frames=frames[sel], verts_c=P["vc"][sel].astype(np.float32), faces=faces)
        print(f"run {r['run']}: gvhmr_run{r['run']}{a.suffix}.npz 저장 (f_mm={m['f_mm']})")


# ============================================================ 2) analyze (repo env)

def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rodrigues(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def kabsch(A, B):
    """B ≈ R A + t 인 R (A, B: (N,3))."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def yaw_rot(v, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return v @ R.T


def angles(J, fwd):
    """J: (L,22,3) 세계 관절, fwd: (L,3) 수평 단위 진행 방향 → dict of (L,) 배열 (정의는 모듈 docstring)."""
    left = unit(np.cross(UP, fwd))
    v = J[:, J_NECK] - J[:, J_PELVIS]
    W = (J[:, J_LWR] + J[:, J_RWR]) / 2
    A = (J[:, J_LANK] + J[:, J_RANK]) / 2
    s = W - A
    dot = lambda x, y: np.sum(x * y, -1)  # noqa: E731
    roll = np.degrees(np.arctan2(dot(v, left), dot(v, UP)))
    pitch = np.degrees(np.arctan2(dot(v, fwd), dot(v, UP)))
    bike = np.degrees(np.arctan2(dot(s, left), dot(s, UP)))
    b = unit(dot(s, left)[:, None] * left + dot(s, UP)[:, None] * UP)
    n_right = unit(np.cross(fwd, b))
    hip = 1000 * dot(J[:, J_PELVIS] - A, n_right)
    return dict(torso_roll=roll, torso_pitch=pitch, bike_lean_body=bike, hip_offset_mm=hip,
                hip_side=dot(J[:, J_LHIP] - J[:, J_RHIP], left))


def selftest():
    """알려진 롤·피치·진행 방향으로 만든 합성 골격에서 각도 함수의 부호를 확인한다."""
    for heading in (0, 37, 140, -100):
        f = yaw_rot(np.array([[0, 0, 1.0]]), heading)[0]
        l = np.cross(UP, f)
        for roll, pitch, bike in ((20, 30, 35), (-10, 45, 5)):
            J = np.zeros((1, 22, 3))
            J[0, J_LHIP], J[0, J_RHIP] = 0.1 * l, -0.1 * l
            tr = math.radians(roll); tp = math.radians(pitch)
            # 몸통: 앞으로 pitch, 왼쪽으로 roll (작은 각에서 독립; 정의대로 tan 성분을 맞춤)
            J[0, J_NECK] = 0.5 * (UP + math.tan(tp) * f + math.tan(tr) * l)
            tb = math.radians(bike)
            A = -0.6 * UP + 0.3 * f
            Wp = A + 0.8 * (UP + math.tan(tb) * l) + 0.5 * f
            J[0, J_LANK] = J[0, J_RANK] = A
            J[0, J_LWR], J[0, J_RWR] = Wp + 0.3 * l, Wp - 0.3 * l
            r = angles(J, f[None])
            assert abs(r["torso_roll"][0] - roll) < 1e-6, (heading, r["torso_roll"], roll)
            assert abs(r["torso_pitch"][0] - pitch) < 1e-6, (heading, r["torso_pitch"], pitch)
            assert abs(r["bike_lean_body"][0] - bike) < 1e-6, (heading, r["bike_lean_body"], bike)
            assert r["hip_side"][0] > 0
            # 골반을 오른쪽으로 옮기면 hip_offset이 +로 커져야 함
            J2 = J.copy(); J2[0, J_PELVIS] = J[0, J_PELVIS] - 0.1 * l
            assert angles(J2, f[None])["hip_offset_mm"][0] > angles(J, f[None])["hip_offset_mm"][0] + 50
    print("selftest ok: roll/pitch/bike/hip 부호와 크기 확인")


def signed_yaw(a, b):
    """수평 단위벡터 a에서 b까지의 yaw(도), + = 위에서 볼 때 반시계 = 라이더 왼쪽으로 돎."""
    return np.degrees(np.arctan2(np.sum(np.cross(a, b) * UP, -1), np.sum(a * b, -1)))


def hproj(v):
    return unit(v - np.sum(v * UP, -1, keepdims=True) * UP)


def travel_dir(P, go_g, J, fps, mode="facing"):
    """진행 방향. GVHMR 세계 골반 궤적은 자전거 라이더에서 거의 멈춰 있어(구간 1.5초에 수평 이동 수 cm)
    속도 방향을 쓸 수 없다 → 기본은 골반이 향한 방향(전역 회전 × 몸 +z)의 수평 성분.
    mode="traj"는 평활한 골반 속도의 수평 성분(속도 0.5 m/s 미만이면 facing)."""
    from scipy.signal import savgol_filter
    win = min(15, len(P) - (1 - len(P) % 2))
    Ps = savgol_filter(P, win, 2, axis=0) if win >= 5 else P
    vel = np.gradient(Ps, axis=0) * fps
    vh = vel - np.sum(vel * UP, -1, keepdims=True) * UP
    speed = np.linalg.norm(vh, axis=-1)
    face = hproj(rodrigues(go_g) @ np.array([0, 0, 1.0]))
    bar = hproj(np.cross(hproj(J[:, J_LWR] - J[:, J_RWR]), UP))   # 손목선(핸들)에 수직인 수평 방향
    if mode == "traj":
        fwd = np.where((speed > 0.5)[:, None], unit(vh), face)
        src = np.where(speed > 0.5, "traj", "facing")
    else:
        fwd, src = face, np.full(len(P), "facing")
    return fwd, face, bar, src, speed, signed_yaw(fwd, face), signed_yaw(fwd, bar)


def project(X, K):
    return np.stack([K[0, 0] * X[..., 0] / X[..., 2] + K[0, 2], K[1, 1] * X[..., 1] / X[..., 2] + K[1, 2]], -1)


def seek_offset(video, start, pad=0.4):
    """analyze_lean.py는 cap.set(POS_FRAMES, f0) 뒤 순차로 읽어 frame 번호를 붙인다. 일부 영상(코치님 클립)은
    시크가 한 프레임 앞에 떨어져서, 기존 CSV의 frame f가 순차 디코드 f+off 번째 장면이다. 그 off를 잰다."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int((start - pad) * fps)
    seq = []
    for _ in range(f0 + 3):
        ok, img = cap.read()
        seq.append(img.astype(np.int16) if ok else None)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, sk = cap.read()
    cap.release()
    d = {o: float(np.abs(sk.astype(np.int16) - seq[f0 + o]).mean()) for o in (-2, -1, 0, 1, 2)
         if 0 <= f0 + o < len(seq) and seq[f0 + o] is not None}
    return min(d, key=d.get)


def compute_run(npz_path, lean_rows, raw=False, off=0):
    z = np.load(npz_path)
    frames, fps, K = z["frames"], float(z["fps"]), z["K"]
    Jg = z["joints_g_raw" if raw else "joints_g"]
    Jc = z["joints_c_raw" if raw else "joints_c"]
    coco = z["coco_c_raw" if raw else "coco_c"]
    head_up = np.median(np.sum((Jg[:, J_HEAD] - Jg[:, J_PELVIS]) * UP, -1))
    assert head_up > 0, f"세계 +y가 위가 아님? 머리-골반 y 중앙값 {head_up}"
    fwd, face, bar, src, speed, yaw, yaw_bar = travel_dir(Jg[:, J_PELVIS], z["go_g"], Jg, fps)
    ang = angles(Jg, fwd)
    ang_bar = angles(Jg, bar)
    sens = {}
    for key in ("torso_roll", "bike_lean_body"):
        alt = [angles(Jg, yaw_rot(fwd, d))[key] for d in (-15, 15)]
        sens[key] = np.max(np.abs(np.stack(alt) - ang[key]), 0)
    Lw, Rw = Jg[:, J_LWR], Jg[:, J_RWR]
    grip = np.degrees(np.arcsin(np.clip(-np.sum((Lw - Rw) * UP, -1) / np.maximum(np.linalg.norm(Lw - Rw, axis=-1), 1e-9), -1, 1)))
    Rc2w = np.stack([kabsch(Jc[i], Jg[i]) for i in range(len(frames))])
    cam_fwd = Rc2w @ np.array([0, 0, 1.0])
    cam_fwd_h = unit(cam_fwd - np.sum(cam_fwd * UP, -1, keepdims=True) * UP)
    view = np.degrees(np.arccos(np.clip(np.sum(cam_fwd_h * fwd, -1), -1, 1)))
    view_s = signed_yaw(fwd, cam_fwd_h)   # + = 광축이 라이더 왼쪽으로 돌아 있음 = 카메라가 라이더 오른쪽 뒤
    up_c = np.einsum("lji,j->li", Rc2w, UP)                  # R^T up
    p0 = project(Jc[:, J_PELVIS], K)
    p1 = project(Jc[:, J_PELVIS] + 0.7 * up_c, K)
    d = p1 - p0
    cam_roll = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
    vp = z["vitpose"]
    conf = vp[:, COCO_CONF, 2].mean(1)
    limb = vp[:, COCO_LIMB, 2].mean(1)
    pc = project(coco, K)
    bh = z["boxes"][:, 3] - z["boxes"][:, 1]
    err = np.linalg.norm(pc - vp[..., :2], axis=-1)
    err = np.where(vp[..., 2] > 0.5, err, np.nan)
    reproj = np.nanmedian(err, 1) / bh
    idx = {int(f): i for i, f in enumerate(frames)}
    rows = []
    for lr in lean_rows:
        fr = int(lr["frame"])
        if fr + off not in idx:
            continue
        i = idx[fr + off]
        rows.append(dict(
            run=lr["run"], frame=fr, seq_frame=fr + off, t=lr["t"], progress=lr["progress"], b3d_conf=round(float(conf[i]), 3),
            torso_roll=round(float(ang["torso_roll"][i]), 2), torso_pitch=round(float(ang["torso_pitch"][i]), 2),
            hip_offset_mm=round(float(ang["hip_offset_mm"][i]), 1), bike_lean_body=round(float(ang["bike_lean_body"][i]), 2),
            separation_b3d=round(float(ang["bike_lean_body"][i] - ang["torso_roll"][i]), 2),
            grip_roll=round(float(grip[i]), 2), heading_src=str(src[i]), travel_vs_facing=round(float(yaw[i]), 1),
            heading_bar_diff=round(float(yaw_bar[i]), 1), bike_lean_barhead=round(float(ang_bar["bike_lean_body"][i]), 2),
            torso_roll_barhead=round(float(ang_bar["torso_roll"][i]), 2),
            speed_mps=round(float(speed[i]), 2), roll_sens15=round(float(sens["torso_roll"][i]), 1),
            bike_sens15=round(float(sens["bike_lean_body"][i]), 1), cam_view_deg=round(float(view[i]), 1),
            cam_view_signed=round(float(view_s[i]), 1),
            cam_roll_deg=round(float(cam_roll[i]), 1), limb_conf=round(float(limb[i]), 3),
            reproj_err=round(float(reproj[i]), 4) if np.isfinite(reproj[i]) else "",
            hip_side_ok=int(ang["hip_side"][i] > 0)))
    extra = dict(z=z, Jc=Jc, up_c=up_c, idx=idx, heading=np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))),
                 seg_disp=float(np.linalg.norm((Jg[-1, J_PELVIS] - Jg[0, J_PELVIS]) * [1, 0, 1])))
    return rows, extra


COLS = ["run", "frame", "seq_frame", "t", "progress", "b3d_conf", "torso_roll", "torso_pitch", "hip_offset_mm", "bike_lean_body",
        "separation_b3d", "grip_roll", "heading_src", "travel_vs_facing", "heading_bar_diff", "bike_lean_barhead",
        "torso_roll_barhead", "speed_mps", "roll_sens15", "bike_sens15",
        "cam_view_deg", "cam_view_signed", "cam_roll_deg", "limb_conf", "reproj_err", "hip_side_ok"]


def lean_rows_of(seg, cfg, run):
    snap = out_dir(seg, cfg) / "lean_snapshot" / f"run{run}.csv"
    src = PATHS.feat_dir(seg, cfg, "lean") / f"run{run}.csv"
    if not snap.exists() and src.exists():  # bike_fit 재실행으로 fit_* 열이 바뀌기 전 상태를 고정
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, snap)
    return read_csv(snap) if snap.exists() else []


# ---------------- 메시 오버레이 (pytorch3d 없이 화가 알고리즘)

FONT = "C:/Windows/Fonts/malgun.ttf"


def render_tile(img, verts, faces, K, Jc, up_c, row, box, tile=420):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    x0, y0, x1, y1 = box
    bh = y1 - y0
    S = max(bh, x1 - x0) * 2.4
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * bh
    k = tile / S
    M = np.float32([[k, 0, -(cx - S / 2) * k], [0, k, -(cy - S / 2) * k]])
    base = cv2.warpAffine(img, M, (tile, tile), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    to_t = lambda p: np.stack([p[..., 0] * k + M[0, 2], p[..., 1] * k + M[1, 2]], -1)  # noqa: E731
    uv = to_t(project(verts, K))
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vis = np.sum(n * tri[:, 0], -1) < 0
    nz = -unit(n)[:, 2]
    order = np.argsort(-tri[:, :, 2].mean(1))
    layer = base.copy()
    mask = np.zeros(base.shape[:2], np.uint8)
    pts = np.round(uv * 16).astype(np.int32)
    for fi in order:
        if not vis[fi]:
            continue
        sh = 0.35 + 0.65 * float(np.clip(nz[fi], 0, 1))
        col = (int(235 * sh), int(200 * sh), int(170 * sh))   # BGR, 연한 하늘색 계열
        cv2.fillConvexPoly(layer, pts[faces[fi]], col, lineType=cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(mask, pts[faces[fi]], 255, shift=4)
    a = (mask[..., None] / 255.0) * 0.55
    out = (base * (1 - a) + layer * a).astype(np.uint8)
    P = lambda X: tuple(np.round(to_t(project(X, K))).astype(int))  # noqa: E731
    pel = Jc[J_PELVIS]
    W = (Jc[J_LWR] + Jc[J_RWR]) / 2
    A = (Jc[J_LANK] + Jc[J_RANK]) / 2
    p0 = P(pel)
    cv2.line(out, p0, (p0[0], p0[1] - 110), (150, 150, 150), 1, cv2.LINE_AA)            # 화면 수직
    cv2.arrowedLine(out, p0, P(pel + 0.9 * up_c), (80, 230, 80), 3, cv2.LINE_AA, tipLength=0.18)  # 중력 위
    cv2.line(out, p0, P(Jc[J_NECK]), (60, 60, 255), 3, cv2.LINE_AA)                     # 몸통
    ext = A + 1.25 * (W - A)
    cv2.line(out, P(A - 0.15 * (W - A)), P(ext), (0, 200, 255), 2, cv2.LINE_AA)        # 발목중점→손목중점
    for X in (W, A):
        cv2.circle(out, P(X), 5, (0, 200, 255), -1, cv2.LINE_AA)
    im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 15)
    dr.rectangle((0, 0, tile, 50), fill=(0, 0, 0))
    dr.text((6, 2), f"f{row['frame']}  p{float(row['progress']):.2f}  시점 {row['cam_view_deg']:.0f}°  conf {row['b3d_conf']:.2f}",
            font=f2, fill=(230, 230, 230))
    dr.text((6, 24), f"몸 롤 {row['torso_roll']:+.0f}°  피치 {row['torso_pitch']:+.0f}°  자전거 {row['bike_lean_body']:+.0f}°  "
            f"엉덩이 {row['hip_offset_mm']:+.0f}mm", font=f1, fill=(255, 255, 255))
    return im


def montage(seg, cfg, run, rows, extra, out_path, n_tiles=12):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    z = extra["z"]
    vf = list(z["verts_frames"])
    cand = [r for r in rows if r["seq_frame"] in vf]
    if not cand:
        return
    pick = [cand[int(round(i))] for i in np.linspace(0, len(cand) - 1, min(n_tiles, len(cand)))]
    want = {r["seq_frame"] for r in pick}
    rr = next(x for x in cfg["runs"] if x["run"] == run)
    cap = cv2.VideoCapture(str(PATHS.video_dir(cfg) / rr["file"]))
    imgs, fi = {}, 0
    while fi <= max(want):
        ok, img = cap.read()
        if not ok:
            break
        if fi in want:
            imgs[fi] = img
        fi += 1
    cap.release()
    tiles = []
    for r in pick:
        i = extra["idx"][r["seq_frame"]]
        vi = vf.index(r["seq_frame"])
        tiles.append(render_tile(imgs[r["seq_frame"]], z["verts_c"][vi], z["faces"], z["K"], extra["Jc"][i], extra["up_c"][i],
                                 r, z["boxes"][i]))
    cols, T = 4, 420
    head = 92
    rows_n = math.ceil(len(tiles) / cols)
    board = Image.new("RGB", (cols * T, head + rows_n * T), (22, 24, 28))
    dr = ImageDraw.Draw(board)
    name = "코치님" if run == 0 else f"Run {run}"
    dr.text((10, 6), f"{name} · GVHMR 메시 오버레이 (좌회전 뱅크, f_mm={float(z['f_mm']):.0f})",
            font=ImageFont.truetype(FONT, 26), fill=(255, 255, 255))
    dr.text((10, 44), "초록 화살표 = GVHMR 중력 위쪽   회색 = 화면 수직   빨강 = 골반→목   주황 = 발목 중점→손목 중점(자전거 평면 근사)",
            font=ImageFont.truetype(FONT, 15), fill=(200, 200, 200))
    dr.text((10, 66), "각도는 측정값(GVHMR 3D). +롤/+자전거 = 라이더 왼쪽, +엉덩이 = 라이더 오른쪽(좌회전 바깥). 시점 0°=바로 뒤, 90°=옆",
            font=ImageFont.truetype(FONT, 15), fill=(200, 200, 200))
    for k, t in enumerate(tiles):
        board.paste(t, ((k % cols) * T, head + (k // cols) * T))
    board.save(out_path, quality=90)


def cmd_analyze(a):
    selftest()
    for segp in [a.segments] + ([a.ref] if a.ref else []):
        seg = (ROOT / segp).resolve() if not Path(segp).is_absolute() else Path(segp)
        cfg = PATHS.load_cfg(seg)
        od = out_dir(seg, cfg)
        for r in cfg["runs"]:
            if a.runs and r["run"] not in a.runs and r["run"] != 0:
                continue
            npz = od / f"gvhmr_run{r['run']}{a.suffix}.npz"
            if not npz.exists():
                print(f"{npz.name} 없음 — infer 먼저"); continue
            off = seek_offset(PATHS.video_dir(cfg) / r["file"], r["start"])
            rows, extra = compute_run(npz, lean_rows_of(seg, cfg, r["run"]), raw=a.raw, off=off)
            suffix = a.suffix + ("_raw" if a.raw else "")
            with open(od / f"run{r['run']}{suffix}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(rows)
            hs = np.mean([x["hip_side_ok"] for x in rows]) if rows else float("nan")
            hd = extra["heading"]
            sel = [extra["idx"][x["seq_frame"]] for x in rows]
            hb = np.array([x["heading_bar_diff"] for x in rows])
            print(f"run {r['run']}: {len(rows)}행, 구간 진행방향(골반) 변화 {hd[sel[-1]] - hd[sel[0]]:+.0f}°, "
                  f"창 전체 골반 수평 이동 {extra['seg_disp']:.2f} m, 손목선 방향−골반 방향 중앙값 {np.median(hb):+.0f}° "
                  f"(|차|중앙 {np.median(np.abs(hb)):.0f}°), hip_side_ok {hs:.0%}")
            if not a.no_montage:
                montage(seg, cfg, r["run"], rows, extra, od / f"montage_run{r['run']}{suffix}.jpg")


def cmd_diag(a):
    """진단: (1) GVHMR 세계 궤적·카메라 기준 궤적이 진행 방향으로 쓸 만한가, (2) GVHMR 중력과 VO의 일관성,
    (3) 시점에 따라 값이 끌려가는가(상관), (4) 실루엣 fit_yaw와 GVHMR 시점의 독립 비교,
    (5) 회전 각도(골반 방향이 구간 시작부터 왼쪽으로 돈 양)를 맞춘 런 비교. 결과는 diag.txt와 turn_matched.csv."""
    from scipy.signal import savgol_filter
    lines, table = [], []
    THETAS = [15, 30, 45, 60, 75, 90]
    keys = ["torso_roll", "body_lean", "torso_pitch", "bike_lean_body", "separation_b3d", "hip_offset_mm", "fit_lean_s"]
    for segp in ([a.ref] if a.ref else []) + [a.segments]:
        seg = (ROOT / segp).resolve(); cfg = PATHS.load_cfg(seg); od = out_dir(seg, cfg)
        for r in cfg["runs"]:
            npz = od / f"gvhmr_run{r['run']}.npz"
            csvp = od / f"run{r['run']}.csv"
            if not npz.exists() or not csvp.exists():
                continue
            z = np.load(npz)
            fr, fps = z["frames"], float(z["fps"])
            Jg, Jc = z["joints_g"], z["joints_c"]
            seg_m = (fr >= int(z["seg_f0"])) & (fr <= int(z["seg_f1"]))
            Rc2w = np.stack([kabsch(Jc[i], Jg[i]) for i in range(len(fr))])
            Rw2c = z["T_w2c"][:, :3, :3].astype(float)
            up_c = np.einsum("lji,j->li", Rc2w, UP)
            up_vo = np.einsum("lji,lj->li", Rw2c, up_c)
            upm = unit(up_vo.mean(0))
            spread = np.degrees(np.arccos(np.clip(up_vo @ upm, -1, 1)))
            P = np.einsum("lji,lj->li", Rw2c, Jc[:, J_PELVIS])     # 카메라 기준 궤적(카메라 위치 고정 가정)
            v = np.gradient(savgol_filter(P, 15, 2, axis=0), axis=0) * fps
            vh = v - (v @ upm)[:, None] * upm
            d_g = hproj(np.einsum("lij,lj->li", Rc2w, np.einsum("lij,lj->li", Rw2c, unit(vh))))
            face = hproj(rodrigues(z["go_g"]) @ np.array([0, 0, 1.0]))
            yc = signed_yaw(d_g, face)[seg_m]
            disp = float(np.linalg.norm((Jg[-1, J_PELVIS] - Jg[0, J_PELVIS]) * [1, 0, 1]))
            rows = read_csv(csvp)
            old = {int(x["frame"]): x for x in lean_rows_of(seg, cfg, r["run"])}
            g = lambda k, src="new": np.array([fnum((x if src == "new" else old.get(int(x["frame"]), {})).get(k))  # noqa: E731
                                              if fnum((x if src == "new" else old.get(int(x["frame"]), {})).get(k)) is not None else np.nan
                                              for x in rows])
            vs, hip, roll, croll = g("cam_view_signed"), g("hip_offset_mm"), g("torso_roll"), g("cam_roll_deg")
            fy, iou = g("fit_yaw", "old"), g("fit_iou", "old")
            lowc = np.array([str(old.get(int(x["frame"]), {}).get("fit_lowconf", "0")).strip() in ("1", "1.0") for x in rows])
            ok = np.isfinite(fy) & ~lowc
            corr = lambda x, y: float(np.corrcoef(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])[0, 1])  # noqa: E731
            name = "coach" if r["run"] == 0 else f"run{r['run']}"
            lines.append(
                f"{name}: GVHMR world pelvis horiz. displacement over window {disp:.2f} m | VO-gravity spread med {np.median(spread[seg_m]):.1f} "
                f"max {spread[seg_m].max():.1f} deg | camera-anchored heading - facing: med {np.median(yc):+.0f}, IQR "
                f"{np.subtract(*np.percentile(yc, [75, 25])):.0f} deg | corr(hip_offset, view_signed) {corr(hip, vs):+.2f} | "
                f"corr(cam_roll, torso_roll) {corr(croll, roll):+.2f} | fit_yaw - view_signed (non-lowconf, n={ok.sum()}): "
                f"med {np.nanmedian((fy - vs)[ok]):+.0f}, corr {corr(fy[ok], vs[ok]):+.2f}")
            # 회전 각도 맞춤: 구간 첫 행부터 골반 방향이 왼쪽으로 돈 누적 yaw
            idx = {int(f): i for i, f in enumerate(fr)}
            si = np.array([idx[int(x["seq_frame"])] for x in rows])
            hd = np.degrees(np.unwrap(np.arctan2(face[:, 0], face[:, 2])))
            turn = hd[si] - hd[si[0]]
            turn = np.maximum.accumulate(turn)   # 단조로 만들어 보간
            vals = {k: g(k, "old" if k in ("body_lean", "fit_lean_s") else "new") for k in keys}
            vals["fit_lean_s"] = np.where(lowc, np.nan, vals["fit_lean_s"])
            for th in THETAS:
                m = np.abs(turn - th) <= 5
                row = dict(run=r["run"], turn_deg=th, n=int(m.sum()),
                           progress=round(float(np.nanmedian(g("progress")[m])), 2) if m.any() else "",
                           view_signed=round(float(np.nanmedian(vs[m])), 0) if m.any() else "")
                for k in keys:
                    x = vals[k][m]
                    row[k] = round(float(np.nanmedian(x)), 1) if np.isfinite(x).any() else ""
                table.append(row)
    od = out_dir((ROOT / a.segments).resolve(), PATHS.load_cfg((ROOT / a.segments).resolve()))
    (od / "diag.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(od / "turn_matched.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["run", "turn_deg", "n", "progress", "view_signed"] + keys); w.writeheader(); w.writerows(table)
    print("\n".join(lines))
    for k in ("torso_roll", "body_lean", "bike_lean_body", "separation_b3d", "hip_offset_mm", "fit_lean_s"):
        print(f"\n{k} at turn angle (deg) " + " ".join(f"{t:>6}" for t in THETAS))
        for run in sorted({x["run"] for x in table}):
            vals_ = [next((x[k] for x in table if x["run"] == run and x["turn_deg"] == t), "") for t in THETAS]
            print(f"  {'coach' if run == 0 else 'run' + str(run):>6}             " + " ".join(f"{v:>6}" for v in vals_))


def cmd_prepcheck(a):
    """infer --stage prep 결과(라이더 박스·ViTPose·다른 사람 트랙)를 구간 프레임 위에 그려 확인한다.
    GVHMR_HOME은 Windows에서 보이는 경로(예: \\\\wsl.localhost\\Ubuntu\\home\\<user>\\gvhmr\\GVHMR)."""
    import cv2
    import torch
    from PIL import Image, ImageDraw, ImageFont
    gh = Path(os.environ["GVHMR_HOME"])
    seg = (ROOT / a.segments).resolve()
    cfg = PATHS.load_cfg(seg)
    od = out_dir(seg, cfg)
    sk = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    for r in cfg["runs"]:
        if a.runs and r["run"] not in a.runs:
            continue
        wd = gh / "outputs" / "aimtb" / PATHS.session_name(seg) / f"run{r['run']}"
        if not (wd / "vitpose.pt").exists():
            continue
        m = json.loads((wd / "meta.json").read_text())
        boxes = np.load(wd / "boxes.npy")
        vp = torch.load(wd / "vitpose.pt").numpy()
        tracks = json.loads((wd / "tracks.json").read_text())
        ia = m["f0"]  # 창 시작(트랙으로 자른 뒤)
        cap = cv2.VideoCapture(str(wd / "clip.mp4"))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        lo, hi = m["seg_f0"] - ia, m["seg_f1"] - ia
        pick = np.linspace(max(0, lo - 15), min(len(frames) - 1, hi + 10), 8).round().astype(int)
        T = 380
        board = Image.new("RGB", (4 * T, 60 + 2 * T), (22, 24, 28))
        dr = ImageDraw.Draw(board)
        dr.text((10, 8), f"run {r['run']} prep: 창 f{m['f0']}-{m['f1']}, 구간 f{m['seg_f0']}-{m['seg_f1']}, 트랙 {m['track_main']}"
                f"+{m['track_used'][1:]}, 구간 커버 {m['seg_cover']:.0%}, f_mm {m['f_mm']}", font=ImageFont.truetype(FONT, 18), fill=(255, 255, 255))
        dr.text((10, 34), "노랑 = GVHMR 입력 라이더 박스, 회색 = 다른 YOLO 트랙(트랙 시작 전 원래 창 기준 번호), 초록 = ViTPose(신뢰도>0.3)",
                font=ImageFont.truetype(FONT, 14), fill=(200, 200, 200))
        for k, i in enumerate(pick):
            img = frames[i].copy()
            b = boxes[i]
            bh = b[3] - b[1]
            S = max(bh * 3.2, 300)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for tid, tb in tracks.get(str(i + m["f0"] - m.get("f0_orig", m["f0"])), {}).items():
                cv2.rectangle(img, tuple(map(int, tb[:2])), tuple(map(int, tb[2:])), (150, 150, 150), 2)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 255), 2)
            for p, q in sk:
                if vp[i, p, 2] > 0.3 and vp[i, q, 2] > 0.3:
                    cv2.line(img, tuple(vp[i, p, :2].astype(int)), tuple(vp[i, q, :2].astype(int)), (60, 230, 60), 2, cv2.LINE_AA)
            M = np.float32([[T / S, 0, -(cx - S / 2) * T / S], [0, T / S, -(cy - S / 2) * T / S]])
            tile = cv2.warpAffine(img, M, (T, T), flags=cv2.INTER_AREA)
            cv2.putText(tile, f"f{i + m['f0']}{' SEG' if lo <= i <= hi else ''}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            board.paste(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)), ((k % 4) * T, 60 + (k // 4) * T))
        board.save(od / f"prep_run{r['run']}.jpg", quality=88)
        print(od / f"prep_run{r['run']}.jpg")


# ============================================================ 3) board (repo env)

def cmd_board(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    C_TORSO, C_BIKE, C_PITCH = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    items = []
    if a.ref:
        rs = (ROOT / a.ref).resolve(); rc = PATHS.load_cfg(rs)
        items += [(rs, rc, r["run"], "코치님") for r in rc["runs"]]
    seg = (ROOT / a.segments).resolve(); cfg = PATHS.load_cfg(seg)
    items += [(seg, cfg, r["run"], f"Run {r['run']}") for r in cfg["runs"] if not a.runs or r["run"] in a.runs]
    data = []
    for s, c, run, name in items:
        p = out_dir(s, c) / f"run{run}.csv"
        if not p.exists():
            continue
        new = {int(x["frame"]): x for x in read_csv(p)}
        old = {int(x["frame"]): x for x in lean_rows_of(s, c, run)}
        fr = sorted(set(new) & set(old))
        g = lambda d, k: np.array([fnum(d[f].get(k)) if fnum(d[f].get(k)) is not None else np.nan for f in fr])  # noqa: E731
        lowc = np.array([str(old[f].get("fit_lowconf", "0")).strip() in ("1", "1.0", "True") for f in fr])
        from separation_board import select as old_select   # 기존 분리각 보드와 같은 프레임 선택
        osel = old_select([old[f] for f in fr])
        ofr = {x[0] for x in osel}
        data.append(dict(old_sep=np.array([x[3] for x in osel]),
                         sep3_oldframes=np.array([fnum(new[f]["separation_b3d"]) for f in fr if f in ofr]),
                         bikeb=g(new, "bike_lean_barhead"), rollb=g(new, "torso_roll_barhead")))
        data[-1].update(name=name, run=run, prog=g(old, "progress"), vr=g(old, "view_ratio"),
                         body=g(old, "body_lean"), fit=np.where(lowc, np.nan, g(old, "fit_lean_s")), fit_all=g(old, "fit_lean_s"),
                         roll=g(new, "torso_roll"), pitch=g(new, "torso_pitch"), bike=g(new, "bike_lean_body"),
                         grip=g(new, "grip_roll"), view=g(new, "cam_view_deg"), camroll=g(new, "cam_roll_deg"),
                         views=g(new, "cam_view_signed"),
                         limb=g(new, "limb_conf"), hip=g(new, "hip_offset_mm"), sens=g(new, "bike_sens15"),
                         rsens=g(new, "roll_sens15"))
    n = len(data)
    H_in = 2.6 * n + 12.5
    fig = plt.figure(figsize=(21, H_in), facecolor=SURF)
    top = 1 - 1.25 / H_in
    gs = fig.add_gridspec(n + 4, 2, width_ratios=[2.0, 1], height_ratios=[1] * n + [0.15, 1.9, 1.9, 2.0], hspace=0.5,
                          wspace=0.16, top=top, bottom=0.01, left=0.06, right=0.99)
    fig.text(0.01, 1 - 0.3 / H_in, "좌회전 뱅크 · GVHMR 3D 각도 vs 기존 2D/실루엣 값", fontsize=22, color=INK, va="top")
    fig.text(0.01, 1 - 0.75 / H_in, "실선 = GVHMR 3D(측정값, 중력 기준), 점선 = 기존 값(2D body_lean, 실루엣 fit_lean_s, LOW-CONF 제외). "
             "+ = 라이더 왼쪽. 회색 띠 = 뒷모습(view_ratio>0.75). 진행 방향 = 골반이 향한 방향.", fontsize=12, color=INK2, va="top")
    for k, d in enumerate(data):
        ax = fig.add_subplot(gs[k, 0]); ax.set_facecolor(SURF)
        rear = d["vr"] > 0.75
        for i in range(len(d["prog"])):
            if rear[i]:
                ax.axvspan(d["prog"][i] - 0.008, d["prog"][i] + 0.008, color="#d9d8d3", lw=0)
        ax.plot(d["prog"], d["roll"], color=C_TORSO, lw=2, label="상체 롤 (GVHMR)")
        ax.plot(d["prog"], d["body"], color=C_TORSO, lw=1.6, ls="--", label="상체 2D body_lean (기존)")
        ax.plot(d["prog"], d["bike"], color=C_BIKE, lw=2, label="자전거 손·발 평면 (GVHMR)")
        ax.plot(d["prog"], d["fit"], color=C_BIKE, lw=1.6, ls="--", label="자전거 실루엣 fit_lean_s (기존)")
        ax.plot(d["prog"], d["pitch"], color=C_PITCH, lw=1.4, label="상체 피치 (GVHMR, +앞숙임)")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(-40, 80)
        ax.grid(True, color=GRID, lw=0.6); ax.tick_params(colors=INK2, labelsize=10)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_ylabel(d["name"], fontsize=15, color=INK, rotation=0, ha="right", va="center", labelpad=12)
        if k == 0:
            ax.legend(loc="lower left", ncol=5, fontsize=10.5, frameon=False, bbox_to_anchor=(0, 1.0))
        if k == n - 1:
            ax.set_xlabel("구간 진행률 (0 = 착지 직후, 1 = 뱅크 탈출)", color=INK2)

    def jit(x):
        x = x[np.isfinite(x)]
        return float(np.median(np.abs(np.diff(x)))) if len(x) > 3 else np.nan

    def med(x, m):
        x = x[m & np.isfinite(x)]
        return (float(np.median(x)), len(x)) if len(x) else (np.nan, 0)

    # 오른쪽 위: 2D - 3D 차이 vs 부호 있는 시점, 앞숙임만으로 예측한 곡선
    def style(ax_, title, xl):
        ax_.set_facecolor(SURF); ax_.axhline(0, color=INK2, lw=0.8); ax_.axvline(0, color=INK2, lw=0.6)
        ax_.grid(True, color=GRID, lw=0.6); ax_.tick_params(colors=INK2, labelsize=10)
        for sp in ax_.spines.values():
            sp.set_color(GRID)
        ax_.set_xlabel(xl, color=INK2); ax_.set_ylabel("차이 (°)", color=INK2)
        ax_.set_title(title, fontsize=13, color=INK, loc="left")
        ax_.set_xlim(-90, 90)

    half = max(2, n // 2)
    ax = fig.add_subplot(gs[0:half, 1])
    allv = np.concatenate([d["views"] for d in data]); alld = np.concatenate([d["body"] - d["roll"] for d in data])
    pmed = float(np.nanmedian(np.concatenate([d["pitch"] for d in data])))
    ax.scatter(allv, alld, s=14, color=C_TORSO, alpha=0.6, label="2D body_lean - 3D 롤 (프레임)")
    xs = np.linspace(-90, 90, 181)
    ax.plot(xs, -np.degrees(np.arctan(np.tan(np.radians(pmed)) * np.sin(np.radians(xs)))), color=INK, lw=2,
            label=f"앞숙임 {pmed:.0f}°가 시점만큼 화면 기울기로 섞일 때의 예측")
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_ylim(-45, 60)
    style(ax, "2D 상체 기울기의 시점 혼입: 2D - 3D 차이", "GVHMR 시점 cam_view_signed (0 = 바로 뒤, + = 카메라가 라이더 오른쪽 뒤)")
    ax2 = fig.add_subplot(gs[half:n, 1])
    allfb = np.concatenate([d["fit"] - d["bike"] for d in data])
    allgb = np.concatenate([d["grip"] - d["bike"] for d in data])
    ax2.scatter(allv, allfb, s=14, color=C_BIKE, alpha=0.65, label="실루엣 - 3D 손·발 평면")
    ax2.scatter(allv, allgb, s=14, color=C_TORSO, alpha=0.55, label="3D 손목선 기울기 - 3D 손·발 평면")
    ax2.legend(fontsize=10, frameon=False, loc="upper right")
    ax2.set_ylim(-35, 50)
    style(ax2, "자전거 기울기: 방법끼리의 차이", "GVHMR 시점 cam_view_signed")

    # 표: 뒷모습 중앙값
    cols = ["", "뒷모습\nn", "손발\nconf≥0.3", "GVHMR\n시점°", "3D 롤", "2D\n상체", "3D 피치", "3D 자전거\n(골반 방향)",
            "3D 자전거\n(손목선 방향)", "손목선\n기울기", "실루엣", "분리각 3D\n(골반 방향)", "분리각 3D\n(손목선 방향)",
            "기존 분리각\n(보드 기준, n)", "3D 분리각\n(같은 프레임)", "엉덩이\nmm", "흔들림 °/f\n3D 자전거·실루엣", "흔들림 °/f\n3D 롤·2D 상체"]
    cell = []
    for d in data:
        rear = (d["vr"] > 0.75)
        gate = rear & (d["limb"] >= 0.3)
        fmt = lambda v: "-" if not np.isfinite(v) else f"{v:+.0f}"  # noqa: E731
        so = d["old_sep"]
        cell.append([d["name"], str(int(rear.sum())), str(int(gate.sum())), f"{med(d['view'], rear)[0]:.0f}",
                     fmt(med(d["roll"], rear)[0]), fmt(med(d["body"], rear)[0]), fmt(med(d["pitch"], rear)[0]),
                     fmt(med(d["bike"], gate)[0]), fmt(med(d["bikeb"], gate)[0]), fmt(med(d["grip"], gate)[0]),
                     fmt(med(d["fit"], rear)[0]), fmt(med(d["bike"] - d["roll"], gate)[0]),
                     fmt(med(d["bikeb"] - d["rollb"], gate)[0]),
                     (f"{np.median(so):+.0f} ({len(so)})" if len(so) else "- (0)"),
                     (fmt(float(np.nanmedian(d["sep3_oldframes"]))) if len(d["sep3_oldframes"]) else "-"),
                     fmt(med(d["hip"], gate)[0]), f"{jit(d['bike']):.1f} · {jit(d['fit_all']):.1f}",
                     f"{jit(d['roll']):.1f} · {jit(d['body']):.1f}"])
    axt = fig.add_subplot(gs[n + 1, :]); axt.axis("off")
    widths = [0.05, 0.035, 0.04, 0.04] + [0.045] * 3 + [0.06] * 2 + [0.045] * 2 + [0.06] * 2 + [0.065, 0.06, 0.045, 0.08, 0.08]
    tb = axt.table(cellText=cell, colLabels=cols, cellLoc="center", colWidths=[w / sum(widths) for w in widths],
                   bbox=[0, 0, 1, 0.93])
    tb.auto_set_font_size(False); tb.set_fontsize(11)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(color=INK, weight="bold"); c.set_facecolor("#f0efec")
    axt.set_title("뒷모습 프레임(view_ratio>0.75) 중앙값(°). 3D 자전거·분리각·엉덩이는 손목·발목 2D 신뢰도 ≥0.3인 프레임만. "
                  "기존 분리각 = separation_board.py 선택 기준(실루엣-2D, lean_snapshot). "
                  "흔들림 = 연속 프레임 차의 중앙값(매끄러움은 모델 사전 영향이지 정확도가 아님)",
                  fontsize=11.5, color=INK2, loc="left")
    # 표 2: 회전 각도 맞춤 (diag가 만든 turn_matched.csv)
    tmp = out_dir(seg, cfg) / "turn_matched.csv"
    if tmp.exists():
        tm = read_csv(tmp)
        TH = ["15", "30", "45", "60"]
        cols2 = [""] + [f"{t}° 회전\n{lab}" for t in TH for lab in ("시점", "3D 롤", "3D 자전거", "분리각")]
        cell2 = []
        for d in data:
            row = [d["name"]]
            for t in TH:
                x = next((q for q in tm if int(q["run"]) == d["run"] and q["turn_deg"] == t), None)
                fv = lambda k: "-" if x is None or fnum(x.get(k)) is None else f"{fnum(x[k]):+.0f}"  # noqa: E731
                row += [fv("view_signed"), fv("torso_roll"), fv("bike_lean_body"), fv("separation_b3d")]
            cell2.append(row)
        axm = fig.add_subplot(gs[n + 2, :]); axm.axis("off")
        tb2 = axm.table(cellText=cell2, colLabels=cols2, cellLoc="center",
                        colWidths=[0.06] + [0.94 / 16] * 16, bbox=[0, 0, 1, 0.9])
        tb2.auto_set_font_size(False); tb2.set_fontsize(11)
        for (i, j), c in tb2.get_celld().items():
            c.set_edgecolor(GRID)
            if i == 0:
                c.set_text_props(color=INK, weight="bold"); c.set_facecolor("#f0efec")
            elif j >= 1 and ((j - 1) // 4) % 2 == 0:
                c.set_facecolor("#f6f5f2")
        axm.set_title("회전 단계를 맞춘 비교: 골반 방향이 구간 시작부터 왼쪽으로 15·30·45·60° 돈 시점(±5°)의 중앙값(°). "
                      "내 런은 15~45°에서 시점이 ±25° 안(뒷모습 쪽). 코치님은 촬영 자리가 달라 같은 회전에서도 시점이 다름",
                      fontsize=11.5, color=INK2, loc="left")
    axn = fig.add_subplot(gs[n + 3, :]); axn.axis("off")
    note = a.note or ""
    if note and Path(note).exists():
        note = Path(note).read_text(encoding="utf-8")
    axn.text(0, 1, note, fontsize=12.5, color=INK, va="top", linespacing=1.6)
    out = Path(a.out) if a.out else out_dir(seg, cfg) / "compare_board.png"
    fig.savefig(out, dpi=90, facecolor=SURF, bbox_inches="tight")
    print(out)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow([c.replace("\n", " ") for c in cols]); w.writerows(cell)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer"); p.add_argument("segments"); p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--stage", choices=["prep", "all"], default="all")
    p.add_argument("--pre", type=float, default=PRE); p.add_argument("--post", type=float, default=POST)
    p.add_argument("--f-mm", default="auto", help="35mm 환산 초점거리(mm) 또는 auto")
    p.add_argument("--track-id", nargs="*", help="RUN:ID 로 라이더 트랙 강제 (예: 0:3)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--suffix", default="", help="변형 실행 이름(예: _f50). 고정 --f-mm과 함께 쓰면 VO·예측만 다시 함")
    p = sub.add_parser("analyze"); p.add_argument("segments"); p.add_argument("--ref"); p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--suffix", default="")
    p.add_argument("--raw", action="store_true", help="GVHMR 후처리(정지 관절·IK) 끈 결과로 계산")
    p.add_argument("--no-montage", action="store_true")
    p = sub.add_parser("board"); p.add_argument("segments"); p.add_argument("--ref"); p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--out"); p.add_argument("--note", help="보드 아래 판단 문장(문자열 또는 텍스트 파일 경로)")
    p = sub.add_parser("prepcheck"); p.add_argument("segments"); p.add_argument("--runs", type=int, nargs="*")
    p = sub.add_parser("diag"); p.add_argument("segments"); p.add_argument("--ref")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"infer": cmd_infer, "analyze": cmd_analyze, "board": cmd_board, "prepcheck": cmd_prepcheck, "diag": cmd_diag,
     "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
