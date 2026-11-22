"""SAM 3D Body(Meta, MHR 인체 모델)로 프레임마다 라이더의 3D 몸 자세를 복원하고 3D 몸 각도를 잰다.

왜: 2D `body_lean`은 라이더가 뒷모습에서 옆모습으로 돌면 앞숙임이 섞인다. 실루엣 자전거 기울기(`fit_lean_s`)는
뒷모습에서 수렴하지 않았다(LOG 2026-11-22). SAM 3D Body는 이미지 한 장에서 카메라 좌표계 3D 메시를 낸다.
손은 그립, 발은 페달에 있으므로 3D 손목·발목으로 자전거 기울기도 따로 잰다.

단계 (환경이 둘이다):
    prep     [저장소 .venv]        analyze_lean과 같은 YOLO 추적·크롭을 다시 만들고, 크롭 재추론 키포인트와 사람 박스를
                                   전체 프레임 좌표로 저장 → prep_run<N>.npz (저장된 크롭과 픽셀 차이도 기록)
    infer    [SAM 3D Body 환경]    전체 프레임 + 라이더 박스(+키포인트 프롬프트)로 추론 → raw_<tag>_run<N>.npz
    angles   [저장소 .venv]        각도 계산·평활 → run<N>.csv
    montage  [저장소 .venv]        메시 오버레이 몽타주 → montage_run<N>.jpg
    board    [저장소 .venv]        비교 보드 → compare_board.png (body3d_board.py)
    hints    [저장소 .venv]        사용자가 그린 핸들바 선(sessions/<세션>/bar_hints.json)과 비교
                                   → bar_hints_eval_<tag>.csv, bar_hints_board_<tag>.jpg
    all      prep → infer(하위 프로세스로 $SAM3DBODY_HOME/.venv 실행) → angles → montage
    selftest 합성 뼈대로 각도 정의·부호 확인

사용법:
    uv run python tools/body3d_sam3d.py all sessions/2026-11-21/segments.json [--runs 1 3] [--prompt kp]
    uv run python tools/body3d_sam3d.py board sessions/2026-11-21/segments.json --ref sessions/reference-coach/segments.json
    uv run python tools/body3d_sam3d.py hints sessions/2026-11-21/segments.json --ref sessions/reference-coach/segments.json
    uv run python tools/body3d_sam3d.py angles sessions/2026-11-21/segments.json --gravity gvhmr --suffix _gvgrav
    # GVHMR와 프레임별 비교·그린 선 채점·Run 5 진단: tools/body3d_vs_gvhmr.py
    # infer만 직접: "$SAM3DBODY_HOME/.venv/Scripts/python.exe" tools/body3d_sam3d.py infer sessions/...
    # 다른 설정: --prompt kp, --focal 2203 → raw_kp_f1500_*, raw_box_f2203_* (board가 민감도 행으로 자동 사용)

환경변수:
    SAM3DBODY_HOME   sam-3d-body 클론 폴더. 그 안에 .venv/(Python 3.11, PyTorch cu128)와
                     checkpoints/sam-3d-body-dinov3/(model.ckpt, model_config.yaml, assets/mhr_model.pt)
    AIMTB_DATA_ROOT  worktree에서 메인 체크아웃의 reference/·outputs/·models/를 쓸 때 (tools/paths.py)

출력: outputs/<세션>/<feature>_sam3dbody/ (git 제외)

입력 선택: 전체 프레임 + 라이더 박스. 768 크롭은 원본 픽셀을 키운 것이라 정보가 늘지 않고, 크롭 변환이 저장돼 있지 않다.
SAM 3D Body의 카메라 머리는 박스 위치와 초점거리로 시선 방향을 정하므로, 전체 프레임과 실제 초점거리를 주는 쪽이
기하학적으로 맞다. 박스는 크롭에서 다시 잡은 YOLO 사람 박스를 전체 프레임 좌표로 되돌린 것이다.
크롭 재검출이 다른 사람을 고르는 프레임이 있어(2026-11-21 run 2 f216), infer에서 앞뒤 3프레임 중앙값 박스와
중심이 박스 높이의 0.35배 넘게 떨어지거나 크기가 35％ 넘게 다르면 중앙값 박스로 바꾼다(box_fixed 열).

프롬프트: 공개 API(process_one_image)는 박스, 사람 마스크(양성), 카메라 내부 행렬만 받는다. 음성 마스크(자전거 제외)
경로는 없다. 키포인트 프롬프트는 내부 함수 run_keypoint_prompt(MHR70 번호)로만 줄 수 있다.
체크포인트 설정의 PROMPT_ENCODER.MAX_NUM_CLICKS는 2다. 12점을 한꺼번에 넣는 kp 설정은 학습 범위 밖일 수 있다.
    --prompt box : 박스만 (기본, 권장)
    --prompt kp  : 박스 + YOLO 키포인트(conf≥0.5, 몸통·팔·다리 12점)를 첫 몸 디코더 뒤와 손 보정 단계에 프롬프트로 추가

카메라: SAM 3D Body는 초점거리를 추정하지 않는다(MoGe2 FOV 추정기는 선택 사항, 설치 안 함). 기본값은 이미지 대각선
(1080p에서 2203px, 수평 화각 47°)이다. 이 도구는 --focal(기본 1500px)을 직접 준다. 1080p 폰 영상, 26mm 환산
메인 카메라, 손떨림 보정 크롭을 가정한 값이다(수평 화각 약 65°). 주점은 이미지 중앙.

좌표·가정:
    - SAM 3D Body 출력은 카메라 좌표(OpenCV: x 오른쪽, y 아래, z 앞), 단위 미터.
    - 중력 g (--gravity):
      image(기본)  카메라 +y(화면 아래). 폰이 수평(롤 0)이고 피치 0이라는 가정. 2D 지표와 같은 가정. --pitch로 민감도.
      gvhmr        GVHMR이 본 프레임별 중력(body3d_gvhmr.py 출력의 incam·global 관절 Kabsch 회전, σ=2프레임 평활).
                   관절을 프레임마다 수평 맞춤(level_rot)한 뒤 같은 정의로 잰다. 열 cam_pitch_g, cam_roll_g에 기록.
      gvhmr_med    같은 중력의 런 중앙값 하나.
      2026-11-21 측정: GVHMR 카메라는 피치 +2~+5°(아래를 봄), 롤 −5~−7°. 화면 아래 가정을 바꾸면 뒷모습 상체 롤이
      5~9° 줄고(Run 5만 1°), 분리각은 2° 안에서만 바뀐다(LOG 2026-11-23 GVHMR 비교).
    - lean CSV(frame·progress·view_ratio·실루엣 값)는 lean_csv(): GVHMR 세션이 고정한 fix4 스냅샷이 있으면 그것을 쓴다.

정의 (u = −g = 위, 각도 단위 도, 키포인트 번호는 MHR70):
    P = 골반 중심 = 양 엉덩이(9, 10) 중점.  N = 목(69).  W = 양 손목(41, 62) 중점.  A = 양 발목(13, 14) 중점.
    진행 방향 t: 골반 앞 축 F = (N−P) × (오른엉덩이 − 왼엉덩이)를 수평면(u에 수직)에 투영한 방향.
                 프레임마다 yaw를 구하고, 펼친 뒤 Hampel(7프레임, 3 MAD) + 가우시안(σ=2프레임)으로 평활.
                 --heading bar면 손목→손목 선에 수직인 수평 방향을 쓴다(민감도 시험용).
                 --heading gvhmr면 GVHMR 골반 방향(진단용, --gravity gvhmr* 필요). 진행 방향 합치기가 아니다.
                 진행 방향 오차는 앞숙임을 롤로 섞는다(상체 앞숙임 ~35°면 1°당 약 0.7°). 손-발 벡터도 비슷하게
                 앞으로 기울어 있어서 분리각에서는 대부분 상쇄된다.
    왼쪽 축 L = u × t (라이더 왼쪽, 수평).
    torso_roll     : v = N−P를 t에 수직인 평면에 투영한 기울기. atan2(v·L, v·u). + = 라이더 왼쪽.
    torso_pitch    : v를 L에 수직인 평면(시상면)에 투영한 기울기. atan2(v·t, v·u). + = 앞숙임.
    bike_lean_body : W와 A를 지나고 t를 포함하는 평면(자전거 중심면 근사)의 중력 대비 기울기.
                     h = W−A, atan2(h·L, h·u). + = 왼쪽. 그립·페달이 자전거 중심면에 대칭이라는 가정.
    hip_offset_mm  : 그 평면에서 P까지의 부호 있는 거리. b̂ = h에서 t 성분을 뺀 단위벡터, n = t × b̂(라이더 오른쪽).
                     (P−A)·n × 1000. + = 라이더 오른쪽(좌회전의 바깥).
    b3d_conf       : SAM 3D Body 2D 투영과 YOLO 크롭 키포인트(conf≥0.5인 몸통·팔·다리 12점)의 일치도.
                     mean(exp(−d²/(2·(0.1·T)²))), T = YOLO 몸통 길이(px). 4점 미만이거나 어깨·엉덩이 conf<0.3이면
                     빈칸(YOLO 근거 없음, 판단 불가 — 블러 프레임에서 흔함). 빈칸은 평활 입력에 포함한다.
                     SAM 3D Body는 자체 신뢰도를 내지 않는다. --prompt kp에서는 일부 순환적이다(같은 점을 프롬프트로 씀).
    교차 확인 열:
    bar_roll       : 손목→손목 벡터(왼→오)의 수평 대비 기울기 = asin(ŵ·u). + = 왼쪽(오른손이 높음).
                     진행 방향을 쓰지 않는다. 핸들 조향각이 작고 손목이 그립에 대칭이라는 가정.
    yaw_bar        : 손목→손목 벡터에 수직인 수평 방향의 yaw. travel_yaw와 차이가 크면 진행 방향(골반 축) 오차를 의심.
    travel_yaw     : 평활한 진행 방향 yaw. 0 = 카메라에서 멀어짐(뒷모습), +90 = 화면 오른쪽으로 감, ±180 = 카메라로 옴.
    *_raw          : 평활 전 프레임별 값(진행 방향만 평활한 t 사용).
    평활 열        : *_raw에 Hampel(7, 3 MAD) + 가우시안(σ=1.5프레임). b3d_conf<0.3 프레임은 평활 입력에서 뺀다.
    box_fixed      : 1 = 박스 이상치를 이웃 중앙값으로 바꾼 프레임.

GVHMR 세션(left_bank_gvhmr)도 같은 열 이름(b3d_conf, torso_roll, torso_pitch, hip_offset_mm, bike_lean_body)을 쓴다.
"""
import argparse
import contextlib
import csv
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

SUFFIX = "sam3dbody"
DEFAULT_FOCAL = 1500.0
CKPT = "sam-3d-body-dinov3"
PAD = 0.4  # analyze_lean.track_rider와 같은 앞뒤 여유(초)

# COCO 17 → MHR70 번호
COCO2MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
BODY12 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # COCO 번호: 어깨·팔꿈치·손목·엉덩이·무릎·발목
M_LHIP, M_RHIP, M_NECK, M_RWR, M_LWR, M_LAN, M_RAN = 9, 10, 69, 41, 62, 13, 14
M_LSH, M_RSH = 5, 6

SHARED = ["b3d_conf", "torso_roll", "torso_pitch", "hip_offset_mm", "bike_lean_body"]
RAW = ["torso_roll_raw", "torso_pitch_raw", "hip_offset_mm_raw", "bike_lean_body_raw", "bar_roll_raw"]
EXTRA = ["bar_roll", "travel_yaw", "yaw_pelvis_raw", "yaw_bar_raw", "cam_dist_m", "wrist_sep_m",
         "shoulder_w_m", "focal_px", "box_fixed", "grav_src", "cam_pitch_g", "cam_roll_g", "heading_src"]


# ---------------------------------------------------------------- 공통

def out_dir(seg, cfg):
    return PATHS.feat_dir(seg, cfg, SUFFIX)


def lean_dir(seg, cfg):
    return PATHS.feat_dir(seg, cfg, "lean")


def lean_csv(seg, cfg, run):
    """lean CSV 경로. bike_fit 재실행이 fit_* 열을 계속 바꾸므로, GVHMR 세션이 2026-11-23 00:10에 고정한
    스냅샷(left_bank_gvhmr/lean_snapshot, fix4 = fit_variants/fix4_orig와 같음)이 있으면 그것을 쓴다.
    frame·t·progress·view_ratio는 어느 쪽이든 같다."""
    snap = PATHS.feat_dir(seg, cfg, "gvhmr") / "lean_snapshot" / f"run{run}.csv"
    return snap if snap.exists() else lean_dir(seg, cfg) / f"run{run}.csv"


def sel_runs(cfg, runs):
    return [r for r in cfg["runs"] if not runs or r["run"] in runs]


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def tag_of(prompt, focal):
    return f"{prompt}_f{int(round(focal))}"


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


# ---------------------------------------------------------------- prep (저장소 .venv)

def cmd_prep(seg, cfg, runs):
    import cv2
    from ultralytics import YOLO
    import analyze_lean as AL

    out = out_dir(seg, cfg)
    out.mkdir(parents=True, exist_ok=True)
    ldir = lean_dir(seg, cfg)
    pose_m = YOLO(str(PATHS.MODELS / "yolo11x-pose.pt"))
    for r in runs:
        want = {int(x["frame"]) for x in read_csv(ldir / f"run{r['run']}.csv")}
        cap = cv2.VideoCapture(str(PATHS.video_dir(cfg) / r["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, rider = AL.track_rider(cap, fps, r["start"], r["end"], pose_m)
        cap.release()
        rec = {k: [] for k in ("frame", "box", "kp", "kpc", "M", "crop_diff")}
        prev = None
        for fi, img, boxes in frames:
            t = fi / fps
            if not (r["start"] <= t <= r["end"]) or rider not in boxes:
                continue
            M, prev = AL.crop_affine(boxes[rider], prev)   # analyze_lean과 같은 순서로 평활 상태를 갱신
            if fi not in want:
                continue
            crop = cv2.warpAffine(img, M, (AL.CROP, AL.CROP), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
            saved = cv2.imread(str(ldir / "crops" / f"run{r['run']}_f{fi:04d}.jpg"))
            diff = float(np.abs(saved.astype(np.float32) - crop).mean()) if saved is not None else np.nan
            Minv = cv2.invertAffineTransform(M)
            pr = pose_m.predict(crop, imgsz=AL.CROP, classes=[0], verbose=False)[0]
            pi = AL.pick_center_person(pr)
            if pi is not None:
                kxy = pr.keypoints.xy[pi].cpu().numpy()
                kcf = pr.keypoints.conf[pi].cpu().numpy()
                bxy = np.array(pr.boxes.xyxy[pi].tolist(), np.float64).reshape(2, 2)
                kfull = kxy @ Minv[:, :2].T + Minv[:, 2]
                bfull = (bxy @ Minv[:, :2].T + Minv[:, 2]).ravel()
            else:
                kfull, kcf, bfull = np.full((17, 2), np.nan), np.zeros(17), np.array(boxes[rider], np.float64)
            rec["frame"].append(fi)
            rec["box"].append(bfull)
            rec["kp"].append(kfull)
            rec["kpc"].append(kcf)
            rec["M"].append(M)
            rec["crop_diff"].append(diff)
        arr = {k: np.array(v) for k, v in rec.items()}
        np.savez(out / f"prep_run{r['run']}.npz", fps=fps, run=r["run"], video=r["file"],
                 start=r["start"], end=r["end"], **arr)
        cd = arr["crop_diff"]
        print(f"  prep run {r['run']}: {len(arr['frame'])}/{len(want)}프레임, 저장 크롭과 평균 차이 "
              f"중앙값 {np.nanmedian(cd):.2f} 최대 {np.nanmax(cd):.2f} (0~255)")


# ---------------------------------------------------------------- infer (SAM 3D Body 환경)

class PromptHook:
    """YOLO 키포인트를 SAM 3D Body 키포인트 프롬프트로 넣는다.

    run_inference는 (1) 몸 디코더 → (2) 손 디코더 → (3) 손목·팔꿈치 프롬프트로 몸 재추론 순서다.
    (1) 직후 YOLO 프롬프트로 한 번 재추론하고, (3)에서도 YOLO 프롬프트를 앞에 붙인다.
    """

    def __init__(self, model):
        import torch
        self.torch = torch
        self.m = model
        self.kp = None  # (K, 3): 전체 프레임 x, y, MHR70 번호
        self._fs, self._rkp = model.forward_step, model.run_keypoint_prompt
        model.forward_step = self.forward_step
        model.run_keypoint_prompt = self.run_keypoint_prompt

    def _prompt(self, batch):
        torch = self.torch
        xy = torch.tensor(self.kp[None, :, :2], dtype=torch.float32, device=batch["img"].device)
        c = (self.m._full_to_crop(batch, xy) + 0.5).clamp(0.0, 1.0)
        lab = torch.tensor(self.kp[None, :, 2:3], dtype=torch.float32, device=c.device)
        return torch.cat([c, lab], -1)

    def forward_step(self, batch, decoder_type="body"):
        out = self._fs(batch, decoder_type=decoder_type)
        if decoder_type == "body" and self.kp is not None and len(self.kp):
            first = dict(out["mhr"])
            out, _ = self._rkp(batch, out, self._prompt(batch))
            for k, v in first.items():   # 재추론 출력에 없는 값(hand_box 등)은 첫 추론 값을 유지
                out["mhr"].setdefault(k, v)
        return out

    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        if self.kp is not None and len(self.kp):
            keypoint_prompt = self.torch.cat([self._prompt(batch), keypoint_prompt], 1)
        return self._rkp(batch, output, keypoint_prompt)


def clean_boxes(boxes, win=3, max_jump=0.35, max_scale=0.35):
    """앞뒤 win프레임 중앙값 박스와 중심이 박스 높이의 max_jump배 넘게 떨어지거나 크기가 max_scale 넘게 다르면
    중앙값 박스로 바꾼다(크롭 재검출이 다른 사람을 고른 경우). 반환: (박스, 바꿨는지)."""
    b = np.asarray(boxes, np.float64)
    out, fixed = b.copy(), np.zeros(len(b), bool)
    c, h = (b[:, :2] + b[:, 2:]) / 2, b[:, 3] - b[:, 1]
    for i in range(len(b)):
        nb = [j for j in range(max(0, i - win), min(len(b), i + win + 1)) if j != i]
        if len(nb) < 2:
            continue
        mc, mh = np.median(c[nb], 0), np.median(h[nb])
        if np.linalg.norm(c[i] - mc) > max_jump * mh or abs(h[i] / mh - 1) > max_scale:
            out[i], fixed[i] = np.median(b[nb], 0), True
    return out, fixed


def load_estimator(ckpt):
    home = os.environ.get("SAM3DBODY_HOME")
    if not home:
        sys.exit("SAM3DBODY_HOME 환경변수가 없다 (sam-3d-body 클론 폴더)")
    home = Path(home)
    sys.path.insert(0, str(home))
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    ck = home / "checkpoints" / ckpt
    with contextlib.redirect_stdout(io.StringIO()):
        model, mcfg = load_sam_3d_body(str(ck / "model.ckpt"), device="cuda", mhr_path=str(ck / "assets" / "mhr_model.pt"))
        est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=mcfg)
    return model, est


def cmd_infer(seg, cfg, runs, prompt, focal, ckpt, limit=None):
    import cv2
    import torch

    out = out_dir(seg, cfg)
    model, est = load_estimator(ckpt)
    hook = PromptHook(model) if prompt == "kp" else None
    tag = tag_of(prompt, focal)
    faces_saved = False
    for r in runs:
        prep = np.load(out / f"prep_run{r['run']}.npz")
        frames = [int(x) for x in prep["frame"]]
        need = dict(zip(frames, range(len(frames))))
        boxes, box_fixed = clean_boxes(prep["box"])
        if box_fixed.any():
            print(f"  run {r['run']}: 박스 이상치 보정 {[frames[i] for i in np.nonzero(box_fixed)[0]]}")
        if limit:
            keep = set(frames[:: max(1, len(frames) // limit)][:limit])
            need = {k: v for k, v in need.items() if k in keep}
        cap = cv2.VideoCapture(str(PATHS.video_dir(cfg) / r["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        f0 = max(0, int((r["start"] - PAD) * fps))   # analyze_lean.track_rider와 같은 탐색 후 순차 읽기
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        H = W = None
        res = {k: [] for k in ("frame", "kp3d", "kp2d", "cam_t", "verts", "jcoords", "global_rot", "box", "box_fixed")}
        fi = f0
        last = max(need) if need else -1
        while fi <= last:
            ok, img = cap.read()
            if not ok:
                break
            if fi in need:
                i = need[fi]
                H, W = img.shape[:2]
                K = torch.tensor([[[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1]]], dtype=torch.float32)
                box = boxes[i].astype(np.float32)
                if hook is not None:
                    kp, kc = prep["kp"][i], prep["kpc"][i] * (0.0 if box_fixed[i] else 1.0)  # 다른 사람 키포인트는 안 씀
                    sel = [j for j in BODY12 if kc[j] >= 0.5 and np.all(np.isfinite(kp[j]))]
                    hook.kp = np.array([[kp[j][0], kp[j][1], COCO2MHR[j]] for j in sel], np.float32).reshape(-1, 3)
                with contextlib.redirect_stdout(io.StringIO()):
                    o = est.process_one_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bboxes=box[None], cam_int=K)
                if o:
                    o = o[0]
                    res["frame"].append(fi)
                    res["kp3d"].append(o["pred_keypoints_3d"])
                    res["kp2d"].append(o["pred_keypoints_2d"])
                    res["cam_t"].append(o["pred_cam_t"])
                    res["verts"].append(o["pred_vertices"].astype(np.float16))
                    res["jcoords"].append(o["pred_joint_coords"])
                    res["global_rot"].append(o["global_rot"])
                    res["box"].append(box)
                    res["box_fixed"].append(bool(box_fixed[i]))
            fi += 1
        cap.release()
        arr = {k: np.array(v) for k, v in res.items()}
        np.savez_compressed(out / f"raw_{tag}_run{r['run']}.npz", focal=focal, width=W, height=H, prompt=prompt,
                            ckpt=ckpt, **arr)
        if not faces_saved:
            np.save(out / "mhr_faces.npy", est.faces.astype(np.int32))
            faces_saved = True
        print(f"  infer run {r['run']} [{tag}]: {len(arr['frame'])}/{len(need)}프레임")


# ---------------------------------------------------------------- 각도 (순수 numpy)

def gravity(pitch_deg=0.0):
    """카메라 좌표의 중력(아래) 방향. pitch_deg + = 폰이 아래를 봄."""
    p = math.radians(pitch_deg)
    return np.array([0.0, math.cos(p), math.sin(p)])


def horiz_basis(u):
    """수평면 기저 e1(카메라 x를 수평으로), e2 = u × e1 (카메라 앞쪽 수평)."""
    e1 = np.array([1.0, 0.0, 0.0])
    e1 = unit(e1 - (e1 @ u) * u)
    e2 = unit(np.cross(u, e1))
    return e1, e2


def yaw_of(vec, u, e1, e2):
    vh = vec - np.outer(vec @ u, u) if vec.ndim == 2 else vec - (vec @ u) * u
    return np.degrees(np.arctan2(vh @ e1, vh @ e2))


def hampel(x, win=7, k=3.0):
    x = np.asarray(x, float).copy()
    h = win // 2
    y = x.copy()
    for i in range(len(x)):
        w = x[max(0, i - h): i + h + 1]
        w = w[np.isfinite(w)]
        if len(w) < 3 or not np.isfinite(x[i]):
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if mad > 0 and abs(x[i] - med) > k * mad:
            y[i] = med
    return y


def gauss_smooth(x, sigma):
    """NaN을 건너뛰는 가우시안 평활(정규화 합성곱)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    pad = lambda a: np.concatenate([np.zeros(r), a, np.zeros(r)])
    num = np.convolve(pad(np.where(ok, x, 0.0)), k, mode="valid")
    den = np.convolve(pad(ok.astype(float)), k, mode="valid")
    y = num / np.maximum(den, 1e-9)
    y[den < 0.3] = np.nan
    return y


def smooth_series(x, sigma=1.5, keep=None):
    x = np.asarray(x, float).copy()
    if keep is not None:
        x[~keep] = np.nan
    return gauss_smooth(hampel(x), sigma)


def body_angles(kp3d, t, u):
    """kp3d: (N, 70, 3) 카메라 좌표, t: (N, 3) 진행 방향(수평 단위), u: (3,) 위."""
    P = (kp3d[:, M_LHIP] + kp3d[:, M_RHIP]) / 2
    Nk = kp3d[:, M_NECK]
    Wm = (kp3d[:, M_LWR] + kp3d[:, M_RWR]) / 2
    A = (kp3d[:, M_LAN] + kp3d[:, M_RAN]) / 2
    L = unit(np.cross(u[None], t))
    v = Nk - P
    h = Wm - A
    deg = np.degrees
    roll = deg(np.arctan2(np.sum(v * L, 1), v @ u))
    pitch = deg(np.arctan2(np.sum(v * t, 1), v @ u))
    bike = deg(np.arctan2(np.sum(h * L, 1), h @ u))
    bhat = unit(h - np.sum(h * t, 1, keepdims=True) * t)
    n = unit(np.cross(t, bhat))
    hip = np.sum((P - A) * n, 1) * 1000.0
    w = unit(kp3d[:, M_RWR] - kp3d[:, M_LWR])
    bar = deg(np.arcsin(np.clip(w @ u, -1, 1)))
    return dict(torso_roll=roll, torso_pitch=pitch, bike_lean_body=bike, hip_offset_mm=hip, bar_roll=bar)


def travel_dir(kp3d, u, yaw_offset=0.0, heading="pelvis"):
    """골반 앞 축의 yaw(프레임별), 평활 yaw, 평활 진행 방향 t, 손목-손목 기준 yaw.

    heading="bar"면 손목→손목 선에 수직인 수평 방향(조향 0 가정)을 평활해 진행 방향으로 쓴다(민감도 시험용).
    """
    e1, e2 = horiz_basis(u)
    P = (kp3d[:, M_LHIP] + kp3d[:, M_RHIP]) / 2
    U = kp3d[:, M_NECK] - P
    R = kp3d[:, M_RHIP] - kp3d[:, M_LHIP]
    F = np.cross(U, R)
    yaw_raw = yaw_of(F, u, e1, e2)
    w = kp3d[:, M_RWR] - kp3d[:, M_LWR]
    yaw_bar = yaw_of(np.cross(np.broadcast_to(u, w.shape), w), u, e1, e2)
    src = yaw_bar if heading == "bar" else yaw_raw
    yu = np.degrees(np.unwrap(np.radians(src)))
    ys = gauss_smooth(hampel(yu, 7, 3.0), 2.0)
    ys = np.where(np.isfinite(ys), ys, yu) + yaw_offset
    yr = np.radians(ys)
    t = np.sin(yr)[:, None] * e1[None] + np.cos(yr)[:, None] * e2[None]
    wrap = lambda a: (a + 180) % 360 - 180
    return wrap(yaw_raw), wrap(ys), t, wrap(yaw_bar)


def level_rot(up_c):
    """카메라 좌표 위쪽 up_c (N,3)를 −gravity(0) = (0,−1,0)으로 보내는 최소 회전 (N,3,3)."""
    a = unit(np.asarray(up_c, float))
    U0 = -gravity(0.0)
    v = np.cross(a, U0)
    s2 = np.sum(v * v, 1)
    c = a @ U0
    Kx = np.zeros((len(a), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    k = np.where(s2 > 1e-12, (1 - c) / np.maximum(s2, 1e-12), 0.5)
    return np.eye(3)[None] + Kx + k[:, None, None] * (Kx @ Kx)


def cam_tilt(up_c):
    """up_c (카메라 좌표) → 카메라 피치(도, + = 아래를 봄)와 롤(도, + = 위쪽이 화면 오른쪽으로 기움).
    롤은 GVHMR CSV의 cam_roll_deg와 같은 방향 정의(그쪽은 골반 위치에서 투영해 원근만큼 조금 다르다)."""
    a = unit(np.asarray(up_c, float))
    return np.degrees(np.arcsin(np.clip(-a[:, 2], -1, 1))), np.degrees(np.arctan2(a[:, 0], -a[:, 1]))


def _kabsch(A, B):
    """B ≈ R A + t 인 R (A, B: (N,3)). body3d_gvhmr.py와 같은 방법."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def _rodrigues(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    Kx = np.zeros(aa.shape[:-1] + (3, 3))
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    Kx[..., 1, 2], Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def gvhmr_frame(seg, cfg, run, frames, mode="frame", sigma=2.0):
    """GVHMR이 본 중력 위쪽 up과 골반 방향 fwd를 카메라 좌표로, 우리 frame 번호마다 돌려준다.

    입력: outputs/<세션>/<feature>_gvhmr/gvhmr_run<N>.npz, run<N>.csv (body3d_gvhmr.py 출력, 읽기만).
    - 카메라 회전 = 같은 프레임의 GVHMR incam 관절과 global 관절의 Kabsch 맞춤(body3d_gvhmr.py와 같음).
      up = R^T (0,1,0), fwd = R^T × (전역 회전 × 몸 +z의 수평 성분) = GVHMR 진행 방향(골반 방향).
    - npz frames는 순차 디코드 번호다. GVHMR CSV의 seq_frame − frame(코치 클립 −1, 런 1~7은 0)으로 맞춘다.
    - mode="frame": up을 성분별 가우시안(σ프레임) 평활 후 정규화. mode="median": 우리 구간의 중앙값 하나.
    - 중력 위쪽은 카메라 회전이라 초점거리와 무관하게 옮길 수 있다고 본다(GVHMR 초점거리 ~100mm 환산 ≠ 우리 1500px).
    """
    gd = PATHS.feat_dir(seg, cfg, "gvhmr")
    z = np.load(gd / f"gvhmr_run{run}.npz")
    offs = {int(r["seq_frame"]) - int(r["frame"]) for r in read_csv(gd / f"run{run}.csv")}
    if len(offs) != 1:
        raise ValueError(f"GVHMR frame 차이가 일정하지 않다: {offs}")
    off = offs.pop()
    Jc, Jg = z["joints_c"].astype(np.float64), z["joints_g"].astype(np.float64)
    R = np.stack([_kabsch(Jc[i], Jg[i]) for i in range(len(Jc))])
    up = np.einsum("lji,j->li", R, np.array([0.0, 1.0, 0.0]))
    face = _rodrigues(z["go_g"].astype(np.float64)) @ np.array([0.0, 0.0, 1.0])
    face[:, 1] = 0.0
    fwd = np.einsum("lji,lj->li", R, unit(face))
    nf = np.asarray(z["frames"]).astype(int)
    take = np.array([int(np.argmin(np.abs(nf - (f + off)))) for f in frames])
    gap = np.abs(nf[take] - (np.asarray(frames) + off))
    if mode == "median":
        up = np.repeat(unit(np.median(up[take], 0))[None], len(nf), 0)
    elif sigma:
        up = unit(np.stack([gauss_smooth(up[:, k], sigma) for k in range(3)], 1))
    return dict(up=up[take], fwd=unit(fwd[take]), src=f"gvhmr_{mode}", missing=int((gap > 0).sum()), off=off)


def conf_2d(kp2d, yolo_kp, yolo_c, kappa=0.1):
    """SAM 3D Body 2D 투영 vs YOLO 키포인트 일치도 (OKS 비슷한 0~1)."""
    out = np.full(len(kp2d), np.nan)  # NaN = YOLO 근거 부족(판단 불가), 0 = 불일치
    for i in range(len(kp2d)):
        kc, kp = yolo_c[i], yolo_kp[i]
        sel = [j for j in BODY12 if kc[j] >= 0.5 and np.all(np.isfinite(kp[j]))]
        if len(sel) < 4 or min(kc[5], kc[6], kc[11], kc[12]) < 0.3:
            continue
        T = np.linalg.norm((kp[5] + kp[6]) / 2 - (kp[11] + kp[12]) / 2)
        if T < 5:
            continue
        d = np.array([np.linalg.norm(kp2d[i][COCO2MHR[j]] - kp[j]) for j in sel])
        out[i] = float(np.mean(np.exp(-d ** 2 / (2 * (kappa * T) ** 2))))
    return out


def compute_rows(raw, prep, lean_rows, pitch=0.0, yaw_offset=0.0, sigma=1.5, heading="pelvis", grav=None):
    """raw/prep npz와 lean CSV 행으로 출력 행(dict)을 만든다. lean CSV의 모든 프레임에 한 행.

    grav: gvhmr_frame()의 결과(dict up, fwd, src). 주면 관절을 프레임마다 GVHMR 중력으로 수평 맞춤한 뒤 잰다
    (pitch 인자는 무시). heading="gvhmr"이면 진행 방향도 GVHMR 골반 방향을 쓴다(진단용, grav 필요)."""
    fr = [int(x) for x in raw["frame"]]
    kp3d = raw["kp3d"].astype(np.float64)
    pidx = {int(f): i for i, f in enumerate(prep["frame"])}
    pi = [pidx[f] for f in fr]
    if grav is not None:
        u = -gravity(0.0)
        Q = level_rot(grav["up"])
        kp3d = np.einsum("nij,nkj->nki", Q, kp3d)
        cpitch, croll = cam_tilt(grav["up"])
    else:
        u = -gravity(pitch)
        cpitch = croll = np.full(len(fr), np.nan)
    yaw_raw, yaw_s, t, yaw_bar = travel_dir(kp3d, u, yaw_offset, "bar" if heading == "bar" else "pelvis")
    if heading == "gvhmr":
        if grav is None:
            raise ValueError("heading=gvhmr에는 grav가 필요하다")
        f = np.einsum("nij,nj->ni", Q, grav["fwd"])
        t = unit(f - np.outer(f @ u, u))
        e1, e2 = horiz_basis(u)
        yaw_s = yaw_of(t, u, e1, e2)
    ang = body_angles(kp3d, t, u)
    conf = conf_2d(raw["kp2d"], prep["kp"][pi], prep["kpc"][pi])
    keep = ~(conf < 0.3)  # NaN(근거 없음)은 평활에 포함
    sm = {k: smooth_series(ang[k], sigma, keep) for k in ang}
    P = (kp3d[:, M_LHIP] + kp3d[:, M_RHIP]) / 2 + raw["cam_t"]
    ridx = {f: i for i, f in enumerate(fr)}
    rows = []
    for lr in lean_rows:
        f = int(lr["frame"])
        row = dict(run=lr["run"], frame=f, t=lr["t"], progress=lr["progress"], view_ratio=lr.get("view_ratio", ""))
        i = ridx.get(f)
        if i is not None:
            r1 = lambda v, d=1: round(float(v), d) if np.isfinite(v) else ""
            row.update(b3d_conf=r1(conf[i], 3),
                       torso_roll=r1(sm["torso_roll"][i]), torso_pitch=r1(sm["torso_pitch"][i]),
                       hip_offset_mm=r1(sm["hip_offset_mm"][i], 0), bike_lean_body=r1(sm["bike_lean_body"][i]),
                       torso_roll_raw=r1(ang["torso_roll"][i]), torso_pitch_raw=r1(ang["torso_pitch"][i]),
                       hip_offset_mm_raw=r1(ang["hip_offset_mm"][i], 0),
                       bike_lean_body_raw=r1(ang["bike_lean_body"][i]), bar_roll_raw=r1(ang["bar_roll"][i]),
                       bar_roll=r1(sm["bar_roll"][i]), travel_yaw=r1(yaw_s[i]), yaw_pelvis_raw=r1(yaw_raw[i]),
                       yaw_bar_raw=r1(yaw_bar[i]), cam_dist_m=r1(np.linalg.norm(P[i]), 2),
                       wrist_sep_m=r1(np.linalg.norm(kp3d[i, M_RWR] - kp3d[i, M_LWR]), 3),
                       shoulder_w_m=r1(np.linalg.norm(kp3d[i, M_RSH] - kp3d[i, M_LSH]), 3),
                       focal_px=r1(float(raw["focal"]), 0),
                       box_fixed=int(bool(raw["box_fixed"][i])) if "box_fixed" in raw.files else "",
                       grav_src=grav["src"] if grav is not None else f"image(pitch {pitch:+.0f})",
                       cam_pitch_g=r1(cpitch[i]), cam_roll_g=r1(croll[i]), heading_src=heading)
        rows.append(row)
    return rows


def load_run(seg, cfg, run, tag):
    out = out_dir(seg, cfg)
    raw = np.load(out / f"raw_{tag}_run{run}.npz")
    prep = np.load(out / f"prep_run{run}.npz")
    lean = read_csv(lean_csv(seg, cfg, run))
    return raw, prep, lean


def grav_of(seg, cfg, run, raw, gmode):
    """gmode: image(화면 아래 가정) | gvhmr(프레임별) | gvhmr_med(런 중앙값)."""
    if gmode == "image":
        return None
    frames = [int(f) for f in raw["frame"]]
    return gvhmr_frame(seg, cfg, run, frames, mode="median" if gmode == "gvhmr_med" else "frame")


def cmd_angles(seg, cfg, runs, tag, pitch=0.0, yaw_offset=0.0, suffix="", heading="pelvis", gmode="image"):
    out = out_dir(seg, cfg)
    keys = ["run", "frame", "t", "progress", "view_ratio"] + SHARED + RAW + EXTRA
    for r in runs:
        raw, prep, lean = load_run(seg, cfg, r["run"], tag)
        grav = grav_of(seg, cfg, r["run"], raw, gmode)
        if grav is not None and grav["missing"]:
            print(f"  run {r['run']}: GVHMR 창 밖 {grav['missing']}프레임은 가장 가까운 프레임 값")
        rows = compute_rows(raw, prep, lean, pitch, yaw_offset, heading=heading, grav=grav)
        p = out / f"run{r['run']}{suffix}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        n = sum(1 for x in rows if x.get("torso_roll_raw", "") != "")
        print(f"  angles run {r['run']} [{tag}]: {n}/{len(rows)}행 → {p.name}")


# ---------------------------------------------------------------- 몽타주 (메시 오버레이)

def project(X, K):
    return X[..., :2] / X[..., 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def render_mesh(img, verts, faces, K, M, color=(255, 200, 150), alpha=0.55):
    """화가 알고리즘으로 메시를 그린다. verts: 카메라 좌표(cam_t 더한 것). M: 전체 프레임→크롭 2x3."""
    import cv2
    uv = project(verts, K) @ M[:, :2].T + M[:, 2]
    tri = uv[faces]
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = unit(np.cross(v1 - v0, v2 - v0))
    shade = 0.35 + 0.65 * np.abs(n[:, 2])
    order = np.argsort(-(v0[:, 2] + v1[:, 2] + v2[:, 2]))
    layer = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    pts = np.round(tri * 16).astype(np.int32)
    col = np.array(color, float)
    for i in order:
        c = tuple(int(x) for x in col * shade[i])
        cv2.fillConvexPoly(layer, pts[i], c, lineType=cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(mask, pts[i], 255, shift=4)
    m = mask > 0
    outimg = img.copy()
    outimg[m] = (img[m] * (1 - alpha) + layer[m] * alpha).astype(np.uint8)
    return outimg


def montage_cell(seg, cfg, run, fi, raw, prep, faces, row, cell=384, label_extra=""):
    import cv2
    ldir = lean_dir(seg, cfg)
    crop = cv2.imread(str(ldir / "crops" / f"run{run}_f{fi:04d}.jpg"))
    if crop is None:
        return None
    ri = {int(f): i for i, f in enumerate(raw["frame"])}[fi]
    pi = {int(f): i for i, f in enumerate(prep["frame"])}[fi]
    M = prep["M"][pi]
    f, W, H = float(raw["focal"]), int(raw["width"]), int(raw["height"])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    vis = crop.copy()
    mpath = ldir / "masks" / f"run{run}_f{fi:04d}.png"
    if mpath.exists():
        bm = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        cs, _ = cv2.findContours((bm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 200, 255), 2, cv2.LINE_AA)
    verts = raw["verts"][ri].astype(np.float64) + raw["cam_t"][ri]
    vis = render_mesh(vis, verts, faces, K, M)
    kp = raw["kp3d"][ri].astype(np.float64) + raw["cam_t"][ri]
    uv = project(kp, K) @ M[:, :2].T + M[:, 2]
    P = (uv[M_LHIP] + uv[M_RHIP]) / 2
    Wm, A = (uv[M_LWR] + uv[M_RWR]) / 2, (uv[M_LAN] + uv[M_RAN]) / 2
    ip = lambda p: tuple(int(round(v)) for v in p)
    cv2.line(vis, ip(A), ip(Wm), (0, 140, 255), 3, cv2.LINE_AA)            # 손-발 선 (자전거 중심면 근사)
    cv2.line(vis, ip(P), ip(uv[M_NECK]), (60, 230, 60), 3, cv2.LINE_AA)    # 상체 (골반→목)
    cv2.line(vis, ip(uv[M_LWR]), ip(uv[M_RWR]), (255, 0, 255), 2, cv2.LINE_AA)  # 손목-손목
    for j in (M_LAN, M_RAN, M_LWR, M_RWR):
        cv2.circle(vis, ip(uv[j]), 5, (255, 255, 255), -1, cv2.LINE_AA)
    vis = cv2.resize(vis, (cell, cell), interpolation=cv2.INTER_AREA)

    def g(k):
        v = row.get(k, "")
        return float(v) if v not in ("", None) else float("nan")
    roll, pitch, bike, conf = g("torso_roll"), g("torso_pitch"), g("bike_lean_body"), g("b3d_conf")
    lines = [f"f{fi} vr{g('view_ratio'):.2f} yaw{g('travel_yaw'):+.0f} c{conf:.2f}{label_extra}",
             f"roll{roll:+.0f} pitch{pitch:+.0f} bike{bike:+.0f} sep{bike - roll:+.0f}",
             f"2D body{g('body_lean_2d'):+.0f} sil{g('fit_lean_s'):+.0f} bar{g('bar_roll'):+.0f}"]
    for k, s in enumerate(lines):
        y = 18 + 18 * k
        cv2.putText(vis, s, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, s, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def merged_rows(seg, cfg, run, suffix=""):
    """sam3dbody CSV + lean CSV(body_lean, fit_lean_s 등)를 프레임으로 합친다."""
    rows = read_csv(out_dir(seg, cfg) / f"run{run}{suffix}.csv")
    lean = {int(x["frame"]): x for x in read_csv(lean_csv(seg, cfg, run))}
    for x in rows:
        lr = lean.get(int(x["frame"]), {})
        x["body_lean_2d"] = lr.get("body_lean", "")
        for k in ("fit_lean_s", "fit_lowconf", "fit_yaw", "fit_iou", "torso_conf"):
            x[k] = lr.get(k, "")
    return rows


def cmd_montage(seg, cfg, runs, tag, n=16, cols=4, frames=None, name=None):
    import cv2
    out = out_dir(seg, cfg)
    faces = np.load(out / "mhr_faces.npy")
    for r in runs:
        raw, prep, _ = load_run(seg, cfg, r["run"], tag)
        rows = {int(x["frame"]): x for x in merged_rows(seg, cfg, r["run"])}
        have = [int(f) for f in raw["frame"]]
        pick = frames or [have[int(round(i))] for i in np.linspace(0, len(have) - 1, min(n, len(have)))]
        cells = [montage_cell(seg, cfg, r["run"], f, raw, prep, faces, rows.get(f, {})) for f in pick if f in have]
        cells = [c for c in cells if c is not None]
        if not cells:
            continue
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        p = out / (name or f"montage_run{r['run']}.jpg")
        cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  montage run {r['run']}: {p}")


# ---------------------------------------------------------------- 자체 시험

def cmd_selftest():
    """합성 뼈대: 알려진 roll·pitch·자전거 기울기·엉덩이 오프셋을 yaw 0/90/180에서 되찾는지 본다."""
    u = -gravity(0.0)
    e1, e2 = horiz_basis(u)
    worst = 0.0
    for yaw in (0.0, 90.0, 180.0, -60.0):
        y = math.radians(yaw)
        t = math.sin(y) * e1 + math.cos(y) * e2
        L = unit(np.cross(u, t))
        Rt = -L
        for roll, pitch, lean, off in [(15, 40, 25, 60), (-10, 20, 5, -30), (0, 0, 0, 0), (30, 55, 40, 120)]:
            v = u + math.tan(math.radians(roll)) * L + math.tan(math.radians(pitch)) * t
            b = math.cos(math.radians(lean)) * u + math.sin(math.radians(lean)) * L
            n = unit(np.cross(t, b))
            A = np.array([0.3, 0.2, 6.0])
            Wm = A + 0.56 * b + 0.5 * t
            P = A + 0.6 * b + 0.1 * t + off / 1000.0 * n
            kp = np.zeros((70, 3))
            kp[M_LHIP], kp[M_RHIP] = P - 0.15 * Rt, P + 0.15 * Rt
            kp[M_NECK] = P + 0.5 * unit(v)
            kp[M_LWR], kp[M_RWR] = Wm - 0.33 * n, Wm + 0.33 * n
            kp[M_LAN], kp[M_RAN] = A - 0.12 * n, A + 0.12 * n
            kps = np.repeat(kp[None], 9, 0)
            yr, ys, tt, yb = travel_dir(kps, u)
            a = body_angles(kps, tt, u)
            got = (a["torso_roll"][4], a["torso_pitch"][4], a["bike_lean_body"][4], a["hip_offset_mm"][4],
                   a["bar_roll"][4], ys[4], yb[4])
            exp = (roll, pitch, lean, off, lean, (yaw + 180) % 360 - 180, (yaw + 180) % 360 - 180)
            err = max(abs(g - e) for g, e in zip(got, exp))
            worst = max(worst, err)
            if err > 1e-6:
                print("  불일치", yaw, (roll, pitch, lean, off), "→", np.round(got, 3))
    # 기울어진 카메라: 같은 뼈대를 카메라 롤 7°·피치 5°로 돌려 놓고 up_c를 주면 같은 각도가 나와야 한다
    roll, pitch, lean, off, yaw = 20, 35, 30, 50, 25.0
    y = math.radians(yaw)
    t = math.sin(y) * e1 + math.cos(y) * e2
    L = unit(np.cross(u, t))
    v = u + math.tan(math.radians(roll)) * L + math.tan(math.radians(pitch)) * t
    b = math.cos(math.radians(lean)) * u + math.sin(math.radians(lean)) * L
    n = unit(np.cross(t, b))
    A = np.array([0.3, 0.2, 6.0])
    P = A + 0.6 * b + 0.1 * t + off / 1000.0 * n
    kp = np.zeros((70, 3))
    kp[M_LHIP], kp[M_RHIP] = P + 0.15 * L, P - 0.15 * L
    kp[M_NECK] = P + 0.5 * unit(v)
    kp[M_LWR], kp[M_RWR] = A + 0.56 * b + 0.5 * t - 0.33 * n, A + 0.56 * b + 0.5 * t + 0.33 * n
    kp[M_LAN], kp[M_RAN] = A - 0.12 * n, A + 0.12 * n
    cr, sr = math.cos(math.radians(7)), math.sin(math.radians(7))
    cp, sp = math.cos(math.radians(5)), math.sin(math.radians(5))
    Rc = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    kps = np.repeat((kp @ Rc.T)[None], 9, 0)
    up_c = np.repeat((u @ Rc.T)[None], 9, 0)
    kpl = np.einsum("nij,nkj->nki", level_rot(up_c), kps)
    _, _, tt, _ = travel_dir(kpl, u)
    a = body_angles(kpl, tt, u)
    got = (a["torso_roll"][4], a["torso_pitch"][4], a["bike_lean_body"][4], a["hip_offset_mm"][4], a["bar_roll"][4])
    err = max(abs(g - e) for g, e in zip(got, (roll, pitch, lean, off, lean)))
    cpitch, croll = cam_tilt(up_c)
    tilt_ok = err < 1e-6 and abs(cpitch[0] - 5) < 1e-6 and abs(croll[0] - 7) < 1e-6
    a0 = body_angles(kps, travel_dir(kps, u)[2], u)
    print(f"selftest: 최대 오차 {worst:.2e}", "OK" if worst < 1e-6 else "FAIL")
    print(f"selftest 카메라 기울기: 수평 맞춤 후 오차 {err:.2e}, 피치 {cpitch[0]:+.2f} 롤 {croll[0]:+.2f} "
          f"(맞춤 없이 재면 롤 {a0['torso_roll'][4]:+.1f}°, 자전거 {a0['bike_lean_body'][4]:+.1f}°)",
          "OK" if tilt_ok else "FAIL")
    return worst < 1e-6 and tilt_ok


# ---------------------------------------------------------------- 메인

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["prep", "infer", "angles", "montage", "board", "hints", "all", "selftest"])
    ap.add_argument("segments", nargs="?")
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--prompt", choices=["box", "kp"], default="box")
    ap.add_argument("--focal", type=float, default=DEFAULT_FOCAL)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--limit", type=int, default=None, help="infer: 런마다 고르게 N프레임만 (시험용)")
    ap.add_argument("--pitch", type=float, default=0.0, help="angles: 카메라 피치 가정(도, + = 아래를 봄)")
    ap.add_argument("--yaw-offset", type=float, default=0.0, help="angles: 진행 방향 yaw에 더할 값(민감도 시험)")
    ap.add_argument("--suffix", default="", help="angles: 출력 CSV 이름 꼬리(민감도 시험용)")
    ap.add_argument("--heading", choices=["pelvis", "bar", "gvhmr"], default="pelvis",
                    help="angles: 진행 방향 출처 (gvhmr = GVHMR 골반 방향, 진단용, --gravity gvhmr* 필요)")
    ap.add_argument("--gravity", choices=["image", "gvhmr", "gvhmr_med"], default="image",
                    help="angles: 중력 출처. image = 화면 아래 가정(기본), gvhmr = GVHMR 프레임별 중력, "
                         "gvhmr_med = GVHMR 중력의 런 중앙값")
    ap.add_argument("--frames", type=int, nargs="*", help="montage: 고를 프레임 번호")
    ap.add_argument("--name", default=None, help="montage: 출력 파일 이름")
    ap.add_argument("--ref", default=None, help="board: 코치 segments.json")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if cmd_selftest() else 1)
    seg = Path(a.segments)
    cfg = PATHS.load_cfg(seg)
    runs = sel_runs(cfg, a.runs)
    tag = tag_of(a.prompt, a.focal)
    if a.cmd in ("prep", "all"):
        cmd_prep(seg, cfg, runs)
    if a.cmd == "infer":
        cmd_infer(seg, cfg, runs, a.prompt, a.focal, a.ckpt, a.limit)
    if a.cmd == "all":
        home = os.environ.get("SAM3DBODY_HOME")
        if not home:
            sys.exit("SAM3DBODY_HOME 환경변수가 없다")
        py = Path(home) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        cmd = [str(py), str(Path(__file__).resolve()), "infer", str(seg), "--prompt", a.prompt,
               "--focal", str(a.focal), "--ckpt", a.ckpt] + (["--runs", *map(str, a.runs)] if a.runs else [])
        subprocess.run(cmd, check=True)
    if a.cmd in ("angles", "all"):
        cmd_angles(seg, cfg, runs, tag, a.pitch, a.yaw_offset, a.suffix, a.heading, a.gravity)
    if a.cmd in ("montage", "all"):
        cmd_montage(seg, cfg, runs, tag, frames=a.frames, name=a.name)
    if a.cmd == "board":
        import body3d_board
        body3d_board.make_board(seg, cfg, a.ref, tag)
    if a.cmd == "hints":
        import body3d_board
        ents = [("coach", Path(a.ref), PATHS.load_cfg(a.ref))] if a.ref else []
        ents.append((cfg.get("session", "run"), seg, cfg))
        body3d_board.eval_bar_hints(ents, tag, out_dir(seg, cfg))


if __name__ == "__main__":
    main()
