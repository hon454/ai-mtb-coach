"""3D 자전거 실루엣을 SAM2 마스크에 맞춰 자전거의 기울기(lean)·방향(yaw)·피치를 추정한다.

시점이 바뀌어도(뒷모습 ↔ 비스듬 ↔ 옆모습) 같은 의미의 기울기를 얻기 위한 방법.
지오메트리는 세션별: segments.json의 "bike" 블록 (없으면 Moterra SL 1 L). 형식:
    "bike": {"preset": "moterra_sl_l" | "generic", "model": "설명(선택)", "note": "(선택)",
             아래 키 중 바꿀 것만 (mm / 도): wheel_radius, tire_width, bb_drop, chainstay, wheelbase,
             reach, stack, head_angle, head_tube, seat_angle, seat_tube, down_tube_width,
             bar_width, bar_rise, bar_backsweep, bar_upsweep, stem_length, stem_stack, crank}
    프리셋 값은 BIKE_PRESETS 참고. 모르는 키는 오류.

사용법 (analyze_lean.py → bike_sam2.py 다음):
    uv run python tools/bike_fit.py sessions/2026-11-21/segments.json [--runs 1] [--workers 24]
    uv run python tools/bike_fit.py sessions/reference-coach/segments.json --runs 0

좌표: 자전거 X=앞, Y=위, Z=라이더 왼쪽. 원점=BB. 단위 mm.
lean +: 라이더 왼쪽으로 기움 (카메라 수평 가정, 중력 기준).
yaw 0: 카메라에서 멀어지는 방향(완전한 뒷모습). ±90: 옆모습. yaw + = 화면 오른쪽으로 진행.

중요한 성질 (정사영 + 좌우대칭 모델):
    (yaw, lean, pitch) 와 (180°−yaw, −lean, pitch) 는 실루엣이 완전히 같다 (깊이 거울상).
    즉 "앞/뒤 뒤집힘"은 실제로는 lean 부호의 선택 문제이고, 마스크만으로는 구분 불가.
알고리즘:
    1) 프레임별로 yaw∈[-90,90] 반공간을 (yaw, lean, pitch) 격자로 거칠게 탐색 (96px, 크기·위치는 bbox로 맞춤)
    2) 상위 국소해 몇 개를 192px에서 Nelder-Mead로 다듬음 (프레임끼리 독립 → 병렬)
    3) 각 후보에 거울쌍(twin)을 추가 (렌더 불필요, IoU 동일)
    4) 런 전체에 Viterbi: 데이터항(1−IoU + 접점 사전) + 연속성(Δyaw, Δlean, Δpitch, Δlog s) + 약한 뒷모습 사전
       (어깨 순서 × view_ratio 신뢰도) + 회전 방향 사전(좌회전이면 yaw 역회전에 L1 벌점).
       옆모습 근처에서는 연속성이 lean 부호를 이어받는다.
    5) 선택 해를 이웃 프레임에 씨앗으로 전파해 다시 다듬고 Viterbi 반복 (PROP_ITERS).
    6) 2패스: 1차 결과의 비스듬/옆모습 고IoU 프레임으로 런 단위 크기를 정하고, 2차에서 모든 프레임을 거기에 묶는다
       (크롭이 라이더 bbox로 정규화되므로 자전거 크기는 거의 일정 — 뒷모습에서 다리에 가린 바퀴로 모델이 줄어드는 것 방지).
    접점 사전(거울쌍 불변): 손목(신뢰도·팔뚝 길이로 거름) ↔ 라이저바 그립, 발목 ↔ 페달 원, 엉덩이 ↔ BB 위·뒤,
    머리가 엉덩이보다 진행 방향 쪽.
    → 옆모습에서 실루엣이 거의 앞뒤 대칭이라 생기는 진행 방향(앞/뒤 바퀴) 혼동을 막는다.
출력 열: fit_yaw/lean/pitch/iou, fit_lean_s(평활), fit_back(뒷모습 사전 −1..1),
    fit_lean_unc(거의 같은 손실의 다른 해와의 lean 차, 하한), fit_sign_ambig(1=부호는 연속성이 정함),
    fit_scale(px/m @192), fit_occl(berm_line occl_wheel: 1=바퀴 아래가 앞 둔덕에 가림), fit_axis_err(뒷모습 2D 주축 차),
    fit_ncomp(주 성분과 떨어진 가늘고 긴 마스크 성분 수, 진단용 — 덮기 항 W_COMP는 현재 0),
    fit_lowconf(1=IoU<0.45, 또는 |yaw|<35에서 IoU<0.55·주축 차>10°·lean_unc>15° → 분리각 통계에서 제외).
    bike 블록: wheel_radius_front/rear(멀렛), bb_height(>0이면 허브 높이 = 반지름 − BB 높이), 예전 wheel_radius도 허용.
한계: 정사영(원근 무시), 카메라 수평 가정 — 옆모습의 lean 크기는 카메라 높이각만큼 오차가 날 수 있다.
"""
import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

GRID = 192   # 다듬기 해상도
CGRID = 96   # 거친 탐색 해상도
SHIFT = 2    # cv2 서브픽셀 비트
SUB = 1 << SHIFT

# --- 단순화한 자전거 모델 (세션별 설정: segments.json의 "bike" 블록, 없으면 Moterra SL L) ---
BIKE_PRESETS = {
    # Cannondale Moterra SL 1 (2025), L (bikeinsights / singletrackworld; 콕핏은 Cannondale 공식 스펙:
    #  HollowGram SAVE 카본 라이저바 780mm, 30mm 라이즈, 8° 백스윕, 5° 업스윕; 스템 Cannondale 1, 0°.
    #  스템 길이는 스펙에 없어 50mm, 스페이서+스템 높이 절반 40mm로 가정)
    #  멀렛(MX): 앞 29×2.5 (DHF, 반지름 ~375), 뒤 27.5×2.4 (Dissector, 584/2+60 ≈ 352). BB 높이 335, 크랭크 165
    #  (사용자 소유 자전거 판매처 스펙표)
    "moterra_sl_l": dict(
        wheel_radius_front=375.0, wheel_radius_rear=352.0, bb_height=335.0,
        tire_width=62.0, bb_drop=30.0, chainstay=453.0, wheelbase=1283.0,
        reach=470.0, stack=648.0, head_angle=62.5, head_tube=135.0, seat_angle=77.0, seat_tube=520.0,
        down_tube_width=95.0, bar_width=780.0, bar_rise=30.0, bar_backsweep=8.0, bar_upsweep=5.0,
        stem_length=50.0, stem_stack=40.0, crank=165.0),
    # 모델을 모를 때: 일반적인 29" 트레일/엔듀로, L 사이즈, 780mm 라이저바 (배터리 없는 다운튜브)
    "generic": dict(
        wheel_radius_front=372.0, wheel_radius_rear=372.0, bb_height=0.0, tire_width=60.0, bb_drop=30.0, chainstay=440.0, wheelbase=1250.0,
        reach=480.0, stack=630.0, head_angle=64.5, head_tube=110.0, seat_angle=77.0, seat_tube=460.0,
        down_tube_width=65.0, bar_width=780.0, bar_rise=30.0, bar_backsweep=8.0, bar_upsweep=5.0,
        stem_length=45.0, stem_stack=35.0, crank=170.0),
}
DEFAULT_PRESET = "moterra_sl_l"
BIKE = {}
NC = 32  # 바퀴 원 꼭짓점 수
_th = np.linspace(0, 2 * np.pi, NC, endpoint=False)
_RISE_Z0, _RISE_Z1 = 45.0, 160.0  # 바: 클램프 평탄부 끝 → 라이즈 끝(이후 스윕 구간), 중심에서 mm
WRIST_OFF = np.array([-45.0, 45.0, 0.0])  # 손목 관절은 그립 축보다 뒤·위
HIP_T = np.array([-150.0, 850.0, 0.0])    # 서서 타는 자세의 엉덩이 대략 위치 (BB 기준)


def bike_config(block):
    """segments.json "bike" 블록 → 치수 dict. {"preset": 이름, 키: 값 ...}; 없으면 기본 프리셋."""
    block = dict(block or {})
    name = block.pop("preset", DEFAULT_PRESET)
    if name not in BIKE_PRESETS:
        raise ValueError(f"알 수 없는 bike preset: {name} (가능: {list(BIKE_PRESETS)})")
    cfg = dict(BIKE_PRESETS[name])
    block.pop("model", None)
    block.pop("note", None)
    if "wheel_radius" in block:  # 앞뒤 같은 바퀴 (예전 형식)
        r = block.pop("wheel_radius")
        block.setdefault("wheel_radius_front", r)
        block.setdefault("wheel_radius_rear", r)
    unknown = set(block) - set(cfg)
    if unknown:
        raise ValueError(f"bike 블록의 알 수 없는 키: {sorted(unknown)}")
    cfg.update({k: float(v) for k, v in block.items()})
    cfg["preset"] = name
    return cfg


def _bar_pt(z):
    """라이저바 중심선 위 점 (z ≥ 0 한쪽, mm). 클램프 → 라이즈 → 백스윕·업스윕 구간."""
    if z <= _RISE_Z0:
        return BARC + [0, 0, z]
    if z <= _RISE_Z1:
        t = (z - _RISE_Z0) / (_RISE_Z1 - _RISE_Z0)
        return BARC + [0, BAR_RISE * (3 * t * t - 2 * t ** 3), z]
    d = z - _RISE_Z1
    return BARC + [-d * math.tan(math.radians(BAR_BACK)), BAR_RISE + d * math.tan(math.radians(BAR_UP)), z]


def set_bike(cfg=None):
    """자전거 치수로 모델 전역값을 (다시) 만든다. 워커 프로세스에서도 initializer로 호출."""
    global WRF, WRR
    global BIKE, WR, TIRE, WB, REAR, FRONT, BB, HTOP, HBOT, SEAT, BARC, BAR_RISE, BAR_BACK, BAR_UP
    global STEM_TOP, BAR_POLY, GRIP_Z, P3, NW, SEG_W, _circle, _OV_WHEELS, _OV_FRAME
    global CRANK, GRIPS, PEDALS, ANCH, NPED, _GRID_CACHE
    b = bike_config(None) if cfg is None else cfg
    BIKE = b
    WRF, WRR = b["wheel_radius_front"], b["wheel_radius_rear"]
    WR = WRF  # (몸통 비 점검 등 '바퀴 하나'가 필요할 때: 앞바퀴)
    TIRE, WB = b["tire_width"], b["wheelbase"]
    hta = math.radians(b["head_angle"])
    if b.get("bb_height", 0) > 0:   # 허브 높이 = 바퀴 반지름 − BB 높이 (멀렛이면 앞뒤가 다름)
        yr, yf = WRR - b["bb_height"], WRF - b["bb_height"]
    else:
        yr = yf = b["bb_drop"]
    cs_h = math.sqrt(b["chainstay"] ** 2 - yr ** 2)
    REAR = np.array([-cs_h, yr, 0.0])
    FRONT = np.array([WB - cs_h, yf, 0.0])
    BB = np.zeros(3)
    HTOP = np.array([b["reach"], b["stack"], 0.0])
    HBOT = HTOP + b["head_tube"] * np.array([math.cos(hta), -math.sin(hta), 0])
    SEAT = b["seat_tube"] * np.array([-math.cos(math.radians(b["seat_angle"])), math.sin(math.radians(b["seat_angle"])), 0])
    steer = np.array([-math.cos(hta), math.sin(hta), 0.0])     # 헤드튜브 축(위쪽)
    stem_dir = np.array([math.sin(hta), math.cos(hta), 0.0])   # 0° 스템: 축에 수직, 앞쪽
    STEM_TOP = HTOP + b["stem_stack"] * steer
    BARC = STEM_TOP + b["stem_length"] * stem_dir              # 바 클램프 중심
    BAR_RISE, BAR_BACK, BAR_UP = b["bar_rise"], b["bar_backsweep"], b["bar_upsweep"]
    half = b["bar_width"] / 2
    zs = [0.0, _RISE_Z0, 90.0, 125.0, _RISE_Z1, half]
    BAR_POLY = np.array([_bar_pt(z) * [1, 1, -1] for z in zs[::-1]] + [_bar_pt(z) for z in zs[1:]])
    GRIP_Z = half - 65.0   # 손 중심 (바 끝에서 65mm 안쪽)
    Z = lambda q, z: q + [0, 0, z]
    segs = [  # (a, b, 폭mm)
        (HBOT, BB, b["down_tube_width"]),  # 다운튜브(배터리)
        (HTOP, SEAT, 55),                  # 탑튜브
        (BB, SEAT, 55),                    # 시트튜브
        (HTOP, STEM_TOP, 40),              # 스페이서
        (STEM_TOP, BARC, 40),              # 스템
    ]
    for sg in (-1, 1):  # 뒷삼각은 뒤 허브 쪽으로 벌어짐(부스트 148 → 드롭아웃 ±74) — 정면 뒷모습의 '쐐기' 실루엣
        segs += [(Z(BB, 45 * sg), Z(REAR, 70 * sg), 35),       # 체인스테이
                 (Z(SEAT, 22 * sg), Z(REAR, 70 * sg), 25),     # 시트스테이
                 (Z(HBOT, 55 * sg), Z(FRONT, 58 * sg), 45)]    # 포크 다리
    segs += [(Z(REAR, -78), Z(REAR, 78), 45),   # 뒤 허브·드롭아웃·변속기 폭
             (Z(FRONT, -58), Z(FRONT, 58), 35)]  # 앞 허브
    segs += [(BAR_POLY[i], BAR_POLY[i + 1], 30) for i in range(len(BAR_POLY) - 1)]  # 라이저바
    circ = lambda r: np.stack([np.cos(_th) * r, np.sin(_th) * r, np.zeros_like(_th)], 1)
    _circle = circ(WRF)
    # 모든 3D 점을 한 배열로: [바퀴 4개 × NC (뒤−, 뒤+, 앞−, 앞+)] + [선분 끝점 쌍들]
    pts = [circ(r) + hub + [0, 0, z] for hub, r in ((REAR, WRR), (FRONT, WRF)) for z in (-TIRE / 2, TIRE / 2)]
    seg_w = []
    for a, c, w in segs:
        pts.append(np.array([a, c]))
        seg_w.append(w)
    P3 = np.concatenate(pts, 0)
    NW = 4 * NC
    SEG_W = np.array(seg_w, float)
    # 오버레이용 중심면 와이어프레임: 바퀴 / 프레임 / 바를 따로
    _OV_WHEELS = [circ(WRR) + REAR, circ(WRF) + FRONT]
    _OV_FRAME = [(HBOT, BB), (HTOP, SEAT), (BB, SEAT), (BB, REAR), (SEAT, REAR), (HBOT, FRONT), (HTOP, HBOT),
                 (HTOP, STEM_TOP), (STEM_TOP, BARC)]
    # 접점 앵커
    CRANK = b["crank"]
    GRIPS = np.array([_bar_pt(GRIP_Z) + WRIST_OFF, _bar_pt(GRIP_Z) * [1, 1, -1] + WRIST_OFF])
    pa = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    PEDALS = np.concatenate([np.stack([np.cos(pa) * CRANK, np.sin(pa) * CRANK, np.full(12, z)], 1) for z in (-110.0, 110.0)])
    ANCH = np.vstack([GRIPS, BB[None], [[1000.0, 0, 0]], HIP_T[None], PEDALS,
                      REAR[None], FRONT[None], [[0, 1000.0, 0]]])
    # 0,1 그립 / 2 BB / 3 진행축 / 4 엉덩이 / 5..28 페달 / -3,-2 허브 / -1 위쪽축
    NPED = len(PEDALS)
    _GRID_CACHE = None  # 거친 격자는 모델이 바뀌면 다시


PITCH_MAX = math.radians(35)
LEAN_MAX = math.radians(75)


def rot(yaw, lean, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cl, sl = math.cos(lean), math.sin(lean)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])        # 피치(앞 들림 +)
    Rx = np.array([[1, 0, 0], [0, cl, -sl], [0, sl, cl]])        # 린(위쪽이 +Z=왼쪽)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])        # 요
    return Ry @ Rx @ Rz


def unit_uv(R):
    """자전거 좌표 → 이미지(단위 배율, mm). 카메라: x오른쪽, y아래. 자전거 X→깊이, Y→위, Z→-x."""
    W = P3 @ R.T
    return np.stack([-W[:, 2], -W[:, 1]], 1)


def twin(p):
    """깊이 거울쌍: 실루엣이 같은 (180°−yaw, −lean, pitch)."""
    q = np.array(p, float).copy()
    q[0] = wrap(math.pi - q[0])
    q[1] = -q[1]
    return q


def raster(uv, s, tx, ty, size, img=None, outline=False):
    canvas = img if img is not None else np.zeros((size, size), np.uint8)
    if outline:
        pts = np.round(uv * s + [tx, ty]).astype(np.int32)
        col = (0, 255, 0)
        for i in range(4):
            cv2.polylines(canvas, [pts[i * NC:(i + 1) * NC]], True, col, 2, cv2.LINE_AA)
        for j in range(len(SEG_W)):
            a, b = pts[NW + 2 * j], pts[NW + 2 * j + 1]
            cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), col, 2, cv2.LINE_AA)
        return canvas
    pts = np.round((uv * s + [tx, ty]) * SUB).astype(np.int32)
    for i in range(4):  # 한 번에 넘기면 겹친 부분이 even-odd로 비워지므로 하나씩
        cv2.fillPoly(canvas, [pts[i * NC:(i + 1) * NC]], 255, 8, SHIFT)
    th = np.maximum(1, np.round(SEG_W * s)).astype(int)
    for j in range(len(SEG_W)):
        a, b = pts[NW + 2 * j], pts[NW + 2 * j + 1]
        cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, int(th[j]), 8, SHIFT)
    return canvas


def draw_overlay(params, img, kpts=None):
    """params는 img 해상도 기준. 바퀴=노랑, 프레임=초록, 바=빨강, 그립·페달=흰 점, 매칭된 손목·발목·엉덩이=하늘색 + 연결선."""
    yaw, lean, pitch, logs, tx, ty = params
    R = rot(yaw, lean, pitch)
    s = math.exp(logs)

    def P(X):
        W = np.atleast_2d(X) @ R.T
        return np.stack([-W[:, 2] * s + tx, -W[:, 1] * s + ty], 1)

    ip = lambda q: (int(round(q[0])), int(round(q[1])))
    for wpts in _OV_WHEELS:
        cv2.polylines(img, [np.round(P(wpts)).astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    for a, b in _OV_FRAME:
        q = P(np.array([a, b]))
        cv2.line(img, ip(q[0]), ip(q[1]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.round(P(BAR_POLY)).astype(np.int32)], False, (60, 60, 255), 3, cv2.LINE_AA)
    ped = P(np.stack([np.cos(_th) * CRANK, np.sin(_th) * CRANK, np.zeros_like(_th)], 1))
    cv2.polylines(img, [np.round(ped).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    a = anchor_uv(R) * s + [tx, ty]
    for g in a[0:2]:
        cv2.circle(img, ip(g), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, ip(a[2]), 5, (255, 255, 255), -1, cv2.LINE_AA)
    for x, y, w, kind in (kpts or []):
        if kind not in (0, 1, 4):
            continue
        m, _ = kp_match(a, x, y, kind)
        cv2.line(img, ip((x, y)), ip(m), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, ip((x, y)), 6, (255, 255, 0), -1, cv2.LINE_AA)
    return img


def render(params, size, outline=False, img=None):
    yaw, lean, pitch, logs, tx, ty = params
    return raster(unit_uv(rot(yaw, lean, pitch)), math.exp(logs), tx, ty, size, img, outline)


def iou(Rm, M, valid):
    inter = np.count_nonzero(Rm & M & valid)
    union = np.count_nonzero((Rm | M) & valid)
    return inter / max(union, 1)


# 라이더 접점 사전 (프레임 안쪽이 라이더에 가려도 바 높이·BB 위치를 고정):
#   손목 ↔ 가까운 그립의 손목 위치(라이저바 그립 중심 + 손목 오프셋), 발목 ↔ 페달 원(BB ± 크랭크 170mm, 좌우 페달면) 위 가장 가까운 점,
#   엉덩이 중점 ↔ BB 위·뒤의 그럴듯한 위치. 거리는 mm(px/s)로 재고 신뢰도로 가중.
#   좌우 라벨은 쓰지 않고 가까운 쪽에 붙이므로 거울쌍에 대해 불변.
#   kpts 항목 (x, y, w, kind): 0=손목, 1=발목, 2=머리(x=(머리−엉덩이).x/몸통), 3=크기 사전(x=log s0), 4=엉덩이 중점,
#   5=마스크 최저 y = 타이어 최저점 (양방향; 현재 미사용 — occl_wheel=0을 믿을 수 없어 끔, run_one 주석),
#   6=마스크 최저 y는 가림선 → 타이어 최저점이 그보다 아래 (occl_wheel=1; 그 아래 픽셀은 IoU에서 제외)
W_KP = 0.3
# 마스크 아래끝 ↔ 타이어 최저점: 뱅크 앞 둔덕이 바퀴 아래를 가리는 프레임이 많다(정면 뒷모습은 거의 전부).
# 가림 여부를 모르고 양방향으로 맞추면 lean이 부풀었다(run4 f212·f216 +40~50°) → berm_line.py의 occl_wheel로 구분해
# 가려진 프레임(occl_wheel=1)만 단방향(6) + 아래쪽 무시. occl_wheel=0은 바닥 정보 없음으로 취급.
W_GROUND = 0.12
KP_TOL = {0: (60.0, 150.0), 1: (110.0, 150.0), 4: (250.0, 200.0)}  # kind: (허용 mm, 척도 mm)
W_HEAD = 0.15  # 머리가 엉덩이보다 진행 방향 쪽에 있다는 사전 (옆모습에서만 작동, lean과 무관)
W_SCALE, SIG_SCALE = 0.01, 0.06  # 런 단위 크기 사전: 6% 벗어나면 0.01, 20%면 0.11


def anchor_uv(R):
    W = ANCH @ R.T
    return np.stack([-W[:, 2], -W[:, 1]], 1)


def kp_match(a, x, y, kind):
    """화면상의 앵커들 a (px)에서 키포인트 (x, y)에 대응하는 점과 거리(px)."""
    if kind == 0:
        cand = a[0:2]
    elif kind == 1:
        cand = a[5:5 + NPED]
    else:
        cand = a[4:5]
    d = np.hypot(cand[:, 0] - x, cand[:, 1] - y)
    i = int(d.argmin())
    return cand[i], float(d[i])


def kp_term(auv, s, tx, ty, kpts):
    if not kpts:
        return 0.0
    a = auv * s + [tx, ty]
    tot, ws, extra = 0.0, 0.0, 0.0
    for x, y, w, kind in kpts:
        if kind == 7:
            continue
        if kind == 2:  # 진행 방향의 화면 x성분과 (머리−엉덩이) 부호가 반대면 벌점
            fx = (auv[3, 0] - auv[2, 0]) / 1000.0
            if fx * x < 0:
                extra += W_HEAD * min(abs(fx) / 0.7, 1.0) * min(abs(x) / 0.4, 1.0)
            continue
        if kind == 3:
            extra += W_SCALE * ((math.log(s) - x) / SIG_SCALE) ** 2
            continue
        if kind in (5, 6):  # x = 마스크 최저 y(px). 바퀴 원의 화면 최저점 = 허브 v + R·|바퀴면 축들의 v성분|
            hv = math.hypot(auv[3, 1] / 1000.0, auv[-1, 1] / 1000.0)
            low = max(auv[-3, 1] + WRR * hv, auv[-2, 1] + WRF * hv) * s + ty
            # 5: 접지가 보임 → 양방향. 6: 아래끝이 앞 둔덕 가림선 → 타이어가 그보다 위에 떠 있으면만 벌점
            d = abs(low - x) if kind == 5 else max(0.0, x - low)
            e = max(0.0, d / s - 40.0) / 120.0
            extra += W_GROUND * w * e * e / (1 + e * e)
            continue
        _, d = kp_match(a, x, y, kind)
        tol, sc = KP_TOL[kind]
        e = max(0.0, d / s - tol) / sc
        tot += w * e * e / (1 + e * e)  # 로버스트 (오검출 키포인트는 포화)
        ws += w
    return W_KP * tot / max(ws, 2.0) + extra


# 따로 떨어진 가는 마스크 성분(예: 다리 옆으로 보이는 앞바퀴 띠)은 IoU에서 픽셀 수가 적어 무시되기 쉽다.
# → 이런 성분은 모델의 어느 부분이든 덮어야 한다는 항. kpts 항목 (성분 bool 배열, 픽셀 수, w, 7).
# 현재 꺼 둠(0): run4 f205–f222 시험에서 이 항이 성분을 yaw가 아니라 lean으로 덮으려 해 lean이 부풀었다
# (W=0.15: f212 +41°, 0.05: f209–f222 +29~45°, 0.03: f212는 유지되나 f216 띠는 여전히 안 덮이고 f221 +30°).
# 성분 탐지(fit_ncomp)는 진단용으로 남긴다.
W_COMP = 0.0


def comp_term(Rm, kpts):
    t = 0.0
    for cm, area, w, kind in kpts:
        if kind == 7:
            t += W_COMP * w * (1.0 - np.count_nonzero(Rm & cm) / max(area, 1))
    return t


def loss(params, M, valid, kpts=None, parts=False):
    if abs(params[2]) > PITCH_MAX or abs(params[1]) > LEAN_MAX:
        return 1.0 + abs(params[2]) + abs(params[1])
    R = rot(*params[:3])
    s = math.exp(params[3])
    Rm = raster(unit_uv(R), s, params[4], params[5], M.shape[0]) > 0
    v = iou(Rm, M, valid)
    h = kp_term(anchor_uv(R), s, params[4], params[5], kpts)
    if kpts:
        h += comp_term(Rm, kpts)
    return (v, h) if parts else 1.0 - v + h


# --- 거친 격자 (워커마다 한 번 계산) ---
C_YAW = np.radians(np.arange(-90, 91, 6))
C_LEAN = np.radians(np.arange(-72, 73, 6))
C_PITCH = np.radians(np.arange(-30, 31, 10))
def coarse_grid():
    global _GRID_CACHE
    if _GRID_CACHE is None:
        uvs, boxes, huvs = [], [], []
        for y in C_YAW:
            for l in C_LEAN:
                for p in C_PITCH:
                    R = rot(y, l, p)
                    uv = unit_uv(R)
                    huvs.append(anchor_uv(R))
                    lo, hi = uv.min(0), uv.max(0)
                    uvs.append(uv)
                    boxes.append((lo[0], lo[1], hi[0], hi[1]))
        _GRID_CACHE = (np.array(uvs), np.array(boxes), np.array(huvs))
    return _GRID_CACHE


def mask_box(M):
    ys, xs = np.nonzero(M)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def coarse_scores(Mc, Ic, kpts):
    uvs, boxes, huvs = coarse_grid()
    valid = ~Ic
    x0, y0, x1, y1 = mask_box(Mc)
    wm, hm = x1 - x0, y1 - y0
    cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
    S = np.zeros(len(uvs))
    params = np.zeros((len(uvs), 3))
    ls0 = next((x for x, _, _, k in kpts if k == 3), None)
    has_comp = any(k == 7 for _, _, _, k in kpts)
    for i, (uv, b, hv) in enumerate(zip(uvs, boxes, huvs)):
        wu, hu = b[2] - b[0], b[3] - b[1]
        if ls0 is None:
            s = math.sqrt(wm * hm / max(wu * hu, 1.0))
            tx = cxm - s * (b[0] + b[2]) / 2
            ty = cym - s * (b[1] + b[3]) / 2
        else:  # 크기 고정: 바퀴 접지(마스크 아래끝)는 거의 항상 보이므로 아래끝·가로 중심을 맞춤
            s = math.exp(ls0)
            tx = cxm - s * (b[0] + b[2]) / 2
            ty = y1 - s * b[3]
        Rc = raster(uv, s, tx, ty, CGRID) > 0
        S[i] = iou(Rc, Mc, valid) - kp_term(hv, s, tx, ty, kpts) - (comp_term(Rc, kpts) if has_comp else 0.0)
        params[i] = (math.log(s), tx, ty)
    return S.reshape(len(C_YAW), len(C_LEAN), len(C_PITCH)), params


STEP = np.array([0.12, 0.12, 0.12, 0.06, 3.0, 3.0])  # 다듬기 초기 심플렉스 (rad / log s / px@GRID)


def nm(x0, M, valid, kpts, iters, step=STEP):
    sim = np.vstack([x0] + [x0 + np.eye(6)[i] * step for i in range(6)])
    return minimize(loss, x0, args=(M, valid, kpts), method="Nelder-Mead",
                    options=dict(maxiter=iters, initial_simplex=sim, xatol=1e-4, fatol=1e-6))


def fit_frame(args):
    """한 프레임: 거친 탐색 → 상위 K개 다듬기. 반환: [(params(6), iou, loss), ...] (반공간 표현)."""
    M, I, K, hand = args
    valid = ~I
    kc = CGRID / GRID
    hand_c = []
    for x, y, w, k in hand:
        if k == 7:  # 성분 마스크를 거친 해상도로
            cm = cv2.resize(x.astype(np.uint8), (CGRID, CGRID), interpolation=cv2.INTER_AREA) > 0
            hand_c.append((cm, max(1, int(cm.sum())), w, k))
        else:
            hand_c.append((x, y, w, k) if k == 2 else (x + math.log(kc), y, w, k) if k == 3
                          else (x * kc, y, w, k) if k in (5, 6) else (x * kc, y * kc, w, k))
    Mc = cv2.resize(M.astype(np.uint8), (CGRID, CGRID), interpolation=cv2.INTER_AREA) > 0
    Ic = cv2.resize(I.astype(np.uint8), (CGRID, CGRID), interpolation=cv2.INTER_AREA) > 0
    S, sp = coarse_scores(Mc, Ic, hand_c)
    # 국소 최대 (yaw 축은 ±90 에서 거울쌍으로 이어지지만 여기서는 단순히 경계 처리)
    peak = (S == maximum_filter(S, size=(3, 3, 3), mode="nearest"))
    idx = np.argsort(-np.where(peak, S, -np.inf).ravel())
    chosen = []
    for f in idx:
        if len(chosen) >= K:  # (kp 벌점 때문에 점수가 음수일 수 있어 0 기준으로 자르지 않음)
            break
        iy, il, ip = np.unravel_index(f, S.shape)
        y, l = C_YAW[iy], C_LEAN[il]
        if any(abs(y - C_YAW[a]) < math.radians(13) and abs(l - C_LEAN[b]) < math.radians(13) for a, b, _ in chosen):
            continue
        chosen.append((iy, il, ip))
    k = GRID / CGRID
    out = []
    for iy, il, ip in chosen:
        f = np.ravel_multi_index((iy, il, ip), S.shape)
        logs, tx, ty = sp[f]
        x0 = np.array([C_YAW[iy], C_LEAN[il], C_PITCH[ip], logs + math.log(k), tx * k, ty * k])
        r = nm(x0, M, valid, hand, 300)
        r2 = nm(r.x, M, valid, hand, 200, STEP * 0.4)  # IoU는 계단 함수라 한 번 재시작
        x = (r2 if r2.fun <= r.fun else r).x.copy()
        x[0] = wrap(x[0])
        v, h = loss(x, M, valid, hand, parts=True)
        out.append((x, v, 1.0 - v + h))
    # 수렴 후 중복 제거
    out.sort(key=lambda t: t[2])
    uniq = []
    for c in out:
        if all(angdist(c[0], u[0]) > math.radians(6) for u in uniq):
            uniq.append(c)
    return uniq


def refine_seeds(args):
    """이웃 프레임에서 고른 해를 씨앗으로 이 프레임에서 다시 다듬기 (후보 전파)."""
    M, I, hand, seeds = args
    valid = ~I
    out = []
    for x0 in seeds:
        r = nm(np.array(x0, float), M, valid, hand, 300)
        r2 = nm(r.x, M, valid, hand, 150, STEP * 0.4)
        x = (r2 if r2.fun <= r.fun else r).x.copy()
        x[0] = wrap(x[0])
        v, h = loss(x, M, valid, hand, parts=True)
        out.append((x, v, 1.0 - v + h))
    return out


def merge_cands(base, extra):
    allc = sorted(base + extra, key=lambda t: t[2])
    uniq = []
    for c in allc:
        if all(angdist(c[0], u[0]) > math.radians(5) for u in uniq):
            uniq.append(c)
    return uniq


def with_twins(cands):
    full = []
    for c in cands:
        full.append(c)
        full.append((twin(c[0]), c[1], c[2]))
    return full


def angdist(a, b):
    return max(abs(wrap(a[0] - b[0])), abs(a[1] - b[1]), abs(a[2] - b[2]))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ignore_mask(row_kp, size):
    """라이더 몸통·팔다리 영역(자전거를 가리는 부분)은 비교에서 제외. kp는 CROP 좌표."""
    import analyze_lean as AL
    I = np.zeros((size, size), np.uint8)
    if row_kp is None:
        return I.astype(bool)
    k, c = row_kp
    k = k * (size / AL.CROP)
    torso = np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)
    th = max(2, int(torso * 0.22))
    for a, b in [(11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (5, 11), (6, 12)]:
        if c[a] > 0.2 and c[b] > 0.2:
            t = max(2, int(torso * 0.12)) if (a, b) in ((7, 9), (8, 10)) else th  # 팔뚝은 가늘게: 바 끝이 보이게
            cv2.line(I, tuple(k[a].astype(int)), tuple(k[b].astype(int)), 1, t)
    if min(c[[5, 6, 11, 12]]) > 0.2:  # 몸통 사각형 채우기
        cv2.fillPoly(I, [k[[5, 6, 12, 11]].astype(np.int32)], 1)
    return I.astype(bool)


def wrist_ok(k, c, i):
    """뒷모습에서 몸통 뒤에 가린 손은 YOLO가 등·팔꿈치 근처에 낮은 신뢰도로 찍는다 → 핸들바를 끌지 않게 거른다.
    조건: 손목 신뢰도 ≥ 0.5, 팔꿈치 신뢰도 ≥ 0.5, 팔뚝/위팔 길이비 0.5–1.5."""
    el, sh = i - 2, i - 4
    if c[i] < 0.5 or c[el] < 0.5:
        return False
    up = np.linalg.norm(k[el] - k[sh])
    fo = np.linalg.norm(k[i] - k[el])
    return up > 1 and 0.5 <= fo / up <= 1.5


def contact_kpts(kp, size):
    """손목(9,10)·발목(15,16)·엉덩이 중점·머리 → [(x, y, w, kind)] (size 좌표). 신뢰도 0.4 이상만."""
    import analyze_lean as AL
    if kp is None:
        return []
    k, c = kp
    out = []
    for i, kind in ((9, 0), (10, 0), (15, 1), (16, 1)):
        if kind == 0 and not wrist_ok(k, c, i):
            continue
        if c[i] > 0.4:
            x, y = k[i] * (size / AL.CROP)
            out.append((float(x), float(y), float(c[i]), kind))
    if min(c[11], c[12]) > 0.4:
        x, y = (k[11] + k[12]) / 2 * (size / AL.CROP)
        out.append((float(x), float(y), 0.5 * float(min(c[11], c[12])), 4))
    head = [i for i in (0, 1, 2, 3, 4) if c[i] > 0.4]
    if head and min(c[11], c[12]) > 0.3 and min(c[5], c[6]) > 0.3:
        hp = (k[11] + k[12]) / 2
        torso = np.linalg.norm((k[5] + k[6]) / 2 - hp)
        if torso > 1:
            out.append((float((k[head].mean(0)[0] - hp[0]) / torso), 0.0, 1.0, 2))
    return out


def back_prior(kp, view_ratio):
    """+1 = 확실한 뒷모습, −1 = 확실한 앞모습, 0 = 모름. 어깨 순서 × 어깨폭 비율 × 신뢰도."""
    if kp is None:
        return 0.0
    k, c = kp
    sgn = 1.0 if k[5][0] < k[6][0] else -1.0  # 뒤에서 보면 왼어깨가 화면 왼쪽
    try:
        vr = float(view_ratio)
    except (TypeError, ValueError):
        return 0.0
    strength = float(np.clip((vr - 0.45) / 0.3, 0, 1)) * float(np.clip((min(c[5], c[6]) - 0.3) / 0.4, 0, 1))
    return sgn * strength


# --- Viterbi ---
W_DATA = 25.0      # (1−IoU) 0.04 차이 = 1
SIG = dict(yaw=math.radians(12), lean=math.radians(8), pitch=math.radians(10), logs=0.08)
TERM_CAP = 12.0    # 한 항의 최대 비용 (진짜 급변 허용)
PROP_ITERS = 2     # 후보 전파 반복 수
W_BACK = 4.0
W_PITCH, SIG_PITCH = 1.0, math.radians(15)  # 지형 피치 약한 사전: 뒷모습에선 피치가 거의 안 보여 바퀴가 가림 영역으로 미끄러짐
TURN = 0           # +1 좌회전, −1 우회전, 0 없음 (main에서 feature 이름으로 설정)
REV_FREE, W_REV = 1.0, 0.3  # 역방향 yaw 변화 허용량(도/프레임), 초과 1도당 비용


def unary(c, b):
    x, v, L = c
    cost = W_DATA * L + W_PITCH * min((x[2] / SIG_PITCH) ** 2, 4.0)
    if b != 0:
        cost += W_BACK * abs(b) * (1 - np.sign(b) * math.cos(x[0])) / 2
    return cost


def pair(x, y, g):
    t = 0.0
    if TURN:  # 회전 방향 사전: 좌회전이면 yaw(카메라 기준 진행 방향)는 줄어드는 쪽. 역회전만 벌점.
        rev = TURN * wrap(y[0] - x[0])   # 좌회전(+1)에서 yaw 증가량
        # L1: 조금씩 계속 거꾸로 도는 것(누적)도 막도록 선형 벌점
        rd = math.degrees(rev)
        if rd < -90:  # 180° 가까운 점프는 "많이 돈 것"이 아니라 뒤집힘 → 역회전으로 취급
            rd = -rd
        t += W_REV * max(0.0, rd - REV_FREE * g)
    for i, (key, d) in enumerate([("yaw", wrap(x[0] - y[0])), ("lean", x[1] - y[1]),
                                  ("pitch", x[2] - y[2]), ("logs", x[3] - y[3])]):
        t += min((d / (SIG[key] * g)) ** 2, TERM_CAP)
    return t


def viterbi(frames):
    """frames: [(frame_no, cands[(x, iou)], b)] → 프레임별 선택 index."""
    if not frames:
        return []
    D = [np.array([unary(c, frames[0][2]) for c in frames[0][1]])]
    back = []
    for t in range(1, len(frames)):
        g = max(1, frames[t][0] - frames[t - 1][0])
        if g > 6:   # 큰 공백이면 연속성 끊기
            g = 1e3
        prev, cur = frames[t - 1][1], frames[t][1]
        T = np.array([[pair(p[0], c[0], g) for p in prev] for c in cur])  # (cur, prev)
        tot = T + D[-1][None, :]
        back.append(tot.argmin(1))
        D.append(tot.min(1) + np.array([unary(c, frames[t][2]) for c in cur]))
    path = [int(D[-1].argmin())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return path[::-1]


def lean_ambiguity(c, cands, dl=0.02):
    """선택 해와 같은 방향(yaw ±60°)인 후보 중 손실이 거의 같은(최저+dl 이내) 것들의 lean 최대 차이(도),
    그리고 부호가 반대인 후보(옆모습 근처의 거울쌍)가 그 안에 있는지. 찾은 국소해만 보므로 하한값.
    → fit_lean_unc, fit_sign_ambig (1이면 lean 부호는 실루엣이 아니라 시간 연속성이 정한 것)."""
    lim = max(c[2], min(d[2] for d in cands)) + dl
    alt = [d for d in cands if d[2] <= lim and abs(wrap(d[0][0] - c[0][0])) < math.radians(60)]
    same = [d for d in alt if d[0][1] * c[0][1] >= 0 or abs(d[0][1]) < math.radians(8)]
    unc = max(abs(math.degrees(d[0][1] - c[0][1])) for d in same) if same else 0.0
    flip = any(d[0][1] * c[0][1] < 0 and abs(d[0][1]) >= math.radians(8) for d in alt)
    return unc, int(flip)


def smooth(vals, frames):
    """중앙값(3) → 가우시안(σ=1프레임). 연속 구간 안에서만."""
    v = np.array(vals, float)
    out = v.copy()
    n = len(v)
    med = v.copy()
    for i in range(1, n - 1):
        if frames[i + 1] - frames[i - 1] <= 4:
            med[i] = np.median(v[i - 1:i + 2])
    w = np.array([0.25, 0.5, 0.25])
    for i in range(n):
        acc, ws = 0.0, 0.0
        for d, wi in zip((-1, 0, 1), w):
            j = i + d
            if 0 <= j < n and abs(frames[j] - frames[i]) <= 2:
                acc += wi * med[j]
                ws += wi
        out[i] = acc / ws
    return out


def solve(jobs, meta, pool):
    """프레임별 후보 → Viterbi → 이웃 해 전파 반복. 반환 (frames, sel)."""
    results = list(pool.map(fit_frame, jobs, chunksize=1))
    cands = [list(c) for c in results]
    cent = [np.argwhere(j[0]).mean(0)[::-1] for j in jobs]  # 마스크 무게중심 (x, y)
    for it in range(PROP_ITERS + 1):
        frames = [(int(m[0]["frame"]), with_twins(c), m[2]) for m, c in zip(meta, cands)]
        path = viterbi(frames)
        sel = [f[1][i] for f, i in zip(frames, path)]
        if it == PROP_ITERS:
            break
        # 후보 전파: 이웃(±1, ±2)에서 고른 해를 이 프레임 위치로 옮겨 씨앗으로
        jobs2 = []
        for t in range(len(frames)):
            seeds = []
            for d in (-2, -1, 1, 2):
                u = t + d
                if 0 <= u < len(frames) and abs(frames[u][0] - frames[t][0]) <= 3:
                    x0 = sel[u][0].copy()
                    x0[4:6] += cent[t] - cent[u]
                    if all(angdist(x0, c[0]) > math.radians(4) for c in with_twins(cands[t])):
                        seeds.append(x0)
            jobs2.append((jobs[t][0], jobs[t][1], jobs[t][3], seeds))
        extra = list(pool.map(refine_seeds, jobs2, chunksize=1))
        cands = [merge_cands(c, e) for c, e in zip(cands, extra)]
    return frames, sel


def torso_len(kp, size):
    import analyze_lean as AL
    if kp is None:
        return float("nan")
    k, c = kp
    if min(c[[5, 6, 11, 12]]) < 0.5:
        return float("nan")
    return float(np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2) * size / AL.CROP)


RATIO_T = 1.5  # 화면상 앞바퀴 지름 / 몸통(어깨–엉덩이) — 학생 7런의 뒷모습 아닌 프레임 중앙 1.45–1.5


def run_scale(frames, sel, torsos, vrs):
    """런 단위 log s(t). 크롭이 라이더 bbox로 정규화되므로 자전거 크기는 거의 일정 — 느린 선형 변화만 허용.
    신뢰 프레임: |sin yaw| > 0.5 (두 바퀴가 다 보임)만, IoU 상위. 정면 뒷모습 프레임은 1차 맞춤에서 크기가
    붕괴(가린 바퀴 대신 보이는 뒷타이어에 맞춰 작아짐)하므로 절대 쓰지 않는다 — 코치 클립에서 30% 작게 나온 원인.
    옆모습 프레임이 3개 미만이면 몸통 길이 × RATIO_T로 대신한다."""
    fr = np.array([f[0] for f in frames], float)
    ls = np.array([c[0][3] for c in sel])
    v = np.array([c[1] for c in sel])
    side = np.abs(np.sin([c[0][0] for c in sel])) > 0.5
    T = np.array(torsos, float)
    vr = np.array([float(x) if x not in ("", None) else np.nan for x in vrs])
    tmask = np.isfinite(T) & ~(vr > 0.75)  # 뒷모습은 몸통이 앞으로 숙여 짧아 보이므로 제외
    ls_t = float(np.log(np.median(RATIO_T * T[tmask] / (2 * WRF)))) if tmask.sum() >= 3 else None
    ok = side & (v >= max(0.65, np.percentile(v[side], 50) if side.sum() else 1))
    if ok.sum() < 6:
        ok = side & (v >= 0.6)
    if ok.sum() < 6 and side.sum() >= 3:
        ok = side & (v >= np.percentile(v[side], 50))
    src = "옆모습"
    if ok.sum() < 3:
        if ls_t is None:
            ok = v >= np.percentile(v, 60)
            src = "IoU 상위(대체)"
        else:
            ok = np.zeros(len(v), bool)
            src = "몸통×%.1f" % RATIO_T
    med = float(np.median(ls[ok])) if ok.any() else ls_t
    slope = 0.0
    if ok.sum() >= 8 and np.ptp(fr[ok]) > 15:  # Theil–Sen 기울기, 런 전체 ±15% 이내로 제한
        xs, ys = fr[ok], ls[ok]
        sl = [(ys[j] - ys[i]) / (xs[j] - xs[i]) for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i]]
        slope = float(np.clip(np.median(sl), -0.14 / max(np.ptp(fr), 1), 0.14 / max(np.ptp(fr), 1)))
    fc = float(np.median(fr[ok])) if ok.any() else float(np.median(fr))
    ls0 = med + slope * (fr - fc)
    ratio = np.exp(ls0) * 2 * WRF / T  # 화면상 앞바퀴 지름 / 몸통 길이
    used = fr[ok].astype(int)
    info = (f"기준={src} {int(ok.sum())}프레임" + (f" (f{used.min()}–f{used.max()}, IoU 중앙 {np.median(v[ok]):.2f})" if ok.any() else "")
            + f", s={np.exp(med) * 1000:.1f}px/m@{GRID}, 변화 {slope * np.ptp(fr) * 100:+.0f}%/런, "
            f"앞바퀴지름/몸통(뒷모습 제외) 중앙 {np.nanmedian(ratio[tmask]) if tmask.any() else float('nan'):.2f} (기대 ~1.4–1.6)"
            + (f", 몸통 기준 s={np.exp(ls_t) * 1000:.1f} ({(np.exp(med - ls_t) - 1) * 100:+.0f}%)" if ls_t is not None else ""))
    if ls_t is not None and abs(med - ls_t) > 0.2:
        info += "  ※ 몸통 기준과 20% 넘게 다름 — 확인 필요"
    return ls0, info


LEAN_UNC_TOL = 15.0  # fit_lean_unc가 이보다 크면(뒷모습) 저신뢰
AXIS_TOL = 10.0  # 뒷모습(|yaw|<20)에서 마스크와 모델의 2D 주축이 이보다 다르면 저신뢰


COMP_MIN_AREA, COMP_MIN_ELONG, COMP_MIN_LEN = 30, 2.2, 0.35  # 192px 기준 / √λ비 / 주 성분 높이 대비


def side_components(M, valid):
    """주 성분과 따로 떨어진 가늘고 긴 마스크 성분 (먼지·신발 조각 배제: 크기·길쭉함·길이·높이 겹침)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats((M & valid).astype(np.uint8), connectivity=8)
    if n <= 2:
        return []
    main = 1 + int(np.argmax(st[1:, 4]))
    my0, mh = st[main, 1], st[main, 3]
    out = []
    for i in range(1, n):
        if i == main or st[i, 4] < COMP_MIN_AREA:
            continue
        cm = lab == i
        ang, el = _axis(cm)
        length = math.hypot(st[i, 2], st[i, 3])
        y0, h = st[i, 1], st[i, 3]
        ov = max(0, min(y0 + h, my0 + mh) - max(y0, my0)) / max(h, 1)
        if el < COMP_MIN_ELONG or length < COMP_MIN_LEN * mh or ov < 0.6:
            continue  # 둥근 조각(먼지·신발), 짧은 조각, 주 성분 아래(그림자)
        out.append(dict(mask=cm, area=int(st[i, 4]), c=cen[i], ang=ang, mh=mh))
    return out


def persistent_components(comps, cents, frames_no):
    """이웃(±2프레임 이내) 중 하나에 비슷한 위치(마스크 무게중심 기준)·방향의 성분이 있는 것만 남김."""
    keep = []
    for t, cs in enumerate(comps):
        k = []
        for c in cs:
            rel = c["c"] - cents[t]
            ok = False
            for u in (t - 2, t - 1, t + 1, t + 2):
                if 0 <= u < len(comps) and abs(frames_no[u] - frames_no[t]) <= 3:
                    for d in comps[u]:
                        da = abs(c["ang"] - d["ang"]) % 180
                        if np.linalg.norm((d["c"] - cents[u]) - rel) < 0.3 * c["mh"] and min(da, 180 - da) < 25:
                            ok = True
            if ok:
                k.append(c)
        keep.append(k[:2])
    return keep


def load_occlusion(out, run):
    """berm_line.py 결과(<feature>_berm/run<N>.csv)의 occl_wheel: 1=바퀴 아래가 앞 둔덕에 가림, 0=접지가 보임."""
    p = out.parent / out.name.replace("_lean", "_berm") / f"run{run}.csv"
    if not p.exists():
        return {}
    d = {}
    for x in csv.DictReader(open(p, encoding="utf-8")):
        if x.get("occl_wheel") in ("0", "1"):
            d[int(x["frame"])] = int(x["occl_wheel"])
    return d


def _axis(mask):
    """2D 주축의 기울기(화면 수직 기준, 도)와 길쭉함(√λ1/λ2)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return float("nan"), 0.0
    P = np.stack([xs, ys], 1).astype(float)
    P -= P.mean(0)
    w, V = np.linalg.eigh(P.T @ P / len(P))
    v = V[:, 1]
    return math.degrees(math.atan2(v[0], v[1])) % 180, math.sqrt(w[1] / max(w[0], 1e-6))


def axis_error(p, M, I):
    """보이는 영역(무시 영역 밖)에서 마스크와 모델 실루엣의 2D 주축 차(도). 둘 다 길쭉할 때만(아니면 nan).
    정면 뒷모습에선 뒷타이어·뒷삼각 '쐐기'의 기울기가 곧 lean이라, 이 값이 크면 lean을 믿기 어렵다."""
    valid = ~I
    Rm = render(p, M.shape[0]) > 0
    a1, e1 = _axis(M & valid)
    a2, e2 = _axis(Rm & valid)
    if min(e1, e2) < 1.6 or a1 != a1 or a2 != a2:
        return float("nan")
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def run_one(r, out, pose_m, pool, K):
    import analyze_lean as AL
    t0 = time.time()
    run = r["run"]
    csv_path = out / f"run{run}.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    (out / "overlay_fit").mkdir(exist_ok=True)
    for x in rows:
        for k in ("fit_yaw", "fit_lean", "fit_pitch", "fit_iou", "fit_lean_s", "fit_back", "fit_lean_unc", "fit_sign_ambig",
                  "fit_scale", "fit_occl", "fit_axis_err", "fit_lowconf", "fit_ncomp"):
            x[k] = ""
    jobs, meta, kps, torsos = [], [], [], []
    occl = load_occlusion(out, run)
    for x in rows:
        name = f"run{run}_f{int(x['frame']):04d}"
        mp = out / "masks" / f"{name}.png"
        if not mp.exists():
            continue
        M = cv2.resize(cv2.imread(str(mp), 0), (GRID, GRID), interpolation=cv2.INTER_AREA) > 127
        if M.sum() < 60:
            continue
        crop = cv2.imread(str(out / "crops" / f"{name}.jpg"))
        pr = pose_m.predict(crop, imgsz=AL.CROP, classes=[0], verbose=False)[0]
        pi = AL.pick_center_person(pr)
        kp = None
        if pi is not None:
            kp = (pr.keypoints.xy[pi].cpu().numpy(), pr.keypoints.conf[pi].cpu().numpy())
        I = ignore_mask(kp, GRID)
        kpl = contact_kpts(kp, GRID)
        ybot = float(np.nonzero(M)[0].max()) + 0.5
        oc = occl.get(int(x["frame"]))
        if oc == 1:     # 아래끝 = 앞 둔덕 가림선: 그 아래는 보이지 않으므로 비교에서 제외, 타이어는 그보다 아래
            I = I.copy()
            I[int(ybot) + 1:, :] = True
            kpl.append((ybot, 0.0, 1.0, 6))
        # oc == 0 (접지가 보인다고 표시된 프레임)에 '타이어 최저점 = 마스크 아래끝'을 양방향으로 걸면 run4 f212가
        # yaw +11 / lean −3으로 뒤집혔다(절제 실험: 이 항 하나만 끄면 fix2와 같은 +24/+16). occl_wheel=0이 실제로는
        # 둔덕 립에 걸린 프레임도 있어 믿기 어려움 → 0이면 바닥 정보 없음으로 취급.
        x["fit_occl"] = "" if oc is None else oc
        jobs.append((M, I, K, kpl))
        kps.append(kp)
        torsos.append(torso_len(kp, GRID))
        meta.append((x, name, back_prior(kp, x.get("view_ratio"))))
    # 따로 떨어진 가는 성분(앞바퀴 띠 등): 이웃 프레임에도 있는 것만 '덮어야 할 성분'으로
    comps = [side_components(j[0], ~j[1]) for j in jobs]
    cents = [np.argwhere(j[0]).mean(0)[::-1] for j in jobs]
    comps = persistent_components(comps, cents, [int(m[0]["frame"]) for m in meta])
    for i, cs in enumerate(comps):
        M_, I_, K_, kp_ = jobs[i]
        jobs[i] = (M_, I_, K_, kp_ + [(c["mask"], c["area"], 1.0, 7) for c in cs])
        meta[i][0]["fit_ncomp"] = len(cs)
    t1 = time.time()
    # 1차: 크기 자유 → 두 바퀴가 다 보이는 비스듬/옆모습 고IoU 프레임에서 런 단위 크기 추정
    frames, sel = solve(jobs, meta, pool)
    ls0, info = run_scale(frames, sel, torsos, [m[0].get("view_ratio") for m in meta])
    print(f"  run {run}: 크기 사전 {info}")
    # 2차: 모든 프레임을 런 크기(느린 변화만 허용)에 묶어서 다시 맞춤
    jobs = [(M, I, K_, kp + [(float(ls0[i]), 0.0, 1.0, 3)]) for i, (M, I, K_, kp) in enumerate(jobs)]
    frames, sel = solve(jobs, meta, pool)
    t2 = time.time()
    fr = [f[0] for f in frames]
    lean_s = smooth([math.degrees(c[0][1]) for c in sel], fr) if sel else []
    scale = GRID / AL.CROP
    for (x, name, b), c, ls, f, kp, job in zip(meta, sel, lean_s, frames, kps, jobs):
        p, v, _ = c
        unc, flip = lean_ambiguity(c, f[1])
        x["fit_lean_unc"], x["fit_sign_ambig"] = round(unc, 1), flip
        ae = axis_error(p, job[0], job[1])
        x["fit_axis_err"] = "" if ae != ae else round(ae, 1)
        rear = abs(math.degrees(wrap(p[0]))) < 35
        # 저신뢰: IoU 낮음 / 뒷모습에서 주축 불일치 / 손실이 거의 같은 해끼리 lean이 15° 넘게 다름(관측 불가)
        low = (v < 0.45) or (rear and (v < 0.55 or (ae == ae and ae > AXIS_TOL) or unc > LEAN_UNC_TOL))
        x["fit_lowconf"] = int(low)
        yaw, lean, pitch = (math.degrees(wrap(a)) for a in p[:3])
        x["fit_yaw"], x["fit_lean"], x["fit_pitch"] = round(yaw, 1), round(lean, 1), round(pitch, 1)
        x["fit_iou"] = round(v, 3)
        x["fit_lean_s"] = round(float(ls), 1)
        x["fit_back"] = round(b, 2)
        x["fit_scale"] = round(math.exp(p[3]) * 1000, 2)  # GRID(192)px / m
        # 오버레이: 원본 해상도 크롭 위에 모델 윤곽
        q = p.copy()
        q[3] += math.log(1 / scale)
        q[4] /= scale
        q[5] /= scale
        crop = cv2.imread(str(out / "crops" / f"{name}.jpg"))
        vis = crop.copy()
        msk = cv2.imread(str(out / "masks" / f"{name}.png"), 0) > 127
        vis[msk] = (vis[msk] * 0.6 + np.array([255, 0, 200]) * 0.4).astype(np.uint8)
        draw_overlay(q, vis, contact_kpts(kp, AL.CROP))
        cv2.putText(vis, f"lean {lean:+.0f}  yaw {yaw:+.0f}  pitch {pitch:+.0f}  IoU {v:.2f}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        txt = f"back {b:+.2f}" + ("  LOW-CONF" if x["fit_lowconf"] else "")
        if x.get("body_lean"):
            txt = f"torso(2D) {float(x['body_lean']):+.0f}  " + txt
        cv2.putText(vis, txt, (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 220, 60), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "overlay_fit" / f"{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    dt = time.time() - t0
    st = run_stats(rows)
    nlow = sum(1 for x in rows if x.get("fit_lowconf") == 1)
    print(f"  run {run}: 저신뢰 {nlow}프레임 (뒷모습에서 주축 불일치·IoU 낮음 — 분리각 통계에서 제외 권장)")
    print(f"  run {run}: {len(sel)}프레임, {dt:.0f}s (포즈 {t1 - t0:.0f}s, 맞춤 {t2 - t1:.0f}s) | IoU 중앙값 {st['iou']:.2f} | "
          f"yaw급변(>90°) {st['yawflip']} (>30° {st['yawjump']}) | lean급변(>25°) {st['leanjump']} | lean {st['lmin']:+.0f}..{st['lmax']:+.0f} | "
          f"뒷모습(vr>0.75) lean 중앙값 {st['rear']:+.0f}")
    save_montage(out, run)
    return dict(run=run, time=dt, **st)


def run_stats(rows):
    fit = [x for x in rows if x.get("fit_iou") not in ("", None)]
    if not fit:
        return dict(iou=float("nan"), yawflip=0, yawjump=0, leanjump=0, signchg=0, lmin=0, lmax=0,
                    rear=float("nan"), rear_max=float("nan"), n=0)
    ious = [float(x["fit_iou"]) for x in fit]
    yaws = [float(x["fit_yaw"]) for x in fit]
    leans = [float(x["fit_lean"]) for x in fit]
    dyaw = [abs(math.degrees(wrap(math.radians(a - b)))) for a, b in zip(yaws, yaws[1:])]
    yf = sum(1 for d in dyaw if d > 90)
    yj = sum(1 for d in dyaw if d > 30)
    lj = sum(1 for a, b in zip(leans, leans[1:]) if abs(a - b) > 25)
    sc = sum(1 for a, b in zip(leans, leans[1:]) if a * b < 0 and abs(a - b) > 10)
    rear = [float(x["fit_lean_s"]) for x in fit if x.get("view_ratio") and float(x["view_ratio"]) > 0.75]
    return dict(iou=float(np.median(ious)), yawflip=yf, yawjump=yj, leanjump=lj, signchg=sc,
                lmin=min(leans), lmax=max(leans),
                rear=float(np.median(rear)) if rear else float("nan"),
                rear_max=float(max(rear)) if rear else float("nan"), n=len(fit))


def save_montage(out, run, n=12):
    fs = sorted((out / "overlay_fit").glob(f"run{run}_f*.jpg"))
    if not fs:
        return
    idx = np.unique(np.linspace(0, len(fs) - 1, n).round().astype(int))
    ims = []
    for i in idx:
        im = cv2.resize(cv2.imread(str(fs[i])), (384, 384), interpolation=cv2.INTER_AREA)
        cv2.putText(im, fs[i].stem.split("_")[1], (6, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ims.append(im)
    while len(ims) % 4:
        ims.append(np.zeros_like(ims[0]))
    grid = np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)])
    cv2.imwrite(str(out / f"fit_montage_run{run}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])


def plot_all(out, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, run in enumerate(runs):
        p = out / f"run{run}.csv"
        if not p.exists():
            continue
        rows = [x for x in csv.DictReader(open(p, encoding="utf-8")) if x.get("fit_iou")]
        if not rows:
            continue
        pr = np.array([float(x["progress"]) for x in rows])
        ln = np.array([float(x.get("fit_lean_s") or x["fit_lean"]) for x in rows])
        yw = np.array([float(x["fit_yaw"]) for x in rows])
        rear = np.array([bool(x.get("view_ratio")) and float(x["view_ratio"]) > 0.75 for x in rows])
        col = cmap(i % 10)
        axs[0].plot(pr, ln, "-", color=col, lw=1.6, label=f"run {run}")
        axs[0].plot(pr[rear], ln[rear], "o", color=col, ms=3.5)
        axs[1].plot(pr, yw, "-", color=col, lw=1.4)
    axs[0].axhline(0, color="0.6", lw=0.8)
    axs[0].set_ylabel("fit_lean (deg, + = rider's left), smoothed")
    axs[0].set_title("Bike lean from 3D silhouette fit  (dots: rear view, view_ratio > 0.75)")
    axs[0].legend(ncol=7, fontsize=8, loc="upper center")
    axs[1].axhline(0, color="0.6", lw=0.8)
    for yv in (-90, 90):
        axs[1].axhline(yv, color="0.8", lw=0.8, ls="--")
    axs[1].set_ylabel("fit_yaw (deg, 0 = heading away, + = right)")
    axs[1].set_xlabel("progress through feature")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fit_lean.png", dpi=110)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("segments")
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--topk", type=int, default=6, help="프레임당 다듬을 후보 수 (거울쌍 제외)")
    ap.add_argument("--turn", choices=["left", "right", "none", "auto"], default="auto",
                    help="회전 방향 사전 (auto: feature 이름의 left/right)")
    a = ap.parse_args()
    seg = Path(a.segments)
    cfg = json.loads(seg.read_text(encoding="utf-8"))
    global TURN
    tn = a.turn
    if tn == "auto":
        f = cfg.get("feature", "")
        tn = "left" if "left" in f else "right" if "right" in f else "none"
    TURN = {"left": 1, "right": -1, "none": 0}[tn]
    print(f"회전 방향 사전: {tn}")
    import paths as PATHS
    out = PATHS.feat_dir(seg, cfg, "lean")
    pose_m = YOLO(str(Path(__file__).resolve().parent.parent / "models" / "yolo11x-pose.pt"))
    stats = []
    bike = bike_config(cfg.get("bike"))
    set_bike(bike)
    print(f"자전거 모델: {bike['preset']}" + (f" ({cfg['bike'].get('model')})" if cfg.get("bike", {}).get("model") else "")
          + f" — 휠베이스 {WB:.0f}, 바 {bike['bar_width']:.0f}mm/라이즈 {BAR_RISE:.0f}")
    with ProcessPoolExecutor(a.workers, initializer=set_bike, initargs=(bike,)) as pool:
        for r in cfg["runs"]:
            if a.runs and r["run"] not in a.runs:
                continue
            stats.append(run_one(r, out, pose_m, pool, a.topk))
    plot_all(out, [r["run"] for r in cfg["runs"]])
    print("\nrun | 프레임 | 시간s | IoU중앙 | yaw>90 | yaw>30 | lean>25 | lean부호변화 | lean 범위 | 뒷모습 lean 중앙/최대")
    for s in stats:
        print(f"{s['run']:>3} | {s['n']:>4} | {s['time']:5.0f} | {s['iou']:.2f} | {s['yawflip']:>3} | {s['yawjump']:>3} | "
              f"{s['leanjump']:>3} | {s['signchg']:>3} | {s['lmin']:+4.0f}..{s['lmax']:+4.0f} | "
              f"{s['rear']:+.0f} / {s['rear_max']:+.0f}")


set_bike()  # 기본 모델 (main에서 세션 설정으로 다시 부름)

if __name__ == "__main__":
    main()
