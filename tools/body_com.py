"""몸 무게중심(CoM) — GVHMR 3D 관절 + 체절 질량 비율 (좌회전 뱅크).

무게중심 계산
- 체절 질량 비율과 체절 안 위치: de Leva (1996) 남성 값. SMPL-X 관절 22개를 체절 끝점으로 쓴다.
  머리 6.94% / 몸통 43.46% (목→척추3→척추2→척추1→엉덩이 중점 꺾은선에서 목부터 44.86%) / 위팔 2.71%×2 /
  아래팔 1.62%×2 / 손 0.61%×2 / 허벅지 14.16%×2 / 정강이 4.33%×2 / 발 1.37%×2.
  손끝·정수리 관절이 없어서 손 중심 = 손목에서 아래팔 방향으로 아래팔 길이의 0.3,
  머리 중심 = 머리 관절에서 목→머리 방향으로 그 길이의 0.25만큼 더 간 점으로 근사한다.
- 장비(신발·옷·헬멧) 무게는 질량비에만 넣고 위치는 따로 두지 않는다.
- 점검: 메시가 있는 프레임에서 메시 부피 중심(밀도 균일)과 비교한다 → check 명령.

기준점과 각도 (세계 좌표, +y = 중력 위, f = 골반이 향한 수평 방향, l = 라이더 왼쪽 = up × f)
- BB: 양발 '발 중간점'(발목 관절과 발 관절의 중점)의 중점. 두 페달은 BB를 사이에 두고 마주 보므로 두 발의 중점 ≈ BB.
- com_lean: BB→무게중심 선을 f에 수직인 평면에 투영했을 때 중력 수직과의 각. + = 왼쪽(좌회전 안쪽). 자전거 면이 필요 없다.
- grip_roll: 손목선 기울기 = 자전거 기울기 (body3d_gvhmr와 같은 정의, + = 왼쪽).
- com_sep = grip_roll − com_lean. + = 자전거를 몸 전체 무게선보다 더 눕힘 (분리각의 몸 전체 판).
  주의: 같은 관절과 같은 손목선에서 나온 값이라 분리각과 독립된 확인이 아니다. 상체 선에 다리·골반·팔이 더해진 효과를 본다.
- lat_wrist: 무게중심이 '손목선 자전거 면'(BB를 지나고 f를 품고 grip_roll만큼 기운 면)에서 떨어진 거리. + = 라이더 오른쪽 = 바깥.
- lat_hf: 같은 거리를 손·발 면(발목 중점·손목 중점·f로 정한 면, hip_offset_mm과 같은 면)에서 잰 값. 두 면에서 같은 방향일 때만 믿는다.
- fwd: BB에서 무게중심까지 f 방향 거리(+ = 앞). 뒷모습에서는 카메라 깊이 방향이라 믿기 어렵다(보드에 안 넣음).
- height: BB에서 무게중심까지 높이. dist: 뒤에서 본 BB→무게중심 길이(f에 수직인 면, = height / cos(com_lean)).
  height는 많이 기울수록 작아지므로 웅크린 정도는 dist로 본다.
- tot_lean (라이더 런만): 라이더+자전거를 합친 무게중심을 타이어 접지점에서 본 기울기.
  가정: 자전거 무게중심 = BB에서 자전거 면을 따라 위로 0.15 m(0.05~0.25로 바꿔 범위를 냄),
  접지점 = BB에서 자전거 면을 따라 아래로 BB 높이 + 0.07 m(발 중간점이 페달 축보다 높은 만큼).
  정상 선회에서 tan(tot_lean) = v²/(gR) → 같은 회전 단계에서 크면 더 빠르거나 더 좁게 돈 것. 속도는 따로 재지 못한다.

크기: 길이는 GVHMR 체형 키(T-pose, stature 명령)를 sessions/rider.local.json(git 제외)의 키로 맞춘 cm.
  코치님은 키를 몰라 같은 비율(라이더 키 기준)로 맞춘다. rider.local.json이 없으면 길이를 키 대비 %로 낸다.

명령:
  stature  (WSL GVHMR 환경) 런별 betas로 T-pose 키를 재서 left_bank_gvhmr/stature.json에 씀
           $GVHMR_HOME/.venv/bin/python tools/body_com.py stature sessions/2026-11-21/segments.json
  analyze  left_bank_gvhmr/com_run<N>.csv와 회전 단계별 요약
           uv run python tools/body_com.py analyze sessions/2026-11-21/segments.json
  check    메시 부피 중심 vs 체절 무게중심 (메시가 있는 프레임, 카메라 좌표)
  board    insights/left_bank_com.png
           uv run python tools/body_com.py board sessions/2026-11-21/segments.json --ref sessions/reference-coach/segments.json
  selftest 부호 점검
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RIDER = ROOT / "sessions" / "rider.local.json"
UP = np.array([0.0, 1.0, 0.0])
TURNS, TURN_WIN, TURN_MAX = (15, 30, 45), 5, 60
BIKE_COM_H, BIKE_COM_RANGE, FOOT_ABOVE_SPINDLE = 0.15, (0.05, 0.25), 0.07
COM_C = (255, 105, 200)   # 무게중심 (보드, RGB)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def dot(x, y):
    return np.sum(x * y, -1)


# ---------------------------------------------------------------- 무게중심

def segment_com(J):
    """J: (L,22,3) 또는 (22,3) SMPL-X 관절 → 무게중심 (L,3) 또는 (3,). 좌표계와 무관(관절의 선형 결합)."""
    J = np.asarray(J, float)
    single = J.ndim == 2
    if single:
        J = J[None]
    L = len(J)
    mid_hip = (J[:, 1] + J[:, 2]) / 2

    def along(a, b, t):
        return a + t * (b - a)

    parts = [(6.94, J[:, 15] + 0.25 * (J[:, 15] - J[:, 12]))]
    poly = np.stack([J[:, 12], J[:, 9], J[:, 6], J[:, 3], mid_hip], 1)
    seg = np.linalg.norm(np.diff(poly, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((L, 1)), np.cumsum(seg, 1)], 1)
    target = 0.4486 * cum[:, -1]
    trunk = np.empty((L, 3))
    for i in range(L):
        k = int(np.clip(np.searchsorted(cum[i], target[i]) - 1, 0, 3))
        t = (target[i] - cum[i, k]) / max(seg[i, k], 1e-9)
        trunk[i] = poly[i, k] + t * (poly[i, k + 1] - poly[i, k])
    parts.append((43.46, trunk))
    for sh, el, wr, hip, kn, an, ft in ((16, 18, 20, 1, 4, 7, 10), (17, 19, 21, 2, 5, 8, 11)):
        parts += [(2.71, along(J[:, sh], J[:, el], 0.5772)), (1.62, along(J[:, el], J[:, wr], 0.4574)),
                  (0.61, J[:, wr] + 0.3 * (J[:, wr] - J[:, el])), (14.16, along(J[:, hip], J[:, kn], 0.4095)),
                  (4.33, along(J[:, kn], J[:, an], 0.4459)), (1.37, (J[:, an] + J[:, ft]) / 2)]
    m = np.array([p[0] for p in parts])
    assert abs(m.sum() - 100) < 1e-6
    C = np.einsum("s,lsd->ld", m / 100, np.stack([p[1] for p in parts], 1))
    return C[0] if single else C


def bb_point(J):
    """양발 '발 중간점'(발목·발 관절 중점)의 중점 ≈ BB."""
    return ((J[..., 7, :] + J[..., 10, :]) / 2 + (J[..., 8, :] + J[..., 11, :]) / 2) / 2


def metrics(J, fwd):
    """J: (L,22,3) 세계 관절, fwd: (L,3) 수평 단위 진행 방향 → 프레임별 값(길이는 모델 m). 정의는 모듈 docstring."""
    left = unit(np.cross(UP, fwd))
    C, BB = segment_com(J), bb_point(J)
    v = C - BB
    com_lean = np.degrees(np.arctan2(dot(v, left), dot(v, UP)))
    Lw, Rw = J[:, 20], J[:, 21]
    grip = np.degrees(np.arcsin(np.clip(-dot(Lw - Rw, UP) / np.maximum(np.linalg.norm(Lw - Rw, axis=-1), 1e-9), -1, 1)))
    g = np.radians(grip)[:, None]
    bw = np.cos(g) * UP + np.sin(g) * left            # 손목선 자전거 면의 '위' 방향
    nw = unit(np.cross(fwd, bw))                      # 그 면의 법선(라이더 오른쪽)
    A = (J[:, 7] + J[:, 8]) / 2
    s = (J[:, 20] + J[:, 21]) / 2 - A
    bh = unit(dot(s, left)[:, None] * left + dot(s, UP)[:, None] * UP)
    nh = unit(np.cross(fwd, bh))
    return dict(C=C, BB=BB, bw=bw, nw=nw, left=left, com_lean=com_lean, grip=grip, com_sep=grip - com_lean,
                lat_wrist=dot(v, nw), lat_hf=dot(C - A, nh), hip_wrist=dot(J[:, 0] - BB, nw),
                fwd=dot(v, fwd), height=dot(v, UP), dist=np.hypot(dot(v, left), dot(v, UP)))


def tot_lean(met, scale, m_r, m_b, bb_h, h_b=BIKE_COM_H):
    """라이더+자전거 합친 무게중심의 접지점 기준 기울기(°). scale: 모델 m → 실제 m."""
    C, BB, bw, left = met["C"] * scale, met["BB"] * scale, met["bw"], met["left"]
    contact = BB - (bb_h + FOOT_ABOVE_SPINDLE) * bw
    Ct = (m_r * C + m_b * (BB + h_b * bw)) / (m_r + m_b)
    u = Ct - contact
    return np.degrees(np.arctan2(dot(u, left), dot(u, UP)))


def volume_centroid(V, F):
    """닫힌 삼각형 메시의 부피 중심(밀도 균일)과 부피."""
    o = V.mean(0)
    a, b, c = V[F[:, 0]] - o, V[F[:, 1]] - o, V[F[:, 2]] - o
    vol = dot(a, np.cross(b, c)) / 6
    return o + (vol[:, None] * (a + b + c) / 4).sum(0) / vol.sum(), float(vol.sum())


# ---------------------------------------------------------------- 불러오기

def rider_info():
    if not RIDER.exists():
        return None
    return json.loads(RIDER.read_text(encoding="utf-8"))


def load_run(seg, run, target_h, masses=None):
    """런 하나: 프레임별 값(run<N>.csv 행 순서), 회전, 크기. target_h=None이면 길이 단위는 키 대비 %."""
    import body3d_gvhmr as G
    cfg = PATHS.load_cfg(seg)
    od = G.out_dir(seg, cfg)
    rows = G.read_csv(od / f"run{run}.csv")
    z = np.load(od / f"gvhmr_run{run}.npz")
    st = od / "stature.json"
    if not st.exists():
        raise SystemExit(f"{st} 없음 → WSL에서 stature 명령을 먼저 실행")
    model_h = json.loads(st.read_text(encoding="utf-8"))[str(run)]
    fwd = G.hproj(G.rodrigues(z["go_g"]) @ np.array([0, 0, 1.0]))
    met = metrics(z["joints_g"], fwd)
    idx = {int(f): i for i, f in enumerate(z["frames"])}
    si = np.array([idx[int(x["seq_frame"])] for x in rows])
    scale = target_h / model_h if target_h else None
    k = 100 * scale if scale else 100 / model_h          # 모델 m → cm 또는 %키
    d = dict(G=G, cfg=cfg, od=od, rows=rows, z=z, met=met, si=si, turn=G.turn_of(z, rows), model_h=model_h,
             scale=scale, unit="cm" if scale else "%키",
             torso=np.array([G.fnum(x["torso_roll"]) for x in rows], float))
    for key in ("com_lean", "grip", "com_sep"):
        d[key] = met[key][si]
    for key in ("lat_wrist", "lat_hf", "hip_wrist", "fwd", "height", "dist"):
        d[key] = met[key][si] * k
    if masses and scale:
        bb_h = cfg["bike"]["bb_height"] / 1000
        d["tot"] = tot_lean(met, scale, *masses, bb_h)[si]
        d["tot_lo"], d["tot_hi"] = (tot_lean(met, scale, *masses, bb_h, h)[si] for h in BIKE_COM_RANGE)
    return d


KEYS = ("com_lean", "grip", "torso", "com_sep", "lat_wrist", "lat_hf", "hip_wrist", "fwd", "height", "dist", "tot", "tot_lo", "tot_hi")


def stage_stats(d):
    out = {}
    for T in TURNS:
        m = (np.abs(d["turn"] - T) <= TURN_WIN) & np.isfinite(d["com_lean"])
        st = dict(n=int(m.sum()), mask=m)
        for key in KEYS:
            if key in d:
                st[key] = float(np.median(d[key][m])) if m.any() else None
        out[T] = st
    return out


def items_of(a):
    info = rider_info()
    target = info["height_m"] if info else None
    masses = (info["mass_with_gear_kg"], info["bike_mass_kg"]) if info else None
    seg = Path(a.segments)
    items = []
    if getattr(a, "ref", None):
        items.append((Path(a.ref), 0, "코치님", True, target, None))
    items += [(seg, r, f"Run {r}", False, target, masses) for r in a.runs]
    return items


# ---------------------------------------------------------------- 명령

def cmd_stature(a):
    """WSL GVHMR 환경: SMPLX_NEUTRAL v_template + shapedirs(10) × 런 betas 중앙값 → T-pose 키(정수리~발바닥)."""
    import body3d_gvhmr as G
    m = np.load(Path(os.environ["GVHMR_HOME"]).expanduser() / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
                allow_pickle=True)
    for s in [a.segments] + (a.more or []):
        seg = Path(s)
        od = G.out_dir(seg, PATHS.load_cfg(seg))
        res = {}
        for p in sorted(od.glob("gvhmr_run?.npz")):
            b = np.load(p)["betas"]
            assert np.abs(b - b[0]).max() < 1e-4, f"{p.name}: 프레임마다 betas가 다름"
            v = m["v_template"] + m["shapedirs"][:, :, :b.shape[1]] @ np.median(b, 0)
            res[p.stem.replace("gvhmr_run", "")] = round(float(v[:, 1].max() - v[:, 1].min()), 4)
        (od / "stature.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(od / "stature.json", res)


def cmd_analyze(a):
    for s, r, name, is_ref, target, masses in items_of(a):
        d = load_run(s, r, target, masses)
        path = d["od"] / f"com_run{r}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            keys = [k for k in KEYS if k in d]
            w.writerow(["frame", "seq_frame", "turn"] + keys)
            for i, row in enumerate(d["rows"]):
                w.writerow([row["frame"], row["seq_frame"], round(float(d["turn"][i]), 1)] + [round(float(d[k][i]), 2) for k in keys])
        st = stage_stats(d)
        print(f"{name} ({path.name}, 단위 {d['unit']})")
        for T in TURNS:
            x = st[T]
            if x["n"] == 0:
                print(f"  {T}°: -")
                continue
            tot = f" 합친 {x['tot']:.1f} ({x['tot_lo']:.1f}~{x['tot_hi']:.1f})" if "tot" in x else ""
            print(f"  {T}° n={x['n']}: 무게선 {x['com_lean']:.1f} 자전거 {x['grip']:.1f} 상체 {x['torso']:.1f} "
                  f"com_sep {x['com_sep']:+.1f} | 바깥 손목면 {x['lat_wrist']:+.1f} 손발면 {x['lat_hf']:+.1f} "
                  f"(골반 {x['hip_wrist']:+.1f}) 앞 {x['fwd']:+.1f} 높이 {x['height']:.1f} 거리 {x['dist']:.1f}{tot}")


def cmd_check(a):
    """메시가 있는 프레임: 체절 무게중심(joints_c) vs 메시 부피 중심(verts_c). 손목선 면 법선 방향 성분도 본다."""
    import body3d_gvhmr as G
    for s, r, name, is_ref, target, masses in items_of(a):
        d = load_run(s, r, target)
        z = d["z"]
        k = 100 * d["scale"] if d["scale"] else 100 / d["model_h"]
        F = z["faces"]
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        idx = {int(f): i for i, f in enumerate(z["frames"])}
        diff, lat, vert = [], [], []
        for j, sf in enumerate(z["verts_frames"]):
            i = idx[int(sf)]
            Cv, vol = volume_centroid(z["verts_c"][j].astype(float), F)
            Cs = segment_com(z["joints_c"][i])
            R = G.kabsch(z["joints_c"][i], z["joints_g"][i])        # 카메라 → 세계
            dw = R @ (Cv - Cs)
            diff.append(np.linalg.norm(dw) * k)
            lat.append(float(dw @ d["met"]["nw"][i]) * k)
            vert.append(float(dw @ UP) * k)
        print(f"{name}: 프레임 {len(diff)}, 경계 모서리 {int((cnt == 1).sum())}개, 부피 {abs(vol):.3f} m³(모델) | "
              f"부피중심−체절중심 거리 중앙값 {np.median(diff):.1f} {d['unit']} (최대 {np.max(diff):.1f}), "
              f"바깥 방향 성분 중앙값 {np.median(lat):+.1f} (범위 {np.min(lat):+.1f}~{np.max(lat):+.1f}), "
              f"위 방향 {np.median(vert):+.1f}")


def sgn(v, nd=0):
    """부호 붙인 숫자. 반올림해서 0이면 '0'(−0 방지)."""
    r = round(float(v), nd)
    return "0" if r == 0 else f"{r:+.{nd}f}"


def pick_frame(d):
    """회전 30° 구간에서 바깥 거리가 중앙값에 가까운 프레임(메시 있는 프레임 중). 없으면 회전 30°에 가장 가까운 메시 프레임."""
    vf = [int(v) for v in d["z"]["verts_frames"]]
    seqs = np.array([int(x["seq_frame"]) for x in d["rows"]])
    has = np.isin(seqs, vf)
    st = stage_stats(d)[30]
    cand = np.nonzero(st["mask"] & has)[0]
    if len(cand):
        return int(cand[np.argmin(np.abs(d["lat_wrist"][cand] - st["lat_wrist"]))])
    hk = np.nonzero(has)[0]
    return int(hk[np.argmin(np.abs(d["turn"][hk] - 30))])


BONES = [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19),
         (18, 20), (19, 21), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11)]


def schematic(d, k, w, h, texts=None, title="뒤에서 본 그림(중력 기준)"):
    """뒤에서 본 그림(진행 방향에 수직인 면, 중력 위가 위쪽, 원근 없음). 골격·자전거 면·무게중심을 같은 프레임으로 그림.
    texts: 아래 띠 글자 [(왼쪽 위, 색), (왼쪽 아래, 색), (오른쪽 위, 색), (오른쪽 아래, 색)]. 없으면 각도·바깥 거리."""
    from PIL import Image, ImageDraw
    from separation_board import BIKE_C, BODY_C, f
    i = int(d["si"][k])
    met, cfg = d["met"], d["cfg"]
    sc = d["scale"] or (1.75 / d["model_h"])
    J, BB, C = d["z"]["joints_g"][i] * sc, met["BB"][i] * sc, met["C"][i] * sc
    bw, left = met["bw"][i], met["left"][i]
    contact = BB - (cfg["bike"].get("bb_height", 340) / 1000 + FOOT_ABOVE_SPINDLE) * bw
    ppm, ox, oy = 124, w * 0.56, h - 112     # 아래 56px는 글자 띠

    def Q(P):
        v = np.asarray(P) - BB
        return (ox - ppm * float(v @ left), oy - ppm * float(v @ UP))

    im = Image.new("RGB", (w, h), (30, 33, 38))
    dr = ImageDraw.Draw(im)
    gy = Q(contact)[1]
    dr.line([Q(contact)[0] - 70, gy, Q(contact)[0] + 70, gy], fill=(90, 90, 90), width=2)
    for yy in range(18, int(gy), 10):                                # 중력 수직(BB를 지남), 점선
        dr.line([ox, yy, ox, yy + 5], fill=(110, 110, 110), width=1)
    dr.line([Q(contact), Q(BB + 1.15 * bw)], fill=BIKE_C, width=4)    # 자전거 면
    for a_, b_ in BONES:
        dr.line([Q(J[a_]), Q(J[b_])], fill=(150, 165, 185), width=3)
    dr.line([Q(J[0]), Q(J[12])], fill=BODY_C, width=4)               # 상체 롤 선
    hx, hy = Q(J[15])
    dr.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], outline=(150, 165, 185), width=3)
    lat = float((C - BB) @ met["nw"][i])
    dr.line([Q(BB), Q(C)], fill=COM_C, width=2)                       # 무게선
    dr.line([Q(C), Q(C - lat * met["nw"][i])], fill=COM_C, width=2)
    cx_, cy_ = Q(C)
    dr.ellipse([cx_ - 8, cy_ - 8, cx_ + 8, cy_ + 8], fill=COM_C, outline=(20, 20, 20), width=2)
    bx, by = Q(BB)
    dr.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255))
    qx, qy = Q(contact)
    dr.ellipse([qx - 5, qy - 5, qx + 5, qy + 5], outline=(255, 255, 255), width=2)
    u = d["unit"]
    dr.rectangle([0, h - 56, w, h], fill=(24, 26, 30))
    dr.text((6, 4), title, font=f(12), fill=(140, 140, 140))
    if texts is None:
        texts = [(f"자전거 {d['grip'][k]:.0f}°", BIKE_C), (f"상체 {d['torso'][k]:.0f}°", BODY_C),
                 (f"무게선 {d['com_lean'][k]:.0f}°", COM_C), (f"바깥 {sgn(d['lat_wrist'][k])}{u}", COM_C)]
    for (t, col), (xx, yy, anc) in zip(texts, ((8, h - 52, "la"), (8, h - 28, "la"), (w - 8, h - 52, "ra"), (w - 8, h - 28, "ra"))):
        if t:
            dr.text((xx, yy), t, font=f(15, True), fill=col, anchor=anc)
    return im


def com_tile(d, name, size, k, header=None):
    """사진 타일: 메시 + 무게중심·자전거 면(원근이 있어 바깥 거리는 오른쪽 그림에서 본다)."""
    import cv2
    G, cfg, z = d["G"], d["cfg"], d["z"]
    vf = [int(v) for v in z["verts_frames"]]
    seqs = np.array([int(x["seq_frame"]) for x in d["rows"]])
    seq, i = int(seqs[k]), int(d["si"][k])
    rr = next(x for x in cfg["runs"] if x["run"] == int(d["rows"][k]["run"]))
    cap = cv2.VideoCapture(str(PATHS.video_dir(cfg) / rr["file"]))
    img = None
    for _ in range(seq + 1):
        ok, img = cap.read()
    cap.release()
    Jc, Jg = z["joints_c"][i], z["joints_g"][i]
    R = G.kabsch(Jc, Jg)                     # 카메라 → 세계
    up_c, bw_c, nw_c = R.T @ UP, R.T @ d["met"]["bw"][i], R.T @ d["met"]["nw"][i]
    C, BB = segment_com(Jc), bb_point(Jc)
    lat_m = float((C - BB) @ nw_c)
    sc = d["scale"] or (1.75 / d["model_h"])
    contact = BB - (cfg["bike"].get("bb_height", 340) / 1000 + FOOT_ABOVE_SPINDLE) / sc * bw_c

    def draw(out, P):
        cv2.line(out, P(contact), P(BB + 1.05 / sc * bw_c), (0, 165, 255), 3, cv2.LINE_AA)   # 자전거 면(손목선 기울기)
        cv2.circle(out, P(contact), 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, P(BB), 5, (255, 255, 255), -1, cv2.LINE_AA)
        foot = C - lat_m * nw_c
        cv2.line(out, P(C), P(foot), COM_C[::-1], 3, cv2.LINE_AA)
        cv2.circle(out, P(C), 7, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, P(C), 5, COM_C[::-1], -1, cv2.LINE_AA)

    head = header or (f"{name} · 회전 {d['turn'][k]:.0f}° · f{d['rows'][k]['frame']}", "분홍 = 무게중심, 흰 원 = 접지점")
    return G.render_tile(img, z["verts_c"][vf.index(seq)], z["faces"], z["K"], Jc, up_c, None, z["boxes"][i],
                         tile=size, line="none", header=head, draw=draw)


def cmd_board(a):
    from PIL import Image, ImageDraw
    from separation_board import CELL, GOLD, BIKE_C, BODY_C, f
    items = items_of(a)
    ROW, CW, SCH, head, foot = 330, 900, 230, 176, 230
    L0 = CELL + SCH + 10                      # 이름 열 시작
    W, H = L0 + 40 + CW + 30, head + ROW * len(items) + foot
    board = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(board)
    dr.text((24, 16), "좌회전 뱅크 · 몸 무게중심 (GVHMR 3D + 체절 질량 비율)", font=f(34, True), fill=(255, 255, 255))
    dr.text((26, 66), "분홍 = 몸 무게선(BB→무게중심) 기울기   노랑 = 자전거 기울기(손목선)   초록 = 상체 롤(골반→목)",
            font=f(18), fill=(210, 210, 210))
    dr.text((26, 94), "큰 숫자 = 무게중심이 자전거 면에서 떨어진 거리(+ = 바깥, 손목선 면 / 괄호 = 손·발 면). "
            "가로축 = 골반이 구간 시작부터 왼쪽으로 돈 각도.", font=f(18), fill=(210, 210, 210))
    dr.text((26, 122), "사진: 회전 30° 근처 프레임(분홍 점 = 무게중심, 주황 = 자전거 면, 흰 점 = BB, 흰 원 = 접지점 가정). "
            "옆 그림: 같은 프레임을 뒤에서 중력 기준으로 본 골격(점선 = BB를 지나는 중력 수직, 분홍 선 = BB→무게중심).",
            font=f(15), fill=(160, 160, 160))
    cx0, cx1 = L0 + 200, W - 40

    def X(t):
        return cx0 + (cx1 - cx0) * t / TURN_MAX

    unit_seen = set()
    for n_i, (s, r, name, is_ref, target, masses) in enumerate(items):
        y0 = head + n_i * ROW
        d = load_run(s, r, target, masses)
        unit_seen.add(d["unit"])
        st = stage_stats(d)
        kf = pick_frame(d)
        tile = com_tile(d, name, CELL, kf)
        if is_ref:
            ImageDraw.Draw(tile).rectangle([1, 1, CELL - 2, CELL - 2], outline=GOLD, width=5)
        board.paste(tile, (0, y0 + 10))
        board.paste(schematic(d, kf, SCH, CELL), (CELL + 5, y0 + 10))
        dr.text((L0 + 30, y0 + 30), name, font=f(28, True), fill=GOLD if is_ref else (255, 255, 255))
        dr.text((L0 + 30, y0 + 74), "프레임 " + "/".join(str(st[T]["n"]) for T in TURNS), font=f(15), fill=(160, 160, 160))
        dr.text((L0 + 30, y0 + 96), "(15/30/45°)", font=f(13), fill=(130, 130, 130))
        if is_ref:
            dr.text((L0 + 30, y0 + 124), "길이는 라이더 키", font=f(13), fill=(150, 150, 150))
            dr.text((L0 + 30, y0 + 142), "비율로 맞춤", font=f(13), fill=(150, 150, 150))
        cy0, cy1 = y0 + 20, y0 + ROW - 118

        def Y(v):
            return cy1 - (cy1 - cy0) * (v + 10) / 60

        for T in TURNS:
            dr.rectangle([X(T - TURN_WIN), cy0, X(T + TURN_WIN), cy1], fill=(34, 37, 44))
        for g in (0, 20, 40):
            dr.line([cx0, Y(g), cx1, Y(g)], fill=(60, 62, 68), width=1)
            dr.text((cx0 - 34, Y(g) - 9), f"{g}°", font=f(13), fill=(120, 120, 120))
        for t in range(0, TURN_MAX + 1, 15):
            dr.text((X(t) - 10, cy1 + 4), f"{t}°", font=f(13), fill=(120, 120, 120))
        m = d["turn"] <= TURN_MAX
        o = np.argsort(d["turn"][m], kind="stable")
        for vals, col, wd in ((d["torso"], BODY_C, 2), (d["grip"], BIKE_C, 2), (d["com_lean"], COM_C, 3)):
            pts = [(X(t), Y(v)) for t, v in zip(d["turn"][m][o], vals[m][o]) if np.isfinite(v)]
            if len(pts) > 1:
                dr.line(pts, fill=col, width=wd)
            for q in pts:
                dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=col)
        u = d["unit"]
        for T in TURNS:
            x = st[T]
            tx = X(T) - 70
            if x["n"] == 0:
                dr.text((tx, cy1 + 26), "프레임 없음", font=f(16), fill=(150, 150, 150))
                continue
            big = f"바깥 {sgn(x['lat_wrist'])}{u}"
            dr.text((tx, cy1 + 22), big, font=f(21, True), fill=COM_C)
            dr.text((tx + dr.textlength(big, font=f(21, True)) + 4, cy1 + 29), f"({sgn(x['lat_hf'])})", font=f(14),
                    fill=(200, 200, 200))
            dr.text((tx, cy1 + 54), f"무게선 {x['com_lean']:.0f}° · 자전거 {x['grip']:.0f}°", font=f(13), fill=(170, 170, 170))
            line3 = f"거리 {x['dist']:.0f}{u}"
            if "tot" in x:
                line3 += f" · 합친 {x['tot']:.0f}°"
            dr.text((tx, cy1 + 74), line3, font=f(13), fill=(150, 150, 150))
        dr.line([0, y0 + ROW - 1, W, y0 + ROW - 1], fill=(50, 52, 58), width=1)
    unit_note = ("길이는 GVHMR 체형 키를 라이더 실제 키로 맞춘 cm(실제 키는 git 제외 파일에서 읽음). "
                 if "cm" in unit_seen else "길이는 키 대비 %. ")
    note = ("측정값(GVHMR 3D 관절) + 가정(체절 질량 비율 de Leva 1996). 무게중심 정답 데이터는 없다. "
            "BB = 두 발 중간점의 중점. 바깥 거리는 자전거 면 정의에 따라 달라진다. 큰 숫자 = 손목선 면(그린 핸들바 선으로 검증된 런 1~5·7), "
            "괄호 = 손·발 면(Run 5에서 그린 선보다 12~18° 더 눕게 나옴). "
            "분리각과 같은 관절·손목선에서 나온 값이라 독립 확인이 아니다. 상체 선에 다리·골반·팔이 더해진 효과를 보는 용도.\n"
            + unit_note +
            "거리 = 뒤에서 본 BB→무게중심 길이(작을수록 웅크림). 앞뒤 거리는 뒷모습에서 카메라 깊이 방향이라 보드에 넣지 않음(CSV에만). 합친 = 라이더+자전거 무게중심을 접지점에서 본 기울기"
            "(자전거 무게중심·접지점은 가정). 같은 회전 단계에서 크면 더 빠르거나 더 좁게 돈 것.\n"
            "회전 단계마다 프레임이 2~9개뿐 → 런별 값은 몇 cm·몇 ° 흔들릴 수 있다. 큰 차이만 본다.")
    if a.ref:
        note += " 코치님: 시점이 더 옆쪽이고 키·자전거를 몰라 참고만."
    if 6 in a.runs:
        note += " Run 6: 다른 촬영 자리이고 그린 핸들바 선이 없어 손목선(자전거 기울기)을 검증하지 못함."
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
    out = Path(a.out) if a.out else PATHS.out_base(Path(a.segments)) / "insights" / "left_bank_com.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out)
    print("saved:", out)


def selftest():
    """합성 변형으로 부호 확인: 몸 전체를 BB 기준 f축으로 왼쪽 10° 돌리면 무게선·자전거 +10°, 바깥 거리는 그대로.
    발·손을 두고 몸통·다리 위쪽을 오른쪽으로 옮기면 바깥 거리가 +로 커지고 무게선이 줄어든다."""
    import body3d_gvhmr as G
    rng = np.random.default_rng(0)
    J = np.zeros((22, 3))
    # 대략적인 공격 자세 골격(y 위, z 앞, x 왼쪽)
    pts = {0: (0, .95, -.10), 1: (.09, .90, -.10), 2: (-.09, .90, -.10), 3: (0, 1.05, -.05), 4: (.12, .70, .10), 5: (-.12, .70, .10),
           6: (0, 1.18, 0), 7: (.12, .42, -.02), 8: (-.12, .42, -.02), 9: (0, 1.30, .05), 10: (.13, .36, .10), 11: (-.13, .36, .10),
           12: (0, 1.42, .12), 13: (.07, 1.38, .10), 14: (-.07, 1.38, .10), 15: (0, 1.52, .18), 16: (.18, 1.36, .10),
           17: (-.18, 1.36, .10), 18: (.28, 1.15, .25), 19: (-.28, 1.15, .25), 20: (.38, 1.05, .45), 21: (-.38, 1.05, .45)}
    for k, p in pts.items():
        J[k] = p
    J += rng.normal(0, 0.003, J.shape)
    f = np.array([[0, 0, 1.0]])
    m0 = metrics(J[None], f)
    assert abs(m0["lat_wrist"][0]) < 0.03 and abs(m0["com_lean"][0]) < 3, (m0["lat_wrist"], m0["com_lean"])
    BB = bb_point(J)
    R = G.rodrigues(np.radians(10) * np.array([0, 0, -1.0]))   # f(+z)축 회전, 위(+y)가 왼쪽(+x)으로
    assert (R @ UP)[0] > 0
    J1 = (J - BB) @ R.T + BB
    m1 = metrics(J1[None], f)
    assert abs(m1["com_lean"][0] - m0["com_lean"][0] - 10) < 0.5 and abs(m1["grip"][0] - m0["grip"][0] - 10) < 0.5
    assert abs(m1["lat_wrist"][0] - m0["lat_wrist"][0]) < 0.005
    J2 = J.copy()
    move = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17]
    J2[move] += np.array([-0.10, 0, 0])                          # 오른쪽 = −x
    m2 = metrics(J2[None], f)
    assert m2["lat_wrist"][0] > m0["lat_wrist"][0] + 0.05 and m2["com_lean"][0] < m0["com_lean"][0] - 3
    assert m2["lat_hf"][0] > m0["lat_hf"][0] + 0.05
    C = segment_com(J)
    assert J[7, 1] < C[1] < J[12, 1], C
    print("selftest ok", f"CoM 높이 {C[1]:.2f} m (골반 {J[0, 1]:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stature", "analyze", "check", "board", "selftest"])
    ap.add_argument("segments", nargs="?")
    ap.add_argument("more", nargs="*", help="stature: 다른 세션 segments.json도 함께")
    ap.add_argument("--ref", help="코치님 segments.json")
    ap.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    {"stature": cmd_stature, "analyze": cmd_analyze, "check": cmd_check, "board": cmd_board}[a.cmd](a)


if __name__ == "__main__":
    main()
