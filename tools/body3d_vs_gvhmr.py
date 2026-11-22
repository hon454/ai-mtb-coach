"""SAM 3D Body vs GVHMR 3D 몸 각도 프레임별 비교, 그린 핸들바 선 채점, Run 5 손-발 차이 진단 (저장소 .venv).

입력(읽기만):
    outputs/<세션>/<feature>_sam3dbody/run<N><suffix>.csv, raw_box_f1500_run<N>.npz, prep_run<N>.npz (body3d_sam3d.py)
        suffix ""          = A: 중력 = 화면 아래 가정, 진행 방향 = SAM 골반 축 (지금까지의 기준 출력)
               "_gvgrav"   = B: A + GVHMR 프레임별 중력 (body3d_sam3d.py angles --gravity gvhmr)
               "_gvgravmed"= B': A + GVHMR 중력의 런 중앙값 (--gravity gvhmr_med)
               "_gvhead"   = C: B + GVHMR 골반 방향을 진행 방향으로 (--heading gvhmr, 진단용. 진행 방향 합치기가 아님)
    outputs/<세션>/<feature>_gvhmr/run<N>.csv, gvhmr_run<N>.npz (body3d_gvhmr.py, main 브랜치)
    outputs/2026-11-21/left_bank_lean/bar_hints/gvhmr_vs_drawn_45_lean.json (그린 핸들바 선 45프레임, drawn_g = 그린 각 + GVHMR cam_roll)
        선 끝점: 같은 폴더 hints_staging_v2.json(내 런 41) + sessions/reference-coach/bar_hints.json(코치 4, 데이터 루트)
    outputs/<세션>/<feature>_lean/crops, masks (장면 칸 배경), <feature>_gvhmr/lean_snapshot (fix4 실루엣, fit_yaw)

분해: B - A = 중력(카메라 기울기) 효과, C - B = 진행 방향 효과, GVHMR - C = 두 모델의 관절 차이.
시점 구간은 GVHMR cam_view_deg(0 = 바로 뒤)로 나눈다: 뒷모습 < 30°, 비스듬 30~55°, 옆 ≥ 55°.

Run 5 진단 (손목 중점 W, 발목 중점 A, 손목선 방향 ŵ = 왼손목 - 오른손목, + = 라이더 왼쪽):
    dA = (A - W)·ŵ (mm). 핸들이 자전거 중심면에 수직이고 그립·페달이 대칭이면 0이어야 한다.
    손-발 기울기를 "손목선에 수직인 수평 방향"으로 재면 손목선 기울기와의 차이는 dA만으로 생긴다.
    진행 방향을 골반 방향으로 재면 차이 = dA 몫 + 진행 방향 몫(손이 발보다 앞에 있어 생김).
    픽셀 시험: 사용자가 그린 선의 오른쪽 끝(보이는 오른손 그립)과 ViTPose/YOLO 발목으로 2D dA를 따로 잰다.
        줄기(핸들 중앙) = 오른쪽 끝에서 선 방향으로 그립 반폭 0.34 m(MTB 바 760~800 mm 가정) 왼쪽.
        px/m는 SAM 복원의 골반 깊이에서 얻는다(f/Z). 뒷모습(시점 작음)일 때만 2D 옆 방향 ≈ 3D 옆 방향이다.

사용:
    uv run python tools/body3d_vs_gvhmr.py all
출력: outputs/2026-11-21/left_bank_sam3dbody/ 의 vs_gvhmr_frames.csv, vs_gvhmr_tables.txt,
      vs_gvhmr_board.png, drawn45_board.png, drawn45_eval.csv, run5_gap_board.png, run5_gap.csv
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import body3d_sam3d as B  # noqa: E402
import paths as PATHS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "sessions/2026-11-21/segments.json"
REF = ROOT / "sessions/reference-coach/segments.json"
TAG = "box_f1500"
VARIANTS = [("A", ""), ("B", "_gvgrav"), ("Bmed", "_gvgravmed"), ("C", "_gvhead")]
VIEW_BINS = [("뒷모습 <30°", 0, 30), ("비스듬 30~55°", 30, 55), ("옆 ≥55°", 55, 181)]
PROG_BINS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
HALF_GRIP_M = 0.34
GJ = dict(pelvis=0, lhip=1, rhip=2, lank=7, rank=8, neck=12, lwr=20, rwr=21)  # SMPL 22
FONT = "Malgun Gothic"


def fnum(x):
    return B.fnum(x)


def entries():
    out = []
    for segp in (REF, SEG):
        cfg = PATHS.load_cfg(segp)
        for r in cfg["runs"]:
            out.append(("코치님" if segp == REF else f"Run {r['run']}", segp, cfg, r["run"]))
    return out


def gdir(seg, cfg):
    return PATHS.feat_dir(seg, cfg, "gvhmr")


def sdir(seg, cfg):
    return PATHS.feat_dir(seg, cfg, "sam3dbody")


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def unit(v):
    return B.unit(v)


# ---------------------------------------------------------------- 프레임별 합치기

def joined(seg, cfg, run):
    """GVHMR CSV 행마다 SAM 변형 A/B/Bmed/C 값을 붙인다 (join = frame)."""
    g = B.read_csv(gdir(seg, cfg) / f"run{run}.csv")
    sam = {v: {int(r["frame"]): r for r in B.read_csv(sdir(seg, cfg) / f"run{run}{suf}.csv")} for v, suf in VARIANTS}
    lean = {int(r["frame"]): r for r in B.read_csv(B.lean_csv(seg, cfg, run))}
    rows = []
    for r in g:
        f = int(r["frame"])
        x = dict(run=run, frame=f, progress=fnum(r["progress"]), view=fnum(r["cam_view_deg"]),
                 view_s=fnum(r["cam_view_signed"]), cam_roll=fnum(r["cam_roll_deg"]),
                 g_roll=fnum(r["torso_roll"]), g_pitch=fnum(r["torso_pitch"]), g_bar=fnum(r["grip_roll"]),
                 g_bike=fnum(r["bike_lean_body"]), g_bike_barhead=fnum(r["bike_lean_barhead"]),
                 g_conf=fnum(r["b3d_conf"]), view_ratio=fnum(lean.get(f, {}).get("view_ratio")),
                 fit_yaw=fnum(lean.get(f, {}).get("fit_yaw")), fit_s=fnum(lean.get(f, {}).get("fit_lean_s")),
                 fit_lowconf=str(lean.get(f, {}).get("fit_lowconf", "")).strip() in ("1", "1.0", "True"))
        for v, _ in VARIANTS:
            s = sam[v].get(f, {})
            for k, kk in (("roll", "torso_roll"), ("pitch", "torso_pitch"), ("bar", "bar_roll"), ("bike", "bike_lean_body"),
                          ("conf", "b3d_conf"), ("cam_pitch", "cam_pitch_g"), ("cam_roll", "cam_roll_g")):
                x[f"{v}_{k}"] = fnum(s.get(kk))
        rows.append(x)
    return rows


def all_joined():
    out = {}
    for lab, seg, cfg, run in entries():
        out[lab] = joined(seg, cfg, run)
    return out


def arr(rows, k):
    return np.array([r[k] for r in rows], float)


# ---------------------------------------------------------------- 점검: GVHMR CSV 재현, 공통 관절 비교

def gv_angles(J, fwd):
    """body3d_gvhmr.angles와 같은 정의(세계 +y 위)."""
    up = np.array([0.0, 1.0, 0.0])
    left = unit(np.cross(up, fwd))
    v = J[:, GJ["neck"]] - J[:, GJ["pelvis"]]
    W = (J[:, GJ["lwr"]] + J[:, GJ["rwr"]]) / 2
    A = (J[:, GJ["lank"]] + J[:, GJ["rank"]]) / 2
    s = W - A
    d = lambda a, b: np.sum(a * b, -1)  # noqa: E731
    return dict(roll=np.degrees(np.arctan2(d(v, left), d(v, up))), pitch=np.degrees(np.arctan2(d(v, fwd), d(v, up))),
                bike=np.degrees(np.arctan2(d(s, left), d(s, up))))


def check_reproduce(log):
    worst = 0.0
    for lab, seg, cfg, run in entries():
        z = np.load(gdir(seg, cfg) / f"gvhmr_run{run}.npz")
        face = B._rodrigues(z["go_g"].astype(np.float64)) @ np.array([0, 0, 1.0])
        face[:, 1] = 0
        a = gv_angles(z["joints_g"].astype(np.float64), unit(face))
        idx = {int(f): i for i, f in enumerate(z["frames"])}
        for r in B.read_csv(gdir(seg, cfg) / f"run{run}.csv"):
            i = idx[int(r["seq_frame"])]
            for k, kk in (("roll", "torso_roll"), ("pitch", "torso_pitch"), ("bike", "bike_lean_body")):
                worst = max(worst, abs(a[k][i] - float(r[kk])))
    log(f"[점검] GVHMR CSV를 npz에서 다시 계산: torso_roll·pitch·bike_lean_body 최대 차이 {worst:.3f}° (반올림 0.005°)")
    return worst


def coco_check(log, J):
    """같은 중력·같은 진행 방향(GVHMR)·같은 COCO 관절(어깨 중점 - 엉덩이 중점)로 두 모델을 비교한다."""
    res = {nm: {k: [] for k in ("roll", "pitch", "bar", "bike")} for nm, _, _ in VIEW_BINS}
    for lab, seg, cfg, run in entries():
        raw = np.load(sdir(seg, cfg) / f"raw_{TAG}_run{run}.npz")
        frames = [int(f) for f in raw["frame"]]
        grav = B.gvhmr_frame(seg, cfg, run, frames)
        Q = B.level_rot(grav["up"])
        z = np.load(gdir(seg, cfg) / f"gvhmr_run{run}.npz")
        zi = {int(f): i for i, f in enumerate(z["frames"])}
        ri = {f: i for i, f in enumerate(frames)}
        u = -B.gravity(0.0)
        for x in J[lab]:
            i = ri.get(x["frame"])
            j = zi.get(x["frame"] + grav["off"])
            if i is None or j is None:
                continue
            t = Q[i] @ grav["fwd"][i]
            t = unit(t - (t @ u) * u)
            L = np.cross(u, t)
            vals = {}
            for nm, P in (("s", (Q[i] @ raw["kp3d"][i].astype(np.float64).T).T[B.COCO2MHR]),
                          ("g", (Q[i] @ z["coco_c"][j].astype(np.float64).T).T)):
                tor = (P[5] + P[6]) / 2 - (P[11] + P[12]) / 2
                W, A = (P[9] + P[10]) / 2, (P[15] + P[16]) / 2
                w = unit(P[10] - P[9])  # 오른손목 - 왼손목 (COCO 9 = 왼손목)
                h = W - A
                vals[nm] = dict(roll=math.degrees(math.atan2(tor @ L, tor @ u)), pitch=math.degrees(math.atan2(tor @ t, tor @ u)),
                                bar=math.degrees(math.asin(np.clip(w @ u, -1, 1))), bike=math.degrees(math.atan2(h @ L, h @ u)))
            for nm, lo, hi in VIEW_BINS:
                if lo <= x["view"] < hi:
                    for k in res[nm]:
                        res[nm][k].append(vals["s"][k] - vals["g"][k])
    log("[점검] 같은 중력·같은 진행 방향(GVHMR)·같은 COCO 관절(몸통 = 어깨 중점 - 엉덩이 중점)로 SAM - GVHMR (평활 전, 중앙값 / |차| 중앙값):")
    for nm, _, _ in VIEW_BINS:
        log("    " + nm + ": " + "  ".join(f"{k} {med(v):+.1f}/{med(np.abs(v)):.1f}" for k, v in res[nm].items())
            + f"  (n={len(res[nm]['roll'])})")
    return res


# ---------------------------------------------------------------- 표: 분해, 중력 효과, 분리각

def decomp(J):
    allr = [x for rows in J.values() for x in rows]
    v = arr(allr, "view")
    out = {}
    for nm, lo, hi in VIEW_BINS:
        m = (v >= lo) & (v < hi)
        d = {}
        for k in ("roll", "pitch", "bar", "bike"):
            gk = {"roll": "g_roll", "pitch": "g_pitch", "bar": "g_bar", "bike": "g_bike"}[k]
            g = arr(allr, gk)[m]
            a, b, c = (arr(allr, f"{p}_{k}")[m] for p in ("A", "B", "C"))
            d[k] = dict(grav=med(b - a), head=med(c - b), joint=med(c - g), joint_abs=med(np.abs(c - g)),
                        A_g=med(a - g), B_g=med(b - g), B_g_abs=med(np.abs(b - g)))
        out[nm] = (int(m.sum()), d)
    return out


def run_tables(J):
    """런별: 카메라 피치·롤, 뒷모습(view<30) 롤·분리각 변화, 같은 진행률 구간 분리각(손-발, 손목선)."""
    out = {}
    for lab, rows in J.items():
        v = arr(rows, "view")
        p = arr(rows, "progress")
        rear = v < 30
        t = dict(n_rear=int(rear.sum()), cam_pitch=med(arr(rows, "B_cam_pitch")), cam_roll=med(arr(rows, "B_cam_roll")),
                 cam_pitch_rng=(np.nanmin(arr(rows, "B_cam_pitch")), np.nanmax(arr(rows, "B_cam_pitch"))))
        for src in ("A", "B", "Bmed", "C", "g"):
            roll = arr(rows, "g_roll" if src == "g" else f"{src}_roll")
            bike = arr(rows, "g_bike" if src == "g" else f"{src}_bike")
            bar = arr(rows, "g_bar" if src == "g" else f"{src}_bar")
            t[src] = dict(roll_rear=med(roll[rear]), sep_rear=med((bike - roll)[rear]), sepbar_rear=med((bar - roll)[rear]),
                          bins={b: (med((bike - roll)[(p >= b[0]) & (p < b[1])]), med((bar - roll)[(p >= b[0]) & (p < b[1])]))
                                for b in PROG_BINS})
        out[lab] = t
    return out


# ---------------------------------------------------------------- 그린 핸들바 선 45프레임

def hint_lines():
    """(run, frame) → 선 목록(크롭 좌표). 내 런은 hints_staging_v2.json, 코치는 sessions/reference-coach/bar_hints.json."""
    base = PATHS.DATA_ROOT / "outputs/2026-11-21/left_bank_lean/bar_hints"
    out = {}
    for sess, p in (("2026-", base / "hints_staging_v2.json"),
                    ("refer", PATHS.DATA_ROOT / "sessions/reference-coach/bar_hints.json")):
        for h in json.loads(p.read_text(encoding="utf-8"))["hints"]:
            out[(sess, h["run"], h["frame"])] = h
    return out


def img_angle(p, q):
    a, b = (p, q) if p[0] <= q[0] else (q, p)
    return math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))


class RunData:
    """한 런의 SAM raw/prep/평활 행, GVHMR npz/CSV를 frame으로 찾게 묶는다."""

    def __init__(self, seg, cfg, run):
        self.seg, self.cfg, self.run = seg, cfg, run
        self.raw = np.load(sdir(seg, cfg) / f"raw_{TAG}_run{run}.npz")
        self.prep = np.load(sdir(seg, cfg) / f"prep_run{run}.npz")
        self.frames = [int(f) for f in self.raw["frame"]]
        self.ri = {f: i for i, f in enumerate(self.frames)}
        self.pi = {int(f): i for i, f in enumerate(self.prep["frame"])}
        self.grav = B.gvhmr_frame(seg, cfg, run, self.frames)
        self.Q = B.level_rot(self.grav["up"])
        self.z = np.load(gdir(seg, cfg) / f"gvhmr_run{run}.npz")
        self.zi = {int(f): i for i, f in enumerate(self.z["frames"])}
        self.gcsv = {int(r["frame"]): r for r in B.read_csv(gdir(seg, cfg) / f"run{run}.csv")}
        self.sB = {int(r["frame"]): r for r in B.read_csv(sdir(seg, cfg) / f"run{run}_gvgrav.csv")}
        self.sA = {int(r["frame"]): r for r in B.read_csv(sdir(seg, cfg) / f"run{run}.csv")}
        self.sC = {int(r["frame"]): r for r in B.read_csv(sdir(seg, cfg) / f"run{run}_gvhead.csv")}
        self.lean = {int(r["frame"]): r for r in B.read_csv(B.lean_csv(seg, cfg, run))}
        f = float(self.raw["focal"])
        self.K = np.array([[f, 0, int(self.raw["width"]) / 2], [0, f, int(self.raw["height"]) / 2], [0, 0, 1]])

    def M(self, f):
        return self.prep["M"][self.pi[f]]

    def sam_2d(self, f):
        """SAM 관절 투영(크롭 좌표) dict, px/m(크롭)."""
        i = self.ri[f]
        kp = self.raw["kp3d"][i].astype(np.float64) + self.raw["cam_t"][i]
        M = self.M(f)
        uv = B.project(kp, self.K) @ M[:, :2].T + M[:, 2]
        P = (kp[B.M_LHIP] + kp[B.M_RHIP]) / 2
        pxm = M[0, 0] * self.K[0, 0] / P[2]
        return dict(lw=uv[B.M_LWR], rw=uv[B.M_RWR], la=uv[B.M_LAN], ra=uv[B.M_RAN]), pxm

    def gv_2d(self, f):
        j = self.zi[f + self.grav["off"]]
        Kg = self.z["K"]
        c = self.z["coco_c"][j].astype(np.float64)
        uv = c[:, :2] / c[:, 2:3] * np.array([Kg[0, 0], Kg[1, 1]]) + np.array([Kg[0, 2], Kg[1, 2]])
        M = self.M(f)
        uv = uv @ M[:, :2].T + M[:, 2]
        return dict(lw=uv[9], rw=uv[10], la=uv[15], ra=uv[16])

    def det_2d(self, f, thr=0.5):
        """검출 키포인트(크롭 좌표): ViTPose(GVHMR npz) 우선, 없으면 YOLO(prep). 값: (xy, conf, 출처)."""
        j = self.zi.get(f + self.grav["off"])
        M = self.M(f)
        vp = self.z["vitpose"][j] if j is not None else None
        p = self.prep
        out = {}
        for nm, ci in (("lw", 9), ("rw", 10), ("la", 15), ("ra", 16)):
            if vp is not None and vp[ci, 2] >= thr:
                out[nm] = (vp[ci, :2] @ M[:, :2].T + M[:, 2], float(vp[ci, 2]), "vitpose")
            elif p["kpc"][self.pi[f]][ci] >= thr:
                out[nm] = (p["kp"][self.pi[f]][ci] @ M[:, :2].T + M[:, 2], float(p["kpc"][self.pi[f]][ci]), "yolo")
            else:
                out[nm] = (None, float(vp[ci, 2]) if vp is not None else 0.0, "")
        return out

    def depth_parts(self, f):
        """dA를 카메라 좌표 성분으로 나눈다: 화면 안(x, y) 몫과 깊이(z) 몫(mm), 발목 중점 − 손목 중점의 깊이,
        왼손목 − 오른손목의 깊이(m, + = 왼손목이 카메라에서 더 멂). 두 모델, 각자의 카메라 좌표."""
        i = self.ri[f]
        j = self.zi[f + self.grav["off"]]
        k = self.raw["kp3d"][i].astype(np.float64)
        Jc = self.z["joints_c"][j].astype(np.float64)
        out = {}
        for nm, (LW, RW, LA, RA) in (("s", (k[B.M_LWR], k[B.M_RWR], k[B.M_LAN], k[B.M_RAN])),
                                     ("g", (Jc[GJ["lwr"]], Jc[GJ["rwr"]], Jc[GJ["lank"]], Jc[GJ["rank"]]))):
            W, A = (LW + RW) / 2, (LA + RA) / 2
            d = A - W
            p = d * unit(LW - RW) * 1000
            out[nm] = dict(img=float(p[0] + p[1]), depth=float(p[2]), dz=float(d[2]), wz=float((LW - RW)[2]))
        return out

    def offsets_3d(self, f):
        """dA, 왼·오른발목 오프셋(mm, 손목선 방향, + = 왼쪽), 손목선 수직 방향 손-발 기울기, 손목선 기울기 — 두 모델."""
        i = self.ri[f]
        u = -B.gravity(0.0)
        k = (self.Q[i] @ self.raw["kp3d"][i].astype(np.float64).T).T
        s = _offs(k[B.M_LWR], k[B.M_RWR], k[B.M_LAN], k[B.M_RAN], u)
        J = self.z["joints_g"][self.zi[f + self.grav["off"]]].astype(np.float64)
        g = _offs(J[GJ["lwr"]], J[GJ["rwr"]], J[GJ["lank"]], J[GJ["rank"]], np.array([0.0, 1.0, 0.0]))
        return s, g


def _offs(LW, RW, LA, RA, up):
    W, A = (LW + RW) / 2, (LA + RA) / 2
    wl = unit(LW - RW)
    bar = math.degrees(math.asin(np.clip(-wl @ up, -1, 1)))
    wh = unit(wl - (wl @ up) * up)
    t = unit(np.cross(wh, up))          # 손목선에 수직인 수평 방향(앞)
    L = np.cross(up, t)
    h = W - A
    return dict(dA=float((A - W) @ wl * 1000), dLA=float((LA - W) @ wl * 1000), dRA=float((RA - W) @ wl * 1000),
                bike_barhead=math.degrees(math.atan2(h @ L, h @ up)), bar=bar)


def drawn45(log):
    """45프레임: 그린 선(중력 기준 drawn_g)과 두 모델의 손목선·손-발 기울기, 그리고 2D 픽셀 시험."""
    D = json.loads((PATHS.DATA_ROOT / "outputs/2026-11-21/left_bank_lean/bar_hints/gvhmr_vs_drawn_45_lean.json")
                   .read_text(encoding="utf-8"))
    lines = hint_lines()
    cache = {}
    recs = []
    for o in D:
        seg, cfg = (REF, PATHS.load_cfg(REF)) if o["sess"] == "refer" else (SEG, PATHS.load_cfg(SEG))
        key = (o["sess"], o["run"])
        if key not in cache:
            cache[key] = RunData(seg, cfg, o["run"])
        rd = cache[key]
        f = o["frame"]
        sb, sa, sc = rd.sB.get(f, {}), rd.sA.get(f, {}), rd.sC.get(f, {})
        rec = dict(sess="coach" if o["sess"] == "refer" else "2026-11-21", run=o["run"], frame=f, drawn=o["drawn"],
                   cam_roll=o["cam_roll"], drawn_g=o["drawn_g"], lean_g=o.get("lean_g", np.nan),
                   view=o["view"], fit_yaw=fnum(rd.lean.get(f, {}).get("fit_yaw")),
                   g_bar=o["grip_roll"], g_bike=o["bike_body"],
                   B_bar=fnum(sb.get("bar_roll")), B_bike=fnum(sb.get("bike_lean_body")),
                   B_bar_raw=fnum(sb.get("bar_roll_raw")), B_bike_raw=fnum(sb.get("bike_lean_body_raw")),
                   A_bar=fnum(sa.get("bar_roll")), A_bike=fnum(sa.get("bike_lean_body")), C_bike=fnum(sc.get("bike_lean_body")),
                   fit_g=o["fit_s"] + o["cam_roll"])
        s2, pxm = rd.sam_2d(f)
        g2 = rd.gv_2d(f)
        rec["s_wrist2d"] = img_angle(s2["lw"], s2["rw"])
        rec["g_wrist2d"] = img_angle(g2["lw"], g2["rw"])
        so, go = rd.offsets_3d(f)
        rec.update(s_dA=so["dA"], g_dA=go["dA"], s_dLA=so["dLA"], s_dRA=so["dRA"], g_dLA=go["dLA"], g_dRA=go["dRA"])
        # 픽셀 시험
        h = lines.get((o["sess"], o["run"], f))
        det = rd.det_2d(f)
        if h is not None:
            ln = h["lines"][-1]
            a_, b_ = np.array(ln["a"], float), np.array(ln["b"], float)
            dvec = unit(b_ - a_)                      # 이미지 왼쪽 → 오른쪽
            stem = b_ - HALF_GRIP_M * pxm * dvec
            rec["pxm"] = pxm
            rec["e_rw_sam"] = float((s2["rw"] - b_) @ dvec / pxm * 1000)
            rec["e_rw_gv"] = float((g2["rw"] - b_) @ dvec / pxm * 1000)
            la, ra = det["la"][0], det["ra"][0]
            if la is not None and ra is not None:
                Ad = (la + ra) / 2
                rec["dA_2d_drawn"] = float(-(Ad - stem) @ dvec / pxm * 1000)       # + = 이미지 왼쪽 = 라이더 왼쪽(뒷모습)
                rec["det_src"] = det["la"][2]
                for nm, pts in (("sam", s2), ("gv", g2)):
                    rec[f"e_A_{nm}"] = float(-((pts["la"] + pts["ra"]) / 2 - Ad) @ dvec / pxm * 1000)
            for nm, pts in (("sam", s2), ("gv", g2)):
                Wm, Am = (pts["lw"] + pts["rw"]) / 2, (pts["la"] + pts["ra"]) / 2
                rec[f"dA_2d_{nm}"] = float(-(Am - Wm) @ dvec / pxm * 1000)
                rec[f"e_W_{nm}"] = float(-(Wm - stem) @ dvec / pxm * 1000)
        recs.append(rec)
    # 채점
    log("")
    log(f"[그린 선 {len(recs)}프레임] 기준 = drawn_g (그린 화면 각 + GVHMR cam_roll). MAE / 평균차 / 최대|차|")
    stats = {}
    for k, nm in (("g_bar", "GVHMR 손목선"), ("g_bike", "GVHMR 손-발"), ("B_bar", "SAM(B) 손목선"),
                  ("B_bike", "SAM(B) 손-발"), ("B_bar_raw", "SAM(B) 손목선 평활 전"), ("B_bike_raw", "SAM(B) 손-발 평활 전"),
                  ("C_bike", "SAM(C, GVHMR 진행 방향) 손-발"), ("A_bar", "SAM(A, 화면 아래 중력) 손목선"),
                  ("A_bike", "SAM(A) 손-발"), ("fit_g", "실루엣 fix4 + cam_roll")):
        d = np.array([r[k] - r["drawn_g"] for r in recs], float)
        ok = np.isfinite(d)
        stats[k] = (np.mean(np.abs(d[ok])), np.mean(d[ok]), np.max(np.abs(d[ok])), int(ok.sum()))
        log(f"    {nm:32s} {stats[k][0]:4.1f}° / {stats[k][1]:+4.1f}° / {stats[k][2]:4.1f}°  n={stats[k][3]}")
    log("  중력 없이 화면 각끼리(그린 화면 각 drawn vs 투영 손목선 화면 각):")
    for k, nm in (("s_wrist2d", "SAM 투영 손목선"), ("g_wrist2d", "GVHMR 투영 손목선")):
        d = np.array([r[k] - r["drawn"] for r in recs], float)
        stats[k] = (np.mean(np.abs(d)), np.mean(d), np.max(np.abs(d)), len(d))
        log(f"    {nm:32s} {stats[k][0]:4.1f}° / {stats[k][1]:+4.1f}° / {stats[k][2]:4.1f}°  n={len(d)}")
    lg = [r for r in recs if np.isfinite(r["lean_g"])]
    log(f"  참고: lean_g(그린 선을 bike_fit 모델로 시점 보정, {len(lg)}프레임) 기준 MAE: "
        + "  ".join(f"{nm} {np.mean([abs(r[k] - r['lean_g']) for r in lg]):.1f}°"
                    for k, nm in (("g_bar", "GV 손목선"), ("g_bike", "GV 손-발"), ("B_bar", "SAM 손목선"), ("B_bike", "SAM 손-발"))))
    return recs, stats


# ---------------------------------------------------------------- Run 5 진단

def run5_series(rd, f0=195, f1=225):
    out = []
    for f in range(f0, f1 + 1):
        if f not in rd.ri or f + rd.grav["off"] not in rd.zi:
            continue
        so, go = rd.offsets_3d(f)
        g = rd.gcsv.get(f, {})
        sb = rd.sB.get(f, {})
        out.append(dict(frame=f, view=fnum(g.get("cam_view_deg")), fit_yaw=fnum(rd.lean.get(f, {}).get("fit_yaw")),
                        view_s=fnum(g.get("cam_view_signed")),
                        g_bar=fnum(g.get("grip_roll")), g_bike=fnum(g.get("bike_lean_body")), g_bike_barhead=go["bike_barhead"],
                        g_dA=go["dA"], g_dLA=go["dLA"], g_dRA=go["dRA"],
                        s_bar=fnum(sb.get("bar_roll")), s_bike=fnum(sb.get("bike_lean_body")), s_bar_raw=so["bar"],
                        s_bike_barhead_raw=so["bike_barhead"], s_dA=so["dA"], s_dLA=so["dLA"], s_dRA=so["dRA"]))
    # SAM 평활 dA (raw가 프레임마다 크게 흔들림)
    sd = B.smooth_series(np.array([r["s_dA"] for r in out]), 1.5)
    sbh = B.smooth_series(np.array([r["s_bike_barhead_raw"] for r in out]), 1.5)
    for r, a, b in zip(out, sd, sbh):
        r["s_dA_sm"], r["s_bike_barhead"] = float(a), float(b)
    return out


def tile(rd, f, size=360, title=""):
    """크롭에 그린 선·두 모델 관절·검출 관절·자전거 마스크를 그린다."""
    import cv2
    ldir = PATHS.feat_dir(rd.seg, rd.cfg, "lean")
    img = cv2.imread(str(ldir / "crops" / f"run{rd.run}_f{f:04d}.jpg"))
    if img is None:
        return None
    mp = ldir / "masks" / f"run{rd.run}_f{f:04d}.png"
    if mp.exists():
        bm = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        cs, _ = cv2.findContours((bm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 200, 255), 2, cv2.LINE_AA)
    ip = lambda p: tuple(int(round(v)) for v in p)  # noqa: E731
    s2, pxm = rd.sam_2d(f)
    g2 = rd.gv_2d(f)
    sess = "refer" if rd.run == 0 else "2026-"
    h = hint_lines().get((sess, rd.run, f))
    if h is not None:
        ln = h["lines"][-1]
        a_, b_ = np.array(ln["a"]), np.array(ln["b"])
        cv2.line(img, ip(a_), ip(b_), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(img, ip(b_), 9, (255, 255, 255), 2, cv2.LINE_AA)
        stem = b_ - HALF_GRIP_M * pxm * unit(b_ - a_)
        cv2.drawMarker(img, ip(stem), (255, 255, 255), cv2.MARKER_CROSS, 22, 3)
    for pts, col in ((s2, (255, 0, 255)), (g2, (255, 200, 0))):
        Wm, Am = (pts["lw"] + pts["rw"]) / 2, (pts["la"] + pts["ra"]) / 2
        cv2.line(img, ip(pts["lw"]), ip(pts["rw"]), col, 2, cv2.LINE_AA)
        cv2.line(img, ip(Am), ip(Wm), col, 2, cv2.LINE_AA)
        for k in ("lw", "rw"):
            cv2.circle(img, ip(pts[k]), 6, col, -1, cv2.LINE_AA)
        for k in ("la", "ra"):
            cv2.rectangle(img, ip(pts[k] - 6), ip(pts[k] + 6), col, -1, cv2.LINE_AA)
        cv2.drawMarker(img, ip(Wm), col, cv2.MARKER_TILTED_CROSS, 18, 3)
        cv2.drawMarker(img, ip(Am), col, cv2.MARKER_TILTED_CROSS, 18, 3)
    for k, (xy, c, src) in rd.det_2d(f).items():
        if xy is not None:
            cv2.circle(img, ip(xy), 11, (0, 255, 0), 2, cv2.LINE_AA)
    # 라이더 주변만
    pts = np.array([s2[k] for k in s2] + [g2[k] for k in g2])
    cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    half = max(150, int(np.ptp(pts[:, 1]) * 0.9))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    img = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img[:, :, ::-1]


# ---------------------------------------------------------------- 보드

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def board_main(J, dec, rt, coco, drawn_sep, out):
    plt = setup_mpl()
    labs = list(J.keys())
    n = len(labs)
    fig = plt.figure(figsize=(19, 2.0 * n + 15), dpi=105)
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1] * n + [8.0], hspace=0.42, wspace=0.08, top=0.97, bottom=0.01)
    fig.suptitle("SAM 3D Body vs GVHMR — 프레임별 3D 몸 각도 (측정값). 회색 띠 = GVHMR 시점 < 30°(뒷모습)", fontsize=15, y=0.995)
    C = dict(A="#9c9c9c", B="#2ca02c", C="#1f77b4", g="#111111")
    for i, lab in enumerate(labs):
        rows = J[lab]
        p, v = arr(rows, "progress"), arr(rows, "view")
        for c, (ttl, k, gk) in enumerate((("상체 롤", "roll", "g_roll"), ("손-발 자전거 기울기 / 손목선", "bike", "g_bike"))):
            ax = fig.add_subplot(gs[i, c])
            for j in range(len(p)):
                if v[j] < 30:
                    lo = p[j - 1] if j else p[j]
                    hi = p[j + 1] if j + 1 < len(p) else p[j]
                    ax.axvspan((lo + p[j]) / 2, (hi + p[j]) / 2, color="#e8e8e8", lw=0, zorder=0)
            ax.axhline(0, color="#bbbbbb", lw=0.8)
            ax.plot(p, arr(rows, f"A_{k}"), color=C["A"], lw=1.2, ls=":", label="SAM A (화면 아래 중력)")
            ax.plot(p, arr(rows, f"B_{k}"), color=C["B"], lw=2, label="SAM B (GVHMR 중력)")
            ax.plot(p, arr(rows, f"C_{k}"), color=C["C"], lw=1.4, ls="--", label="SAM C (B + GVHMR 진행 방향)")
            ax.plot(p, arr(rows, gk), color=C["g"], lw=2, label="GVHMR")
            if k == "bike":
                ax.plot(p, arr(rows, "B_bar"), color="#c03fbf", lw=1.1, label="SAM B 손목선")
                ax.plot(p, arr(rows, "g_bar"), color="#c03fbf", lw=1.1, ls="--", label="GVHMR 손목선")
            ax.set_xlim(0, 1)
            ax.set_ylim(-30, 60)
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_ylabel(lab, fontsize=12, rotation=0, ha="right", va="center")
            if i == 0:
                ax.set_title(f"{ttl} (+ = 왼쪽, 도)", fontsize=11, loc="left")
                ax.legend(fontsize=7.5, ncol=3 if k == "roll" else 3, loc="upper left", framealpha=0.9)
            if i == n - 1:
                ax.set_xlabel("진행률", fontsize=9)
    axt = fig.add_subplot(gs[n, :])
    axt.axis("off")
    f0 = lambda x: "" if not np.isfinite(x) else f"{x:+.0f}"  # noqa: E731
    f1 = lambda x: "" if not np.isfinite(x) else f"{x:+.1f}"  # noqa: E731
    names = {"roll": "상체 롤", "pitch": "상체 피치", "bar": "손목선 기울기", "bike": "손-발 자전거 기울기"}
    rows_t = []
    for k in ("roll", "pitch", "bar", "bike"):
        cells = [names[k]]
        for nm, (cnt, d) in dec.items():
            x = d[k]
            cells += [f1(x["A_g"]), f1(x["grav"]), f1(x["head"]), f1(x["joint"]), f1(med(coco[nm][k]))]
        rows_t.append(cells)
    cl = ["각도 (중앙값, 도)"]
    for nm, (cnt, d) in dec.items():
        short = nm.split()[0]
        cl += [f"[{short}]\nA - GV", f"[{short}] 중력\nB - A", f"[{short}] 진행\nC - B", f"[{short}] 관절\nC - GV",
               f"[{short}] COCO\nSAM - GV"]
    ncnt = ", ".join(f"{nm} n={cnt}" for nm, (cnt, _) in dec.items())
    t1 = axt.table(cellText=rows_t, colLabels=cl, bbox=[0.0, 0.70, 1.0, 0.27], cellLoc="center",
                   colWidths=[0.1] + [0.06] * 15)
    rows_r = []
    for lab in labs:
        t = rt[lab]
        ds = drawn_sep.get(lab)
        rows_r.append([lab, f"{t['cam_pitch']:+.1f} ({t['cam_pitch_rng'][0]:+.0f}~{t['cam_pitch_rng'][1]:+.0f})",
                       f"{t['cam_roll']:+.1f}", f"{f0(t['A']['roll_rear'])} → {f0(t['B']['roll_rear'])} | {f0(t['g']['roll_rear'])}",
                       f"{f0(t['A']['sep_rear'])} → {f0(t['B']['sep_rear'])} ({f0(t['Bmed']['sep_rear'])}) | {f0(t['g']['sep_rear'])}",
                       f"{f0(t['B']['sepbar_rear'])} | {f0(t['g']['sepbar_rear'])}",
                       (f"{f0(ds['g'])} | {f0(ds['B'])}  (n={ds['n']}, 진행률 {ds['p0']:.2f}~{ds['p1']:.2f})" if ds else ""),
                       str(t["n_rear"])])
    t2 = axt.table(cellText=rows_r, colLabels=["", "카메라 피치 (범위)", "카메라 롤", "뒷모습 상체 롤\nSAM A → B | GVHMR",
                                               "뒷모습 손-발 분리각\nSAM A → B (B런중앙값) | GVHMR",
                                               "뒷모습 손목선 분리각\nSAM B | GVHMR",
                                               "그린 선 분리각 = 그린 선 - 상체 롤\nGVHMR 롤 | SAM B 롤 (그린 프레임 중앙값)", "뒷모습 n"],
                   bbox=[0.0, 0.33, 1.0, 0.34], cellLoc="center", colWidths=[0.07, 0.1, 0.07, 0.14, 0.18, 0.12, 0.26, 0.06])
    rows_b = []
    for lab in labs:
        t = rt[lab]
        cells = [lab]
        for b in PROG_BINS:
            cells.append(" / ".join(f0(t[s]["bins"][b][0]) for s in ("A", "B", "g")))
            cells.append(" / ".join(f0(t[s]["bins"][b][1]) for s in ("B", "g")))
        rows_b.append(cells)
    cl = [""]
    for b in PROG_BINS:
        cl += [f"진행률 {b[0]:.1f}~{b[1]:.1f}\n손-발 분리각 SAM A / B / GV", f"진행률 {b[0]:.1f}~{b[1]:.1f}\n손목선 분리각 SAM B / GV"]
    t3 = axt.table(cellText=rows_b, colLabels=cl, bbox=[0.0, 0.0, 1.0, 0.30], cellLoc="center",
                   colWidths=[0.07] + [0.155] * 6)
    for t in (t1, t2, t3):
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        for (r, c), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_height(cell.get_height() * 1.6)
    axt.text(0.0, 0.985, f"위 표: 시점 구간별 SAM - GVHMR 차이 분해 ({ncnt}). B - A = 카메라 기울기, C - B = 진행 방향, "
             "C - GV = 두 모델 관절 차이, COCO = 같은 중력·진행 방향에서 어깨 중점 - 엉덩이 중점으로 잰 차이.\n"
             "가운데 표: 런별 GVHMR 카메라와 뒷모습(<30°) 중앙값. 아래 표: 같은 진행률 구간 분리각(자전거 - 상체 롤) 중앙값. "
             "손목선 분리각 = 손목선 기울기 - 상체 롤. 모두 측정값.",
             transform=axt.transAxes, fontsize=9.5, va="bottom")
    fig.savefig(out, bbox_inches="tight")
    print("  board:", out)


def board_drawn45(recs, stats, out):
    plt = setup_mpl()
    fig, axs = plt.subplots(1, 4, figsize=(19, 6.0), dpi=105)
    coach = np.array([r["sess"] == "coach" for r in recs])
    x = np.array([r["drawn_g"] for r in recs])
    for ax, (k, ttl) in zip(axs, (("g_bar", "GVHMR 손목선"), ("B_bar", "SAM(B) 손목선"),
                                  ("g_bike", "GVHMR 손-발"), ("B_bike", "SAM(B) 손-발"))):
        y = np.array([r[k] for r in recs])
        ax.fill_between([-10, 55], [-15, 50], [-5, 60], color="#eeeeee", lw=0)
        ax.plot([-10, 55], [-10, 55], ls="--", color="#888888", lw=1)
        ax.scatter(x[~coach], y[~coach], s=40, color="#1f77b4", label="내 런")
        ax.scatter(x[coach], y[coach], s=55, marker="^", color="#ff7f0e", label="코치님")
        r5 = np.array([r["run"] == 5 and r["sess"] != "coach" for r in recs])
        ax.scatter(x[r5], y[r5], s=110, facecolor="none", edgecolor="#d62728", lw=1.6, label="Run 5")
        m = stats[k]
        ax.set_title(f"{ttl}\nMAE {m[0]:.1f}° · 평균차 {m[1]:+.1f}° · 최대 {m[2]:.0f}°", fontsize=11.5, loc="left")
        ax.set_xlim(-5, 50)
        ax.set_ylim(-5, 55)
        ax.set_aspect("equal")
        ax.grid(color="#e6e6e6")
        ax.set_xlabel("그린 핸들바 각도 (°, GVHMR 중력 기준)")
    axs[0].set_ylabel("모델 값 (°)")
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"그린 핸들바 선 {len(recs)}프레임 vs 두 모델 (측정값). 회색 띠 = ±5°", fontsize=14, x=0.01, ha="left")
    s2, g2 = stats["s_wrist2d"], stats["g_wrist2d"]
    fig.text(0.01, 0.005, f"중력 없이 화면 각끼리 비교하면 투영 손목선 MAE: SAM {s2[0]:.1f}°, GVHMR {g2[0]:.1f}°. "
             f"SAM 값은 B(GVHMR 중력) 평활 열. 그린 선은 사람 판단이고, 뒷모습에서 왼손은 몸에 가려 두 모델 모두 추정이다. 정답(IMU)이 아니다.",
             fontsize=9.5, color="#555555")
    plt.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.12, wspace=0.18)
    fig.savefig(out)
    print("  board:", out)


def board_run5(series, tiles, out, ctrl_note, dtab):
    plt = setup_mpl()
    fig = plt.figure(figsize=(17, 24), dpi=105)
    gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 0.8, 1.55, 1.55, 0.75], hspace=0.28, wspace=0.04, top=0.955, bottom=0.03)
    f = np.array([r["frame"] for r in series])
    g = lambda k: np.array([r[k] for r in series], float)  # noqa: E731
    ax = fig.add_subplot(gs[0, :])
    ax.plot(f, g("g_bike") - g("g_bar"), color="#111111", lw=2, label="GVHMR: 손-발(골반 방향) - 손목선")
    ax.plot(f, g("g_bike_barhead") - g("g_bar"), color="#111111", lw=1.3, ls="--", label="GVHMR: 손-발(손목선 수직 방향) - 손목선 = dA 몫")
    ax.plot(f, g("s_bike") - g("s_bar"), color="#2ca02c", lw=2, label="SAM B: 손-발(골반 방향) - 손목선")
    ax.plot(f, g("s_bike_barhead") - g("s_bar"), color="#2ca02c", lw=1.3, ls="--", label="SAM B: 손-발(손목선 수직 방향) - 손목선 = dA 몫")
    ax.axhline(0, color="#bbbbbb")
    ax.axvspan(204.5, 212.5, color="#fde2e2", lw=0, zorder=0)
    ax.axvspan(212.5, 223.5, color="#e2ecfd", lw=0, zorder=0)
    ax.text(208.5, 0.97, "f205~212", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
    ax.text(218, 0.97, "f213~223", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
    ax.set_ylabel("차이 (°)")
    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.9)
    ax.set_title("Run 5: 손-발 자전거 기울기 - 손목선 기울기 (측정값, SAM은 GVHMR 중력 B·평활)", fontsize=12, loc="left")
    ax.set_xlim(f[0], f[-1])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax)
    ax2.plot(f, g("g_dA"), color="#111111", lw=2, label="GVHMR dA")
    ax2.plot(f, g("g_dLA"), color="#111111", lw=1, ls=":", label="GVHMR 왼발목")
    ax2.plot(f, g("g_dRA"), color="#111111", lw=1, ls="-.", label="GVHMR 오른발목")
    ax2.plot(f, g("s_dA_sm"), color="#2ca02c", lw=2, label="SAM dA (평활)")
    ax2.plot(f, g("s_dA"), ".", color="#2ca02c", ms=4)
    ax2.plot(f, g("s_dLA"), color="#2ca02c", lw=1, ls=":", label="SAM 왼발목")
    ax2.plot(f, g("s_dRA"), color="#2ca02c", lw=1, ls="-.", label="SAM 오른발목")
    ax2.axhline(0, color="#bbbbbb")
    ax2.set_ylabel("손목 중점 기준\n손목선 방향 위치 (mm, + = 왼쪽)")
    ax2.legend(fontsize=8.5, ncol=4, loc="lower left")
    ax2.set_xlabel("프레임")
    for i, (img, cap) in enumerate(tiles):
        a = fig.add_subplot(gs[2 + i // 3, i % 3])
        a.imshow(img)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(cap, fontsize=10.5, loc="left")
    axt = fig.add_subplot(gs[4, :])
    axt.axis("off")
    t = axt.table(cellText=dtab, colLabels=["프레임", "GVHMR\n골반 시점", "실루엣 자전거\nyaw (fix4)", "실루엣\n- 골반",
                                            "GVHMR dA (mm)\n= 화면 안 + 깊이", "SAM dA (mm)\n= 화면 안 + 깊이",
                                            "발목 중점 - 손목 중점\n깊이 (m) GVHMR / SAM",
                                            "왼손목 - 오른손목\n깊이 (m) GVHMR / SAM"],
                  bbox=[0, 0, 1, 0.88], cellLoc="center", colWidths=[0.13, 0.08, 0.1, 0.07, 0.16, 0.16, 0.15, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_height(cell.get_height() * 1.5)
    axt.set_title("dA 분해 (측정값, 각 모델의 카메라 좌표: 화면 안 = x·y 성분, 깊이 = z 성분. 깊이 - = 카메라에 가까움)",
                  fontsize=11, loc="left")
    fig.suptitle("Run 5: 손-발 기울기가 손목선보다 큰 이유", fontsize=16, x=0.01, ha="left")
    fig.text(0.01, 0.005, "장면 칸: 흰 선 = 그린 핸들바(오른쪽 동그라미 = 보이는 오른손 그립, + = 그립 반폭 0.34 m 왼쪽의 추정 줄기),\n"
             "자주 = SAM, 하늘 = GVHMR(점 = 손목, 네모 = 발목, X = 중점, 선 = 발목 중점→손목 중점), 초록 원 = 검출 관절(ViTPose/YOLO conf≥0.5), "
             "주황 = 자전거 마스크.\n" + ctrl_note, fontsize=10, color="#444444")
    fig.savefig(out, bbox_inches="tight")
    print("  board:", out)


# ---------------------------------------------------------------- 메인

def main():
    od = sdir(SEG, PATHS.load_cfg(SEG))
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    check_reproduce(log)
    J = all_joined()
    with open(od / "vs_gvhmr_frames.csv", "w", newline="", encoding="utf-8") as fh:
        keys = list(next(iter(J.values()))[0].keys())
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for rows in J.values():
            for r in rows:
                w.writerow({k: (round(v, 2) if isinstance(v, float) and np.isfinite(v) else ("" if isinstance(v, float) else v))
                            for k, v in r.items()})
    coco = coco_check(log, J)
    dec = decomp(J)
    log("")
    log("[분해] 시점 구간별 중앙값(도): A - GVHMR | 중력 B - A | 진행 방향 C - B | 관절 C - GVHMR (|C - GVHMR|)")
    for nm, (cnt, d) in dec.items():
        for k, x in d.items():
            log(f"    {nm} n={cnt} {k:5s}: {x['A_g']:+5.1f} | {x['grav']:+5.1f} | {x['head']:+5.1f} | {x['joint']:+5.1f} ({x['joint_abs']:.1f})"
                f"   [B - GVHMR {x['B_g']:+5.1f}, |B - GVHMR| {x['B_g_abs']:.1f}]")
    rt = run_tables(J)
    log("")
    log("[런별] 카메라 피치·롤(GVHMR, 중앙값), 뒷모습(<30°) 상체 롤 A→B→GVHMR, 분리각 A→B(B런중앙값)→GVHMR, 손목선 분리각 B / GVHMR")
    for lab, t in rt.items():
        log(f"    {lab:6s} 피치 {t['cam_pitch']:+.1f} ({t['cam_pitch_rng'][0]:+.0f}~{t['cam_pitch_rng'][1]:+.0f}) 롤 {t['cam_roll']:+.1f} | "
            f"롤 {t['A']['roll_rear']:+.0f}→{t['B']['roll_rear']:+.0f}→{t['g']['roll_rear']:+.0f} | "
            f"분리각 {t['A']['sep_rear']:+.0f}→{t['B']['sep_rear']:+.0f}({t['Bmed']['sep_rear']:+.0f})→{t['g']['sep_rear']:+.0f} | "
            f"손목선 분리각 {t['B']['sepbar_rear']:+.0f} / {t['g']['sepbar_rear']:+.0f} (n={t['n_rear']})")
    log("  같은 진행률 구간 분리각 (손-발 A/B/GV ; 손목선 B/GV):")
    for lab, t in rt.items():
        log(f"    {lab:6s} " + "  ".join(
            f"{b[0]:.1f}~{b[1]:.1f}: " + "/".join(f"{t[s]['bins'][b][0]:+.0f}" for s in ("A", "B", "g")) + " ; "
            + "/".join(f"{t[s]['bins'][b][1]:+.0f}" for s in ("B", "g")) for b in PROG_BINS))
    recs, stats = drawn45(log)
    with open(od / "drawn45_eval.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in recs for k in r}, key=lambda k: list(recs[0].keys()).index(k) if k in recs[0] else 999)
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: (round(v, 1) if isinstance(v, float) and np.isfinite(v) else ("" if isinstance(v, float) else v))
                        for k, v in r.items()})
    log("  픽셀 시험(그린 선 오른쪽 끝 + 검출 발목, mm, + = 이미지 왼쪽): 프레임별")
    log("    run frame view fit_yaw | dA_2d(그린선) | dA 3D SAM / GV | dA 2D SAM / GV | 오른손목-그린끝 SAM / GV | 발목중점-검출 SAM / GV | 손목중점-줄기 SAM / GV")
    for r in recs:
        if "dA_2d_drawn" not in r:
            continue
        log(f"    {r['sess'][:5]} r{r['run']} f{r['frame']} {r['view']:4.0f} {r['fit_yaw']:+4.0f} | {r['dA_2d_drawn']:+5.0f} | "
            f"{r['s_dA']:+5.0f} / {r['g_dA']:+5.0f} | {r['dA_2d_sam']:+5.0f} / {r['dA_2d_gv']:+5.0f} | "
            f"{r['e_rw_sam']:+5.0f} / {r['e_rw_gv']:+5.0f} | {r['e_A_sam']:+5.0f} / {r['e_A_gv']:+5.0f} | "
            f"{r['e_W_sam']:+5.0f} / {r['e_W_gv']:+5.0f}")
    # Run 5
    cfg = PATHS.load_cfg(SEG)
    rd5 = RunData(SEG, cfg, 5)
    ser = run5_series(rd5)
    with open(od / "run5_gap.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(ser[0].keys()))
        w.writeheader()
        for r in ser:
            w.writerow({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
    for lo, hi in ((205, 212), (213, 223)):
        sel = [r for r in ser if lo <= r["frame"] <= hi]
        g = lambda k: med([r[k] for r in sel])  # noqa: E731
        log(f"[Run 5 f{lo}~{hi}] 중앙값: GVHMR 손-발-손목선 {g('g_bike') - g('g_bar'):+.1f}° (손목선 수직 방향이면 "
            f"{med([r['g_bike_barhead'] - r['g_bar'] for r in sel]):+.1f}°), dA {g('g_dA'):+.0f} mm (왼 {g('g_dLA'):+.0f}, 오른 {g('g_dRA'):+.0f}) | "
            f"SAM 손-발-손목선 {med([r['s_bike'] - r['s_bar'] for r in sel]):+.1f}° (손목선 수직 방향 "
            f"{med([r['s_bike_barhead'] - r['s_bar'] for r in sel]):+.1f}°), dA {g('s_dA_sm'):+.0f} mm (왼 {g('s_dLA'):+.0f}, 오른 {g('s_dRA'):+.0f}) | "
            f"시점 {g('view'):.0f}° fit_yaw {g('fit_yaw'):+.0f}°")
    rd1 = RunData(SEG, cfg, 1)
    rd3 = RunData(SEG, cfg, 3)
    tiles, dtab = [], []
    byf = {(r["run"], r["frame"]): r for r in recs if r["sess"] != "coach"}
    for rd, f in ((rd5, 205), (rd5, 208), (rd5, 210), (rd5, 216), (rd1, 181), (rd3, 216)):
        img = tile(rd, f)
        r = byf.get((rd.run, f), {})
        so, go = rd.offsets_3d(f)
        gr, sb = rd.gcsv[f], rd.sB[f]
        gap_g = fnum(gr["bike_lean_body"]) - fnum(gr["grip_roll"])
        gap_s = fnum(sb["bike_lean_body"]) - fnum(sb["bar_roll"])
        cap = (f"Run {rd.run} f{f}" + ("  (대조)" if rd.run != 5 else "") + f"   시점 {fnum(gr['cam_view_signed']):+.0f}°, "
               f"실루엣 yaw {fnum(rd.lean.get(f, {}).get('fit_yaw')):+.0f}°\n손-발 - 손목선: GVHMR {gap_g:+.0f}°, SAM {gap_s:+.0f}°\n"
               f"dA 3D: GVHMR {go['dA']:+.0f}, SAM {so['dA']:+.0f} mm / 2D(그린 선+검출) {r.get('dA_2d_drawn', float('nan')):+.0f} mm")
        if img is not None:
            tiles.append((img, cap))
    for rd, f in ((rd5, 205), (rd5, 208), (rd5, 210), (rd5, 212), (rd5, 216), (rd1, 181), (rd3, 216)):
        dp = rd.depth_parts(f)
        vs, fy = fnum(rd.gcsv[f]["cam_view_signed"]), fnum(rd.lean.get(f, {}).get("fit_yaw"))
        low = str(rd.lean.get(f, {}).get("fit_lowconf", "")).strip() in ("1", "1.0", "True")
        dtab.append([f"Run {rd.run} f{f}" + (" (대조)" if rd.run != 5 else ""), f"{vs:+.0f}°", f"{fy:+.0f}°" + (" LOW" if low else ""),
                     f"{fy - vs:+.0f}°",
                     f"{dp['g']['img'] + dp['g']['depth']:+.0f} = {dp['g']['img']:+.0f} {dp['g']['depth']:+.0f}",
                     f"{dp['s']['img'] + dp['s']['depth']:+.0f} = {dp['s']['img']:+.0f} {dp['s']['depth']:+.0f}",
                     f"{dp['g']['dz']:+.2f} / {dp['s']['dz']:+.2f}", f"{dp['g']['wz']:+.2f} / {dp['s']['wz']:+.2f}"])
        log(f"  [dA 분해] {dtab[-1]}")
    board_run5(ser, tiles, od / "run5_gap_board.png",
               "대조 칸 Run 1 f181, Run 3 f216은 손-발과 손목선 차이가 작은 프레임이다. 마스크는 2026-11-22 21:15 버전(그림자 제거 전).",
               dtab)
    # 그린 선 분리각: 그린 선(중력 기준) - 상체 롤, 그린 프레임 중앙값
    jmap = {(lab, x["frame"]): x for lab, rows in J.items() for x in rows}
    drawn_sep = {}
    for lab in J:
        run = 0 if lab == "코치님" else int(lab.split()[-1])
        sel = [(r, jmap[(lab, r["frame"])]) for r in recs if r["run"] == run and (lab, r["frame"]) in jmap
               and (r["sess"] == "coach") == (lab == "코치님")]
        if sel:
            p = [x["progress"] for _, x in sel]
            drawn_sep[lab] = dict(n=len(sel), p0=min(p), p1=max(p), g=med([r["drawn_g"] - x["g_roll"] for r, x in sel]),
                                  B=med([r["drawn_g"] - x["B_roll"] for r, x in sel]))
            log(f"  [그린 선 분리각] {lab}: 그린 선 - GVHMR 롤 {drawn_sep[lab]['g']:+.0f}°, - SAM B 롤 {drawn_sep[lab]['B']:+.0f}° "
                f"(n={len(sel)}, 진행률 {min(p):.2f}~{max(p):.2f}); 프레임별 " +
                " ".join(f"{x['progress']:.2f}:{r['drawn_g'] - x['g_roll']:+.0f}" for r, x in sel))
    board_main(J, dec, rt, coco, drawn_sep, od / "vs_gvhmr_board.png")
    board_drawn45(recs, stats, od / "drawn45_board.png")
    (od / "vs_gvhmr_tables.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  tables:", od / "vs_gvhmr_tables.txt")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("all",):
        sys.exit(__doc__)
    main()
