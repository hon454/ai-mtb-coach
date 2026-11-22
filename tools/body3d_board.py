"""SAM 3D Body 3D 몸 각도 vs 기존 2D body_lean·실루엣 fit_lean_s 비교 보드 (body3d_sam3d.py board에서 부름).

한 줄에 한 런(코치님 run 0 + 2026-11-21 run 1~7):
    왼쪽  기울기(진행률 축): 3D torso_roll·bike_lean_body(실선, 점 = 평활 전), 2D body_lean(점선),
          실루엣 fit_lean_s(파선, × = fit_lowconf), bar_roll(가는 자주색). 회색 띠 = 뒷모습(view_ratio > 0.75).
    오른쪽 3D torso_pitch와 travel_yaw(시점: 0 = 뒷모습, ±90 = 옆모습).
아래 표:
    A. 뒷모습 분리각(자전거 − 상체) 런별 중앙값: 실루엣(separation_board 기준 프레임), 같은 프레임의 3D,
       view_ratio > 0.75 전체 3D, 설정 바꿈(프롬프트·초점거리·피치 ±10°·진행 방향 ±10°)에 따른 범위.
    B. 프레임 간 흔들림: 연속 프레임 차이의 중앙값(평활 전 값).
    C. 시점 혼입: 3D torso_roll − 2D body_lean, 뒷모습 vs 그 밖.
"""
import math
from pathlib import Path

import numpy as np

import body3d_sam3d as B
import paths as PATHS

FONT = "Malgun Gothic"
GREEN, ORANGE, MAG, BLUE, GREY = "#2ca02c", "#ff7f0e", "#c03fbf", "#1f77b4", "#888888"


def col(rows, k):
    return np.array([B.fnum(r.get(k)) for r in rows])


def sil_frames(lean):
    """separation_board.select와 같은 기준의 뒷모습 프레임."""
    out = set()
    for x in lean:
        vr, yaw, iou = B.fnum(x.get("view_ratio")), B.fnum(x.get("fit_yaw")), B.fnum(x.get("fit_iou"))
        if not all(np.isfinite([vr, yaw, iou, B.fnum(x.get("body_lean")), B.fnum(x.get("fit_lean_s"))])):
            continue
        if str(x.get("fit_lowconf", "0")).strip() in ("1", "1.0", "True"):
            continue
        if vr > 0.75 and abs(yaw) < 25 and iou >= 0.55:
            out.add(int(x["frame"]))
    return out


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.median(a)), len(a)) if len(a) else (np.nan, 0)


def jitter(a):
    a = np.asarray(a, float)
    d = np.abs(np.diff(a))
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else np.nan


def run_stats(seg, cfg, run, tag, tags_alt):
    raw, prep, lean = B.load_run(seg, cfg, run, tag)
    rows = B.compute_rows(raw, prep, lean)
    fr = col(rows, "frame").astype(int)
    vr = col(rows, "view_ratio")
    conf = col(rows, "b3d_conf")
    rear = (vr > 0.75) & (conf >= 0.3)
    sil = sil_frames(lean)
    in_sil = np.array([f in sil for f in fr])
    lmap = {int(x["frame"]): x for x in lean}
    body2d = np.array([B.fnum(lmap[f].get("body_lean")) for f in fr])
    fit = np.array([B.fnum(lmap[f].get("fit_lean_s")) for f in fr])
    lowc = np.array([str(lmap[f].get("fit_lowconf", "0")).strip() in ("1", "1.0", "True") for f in fr])
    sep3 = col(rows, "bike_lean_body") - col(rows, "torso_roll")
    st = dict(rows=rows, fr=fr, vr=vr, rear=rear, body2d=body2d, fit=fit, lowc=lowc,
              sil=med((fit - body2d)[in_sil]), s3_same=med(sep3[in_sil & (conf >= 0.3)]), s3_rear=med(sep3[rear]))
    # 설정 바꿈
    alt = {}
    for name, kw in (("pitch+10", dict(pitch=10)), ("pitch-10", dict(pitch=-10)),
                     ("yaw+10", dict(yaw_offset=10)), ("yaw-10", dict(yaw_offset=-10))):
        r2 = B.compute_rows(raw, prep, lean, **kw)
        alt[name] = med((col(r2, "bike_lean_body") - col(r2, "torso_roll"))[rear])[0]
    for t2 in tags_alt:
        p = B.out_dir(seg, cfg) / f"raw_{t2}_run{run}.npz"
        if p.exists():
            r2 = B.compute_rows(np.load(p), prep, lean)
            vr2, c2 = col(r2, "view_ratio"), col(r2, "b3d_conf")
            alt[t2] = med((col(r2, "bike_lean_body") - col(r2, "torso_roll"))[(vr2 > 0.75) & (c2 >= 0.3)])[0]
    st["alt"] = alt
    fitv = fit.copy()
    fitv[lowc] = np.nan
    st["jit"] = dict(bike3=jitter(col(rows, "bike_lean_body_raw")), bar3=jitter(col(rows, "bar_roll_raw")),
                     fit=jitter(fitv), roll3=jitter(col(rows, "torso_roll_raw")), body2d=jitter(body2d))
    st["jit_rear"] = dict(bike3=jitter(np.where(rear, col(rows, "bike_lean_body_raw"), np.nan)),
                          fit=jitter(np.where(vr > 0.75, fitv, np.nan)))
    dv = col(rows, "torso_roll") - body2d
    st["confound"] = dict(rear=med(dv[rear])[0], side=med(np.abs(dv)[(vr < 0.5) & (conf >= 0.3)])[0],
                          rear_abs=med(np.abs(dv)[rear])[0])
    return st


def make_board(seg, cfg, ref, tag, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.unicode_minus"] = False

    prompt, focal = tag.split("_f")
    tags_alt = [t for t in (B.tag_of("kp" if prompt == "box" else "box", float(focal)),
                            B.tag_of(prompt, 2203.0)) if t != tag]
    entries = []
    if ref:
        rseg = Path(ref)
        rcfg = PATHS.load_cfg(rseg)
        for r in rcfg["runs"]:
            entries.append(("코치님", rseg, rcfg, r["run"]))
    for r in cfg["runs"]:
        entries.append((f"Run {r['run']}", seg, cfg, r["run"]))
    stats = []
    for lab, s, c, rn in entries:
        try:
            stats.append((lab, rn, run_stats(s, c, rn, tag, tags_alt)))
        except FileNotFoundError as e:
            print("  건너뜀", lab, e)

    n = len(stats)
    fig = plt.figure(figsize=(17, 2.25 * n + 7.5), dpi=110)
    gs = fig.add_gridspec(n + 1, 2, width_ratios=[3.2, 1.3], height_ratios=[1] * n + [3.1], hspace=0.45, wspace=0.12)
    fig.suptitle(f"SAM 3D Body 3D 몸 각도 vs 2D·실루엣  (측정값, 설정 {tag}, 중력 = 화면 아래 가정)", fontsize=15, y=0.995)
    for i, (lab, rn, st) in enumerate(stats):
        rows = st["rows"]
        p = col(rows, "progress")
        ax = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        for a in (ax, ax2):
            rr = st["vr"] > 0.75
            for j in range(len(p)):
                if rr[j]:
                    lo = p[j - 1] if j else p[j]
                    hi = p[j + 1] if j + 1 < len(p) else p[j]
                    a.axvspan((lo + p[j]) / 2, (hi + p[j]) / 2, color="#e8e8e8", lw=0, zorder=0)
            a.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
            a.set_xlim(0, 1)
        ax.plot(p, col(rows, "torso_roll"), color=GREEN, lw=2, label="3D 상체 롤")
        ax.plot(p, col(rows, "torso_roll_raw"), ".", color=GREEN, ms=3, alpha=0.6)
        ax.plot(p, st["body2d"], ":", color=GREEN, lw=1.6, label="2D body_lean")
        ax.plot(p, col(rows, "bike_lean_body"), color=ORANGE, lw=2, label="3D 손-발 자전거")
        ax.plot(p, col(rows, "bike_lean_body_raw"), ".", color=ORANGE, ms=3, alpha=0.6)
        fit = st["fit"].copy()
        fl = fit.copy()
        fl[st["lowc"]] = np.nan
        ax.plot(p, fl, "--", color="#b35900", lw=1.3, label="실루엣 fit_lean_s")
        ax.plot(p[st["lowc"]], fit[st["lowc"]], "x", color="#b35900", ms=4)
        ax.plot(p, col(rows, "bar_roll"), color=MAG, lw=0.9, alpha=0.8, label="3D 손목-손목")
        ax.set_ylim(-35, 65)
        ax.set_ylabel(lab, fontsize=12, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=8)
        s3, sl = st["s3_rear"], st["sil"]
        ax.text(0.005, 0.96, f"뒷모습 분리각 3D {s3[0]:+.0f}° (n={s3[1]})  실루엣 {sl[0]:+.0f}° (n={sl[1]})",
                transform=ax.transAxes, fontsize=9, va="top", bbox=dict(fc="white", ec="none", alpha=0.8))
        ax2.plot(p, col(rows, "torso_pitch"), color=BLUE, lw=2, label="3D 상체 피치")
        ax2.plot(p, col(rows, "torso_pitch_raw"), ".", color=BLUE, ms=3, alpha=0.6)
        ax2.set_ylim(-10, 90)
        ax2.tick_params(labelsize=8)
        ax3 = ax2.twinx()
        ax3.plot(p, col(rows, "travel_yaw"), color=GREY, lw=1.2, ls="-.", label="진행 yaw")
        ax3.set_ylim(-180, 180)
        ax3.set_yticks([-90, 0, 90])
        ax3.tick_params(labelsize=7, colors=GREY)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=5, framealpha=0.9)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax3.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.9)
            ax.set_title("기울기 (+ = 왼쪽, 도). 회색 띠 = 뒷모습(view_ratio > 0.75). 점 = 평활 전", fontsize=10, loc="left")
            ax2.set_title("상체 피치(+ = 앞숙임)와 진행 yaw(0 = 뒷모습)", fontsize=10, loc="left")
        if i == n - 1:
            ax.set_xlabel("진행률", fontsize=9)
            ax2.set_xlabel("진행률", fontsize=9)

    # 표
    axt = fig.add_subplot(gs[n, :])
    axt.axis("off")
    labs = [lab for lab, _, _ in stats]
    fmt = lambda v: "" if not np.isfinite(v) else f"{v:+.0f}"
    fmt1 = lambda v: "" if not np.isfinite(v) else f"{v:.1f}"
    alt_names = ["pitch+10", "pitch-10", "yaw+10", "yaw-10"] + tags_alt
    tA = [["실루엣 − 2D (separation_board 프레임, 현재 CSV)"] + [fmt(st["sil"][0]) for _, _, st in stats],
          ["3D 손-발 − 3D 롤 (같은 프레임)"] + [fmt(st["s3_same"][0]) for _, _, st in stats],
          ["3D 손-발 − 3D 롤 (view_ratio>0.75 전체)"] + [fmt(st["s3_rear"][0]) for _, _, st in stats],
          ["  n (실루엣 / 3D 전체)"] + [f"{st['sil'][1]} / {st['s3_rear'][1]}" for _, _, st in stats]]
    for an in alt_names:
        tA.append([f"  3D, 설정 바꿈: {an}"] + [fmt(st["alt"].get(an, np.nan)) for _, _, st in stats])
    rng = []
    for _, _, st in stats:
        vals = [st["s3_rear"][0]] + [v for v in st["alt"].values()]
        vals = [v for v in vals if np.isfinite(v)]
        rng.append(f"{min(vals):+.0f}~{max(vals):+.0f}" if vals else "")
    tA.append(["  3D 범위 (기본 + 설정 바꿈)"] + rng)
    tB = [["흔들림: 3D 손-발 (평활 전)"] + [fmt1(st["jit"]["bike3"]) for _, _, st in stats],
          ["흔들림: 실루엣 fit_lean_s (LOW-CONF 제외)"] + [fmt1(st["jit"]["fit"]) for _, _, st in stats],
          ["흔들림: 3D 손-발, 뒷모습만"] + [fmt1(st["jit_rear"]["bike3"]) for _, _, st in stats],
          ["흔들림: 실루엣, 뒷모습만"] + [fmt1(st["jit_rear"]["fit"]) for _, _, st in stats],
          ["흔들림: 3D 롤 (평활 전) / 2D body_lean"] + [f"{fmt1(st['jit']['roll3'])} / {fmt1(st['jit']['body2d'])}"
                                                    for _, _, st in stats],
          ["시점 혼입: 3D 롤 − 2D, 뒷모습 중앙값"] + [fmt(st["confound"]["rear"]) for _, _, st in stats],
          ["시점 혼입: |3D 롤 − 2D|, 옆모습(vr<0.5) 중앙값"] + [fmt(st["confound"]["side"]) for _, _, st in stats]]
    table = tA + [[""] * (len(labs) + 1)] + tB
    tb = axt.table(cellText=table, colLabels=["뒷모습 분리각(자전거 − 상체, 도, 중앙값) / 흔들림(도/프레임)"] + labs,
                   loc="upper center", cellLoc="center", colLoc="center",
                   colWidths=[0.30] + [0.70 / len(labs)] * len(labs))
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.25)
    for (r, c), cell in tb.get_celld().items():
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if r == 0:
            cell.set_facecolor("#f0f0f0")
    out = out or (B.out_dir(seg, cfg) / "compare_board.png")
    fig.savefig(out, bbox_inches="tight")
    print("  board:", out)
    return out
