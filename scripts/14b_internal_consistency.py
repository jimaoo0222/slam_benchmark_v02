#!/usr/bin/env python3
"""内部一致性评测（不依赖真值坐标）。

四个指标：

  ① 静止散布     同一停车窗口内位姿的 std（位置 mm / 航向 °）
  ② 重访一致性   同一物理点被访问两次（由路线拓扑确定）时的位置差
  ③ 闭合误差     轨迹首尾位姿差（位姿层面，不只看停车点）
  ④ 段长自洽     相邻停车点间距 / 真值间距（只用长度，不用绝对坐标）

用法：
    python3 scripts/14b_internal_consistency.py \
        --windows results/xxx/imu_windows.json \
        --site config/site_c01_rect.json --bag-key 03_01 \
        --traj "GLIM:path/to/traj.txt" --traj "FAST-LIO2:path/to/traj.txt" \
        --json-out out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def load_traj(path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append([float(v) for v in parts[:8]])
    if not rows:
        raise SystemExit(f"❌ {path} 里没读到轨迹（需要 TUM 8 列格式）")
    return np.asarray(rows)


def yaw_of(q):
    x, y, z, w = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def window_stats(a: np.ndarray, windows: list[dict]) -> list[dict]:
    """每个停车窗口内的散布。"""
    t, p, q = a[:, 0], a[:, 1:4], a[:, 4:8]
    out = []
    for w in windows:
        t0, t1 = float(w["t0"]), float(w["t1"])
        m = (t >= t0) & (t <= t1)
        n = int(m.sum())
        if n < 5:
            out.append(dict(t0=t0, dur=t1 - t0, n=n, ok=False))
            continue
        pp = p[m]
        mean = pp.mean(0)
        dev = pp - mean
        # 水平 / 三维
        std_xy = float(np.sqrt((dev[:, :2] ** 2).sum(axis=1).mean())) * 1000.0
        std_3d = float(np.sqrt((dev ** 2).sum(axis=1).mean())) * 1000.0
        # 航向散布（四元数有正负二义，先统一下符号）
        qq = q[m].copy()
        if np.dot(qq[0], qq.mean(0)) < 0:
            qq = -qq
        yaws = np.array([yaw_of(v) for v in qq])
        y0 = yaws[0]
        yaws = np.array([y0 + wrap_deg(v - y0) for v in yaws])
        yaw_std = float(yaws.std())
        # 窗口内平均速度（静止时应该接近 0）
        if n > 1:
            dt = np.diff(t[m])
            dp = np.linalg.norm(np.diff(pp, axis=0), axis=1)
            spd = float(np.median(dp / np.maximum(dt, 1e-6))) if len(dp) else 0.0
        else:
            spd = 0.0
        out.append(dict(t0=t0, dur=t1 - t0, n=n, ok=True, mean=mean.tolist(),
                        std_xy_mm=std_xy, std_3d_mm=std_3d, yaw_std_deg=yaw_std,
                        speed_mm_s=spd * 1000.0))
    return out


def revisit_pairs(site: dict) -> list[tuple[int, int, int]]:
    """路线拓扑里同一场地点的两个窗口索引：[(i, j, place), ...]。"""
    order = site["order"]
    seen: dict[int, int] = {}
    pairs = []
    for k, place in enumerate(order):
        if place in seen:
            pairs.append((seen[place], k, place))
        else:
            seen[place] = k
    return pairs


def summarize(a: np.ndarray, windows: list[dict], site: dict, span=None) -> dict:
    st = window_stats(a, windows)
    ok = [s for s in st if s["ok"]]
    res: dict = dict(
        poses=int(len(a)),
        dur=float(a[-1, 0] - a[0, 0]),
        n_windows=len(st),
        n_ok=len(ok),
        window_stats=st,
    )
    if ok:
        for key in ("std_xy_mm", "std_3d_mm", "yaw_std_deg", "speed_mm_s"):
            vals = np.array([s[key] for s in ok])
            res[f"{key}_median"] = float(np.median(vals))
            res[f"{key}_max"] = float(vals.max())

    # ② 重访一致性
    rv = []
    for i, j, place in revisit_pairs(site):
        if not (st[i]["ok"] and st[j]["ok"]):
            continue
        pi = np.array(st[i]["mean"])
        pj = np.array(st[j]["mean"])
        rv.append(dict(place=place, w0=i, w1=j,
                       gap_m=float(np.linalg.norm(pi - pj)),
                       gap_mm=float(np.linalg.norm(pi - pj)) * 1000.0,
                       t0=st[i]["t0"], t1=st[j]["t0"]))
    res["revisit"] = rv

    # ③ 闭合误差：首尾各 1 s 平均位置之差
    #    两个前提，缺一即无定义：
    #      a. 起点必须落在某个停车窗口内 —— 否则「起点」只是行进中的某一帧，
    #         把它当原点会让闭合误差变成「这段时间走了多远」；
    #      b. 各系统取同一时间区间（span）—— 不同实现丢帧的起点不同
    #         （GLIM 的 map 轨迹要等初始化完成，起点比其它系统晚 2 s），
    #         不统一就只能比到各自的文件端点。
    t = a[:, 0]
    t0 = t[0] if span is None else max(t[0], span[0])
    t1 = t[-1] if span is None else min(t[-1], span[1])
    res["closure_span"] = [float(t0), float(t1)]
    anchored = any(w["t0"] - 0.5 <= t0 <= w["t1"] + 0.5 for w in windows)
    res["closure_anchored"] = bool(anchored)
    if anchored and t1 - t0 >= 2.0:
        first = a[(t >= t0) & (t <= t0 + 1.0), 1:4].mean(0)
        last = a[(t >= t1 - 1.0) & (t <= t1), 1:4].mean(0)
        res["closure_pose_mm"] = float(np.linalg.norm(a[-1, 1:4] - a[0, 1:4])) * 1000.0
        res["closure_1s_mm"] = float(np.linalg.norm(last - first)) * 1000.0
    else:
        res["closure_pose_mm"] = None
        res["closure_1s_mm"] = None

    # ④ 段长自洽（只用真值长度）
    if all("mean" in s for s in st):
        means = [np.array(s["mean"]) for s in st]
        pts = np.array(site["points"], dtype=float)
        order = site["order"]
        segs = []
        for k in range(len(order) - 1):
            alg = float(np.linalg.norm(means[k + 1] - means[k]))
            gt = float(np.linalg.norm(pts[order[k + 1]] - pts[order[k]]))
            if gt > 1e-6:
                segs.append(dict(seg=f"{order[k]}→{order[k+1]}", alg_m=alg, gt_m=gt,
                                 err_pct=(alg / gt - 1) * 100))
        if segs:
            e = np.array([s["err_pct"] for s in segs])
            res["segments"] = segs
            res["seg_scale_bias_pct"] = float(e.mean())
            res["seg_scale_absmax_pct"] = float(np.abs(e).max())
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="SLAM 轨迹内部一致性评测（不依赖真值坐标）")
    ap.add_argument("--windows", required=True, help="IMU 停车窗口 JSON")
    ap.add_argument("--site", required=True, help="场地 JSON（只用来定路线拓扑与段长）")
    ap.add_argument("--bag-key", required=True, help="windows JSON 里的键")
    ap.add_argument("--traj", action="append", required=True, help='"名字:轨迹文件"')
    ap.add_argument("--span", default="",
                    help='闭合误差用的统一时间区间 "t0:t1"；不给则各用自身首末帧')
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    windows = json.loads(Path(args.windows).read_text())[args.bag_key]["windows"]
    windows = [dict(t0=float(w["t0"]), t1=float(w["t1"])) for w in windows]
    site = json.loads(Path(args.site).read_text())

    span = None
    if args.span:
        lo, _, hi = args.span.partition(":")
        span = (float(lo), float(hi))

    results = {}
    for spec in args.traj:
        name, _, path = spec.partition(":")
        a = load_traj(path)
        r = summarize(a, windows, site, span)
        r["path"] = path
        results[name] = r

    print("=" * 86)
    print(f" {args.bag_key} · 内部一致性（不依赖真值坐标）   窗口 {len(windows)} 个")
    print("=" * 86)
    hdr = (f"{'算法':<11}{'静止散布':>10}{'(max)':>8}{'航向':>8}"
           f"{'重访':>8}{'闭合(1s)':>10}{'段长偏':>9}{'段|max|':>9}")
    print(hdr)
    print("-" * 86)
    for name, r in results.items():
        if r["n_ok"] == 0:
            print(f"{name:<11}{'停车窗口内帧数不足':>20}")
            continue
        rv = min([x["gap_mm"] for x in r["revisit"]], default=None)
        rv_s = f"{rv:8.1f}" if rv is not None else f"{'—':>8}"
        seg = r.get("seg_scale_bias_pct")
        seg_s = f"{seg:+9.2f}" if seg is not None else f"{'—':>9}"
        mx = r.get("seg_scale_absmax_pct")
        mx_s = f"{mx:9.2f}" if mx is not None else f"{'—':>9}"
        cl = r["closure_1s_mm"]
        cl_s = "        —  " if cl is None else f"{cl:8.1f}mm"
        print(f"{name:<11}{r['std_xy_mm_median']:9.1f}mm{r['std_xy_mm_max']:7.1f}"
              f"{r['yaw_std_deg_median']:7.2f}°{rv_s}{cl_s}{seg_s}{mx_s}")
    print("-" * 86)
    print("静止散布 = 窗口内水平位置 std 中位数；航向 = 同窗口航向 std 中位数")
    print("重访     = 同一物理点两次访问的位置差（路线闭环才有）")
    print("闭合(1s) = 统一时间区间内首尾各 1 s 平均位置之差（起点须在停车窗口内）")
    print("段长偏   = 相邻停车点间距 / 真值间距 的平均偏差（只用长度，不用绝对坐标）")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            dict(bag=args.bag_key, windows=len(windows), results=results),
            ensure_ascii=False, indent=1))
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
