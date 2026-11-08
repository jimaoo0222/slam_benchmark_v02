#!/usr/bin/env python3
"""new_benchmark 构建脚本：生成 IMU 停车窗口 → 内部一致性评测 → 出表。

用法（需要 ROS 2 环境，因为要读 bag）：
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 scripts/build.py            # 全量
    /usr/bin/python3 scripts/build.py --windows  # 只重建窗口
    /usr/bin/python3 scripts/build.py --eval     # 只重跑评测（用已有窗口）

依赖：rosbag2_py、sensor_msgs（/usr/bin/python3，不是 conda 的）
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent      # new_benchmark/
REPO = HERE.parent                                  # benchmark/
SLAM = REPO / 'slam_benchmark'                      # 原工作区（bag 和 site 真值在这）
MANIFEST = HERE / 'config/manifest.json'
WINDOWS = HERE / 'config/imu_windows.json'

# IMU 判静止参数（与算法无关）。多组候选，挑选与场地拓扑最吻合的一组。
THRS = [(0.05, 0.03, 2.0), (0.06, 0.04, 2.0), (0.08, 0.05, 1.5), (0.10, 0.06, 1.5),
        (0.12, 0.08, 1.5), (0.15, 0.10, 1.0), (0.20, 0.12, 1.0)]
def num(v, prec=1, null='—'):
    """定点输出；负号用 U+2212，且抹掉 -0.00 这类伪负零。"""
    if v is None:
        return null
    s = f'{v:.{prec}f}'
    if float(s) == 0:
        s = f'{0:.{prec}f}'
    return s.replace('-', '\u2212')


MERGE_GAPS = (0.5, 1.0, 2.0)
TRAJ_SPD_THR = 0.020     # 窗口内中位速度 < 2 cm/s（含静止期 SLAM 漂移；MID-360s 实测 <0.5）
MAX_P90_DEV = 0.30       # 90% 分位偏离 < 30 cm：只用来剔除启动收敛段，不用于判静止
GOOD_SEG_CM = 2.0        # 段长残差 < 2 cm 即认为窗口配对可信


# ── IMU 静止段 ────────────────────────────────────────────────
def imu_series(bag: Path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    r.set_filter(rosbag2_py.StorageFilter(topics=['/livox/imu']))
    ts, g, a = [], [], []
    while r.has_next():
        _, d, _ = r.read_next()
        m = deserialize_message(d, Imu)
        ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        g.append(float(np.linalg.norm([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])))
        a.append(float(np.linalg.norm([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])))
    return np.asarray(ts), np.asarray(g), np.asarray(a)


def segments(ts, g, a, gthr, atol, min_dur, merge_gap):
    k = max(1, int(0.1 / np.median(np.diff(ts))))
    smooth = lambda v: np.convolve(v, np.ones(k) / k, mode='same')
    gs, as_ = smooth(g), smooth(a)
    ref = 1.0 if abs(np.median(as_) - 1.0) < 0.2 else 9.80665     # 自动识别 g / m-s2
    ok = (gs < gthr) & (np.abs(as_ / ref - 1.0) < atol)
    raw, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if ts[j] - ts[i] >= min_dur:
                raw.append((float(ts[i]), float(ts[j])))
            i = j + 1
        else:
            i += 1
    merged = []
    for s in raw:
        if merged and s[0] - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def window_speed(t, p, w):
    """返回 (中位速度 m/s, 窗口均值位置, 90% 分位偏离 m)。"""
    m = (t >= w[0]) & (t <= w[1])
    if m.sum() < 5:
        return 9.9, None, 9.9
    pp = p[m]
    dt, dp = np.diff(t[m]), np.linalg.norm(np.diff(pp, axis=0), axis=1)
    spd = float(np.median(dp / np.maximum(dt, 1e-6))) if len(dp) else 0.0
    dev = np.linalg.norm(pp - pp.mean(0), axis=1)
    return spd, pp.mean(0), float(np.percentile(dev, 90))


def match_topology(P, site):
    """从候选点里挑 N 个，使相邻间距最贴合真值序列（只用长度，不用绝对坐标）。"""
    order = site['order']
    G = np.array(site['points'], float)
    truth = [float(np.linalg.norm(G[order[i + 1]] - G[order[i]])) for i in range(len(order) - 1)]
    best = None
    for combo in itertools.combinations(range(len(P)), len(order)):
        pts = P[list(combo)]
        d = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(order) - 1)]
        err = float(np.mean(np.abs(np.array(d) - np.array(truth))))
        if best is None or err < best[0]:
            best = (err, combo)
    return best


def build_windows():
    m = json.loads(MANIFEST.read_text())
    out = {}
    for bag, info in m['bags'].items():
        site = json.loads((REPO / info['site_path']).read_text())
        want = len(site['order'])
        ref_key = 'GLIM(map)' if 'GLIM(map)' in info['traj'] else next(iter(info['traj']))
        ref = HERE / info['traj'][ref_key]
        if not ref.is_file():
            print(f'{bag}: 缺参考轨迹 {ref}')
            continue
        a = np.loadtxt(ref, comments='#')
        tt, pp = a[:, 0], a[:, 1:4]

        ts, g, acc = imu_series(SLAM / f'rosbag/{bag}')
        best = None
        for gthr, atol, md in THRS:
            for gap in MERGE_GAPS:
                segs = segments(ts, g, acc, gthr, atol, md, gap)
                cand = []
                for s in segs:
                    spd, mp, p90 = window_speed(tt, pp, s)
                    if spd < TRAJ_SPD_THR and p90 < MAX_P90_DEV and mp is not None:
                        cand.append((s, mp, spd))
                if len(cand) < want:
                    continue
                P = np.array([c[1] for c in cand])
                keep = []
                for i, p in enumerate(P):
                    if any(np.linalg.norm(p - P[j]) < 0.3 and abs(cand[i][0][0] - cand[j][0][0]) < 15
                           for j in keep):
                        continue
                    keep.append(i)
                if len(keep) < want:
                    continue
                err, combo = match_topology(P[keep], site)
                maxspd = float(max(cand[keep[c]][2] for c in combo))
                cost = err + maxspd * 0.2
                if best is None or cost < best[0]:
                    best = (cost, err, [cand[keep[c]][0] for c in combo],
                            dict(gthr=gthr, atol=atol, min_dur=md, merge_gap=gap,
                                 n_cand=len(keep), max_spd_cms=maxspd * 100))
            # 已经足够好就不再放松阈值（保持 IMU 判据尽量严格）
            if best is not None and best[1] * 100 < GOOD_SEG_CM:
                break
        if best is None:
            print(f'{bag}: 无法匹配 {want} 个停车点，跳过')
            continue
        _, err, wins, meta = best
        out[bag] = dict(site=info['site_path'], n_windows=len(wins),
                        windows=[dict(t0=x, t1=y) for x, y in wins],
                        seg_err_cm=err * 100, params=meta)
        print(f'{bag}: {want} 窗  段长残差 {err*100:5.1f} cm  '
              f'候选 {meta["n_cand"]} 最大窗速 {meta["max_spd_cms"]:.2f} cm/s')
    WINDOWS.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\n→ {WINDOWS}')
    return out


# ── 评测 ────────────────────────────────────────────────────
def load_span(path):
    """只读首末时间戳，用来求各系统共同覆盖的时间区间。"""
    first = last = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            ts = float(line.split()[0])
            if first is None:
                first = ts
            last = ts
    return first, last


def run_eval():
    win = json.loads(WINDOWS.read_text())
    m = json.loads(MANIFEST.read_text())
    (HERE / 'eval').mkdir(exist_ok=True)
    rows = []
    for bag, info in m['bags'].items():
        if bag not in win:
            print(f'{bag}: 无窗口，跳过')
            continue
        site_path = REPO / info['site_path']
        args = [sys.executable, str(HERE / 'scripts/14b_internal_consistency.py'),
                '--windows', str(WINDOWS), '--site', str(site_path), '--bag-key', bag,
                '--json-out', str(HERE / f'eval/consistency_{bag}.json')]
        used, first_ts, last_ts = [], [], []
        for name, rel in info['traj'].items():
            p = HERE / rel
            if p.is_file():
                args += ['--traj', f'{name}:{p}']
                used.append(name)
                a = load_span(p)
                first_ts.append(a[0])
                last_ts.append(a[1])
        if not used:
            print(f'{bag}: 无轨迹'); continue
        # 闭合误差取各系统共同覆盖的时间区间：不同实现丢帧起点不同
        # （GLIM 的 map 轨迹要等初始化完成，起点比其它系统晚约 2 s），
        # 各用各的首末帧就等于比到不同位置。
        args += ['--span', f'{max(first_ts):.6f}:{min(last_ts):.6f}']
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'{bag}: 评测失败\n{r.stderr[-400:]}'); continue
        print(r.stdout)
        res = json.loads((HERE / f'eval/consistency_{bag}.json').read_text())['results']
        for name, v in res.items():
            if v.get('n_ok', 0) == 0:
                continue
            rows.append(dict(bag=bag, radar=info['radar'], algo=name,
                             std_xy_mm=v['std_xy_mm_median'], std_xy_max=v['std_xy_mm_max'],
                             yaw_deg=v['yaw_std_deg_median'],
                             revisit_mm=min([x['gap_mm'] for x in v['revisit']], default=None),
                             closure_mm=v['closure_1s_mm'],
                             seg_bias_pct=v.get('seg_scale_bias_pct'),
                             seg_absmax_pct=v.get('seg_scale_absmax_pct'),
                             poses=v['poses'], dur=v['dur']))
    (HERE / 'eval/per_run.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


def write_per_bag_tables(rows):
    """输出四张逐序列指标表。"""
    ALGOS = ['GLIM(odom)', 'GLIM(map)', 'FAST-LIO2', 'LIO-SAM', 'Point-LIO']
    idx = {(r['bag'], r['algo']): r for r in rows}
    bags = list(dict.fromkeys(r['bag'] for r in rows))
    DIVERGED = 500.0
    specs = [('std_xy_mm', 1, '静止散布（mm，停车窗口内水平位置 RMS 偏差）', '静止散布'),
             ('revisit_mm', 1, '重访一致性（mm，同一物理点两次访问的位置差）', '重访一致性'),
             ('closure_mm', 1, '闭合误差（mm，轨迹首末各 1 s 平均位置差）', '闭合误差'),
             ('seg_bias_pct', 2, '段长偏差（%，相邻停车点间距 / 图纸间距 − 1）', '段长偏差')]
    (HERE / 'tables').mkdir(exist_ok=True)
    for key, prec, title, fname in specs:
        lines = [f'# {title}', '',
                 '| 序列 | 传感器 | ' + ' | '.join(ALGOS) + ' |',
                 '|---|---|' + '---|' * len(ALGOS)]
        for b in bags:
            radar = idx.get((b, ALGOS[0])) or idx.get((b, 'GLIM(map)')) or {}
            radar = radar.get('radar', '—')
            cells = []
            for a in ALGOS:
                r = idx.get((b, a))
                if not r or r[key] is None:
                    cells.append('—')
                elif r['std_xy_mm'] >= DIVERGED:
                    cells.append('✗')
                else:
                    cells.append(num(r[key], prec))
            lines.append(f'| `{b}` | {radar} | ' + ' | '.join(cells) + ' |')
        (HERE / f'tables/逐序列_{fname}.md').write_text('\n'.join(lines) + '\n')
    print('已输出 4 张逐序列表 → tables/')


def summarize(rows):
    ALGOS = ['GLIM(odom)', 'GLIM(map)', 'FAST-LIO2', 'LIO-SAM', 'Point-LIO']
    DIVERGED = 500.0                      # std_xy > 0.5 m 视为发散
    summ = {}
    for a in ALGOS:
        rs = [r for r in rows if r['algo'] == a]
        good = [r for r in rs if r['std_xy_mm'] < DIVERGED]
        bad = [r['bag'] for r in rs if r['std_xy_mm'] >= DIVERGED]
        if not good:
            continue
        med = lambda k: float(np.median([g[k] for g in good if g[k] is not None]))
        summ[a] = dict(
            std_xy_mm=med('std_xy_mm'), yaw_deg=med('yaw_deg'),
            revisit_mm=(float(np.median([g['revisit_mm'] for g in good if g['revisit_mm'] is not None]))
                        if any(g['revisit_mm'] is not None for g in good) else None),
            closure_mm=med('closure_mm'),
            seg_bias_pct=med('seg_bias_pct'),
            seg_absmax_pct=float(max(g['seg_absmax_pct'] for g in good if g['seg_absmax_pct'] is not None)),
            bags=len(good), diverged=bad)
    (HERE / 'eval/summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))

    def fmt(v, p=1):
        return num(v, p)
    lines = ['| 系统 | 静止散布 | 航向 | 重访 | 闭合 | 段长偏 | 段最差 | 有效序列 |',
             '|---|---|---|---|---|---|---|---|']
    for a, v in summ.items():
        mark = '※' if v['diverged'] else ''
        lines.append(f"| **{a}**{mark} | {fmt(v['std_xy_mm'])} mm | {fmt(v['yaw_deg'],2)}° | "
                     f"{fmt(v['revisit_mm'])} mm | {fmt(v['closure_mm'])} mm | "
                     f"{fmt(v['seg_bias_pct'],2)}% | {fmt(v['seg_absmax_pct'],2)}% | {v['bags']} |")
    md = '\n'.join(lines)
    (HERE / 'tables/汇总.md').write_text('# 内部一致性汇总（中位数）\n\n' + md + '\n')
    print('\n' + md)
    for a, v in summ.items():
        if v['diverged']:
            print(f"※ {a} 在 {v['diverged']} 上发散，已剔除")
    write_per_bag_tables(rows)
    return summ


def build_site():
    """把评测产物渲染成静态网站（site/index.html）。"""
    script = HERE / 'scripts/make_site.py'
    if not script.is_file():
        return
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', action='store_true', help='只重建 IMU 停车窗口')
    ap.add_argument('--eval', action='store_true', help='只重跑评测（需已有窗口）')
    args = ap.parse_args()
    if args.windows:
        build_windows()
    elif args.eval:
        summarize(run_eval())
        build_site()
    else:
        build_windows()
        summarize(run_eval())
        build_site()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
