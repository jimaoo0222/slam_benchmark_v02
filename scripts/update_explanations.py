#!/usr/bin/env python3
"""Generate reader-facing comparison notes from the ground-truth-free JSON."""
import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from statistics import median
from urllib.parse import unquote

HERE = Path(__file__).resolve().parents[1]
ALG = ['GLIM(map)', 'GLIM(odom)', 'FAST-LIO2', 'LIO-SAM', 'Point-LIO']

def fmt(x, p=1):
    return '—' if x is None else f'{x:.{p}f}'

def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def med(values):
    v = [x for x in values if x is not None]
    return median(v) if v else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-html', action='store_true', help='只更新 Markdown 说明')
    args = parser.parse_args()
    src = HERE / 'eval/no_truth/metrics_no_truth.json'
    m = json.loads(src.read_text())
    runs = m['runs']
    docs = HERE / 'docs'
    docs.mkdir(exist_ok=True)
    intro = '[返回项目说明](../README.md) · [数据集来源](数据集说明.md) · [指标说明](指标说明.md) · [数据对比](数据对比.md) · [雷达对比](雷达对比.md)\n\n'
    rows = []
    for a in ALG:
        s = m['summary'][a]; p = s['profiles']['primary']
        rows.append([a, fmt(p['xy_mm']), fmt(p['xyz_mm']), fmt(p['yaw_deg'], 2),
                     fmt(s['closure_mm']), s['closure_n'], s['valid_runs']])
    content = '# 数据对比：在同一录包内比较算法\n\n' + intro
    content += '本页从 `eval/no_truth/metrics_no_truth.json` 生成，主口径为固定 IMU 阈值。单位、窗口与公式见指标说明。使用已有 TUM 轨迹重算，未重新回放算法；不读取图纸坐标或段长。\n\n'
    content += '## 1. 汇总与有效样本数\n\n'
    content += table(['输出', '水平散布 mm', '三维散布 mm', '航向 °', '闭合 mm', '闭合序列数', '有效轨迹数'], rows)
    content += '\n\n散布先取每条序列的窗口中位数，再取跨序列中位数；闭合只汇总通过检查的序列。GLIM(odom) 少两条输入，Point-LIO 少一条稳定输出，表格不能直接读成同等样本下的总排名。\n\n'
    for title, key in [('2. 逐序列水平散布（mm）', 'xy'), ('3. 逐序列闭合距离（mm）', 'closure')]:
        content += '## ' + title + '\n\n'
        rr = []
        for b, run in runs.items():
            row = [b, run['radar']]
            for a in ALG:
                v = run['algorithms'].get(a)
                row.append('缺轨迹' if v is None else '发散' if v['diverged'] else fmt(v['profile_medians']['primary']['xy_mm'] if key == 'xy' else v['closure_mm']))
            rr.append(row)
        content += table(['序列', '雷达'] + ALG, rr) + '\n\n'
    content += '闭合缺项不是零误差。`03_01` 公共首 1 s 的 IMU 低动态占比不足 80%，所以不进入闭合汇总；`03_02`、`04_02` 没有起点停车锚点。\n\n'
    content += '## 4. 固定阈值敏感性\n\n'
    content += table(['输出', '严格阈值水平散布 mm', '主阈值 mm', '宽松阈值 mm'], [[a] + [fmt(m['summary'][a]['profiles'][k]['xy_mm']) for k in ['strict', 'primary', 'loose']] for a in ALG])
    content += '\n\n阈值变化会改变窗口集合与窗口长度，细小名次变化不应解释成稳定的硬件或算法优势。\n\n'
    content += '## 5. 必须和中位数一起看的异常\n\n'
    for b, a in [('03_02', 'Point-LIO'), ('02_1', 'LIO-SAM')]:
        v = runs[b]['algorithms'][a]
        content += f"- **{b} / {a}**：最大相邻位姿距离 {v['max_step_m']:.3f} m，发生在相对首条 IMU 约 {v['max_step_time_s']:.2f} s；P99 位移速度 {v['p99_speed_m_s']:.2f} m/s。" + ('触发数值失稳条件，原始明细保留，稳定轨迹汇总中剔除。' if v['diverged'] else '未整体剔除；局部跳变仍可能显著影响首末距离。仅凭输出不能确定其内部原因。') + '\n'
    content += '- **04_01 / Point-LIO**：需单独检查中段高度变化。XY 分歧与 XYZ 分歧可能差异很大，平面一致不代表高度稳定。\n\n'
    content += '## 6. 怎样对齐别的报告\n\n同一指标需对齐单位、窗口长度、共同时间区间、坐标维度和输出阶段。当前闭合是首末各 1 s 均值，不是完整轨迹首末单帧；当前散布是 IMU 窗口汇总，不是固定首末各 10 s。当前 `pairs` 为三维刚体对齐，不能直接与二维 XY 表格混比。\n\n'
    content += '## 7. 从表格到原始记录\n\n`runs[序列].windows` 保存三个阈值版本的窗口；`algorithms[输出].window_stats` 保存逐窗口统计；`pairs` 保存轨迹间三维分歧。`source_files` 保存输入轨迹路径与 SHA-256。历史 `tables/` 与 `eval/summary.json` 属于图纸辅助评测，不是本页的数据源。\n'
    (docs / '数据对比.md').write_text(content)

    def group(a, radar, bags=None):
        return [r['algorithms'][a] for b, r in runs.items() if r['radar'] == radar and (bags is None or b in bags) and a in r['algorithms'] and not r['algorithms'][a]['diverged']]
    content = '# 两个雷达对比：MID-360s 与 C01\n\n' + intro
    content += '这里比较的是传感器、场景、采集过程与算法配置共同作用下的轨迹输出。两颗雷达独立录制，没有同步对照或独立参考轨迹，因此不能据此宣布哪颗雷达绝对精度更高。\n\n'
    content += '理解这个对比前，先看采集设计：MID-360s 于 2026-10-21 在小场地录制两条路线各两次；C01 于 2026-11-02 在走廊正反向和小场地两条路线各录一次。两者不是同一天、同一次运动的同步 A/B 测试。平台、场地照片和完整序列表见[数据集说明](数据集说明.md)。\n\n'
    content += '## 1. 数据构成与配置差异\n\n'
    content += table(['项目', 'MID-360s', 'C01'], [['序列', '01_1、01_2、02_1、02_2', '03_01、03_02、04_01、04_02'], ['场景', '四条均为小场地', '两条走廊、两条小场地'], ['起点停车 / 返回记录', '4 条', '2 条'], ['通过当前闭合检查', '4 条', '1 条（04_01）'], ['主清单 GLIM', 'CPU GICP', 'C01 适配版 CPU VGICP'], ['GLIM 零速更新', '归档配置关闭', '归档配置开启'], ['Point-LIO 稳定轨迹', '4 条', '3 条，03_02 发散']])
    content += '\n\nGICP/VGICP 与 ZUPT 的差异会影响输出波动；参数还需按各传感器外参和点云格式适配，不能认为两组只改变了雷达硬件。\n\n'
    content += '## 2. 按雷达分组：保留每项指标的样本数\n\n'
    rr=[]
    for a in ALG:
        for radar in ['MID-360s', 'C01']:
            vs=group(a,radar);cs=[v['closure_mm'] for v in vs if v['closure_mm'] is not None]
            rr.append([a,radar,len(vs),fmt(med([v['profile_medians']['primary']['xy_mm'] for v in vs])),fmt(med(cs)),len(cs)])
    content += table(['输出','雷达','有效轨迹数','水平散布 mm','闭合 mm','闭合样本数'],rr)
    content += '\n\nC01 的闭合列只有 04_01 一条，不能把单条结果与 MID-360s 四条的中位数当作等样本比较。C01 混入走廊，直接比较整组散布也包含场景差异。\n\n'
    content += '## 3. 同场地、同路线分组\n\n路线 A：MID-360s 的 01_1、01_2 对 C01 的 04_02；路线 B：MID-360s 的 02_1、02_2 对 C01 的 04_01。两组并非同步采集，路线 A 的 C01 还缺起点停车，所以此处只比较低动态水平散布，不比较整条路线闭合。\n\n'
    rr=[]
    for label, mb, cb in [('A',['01_1','01_2'],['04_02']),('B',['02_1','02_2'],['04_01'])]:
        for a in ALG:
            x=group(a,'MID-360s',mb);y=group(a,'C01',cb)
            rr.append([label,a,fmt(med([v['profile_medians']['primary']['xy_mm'] for v in x])),len(x),fmt(med([v['profile_medians']['primary']['xy_mm'] for v in y])),len(y)])
    content += table(['路线','输出','MID-360s 散布 mm','序列数','C01 散布 mm','序列数'],rr)
    content += '\n\n相同路线只是减少了一个混杂因素，并没有控制录制动作、停车时长和配置。每条路线 C01 只有一次采集，不能据此建立统计显著的雷达排名。\n\n'
    content += '## 4. 当前可以得到什么结论\n\n- 可以检查同一算法换雷达及场景后，低动态波动和失稳情况是否改变。\n- 可以区分小场地与走廊结果，定位更难的序列，并结合原始轨迹复核。\n- GLIM 的比较包含 GICP/VGICP、零速更新及预处理差异，无法分离单一因素的贡献。\n- 不能把波动、闭合或算法间一致性直接称作雷达绝对精度；缺少受控对照时，也不能把异常单独归因于雷达。\n\n若要进一步分离硬件影响，需要匹配场景、采集动作、输出阶段和采样口径，并做多次重复采集；有绝对定位精度需求时还需要独立参考测量。\n'
    (docs / '雷达对比.md').write_text(content)
    if not args.no_html:
        render_html()
    print('已更新指标入口、数据对比、雷达对比' + ('及两份静态说明站点。' if not args.no_html else ' Markdown。'))

def render_html():
    pandoc=shutil.which('pandoc')
    if not pandoc:
        raise SystemExit('静态页生成需要 pandoc；Markdown 已生成，可用 --no-html 跳过。')
    names=['数据集说明','指标说明','数据对比','雷达对比']
    files=[HERE/'README.md']+[HERE/'docs'/f'{name}.md' for name in names]
    nav='<a href="index.html">项目说明</a> <a href="数据集说明.html">数据集来源</a> <a href="指标说明.html">指标说明</a> <a href="数据对比.html">数据对比</a> <a href="雷达对比.html">两个雷达对比</a>'
    for path in files:
        body=subprocess.run([pandoc,'--from=gfm','--to=html5','--wrap=none',str(path)],check=True,capture_output=True,text=True).stdout
        # Rewrite navigation attributes only; source paths in prose/code remain exact.
        def rewrite_attribute(match):
            attr, url = match.groups()
            decoded = unquote(html.unescape(url))
            mapping = {'../README.md': 'index.html',
                       '../config/dataset_provenance.json': 'dataset_provenance.json',
                       'config/dataset_provenance.json': 'dataset_provenance.json',
                       'eval/no_truth/metrics_no_truth.json': 'metrics_no_truth.json'}
            for name in names:
                mapping[name+'.md'] = name+'.html'
                mapping['docs/'+name+'.md'] = name+'.html'
            target = mapping.get(decoded, decoded)
            if target.startswith('site/assets/'):
                target = target.removeprefix('site/')
            return attr+'="'+html.escape(target, quote=True)+'"'
        body=re.sub(r'(href|src)="([^"]+)"', rewrite_attribute, body)
        title=path.read_text().splitlines()[0].lstrip('# ')
        document='<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><link rel="stylesheet" href="assets/site.css"></head><body><header class="top"><div class="wrap"><nav>'+nav+'</nav></div></header><main class="wrap">'+body+'</main></body></html>\n'
        for dest in [HERE/'site',HERE/'docs']:
            dest.mkdir(exist_ok=True)
            (dest/('index.html' if path.name=='README.md' else path.stem+'.html')).write_text(document)
    for dest in [HERE/'site',HERE/'docs']:
        shutil.copyfile(HERE/'eval/no_truth/metrics_no_truth.json', dest/'metrics_no_truth.json')
        shutil.copyfile(HERE/'config/dataset_provenance.json', dest/'dataset_provenance.json')

if __name__=='__main__':
    main()
