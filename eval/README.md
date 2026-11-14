# 评测结果的两种口径

- `no_truth/metrics_no_truth.json`：当前主评测，由 `scripts/evaluate_no_truth.py` 生成，只使用固定 IMU 阈值与已有轨迹，不读取图纸坐标或段长。
- 本目录根部的 `summary.json`、`per_run.json`、`consistency_*.json`：历史图纸辅助窗口评测，由 `scripts/build.py` 生成，供历史复核。

两种窗口产生方式不同，数值不能直接拼接。当前指标与缺项含义见 [指标说明](../docs/指标说明.md)，有效样本数与逐序列结果见 [数据对比](../docs/数据对比.md)。
