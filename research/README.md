# 研究实验

本目录保存可复现的研究脚本、固定数据划分和汇总结果。原始数据及权重另行获取，放在本地忽略目录。

| 实验 | 入口 | 结果与解释 |
|---|---|---|
| 公开IMU到真实视觉与Agent的扩展链路（5段×6窗口×3版本，90张图） | [复现指南](live_chain_20261110_expanded/README.md) | [完整报告](../docs/公开IMU链路扩展实测_20261110.md)、[演示页](live_chain_20261110_expanded/results/run_A/index.html) |
| 六家视觉模型骨架读图横评 | [复现指南](six_model_vision_20261110/README.md) | [横评报告](six_model_vision_20261110/横评报告.md)、[演示页](six_model_vision_20261110/index.html) |
| 公开数据历史归档（CIP / FusionPoser / 标记工具） | [复现指南](continuous_20261108/README.md) | 结果 JSON 在该目录；保留历史值，不作为当前主线 |

`live_chain_20261110_expanded` 是当前端到端链路实验；`six_model_vision_20261110` 是当前视觉模型横评；`continuous_20261108` 仅保留 2026-11-08 的公开数据研究结果，脚本和统计原样保留。
