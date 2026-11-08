# 研究实验

本目录保存可复现的研究脚本、固定数据划分和汇总结果。原始数据及权重另行获取，放在本地忽略目录。

| 实验 | 入口 | 结果与解释 |
|---|---|---|
| CIP 按受试者划分的预测 | [复现指南](continuous_20261108/README.md) | [结果 JSON](continuous_20261108/cip_results.json)、[选模记录](continuous_20261108/cip_selection_frozen.json) |
| CMU 图片升降判断与标记读取 | [复现指南](continuous_20261108/README.md) | [平衡测试结果](continuous_20261108/balanced_live.jsonl)、两个 `marker_*holdout_results.json` |
| FusionPoser 固定录制划分 | [训练脚本](../examples/train_pose_research.py) | [结果 JSON](continuous_20261108/fusionposer_results.json)、[实验报告](../docs/三项优化_结果报告.md) |
| Agent 候选依据检查 | [模拟评估脚本](../examples/evaluate_agent_research.py) | [真实小样本记录](continuous_20261108/agent_live_refined.json)、[补充实验报告](../docs/三项优化_跨人和视觉补充实验_20261108.md) |

`continuous_20261108` 是实验日期标识。历史结果保留原值，整理工程时不重新命名样本或修改已有评分；脚本的可移植性调整记录在 [改进记录](../CHANGELOG.md)。
