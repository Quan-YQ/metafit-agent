# 公开 IMU → 骨架图 → 视觉模型 → Agent · 扩展实测（2026-11-10）

本目录用 5 段未参与训练的公开 FusionPoser 录制、每段 6 个自然窗口，以及每窗口的 forward/reverse/static 三个版本，共 **90 张**预测骨架图，比较 `qwen3.8-flash` 与 `qwen3.8-max-2026-09-02` 的读图结果，并把一张图的观察接入文本选择与 R3 检查。

- [完整实验报告](../../docs/公开IMU链路扩展实测_20261110.md)
- [双击可看的演示页](results/run_A/index.html)
- [完整链路记录](results/run_A/chain_trace.json)
- [预登记方案](results/run_A/freeze.json) · [数据清单](results/run_A/manifest.json) · [费用账本](results/run_A/budget.json)

## 数据与协议

- 训练仍使用已有 5 段录制的固定岭回归基线（alpha=200），不重新训练、不调参。
- 测试为 `0625173852`、`0625174451`、`0625181450`、`0625181811`、`0625182119` 五段录制，每段 6 个自然时间窗口。
- 每窗口生成 forward、reverse、static 三个版本；static 为重复首帧对照，reverse 与 forward 共用源窗口。同源窗口不是独立人物。
- 这些录制此前已用于公开数据的开发或留出评估，本次是扩展样本与链路复测，不是全新盲测。
- 参考方向只比较预测骨架首末帧的髋部高度，阈值 ±5%；不是动作质量或安全标签。
- 请求前冻结 90 张图、提示词、模型配置、代码摘要和预算。三张 pilot 用于检查传输/格式/链路，**不淘汰模型**；全部 90 张都计入每个模型的评分。

<!-- RESULTS_START -->
## 结果

| 项目 | qwen3.8-max | qwen3.8-flash |
|---|---:|---:|
| 全量 90 张正确率 | **75/90（83.3%）** | **62/90（68.9%）** |
| 三类平均 | **73.5%** | **48.1%** |
| 上升 / 下降 / 无明显变化 | 9/14 · 9/14 · 57/62 | 3/14 · 5/14 · 54/62 |
| 中位耗时 | 2.99 秒 | 1.68 秒 |
| usage 估算 | ¥0.937008 | ¥0.06156 |

- 90 张图来自 30 个源窗口；同源 forward/reverse/static 不是独立人物，类别中无明显变化占 62 张，因此主要看三类平均。
- 30 窗口聚类 bootstrap 的三类平均 95% CI：Max 56.6%–89.2%，Flash 36.5%–59.8%。
- 182 次真实调用，usage 估算 ¥1.0097918 / ¥10；另有首轮 pilot 6 次 ¥0.0331506。
- 完整链路：`imu_x2_4_forward`（参考 down）→ Max 读为 down → Flash 选择 silent → R3 reviewed；候选引用事实 `fact_1`、`fact_2`，视觉观察进入了依据。链路图不参与准确率统计。

首轮冻结的 pilot 在 3 张边缘样本上两款均只有 1/3；为避免用 3 张图否定 90 张评估，扩展协议把 pilot 改为诊断、全部 90 张评分，图片、提示词和模型未改。详见[完整报告](../../docs/公开IMU链路扩展实测_20261110.md)。

<!-- RESULTS_END -->

## 无 Key 重放

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\replay.py --run research\live_chain_20261110_expanded\results\run_A
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\present.py --run research\live_chain_20261110_expanded\results\run_A
Start-Process research\live_chain_20261110_expanded\results\run_A\index.html
```

默认核对归档图片、响应、分数、绑定与 R3，不联网调用 API。数据目录存在时可加 `--data-dir data\fusionposer`，重新生成图片摘要：

```powershell
.\.venv\Scripts\python.exe examples\fusionposer_fetch.py --download
.\.venv\Scripts\python.exe examples\fusionposer_extra_fetch.py --download
```

## 在新目录重跑

`models.json` 保存本次核对的官方模型 ID、端点、单价和来源；密钥只从 Windows 用户环境变量 `DASHSCOPE_API_KEY` 或当前进程环境读取。

```powershell
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\dataset.py --output outputs\my_chain_expanded_images
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\run.py freeze --manifest outputs\my_chain_expanded_images\manifest.json --run outputs\my_chain_expanded --models qwen3.8-flash qwen3.8-max-2026-09-02
# 下面开始产生真实请求；每个新实验目录独立计费，预算上限 10 元。
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\run.py pilot --run outputs\my_chain_expanded
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\run.py confirmation --run outputs\my_chain_expanded --approve-expansion
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\run.py chain --run outputs\my_chain_expanded --model qwen3.8-max-2026-09-02 --case CASE_ID
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\replay.py --run outputs\my_chain_expanded --data-dir data\fusionposer
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110_expanded\present.py --run outputs\my_chain_expanded
```

`chain` 的案例按预登记规则选第一张“观察与几何参考一致”的确认图，只用于验证流程，不参与准确率统计。失败请求保留记录，付费请求不自动重试；中断批次用 `*_partial.json` 和 `calls/` 审计，不删除账本重跑。

## 归档

90 张派生 PNG、请求内容摘要、去密钥原回答、usage 和时间在 `results/run_A/`，不含原始 IMU、参考关节数组或权重。原始数据、人体模型、权重和本机输出不进入 Git。

本实验不重训骨架模型，也不把预测骨架当作真实动作真值；实验室专有动捕数据的接入见[后续工作](../../docs/ROADMAP.md)。
