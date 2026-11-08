# 2026-11-08 固定实验与复现

在仓库根目录执行下面命令。实验结果保留在本目录；重新运行时将输出写入 `outputs/` 的新目录。查看报告见 [补充实验结果](../../docs/三项优化_跨人和视觉补充实验_20261108.md)。

## 安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research]"
.\.venv\Scripts\python.exe -m pytest -q research\continuous_20261108
```

运行 CIP 神经网络训练还需安装 `.[train]`。基础包、研究计算和训练依赖分开，便于只运行需要的部分。

## CIP 训练

先按 [CIP 数据说明](../../data/cip/README.md) 获取并重命名前 5 位受试者 ZIP，再运行：

```powershell
$env:CIP_DATA_DIR = 'C:\Datasets\CIP'
.\.venv\Scripts\python.exe research\continuous_20261108\cip_cross_subject.py --output-dir outputs\cip_new_run
```

也可以把文件放在仓库根目录的 `data/cip/`，不设置该环境变量。输出目录必须为空。脚本先核对官方摘要，按 00/01/02 训练、03 验证，再固定模型选择，最后读取 04 测试；每个人选取四类动作的首个未标异常录制，移除最初 300 个锚点样本，不插补缺失包。

输入含 6 个传感器的四元数/加速度和参考头高，时间匹配借助参考姿态；目标为相对骨盆的 22 个 Xsens 节段原点。模型、选模记录和结果会写入输出目录。已有 [结果](cip_results.json) 与 [选模记录](cip_selection_frozen.json) 保留原始统计。

## CMU 视觉与标记实验

以下下载只获取公开 CMU 转换数据并核对固定摘要，没有模型请求；已有目标文件时停止，避免覆盖：

```powershell
.\.venv\Scripts\python.exe research\continuous_20261108\balanced_fetch.py --download
.\.venv\Scripts\python.exe research\continuous_20261108\marker_holdout_fetch.py --download
```

然后复现固定颜色标记工具的两组离线检查：

```powershell
.\.venv\Scripts\python.exe research\continuous_20261108\marker_holdout_eval.py --group initial --output outputs\marker_initial.json
.\.venv\Scripts\python.exe research\continuous_20261108\marker_holdout_eval.py --group moving --output outputs\marker_moving.json
```

这两组共 160 个窗口：154 个与参考一致，6 个接近边界而返回不确定。读入的图像由同一渲染器生成，带指定颜色标记和参考线；同一动作中的窗口及倒序相关。

准备 9 图平衡测试的新清单，不调用 API：

```powershell
.\.venv\Scripts\python.exe research\continuous_20261108\vision_balanced_eval.py prepare --manifest outputs\balanced_preview.json
```

真实视觉调用是另一项显式操作。先通过环境变量提供自己的 `DEEPSEEK_API_KEY`，再使用以下命令；一次最多 9 次图片请求，发生调用或格式错误即停止，不自动重试：

```powershell
.\.venv\Scripts\python.exe research\continuous_20261108\vision_balanced_eval.py live --manifest research\continuous_20261108\balanced_manifest.json --output outputs\balanced_live_new.jsonl
```

它会逐图核对清单中的参考、帧范围与图片 SHA-256，历史测试为 4/9 一致。实际费用以供应商账单为准。`vision_guided_eval.py` 保留开发/历史留出组入口；可用 `--data-dir` 指定含对应 BVH 的目录，这些组已查看过，不能重新称为盲测。

## Agent 和 FusionPoser

```powershell
.\.venv\Scripts\python.exe examples\evaluate_agent_research.py --output outputs\agent_offline_new.json
```

默认是离线替身。`--live` 会显式请求真实文本模型，需先设置密钥。模拟候选仍经过引用和免打扰检查。

FusionPoser 的历史划分和取数方式见 [实验报告](../../docs/三项优化_结果报告.md)、[误差诊断](../../docs/FusionPoser误差诊断报告.md) 与 [数据来源](../../data/fusionposer/来源说明.md)。安装 `.[train]` 后，可运行：

```powershell
.\.venv\Scripts\python.exe examples\fusionposer_fetch.py --download
.\.venv\Scripts\python.exe examples\fusionposer_extra_fetch.py --download
.\.venv\Scripts\python.exe examples\train_pose_research.py --output-dir outputs\fusionposer_new_run
```

前两条联网恢复本轮所需的 12 段录制，核验固定摘要；第三条训练和评估。原始统计另存为 [fusionposer_results.json](fusionposer_results.json)。Agent 后续真实小样本结果见 [agent_live_refined.json](agent_live_refined.json)。

## 工程整理与历史摘要

工程整理移除了 5 份与根目录 `examples/` 逐字相同的脚本副本，现在复用共享实现；绘图与分类规则保持一致，使用已保存的图片摘要核对。原实验 guided 代码摘要仍保留于 `balanced_manifest.json`；可移植版本的摘要由 `vision_balanced_eval.py` 记录，Python 文件使用 LF 换行。

原始数据、权重和新输出目录不进入 Git；本目录的 JSON/JSONL 只包含公开样本标识、汇总统计、摘要和模拟结果。
