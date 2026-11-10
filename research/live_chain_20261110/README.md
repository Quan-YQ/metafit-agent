# 公开 IMU → 骨架图 → 真实视觉模型 → Agent · 2026-11-10

本目录把此前分开的骨架预测、看图和决策实验串成一次真实模型运行。输入来自 FusionPoser 公开六 IMU 数据与头部高度；本次演示使用已有五录制岭回归基线，保留 DynaIP 独立对比结果。

- [完整实验报告](../../docs/公开IMU完整链路实测_20261110.md)
- [双击可看的演示页](results/run_A/index.html)
- [完整链路记录](results/run_A/chain_trace.json)
- [预登记方案](results/run_A/freeze.json)、[零费用预跑检查](results/run_A/dry_run.json)、[费用账本](results/run_A/budget.json)

## 已完成的真实运行

三图预跑后，千问 Flash 和 Max 都通过 2/3 门槛；豆包首个请求返回404，停止该平台。用户选择完成两款千问对比。剩余15图确认：Max **14/15**，Flash **11/15**；三类平均正确率分别 **66.7% / 59.0%**。类别不均衡，需同时读两种分数。

完整链路使用 Max 读第一张按时间顺序的确认图片 `imu_1_2_forward`，Flash 做文本选择。两次真实请求、绑定匹配、R3 reviewed，输出 silent。原回答及引用全部保留；决策最终只引用自报事实，没有引用视觉事实，因此本次证明了流程连接，未证明视觉改变了决策。

已确认usage估算 **¥0.2082174**；404未确认费用的预留 **¥0.638976**；合计 **¥0.8471934 / ¥10**。没有自动重试。

## 无需 Key 查看和重放

在仓库根目录运行，Python3.12，使用前次评测的版本锁：

```powershell
.\.venv\Scripts\python.exe -m pip install -r research\benchmark_20261108\requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\replay.py --run research\live_chain_20261110\results\run_A
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\present.py --run research\live_chain_20261110\results\run_A
Start-Process research\live_chain_20261110\results\run_A\index.html
```

默认重放核对归档图像、响应、分数、绑定与 R3，不联网调用 API。已经准备好公开 FusionPoser 数据时，再加 `--data-dir data\fusionposer`，可重新拟合固定基线并复现18张图片摘要。此前已下载的数据可直接使用；新环境先运行：

```powershell
.\.venv\Scripts\python.exe examples\fusionposer_fetch.py --download
.\.venv\Scripts\python.exe examples\fusionposer_extra_fetch.py --download
```

## 在新目录重新实测

`models.json` 保存2026-11-10核对的官方模型ID、地区、端点、单价和来源。百炼使用 Windows 用户环境变量 `DASHSCOPE_API_KEY`，豆包使用 `ARK_API_KEY`。切勿将密钥写进代码、命令历史或Git；同一用户的脚本会读取用户环境变量，无需打印它们。豆包404通常需要先检查账号模型开通及调用权限。

```powershell
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\dataset.py --output outputs\my_chain_images
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\run.py freeze --manifest outputs\my_chain_images\manifest.json --run outputs\my_chain_run --models qwen3.8-flash qwen3.8-max-2026-09-02
# 下面才会发出真实模型请求；本机示例预算上限为10元，每次新实验需另行批准费用。
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\run.py pilot --run outputs\my_chain_run
# 阅读 pilot.json 和 budget.json，获得扩展批准后再执行：
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\run.py confirmation --run outputs\my_chain_run --approve-expansion
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\run.py chain --run outputs\my_chain_run --model qwen3.8-max-2026-09-02 --case imu_1_2_forward
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\replay.py --run outputs\my_chain_run --data-dir data\fusionposer
.\.venv\Scripts\python.exe -X utf8 research\live_chain_20261110\present.py --run outputs\my_chain_run
```

确认后按 `freeze.json` 的排序规则选模型；命令中的 Max 是本次结果示例。冻结后不改变提示词、模型、图片或评分。失败记录保留，付费请求不自动重试；中断批次使用 `*_partial.json` 和 `calls/` 审计，不删除账本重跑。每个新输出目录是独立实验；不要并行新建多个目录绕过同一笔授权预算。图片固定1200×460，预留65,536输入及8,192输出tokens；这是保守工程估算，usage或价格异常会停止。实际费用以平台账单为准。

## 归档与修订

最终工程完整离线测试753通过、0失败、0跳过。完整记录见 `results/run_A/verification.json`；上游DynaIP的torch.cross弃用提示单独记录。

18张派生PNG、请求内容的SHA-256、去密钥原回答、usage和时间均在 `results/run_A/`，不含原始 IMU、参考关节数组或权重。静止与倒序对照来自相同源，人物身份未知。

首次离线重放发现观察器名称变化会改变观察ID。仅修正重放时保留原观察身份，并添加离线修订核对；原冻结文件、分数和提供方响应没有改写，也没有重发API。两份原脚本在 `frozen_source/`，对应前后摘要和原因在 `postrun_amendments.json`。该归档只允许重放，禁止按修订后的代码继续付费调用。

本演示不重训DynaIP、不重新计算姿态预测误差。FusionPoser数据和许可来源见 [来源说明](../../data/fusionposer/来源说明.md)；本实验图像与原始受限 AMASS 数据分开。
