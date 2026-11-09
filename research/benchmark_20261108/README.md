# DynaIP 与视觉模型公平评测 · 2026-11-08

原实验、小模型和结果仍在 `research/continuous_20261108/` 与原本机输出中。本目录提供新的共同输入协议、官方来源审计、独立结果和复现入口。主程序的离线模拟行为不受影响。

## 环境

使用 Python 3.12；本次实际版本固定在 [requirements-lock.txt](requirements-lock.txt)。在仓库根目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r research\benchmark_20261108\requirements-lock.txt
```

锁文件包含 PyTorch CPU 运行环境。底层 BLAS、CPU 与网络变化会影响耗时；不同硬件的时间不能直接归因于模型。

## 1. DynaIP

```powershell
.\.venv\Scripts\python.exe research\benchmark_20261108\fetch_dynaip.py --download
# CIP_DATA_DIR 指向包含现有五份 CIP ZIP 的本机目录，数据不进入 Git。
.\.venv\Scripts\python.exe research\benchmark_20261108\pose_benchmark.py --output outputs\my_pose_run
```

也可用 `--data-dir 本机目录` 显式传参。新输出目录必须不存在；训练权重、源数据均忽略。下载约 46 MB，仅使用标准 `DynaIP.pth`，严格校验权重 SHA-256 和源码归档；保留 MIT 许可。无需下载 SMPL 模型。

官方固定提交 `ddfff5b88b77e124744bc74a792967cb8bad6735`。标准权重的公开训练配方包含 AnDy、UNIPD、CIP、Emokine、Virginia；`DynaIP_s.pth` 另加 DIP-IMU，本次不使用。依据是官方 README、`utils/config.py`、`train.py` 和 `datasets/split_info`，不等于独立追溯了权重的实际全部训练日志。完整来源、各划分原文与摘要见 [source_audit](source_audit/)。

CIP 官方训练为 00/01/02/03/05/06/07/09，测试为 04/08。对齐小模型训练00/01/02、验证03、测试04；程序在读取测试工作簿前冻结模型选择及权重摘要。04 同时不在双方训练集内，但此前旧实验已报告过04的其他指标，不能称为本项目从未看过的全新受试者。只读取固定规则选出的 sequence/circuit/task/random 各一段。另在冻结模型后获取08并进行新受试者确认，不重训、不调参：

```powershell
.\.venv\Scripts\python.exe research\benchmark_20261108\fetch_cip08.py --download
.\.venv\Scripts\python.exe research\benchmark_20261108\pose_confirmation.py --frozen-run outputs\my_pose_run --output outputs\my_pose_subject08
```

08下载约1.4GB；分段与完整文件会同时保留，全新下载约需2.8GB可用空间。数据为官方CC BY4.0，下载器核验固定大小和官方MD5，可用已完成的HTTP分段恢复。原始ZIP、分段和训练权重都不会进入Git。

共同输入为骨盆、左小腿、右小腿、头、左前臂、右前臂的旋转矩阵与世界系去重力加速度，每帧6×12。坐标 `[x,y,z]→[y,z,x]`，四元数 wxyz、矩阵列向量；非根旋转乘根旋转转置，行向量加速度右乘根旋转。根的全局旋转保留。原有手/脚位置、原始加速度和参考头高43维特征不参与新对比。

严格复用官方的前150帧参考姿态辅助传感器校准、首帧参考姿态与速度初始化；两个模型得到相同信息。速度初始化使用前两帧位置。前300帧不计分。参考辅助信息只来自初始化/校准区间，测试约束验证评分区间的参考姿态改变不会改变模型输入。同步使用同一工作簿 Frame 字段，不做目标相关的时间搜索。CIP 存在范数约1/3的非零四元数，按官方数学库归一化，同时记录数量；零四元数拒绝。

位置指标为官方平均 Xsens 骨架上的22个非根关节点 MPJPE（cm），对预测和参考都用同一骨架正向运动学，不把未知人体尺寸误差混入姿态误差，也不把忽略关节替换为真值。角度指标为11个需要预测的节段的 SO(3) 测地角误差（deg）。与原来实际节段位置/源单位指标不直接比较。速度为同一CPU4线程、600帧、一次预热后交替顺序5次的整段推理耗时，包含输出旋转重建，不含磁盘读取与共同校准；不是在线逐帧延迟。

## 2. 视觉模型

模型能力、地区、官方价格、排除理由及请求参数见 [vision_models.json](vision_models.json)。两家均使用官方接口。密钥只从 `ZHIPUAI_API_KEY`、`DASHSCOPE_API_KEY` 环境变量读取；Windows 支持读取刚保存的用户变量，避免把密钥写进命令历史。程序不自动读取 `.env`。

先生成选型图片，默认只预览、不调用API：

```powershell
.\.venv\Scripts\python.exe research\benchmark_20261108\vision_dataset.py selection --output outputs\my_vision_selection_images
.\.venv\Scripts\python.exe research\benchmark_20261108\vision_benchmark.py run --manifest outputs\my_vision_selection_images\manifest.json
```

真实调用必须显式 `--live`，会产生费用：

```powershell
.\.venv\Scripts\python.exe research\benchmark_20261108\vision_benchmark.py run --manifest outputs\my_vision_selection_images\manifest.json --output outputs\my_vision_selection --live
.\.venv\Scripts\python.exe research\benchmark_20261108\vision_benchmark.py freeze --results outputs\my_vision_selection\results.json --output outputs\my_vision_freeze.json
.\.venv\Scripts\python.exe research\benchmark_20261108\vision_dataset.py confirmation --selection-freeze outputs\my_vision_freeze.json --output outputs\my_vision_confirmation_images
.\.venv\Scripts\python.exe research\benchmark_20261108\vision_benchmark.py run --manifest outputs\my_vision_confirmation_images\manifest.json --selection-freeze outputs\my_vision_freeze.json --output outputs\my_vision_confirmation --free-interval 55 --free-attempts 4 --live
```

18张选型图和18张确认图各有升高/降低/无明显变化6张。先按预登记规则选择前两名，再允许生成确认图片；两个阶段的人物源不同，提示词与请求参数固定。来源为固定提交的公开 CMU BVH，使用本项目共享骨架绘图器，加髋部标记和5%阈值线；这是受控图示识别，不代表真实训练照片理解。正放/倒放和同一录制的不同窗口按同源簇解释，不能当作独立人物扩大样本量。

排序规则在脚本中固定：正确率降序、不确定率升序、无效率升序、费用升序、耗时升序。`no_clear_change` 是确定的第三类；`uncertain` 是模型主动无法判断。额外文字、截断或非JSON等计为无效，失败和缺失仍计入正确率分母。耗时包含网络、服务排队和模型思考时间。

本轮用户授权总费用上限10元。共享持久账本位于 `outputs/benchmark_20261108/vision_budget.json`，跨运行目录累计。每次发请求前按最大上下文和最贵阶梯预留，收到usage后用官方阶梯价结算估算值。超时或缺usage保留预留并停止该平台后续调用；付费请求无自动重试。免费HTTP429可显式指定 `--free-attempts 4`，最多4次、间隔至少55秒，失败尝试另存 `quota_attempts.jsonl`，不因回答错误或格式错误而重问。如果实际估算超过预留，账本记录实际估算并锁定，必须先核对账单；不要删除账本以绕过上限。零价模型也记录请求。估算不计赠送额度与缓存折扣，最终以平台账单为准。

已结束但有缺失的批次可以在**新输出目录**使用 `--resume-from 旧输出目录 --retry-rate-limits`：只重试零价429，复用其他相同图片/提示词/配置及代码版本的已完成回答，并保留来源摘要与失败记录。确认阶段建议55秒间隔；免费接口吞吐限制可能随平台变化。`elapsed_ms` 是单次HTTP耗时，新增调用另记录 `elapsed_with_scheduling_ms` 和尝试次数，批量总等待会更长。

## 离线审计

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe research\benchmark_20261108\replay.py --manifest outputs\my_vision_selection_images\manifest.json --responses outputs\my_vision_selection\responses.jsonl --results outputs\my_vision_selection\results.json --rerender
.\.venv\Scripts\python.exe tools\check_repository.py
```

结果归档后，可将上述路径换为本目录 `results/` 中的清单和响应，不需要API密钥重算分数。`--rerender` 需要 NumPy/Pillow 和公开BVH；缓存缺失时会下载固定公开源，但不会调用模型API。仓库保存去密钥的标签、简短解释、usage、图片/提示词/文件摘要和耗时，完整推理链与请求认证头不保存。图片和原始数据保留在本机忽略目录，可按源摘要复现。

数据署名：CMU Graphics Lab Motion Capture Database；BVH 转换来源 `una-dinosauria/cmu-mocap`（MIT转换代码），CIP 为 Palermo 等的 Complete Inertial Pose 数据集（CC BY4.0），DynaIP 为 Zhang 等 CVPR2024 工作（MIT代码）。各自数据与软件许可分别适用。

`archive_results.py` 专用于原始实验目录布局，要求原始两份模型文件和完整的原批次证据；`my_*` 复现实验应单独保存，不应覆盖冻结归档。
