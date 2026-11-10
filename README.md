# MetaFit Agent

MetaFit Agent 是一个非商业研究原型，探索“训练事件 → 动作骨架与视觉观察 → Agent 候选反馈”的链路。仓库提供可运行的本机模拟接口、公开数据实验、自动化测试与实验记录。

**Python 3.12** · 本机接口契约 **`0.3.0-draft`** · 默认使用模拟数据与离线模型

> 本项目不向真实训练者投递消息，不提供健康或训练指导。候选文字始终不可展示、不可投递（`candidate.may_display` 恒为 `false`）；模拟数据库是本地明文文件，请勿放入真实健康数据或密钥。

## 它能做什么

Agent 是一套以程序为主的流程，大模型只负责提出候选，规则检查与安全控制由普通代码完成：

| 能力 | 说明 |
|---|---|
| 事件与候选 | 校验模拟训练事件；用户报告不适时优先返回固定安全提示；正常路径生成 `silent`、`notify` 或 `defer` 候选。 |
| 引用与规则检查 | 核对候选引用的 `fact_id`、免打扰状态、模型输出结构和失败路径；引用能找到不代表文字正确，规则失败即保守停止。 |
| 会话与安全状态 | 确认必须由用户主动发起；待确认事件可以保存但不调用模型；无法归属会话的不适报告明确提示“未记录”。 |
| 本机模拟接口 | 用 SQLite 保存事件、决策、引用快照和安全状态，提供会话、问题、事件接口，以及示例客户端、自检脚本和静态 fixtures。 |
| 文本模型客户端 | DeepSeek 与智谱 GLM，共用请求预算、HTTP 错误映射和响应解析；只用于独立的演示与研究脚本，本机接口默认使用离线模型。 |
| 动作研究 | 用 FusionPoser 六 IMU 数据预测骨架，在 CIP、CMU 公开数据上做跨受试者预测和骨架读图实验。 |

自动化测试覆盖提醒、静默、免打扰延后、不适优先与错误引用等分支。一次事件的主路径：

```text
模拟客户端 → 本机 HTTP 接口（R5）→ 会话与安全状态（R4/R5）
  → 事件校验（R1）→ 上下文（R4）→ 候选（R2）→ 引用与规则检查（R3）
  → 安全复核、SQLite 记录（R5）→ 结构化响应
```

并非每次请求都会走到模型：待确认会话只保存事件，不适报告直接进入安全提示。

## 当前实验结果

### 视觉模型骨架读图横评

六家模型读同一批 **90 张 CMU 骨架图**（上升、下降、无明显变化各 30 张，要求严格 JSON 输出），按三类平均正确率排序：

| 模型 | CMU 正确/总数 | 三类平均 | IMU 补充图（14 张） | 中位响应 |
|---|---:|---:|---:|---:|
| Kimi `kimi-k3.1-pro` | 87/90 | **96.7%** | 11/14 | 7.52 秒 |
| 豆包 `doubao-seed-2-1-pro-260915` | 81/90 | 90.0% | 6/14 | 3.19 秒 |
| 千问 `qwen3.8-max-2026-09-02` | 75/90 | 83.3% | 11/14 | 2.78 秒 |
| 智谱 `glm-5v-turbo` | 71/90 | 78.9% | 11/14 | 2.06 秒 |
| DeepSeek `deepseek-flash` | 68/90 | 75.6% | 11/14 | 1.14 秒 |
| 腾讯混元 `hy4` | 49/90 | 54.4% | 8/14 | 3.94 秒 |

格式错误、主动不确定和请求失败都进入分母。完整的有线/无线分项、耗时、费用和逐图记录见[横评报告](research/six_model_vision_20261110/横评报告.md)，也可直接打开[本机演示页](research/six_model_vision_20261110/index.html)。

### 端到端真实链路

已完成一次真实串联：公开 IMU 与头部高度 → 预测骨架图 → 视觉模型观察 → 文本模型选择 → R3 检查，最终输出 `silent`。

- 15 张新图确认：`qwen3.8-max` **14/15**，`qwen3.8-flash` **11/15**；三类平均正确率 **66.7% / 59.0%**。
- 下降片段仍是明确短板；完整 18 张（含三图预跑）Max 16/18、Flash 13/18。
- 最终候选只引用了自报事实，没有引用视觉观察，因此这次验证的是链路连通，不是视觉改变了决策。

可直接打开[演示页](research/live_chain_20261110/results/run_A/index.html)，详细口径见[完整报告](docs/公开IMU完整链路实测_20261110.md)，无密钥重放步骤见[重放指南](research/live_chain_20261110/README.md)。

### 姿态预测与几何对照

| 实验 | 当前结果 | 范围 |
|---|---|---|
| DynaIP 对比 | DynaIP **4.290 cm**，对齐小模型 **8.773 cm** | 2 名共同未训练受试者、8 段 CIP 录制；统一骨架的 22 个关节位置误差（MPJPE） |
| CIP 跨受试者预测 | 测试均值 **0.109054**，固定平均姿态基线 **0.146297** | 3 人训练、1 人验证、1 人测试；使用参考头高与辅助对齐 |
| FusionPoser 留出 | 窗口模型 **10.063**，单帧模型 **13.522** | 同一组 3 段留出录制；单位暂按厘米解释 |
| 固定颜色标记工具 | 160 个窗口中 **154 个一致、6 个不确定** | 指定颜色和参考线的受控图片，与视觉模型成绩分别计算 |

> 这些指标来自不同数据、身体点定义和任务，不能相互换算或拼成同一次对比。骨架读图只比较首尾髋部高度，不代表动作质量、安全性或对真实训练画面的理解能力；部分录制者身份未知，窗口可能重叠。

逐项方法和完整结果见 [DynaIP 与视觉模型对比](docs/两项优化_DynaIP与视觉模型对比_20261108.md)、[FusionPoser 结果报告](docs/三项优化_结果报告.md) 与 [跨人和视觉补充实验](docs/三项优化_跨人和视觉补充实验_20261108.md)。

## 快速开始

需要 **Python 3.12**，主程序运行时只使用标准库。在仓库根目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m metafit_agent.demo_training_event examples\set_completed.json
```

测试默认离线，不下载数据、不调用真实模型；缺少可选数据或依赖时，相关用例会跳过。可选依赖：

- `.[dev,research]`：NumPy、Pillow，用于研究计算与图片检查。
- `.[dev,train]`：增加 PyTorch，用于自训练骨架模型。
- `.[dev,benchmark]`：增加 SciPy，用于 DynaIP 评测。

提交前运行 `python tools/check_repository.py` 检查候选文件、文档链接和密钥格式。

### 启动本机模拟服务

```powershell
.\.venv\Scripts\python.exe -m metafit_agent.simulation_api --port 8000
```

在另一个同目录 PowerShell 窗口运行示例客户端：

```powershell
.\.venv\Scripts\python.exe examples\r5_api_client.py --port 8000
```

服务只监听 `127.0.0.1`，默认数据库为 `outputs/r5_simulation/api_demo.db`。`examples\r5_api_doctor.py` 可启动独立临时服务并输出 `PASS/FAIL` 自检结果。macOS/Linux 使用 `python3.12` 运行同一命令，服务本身目前只在 Windows 上实测。字段、状态码与 App 侧处理见 [App 对接指南](docs/R5_App对接指南.md) 和 [OpenAPI 草案](docs/openapi.r5-draft.yaml)。

## 研究复现

先读[研究目录说明](research/README.md)。脚本会把原始数据、权重和输出放入被忽略的本地目录；仓库只保留脚本、固定划分和汇总结果。

```powershell
# 公开样本自检：默认离线，核对数据摘要并跑通模拟候选链
.\.venv\Scripts\python.exe examples\research_doctor.py

# 首次使用可下载固定版本的 FusionPoser 公开样本并校验摘要
.\.venv\Scripts\python.exe examples\fusionposer_fetch.py --download
```

无密钥复核已归档结果：

```powershell
.\.venv\Scripts\python.exe research\six_model_vision_20261110\replay.py
.\.venv\Scripts\python.exe research\live_chain_20261110\replay.py --run research\live_chain_20261110\results\run_A
.\.venv\Scripts\python.exe research\live_chain_20261110\present.py --run research\live_chain_20261110\results\run_A
```

各实验的数据准备、完整命令和版本锁见对应指南：[DynaIP 与视觉评测](research/benchmark_20261108/README.md)、[CIP 与视觉补充](research/continuous_20261108/README.md)、[完整链路](research/live_chain_20261110/README.md)。

真实模型调用需要显式启用并自行配置密钥与预算；已付费请求不会自动重试。重新训练或重新调用会产生新结果，应使用新的输出目录，并记录数据划分、提示词和依赖版本。

## 数据、密钥与许可

- **FusionPoser**：公开真实 IMU 数据，通过脚本获取并校验摘要；来源与 MIT 许可见[数据说明](data/fusionposer/来源说明.md)。
- **CIP**：CC BY 4.0，保留数据作者署名和来源；下载、命名与参考定义见 [CIP 数据说明](data/cip/README.md)。
- **CMU**：仓库保留少量骨架示例，来源与转换说明见[公开动作实验](docs/公开动作数据骨架实验.md)。
- **AMASS / SMPL+H**：由使用者自行取得许可并本地下载，见[本地使用说明](docs/AMASS下载与交接.md)；不要把受限数据、模型或衍生图提交或上传。
- **密钥**：通过 `DEEPSEEK_API_KEY`、`ZHIPUAI_API_KEY`、`DASHSCOPE_API_KEY`、`ARK_API_KEY` 环境变量提供；`.env.example` 只是示例，程序不会自动加载。显式真实调用会产生费用。
- 项目自身暂未指定开源许可证；第三方数据和软件遵循各自许可。

原始数据、人体模型、训练权重、数据库和本机输出已列入忽略规则。

## 目录

```text
metafit_agent/   事件、候选、检查、状态、存储与本机接口
examples/        模拟客户端、数据获取与研究入口
research/        固定实验方案、复现脚本和不含原始数据的结果
tests/           自动化测试
docs/            接口契约、实验报告、评审说明与历史记录
data/            数据来源、许可与本地数据目录
tools/           提交前检查工具
```

## 文档

- [文档总目录](docs/README.md) · [项目一页通](docs/MetaFit_Agent_项目一页通.md)
- [App 对接指南](docs/R5_App对接指南.md) · [OpenAPI 契约](docs/openapi.r5-draft.yaml) · [契约变更记录](docs/CONTRACT_CHANGELOG.md)
- [研究评审说明](docs/研究原型评审交付说明.md) · [后续工作](docs/ROADMAP.md)
- [贡献指南](CONTRIBUTING.md) · [改进记录](CHANGELOG.md)
