# MetaFit Agent

MetaFit Agent 是一个**非商业研究原型**，用于探索训练事件、动作骨架、视觉模型和 Agent 决策如何协同工作。项目同时提供供 App 团队评审的本地模拟接口。

> 当前仅供研究与模拟：尚未完成真实设备接入、真实用户身份与数据保护、实际消息投递或专业安全评审，不能用于训练指导。

## 能做什么

- **模拟训练事件**：验证事件、处理用户报告的不适、生成候选，并检查引用是否属于本次事件。
- **接入大模型**：支持 DeepSeek 和智谱文本接口；独立实验支持 DeepSeek 骨架图观察。真实请求需要自行配置密钥，可能产生费用。
- **公开数据实验**：将 CMU 动作捕捉样本及 FusionPoser 公开 IMU 数据用于骨架和视觉实验；也可在本地使用合法取得的 AMASS 和 SMPL+H 模型绘制骨架。
- **对照与评估**：比较不同录制上的预测误差，以及同一模拟事件在有／无视觉观察时的候选。当前 IMU 预测在部分录制上不如简单基线。
- **App 接口模拟**：提供 `0.3.0-draft` 接口草案与仅监听本机的模拟服务；候选不能展示或投递给真实训练者。

研究链路概览：`公开动作数据或 IMU → 骨架图 → 视觉观察 → 模拟 Agent 候选 → 引用检查`。AMASS 图像默认仅在本地处理；不同数据来源和模拟事件之间没有真实身份或设备绑定。

本机 App 模拟接口与动作研究脚本是两条独立路径。视觉观察在研究脚本内作为模型报告参与模拟引用检查，并未接入 App 接口；引用检查也不能证明动作描述真实或安全。

### 当前实验结果

FusionPoser 的一次跨录制测试中，两个测试片段的平均关节误差分别约为 **69.3 cm** 和 **58.8 cm**，均高于简单姿态基线；坐标布局和单位仍待作者确认。两组公开骨架图的真实模型对照中，有图观察与纯文本候选均为 `silent`。这些小样本结果只说明当前研究的可行性和局限，不代表模型可靠性或训练安全。

## 快速开始

主项目使用 Python **3.12**，运行时无第三方依赖。以下命令适用于 Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

测试依赖需要访问 Python 包源。没有下载可选研究数据或安装可选依赖时，相关测试可能跳过；测试通过不代表真实训练安全已获验证。

仓库包含 Windows / Python 3.12 的 GitHub Actions 测试配置；它不调用真实模型或下载受限数据。

启动 App 模拟服务：

```powershell
.\.venv\Scripts\python.exe -m metafit_agent.simulation_api --port 8000
```

在另一个 PowerShell 窗口的同一项目目录中运行示例客户端：

```powershell
.\.venv\Scripts\python.exe examples\r5_api_client.py --port 8000
```

服务只监听运行它的电脑的 `127.0.0.1`，不具备真实身份鉴权。完整说明见 [App 接入指南](docs/R5_App对接指南.md)。

## 运行公开数据研究演示

本仓库不附带 FusionPoser 原始数据。需联网获取时，可运行：

```powershell
.\.venv\Scripts\python.exe examples\fusionposer_fetch.py --download
.\.venv\Scripts\python.exe examples\research_doctor.py
```

下载工具校验固定版本和文件摘要，不覆盖已有文件；自检默认不联网。它使用离线模型替身，结果仅表示演示可运行。详细结果和限制见 [研究原型评审说明](docs/研究原型评审交付说明.md)。

## 数据、密钥和许可

- **FusionPoser**：数据来源及作者仓库的 MIT 许可见 [数据来源说明](data/fusionposer/来源说明.md)。
- **CMU 动作捕捉样本**：仓库内少量骨架示例的来源与使用说明见 [公开动作数据骨架实验](docs/公开动作数据骨架实验.md)。
- **AMASS / SMPL+H**：须由使用者自行按官网许可注册、下载和使用；原始数据、模型及派生图片不随本仓库发布。具体见 [AMASS 下载与交接](docs/AMASS下载与交接.md)。
- **API 密钥**：使用 `DEEPSEEK_API_KEY` 或 `ZHIPUAI_API_KEY` 环境变量；不要写入源代码或提交 Git。`.env.example` 仅为不含密钥的示例，程序不会自动读取。
- 项目自身暂未指定开源许可证；第三方数据与软件分别遵循其各自许可。

## 项目结构

```text
├── metafit_agent/       事件、模型客户端、状态与本地模拟接口
├── examples/            骨架实验、模拟请求和评估工具
├── tests/               自动化测试
├── docs/                研究说明、App 接口草案及对接材料
├── data/fusionposer/     公开数据来源和许可（不含数据文件）
└── pyproject.toml       项目配置和测试依赖
```

## 进一步阅读

- [项目一页通](docs/MetaFit_Agent_项目一页通.md)
- [研究原型评审说明](docs/研究原型评审交付说明.md)
- [App 接入指南](docs/R5_App对接指南.md) · [OpenAPI 契约草案](docs/openapi.r5-draft.yaml)
- [模拟动作观察与候选选择](docs/模拟动作观察与候选选择.md)
- [App 对接与安全待确认问题](docs/R5_App_待确认问题清单.md)
