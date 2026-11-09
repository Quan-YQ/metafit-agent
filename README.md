# MetaFit Agent

MetaFit Agent 是一个非商业研究原型，探索“训练事件 → 动作骨架与视觉观察 → Agent 候选反馈”的流程。仓库包含可运行的本机模拟接口、公开数据实验、测试和结果记录。

## 已完成的工作

- **事件与决策**：校验模拟训练事件、处理不适报告、生成提醒/静默/延后候选，检查引用依据和免打扰状态。
- **本机接口**：提供会话确认、事件去重、SQLite 记录，以及 `0.3.0-draft` HTTP 契约和示例客户端。
- **模型客户端**：提供 DeepSeek、智谱文本客户端；研究入口支持智谱/百炼四款官方视觉模型，另保留原 DeepSeek 骨架观察实验。
- **动作研究**：开展 FusionPoser 跨录制预测、CIP 按受试者划分的预测评估、CMU 骨架读图和固定颜色标记读取实验。

本机接口使用模拟数据和离线模型；动作研究脚本单独运行。候选反馈目前不向真实训练者投递。

## 最新实验结论

两项优化的统一对比见 [完整报告](docs/两项优化_DynaIP与视觉模型对比_20261108.md)，复现入口见 [DynaIP与视觉评测指南](research/benchmark_20261108/README.md)。

| 方向 | 本轮结果 | 建议与范围 |
|---|---|---|
| DynaIP 对齐比较 | 两名共同未训练受试者、8段录制：DynaIP **4.290 cm**，对齐小模型 **8.773 cm** | 精度优先选DynaIP；小模型同机CPU整段推理约快 **20.2倍**；共享参考校准和初始化条件 |
| 四款视觉模型 | 冻结前两名后另测18张新图；最终实验候选 **Qwen3-VL-Plus**：**12/18（66.7%）** | 仅限带髋部标记与阈值线的受控骨架图，升降判断仍需几何核验 |

新结果不替换下表中不同输入/数据/指标的历史记录。API总估算费用 **¥0.0435868**，上限¥10；逐图/逐段结果、官方训练划分和文件摘要已归档。

## 既有实验（保留原结果）

| 方向 | 当前结果 | 阅读时需要知道 |
|---|---|---|
| FusionPoser 骨架预测 | 同一组 3 段留出录制，旧单帧模型均值 **13.522**，新窗口模型 **10.063** | 单位暂按厘米解释；人物身份未知，窗口可能包含未来帧 |
| CIP 骨架预测 | 3 人训练、1 人验证、1 人测试；测试均值 **0.109054**，固定平均姿态基线 **0.146297** | 源数据位置单位待独立核实；使用参考头高与参考辅助时间对齐 |
| 视觉大模型 | 新的上升/下降/无明显变化平衡测试中，**4/9** 与几何参考一致 | 该方案判断升降仍不稳定；重叠窗口及倒序不算独立人物 |
| 固定颜色标记工具 | 16 个新动作源的 160 个窗口中，**154 个一致、6 个不确定** | 适用于带指定颜色和参考线的图片，与视觉大模型成绩分别计算 |
| Agent | 已检查提醒、静默、免打扰延后、不适优先和错误引用等分支 | 候选文字由已引用的模拟次数构造 |

详细依据见 [补充实验报告](docs/三项优化_跨人和视觉补充实验_20261108.md) 与 [FusionPoser 实验报告](docs/三项优化_结果报告.md)。这些指标来自不同数据和身体点定义，不能串成“一米降到十厘米”的同一次对比。早期实验记录保留在 [文档目录](docs/README.md)。

## 快速开始

需要 **Python 3.12**。主程序运行时仅使用标准库。在仓库根目录打开 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m metafit_agent.demo_training_event examples\set_completed.json
```

基础测试不下载数据、不调用真实模型。缺少可选研究依赖或数据时，相关测试会跳过。可安装 `.[dev,research]` 运行 NumPy/Pillow 研究计算测试，安装 `.[dev,train]` 增加 PyTorch 训练依赖；`.[dev,benchmark]` 增加 DynaIP 评测所需 SciPy。精确复现使用研究指南中的版本锁。

启动本机模拟服务：

```powershell
.\.venv\Scripts\python.exe -m metafit_agent.simulation_api --port 8000
```

在另一个同目录的 PowerShell 窗口运行：

```powershell
.\.venv\Scripts\python.exe examples\r5_api_client.py --port 8000
```

服务地址为本机 `127.0.0.1`。字段和状态说明见 [App 对接指南](docs/R5_App对接指南.md) 与 [OpenAPI 草案](docs/openapi.r5-draft.yaml)。

## 运行研究实验

先阅读 [研究目录说明](research/README.md)。CIP 训练、视觉样本获取、离线结果重放和真实 API 调用的命令见 [研究复现指南](research/continuous_20261108/README.md)。

早期公开数据演示入口：

```powershell
.\.venv\Scripts\python.exe examples\fusionposer_fetch.py --download
.\.venv\Scripts\python.exe examples\research_doctor.py
```

第一条联网下载固定公开样本并核验摘要；第二条默认离线运行。重新运行训练或真实模型可能得到不同结果，应使用新输出目录，并记录数据划分和依赖版本。

## 数据、密钥与许可

- **FusionPoser**：原始数据通过脚本另行获取，来源和 MIT 许可见 [数据说明](data/fusionposer/来源说明.md)。
- **CIP**：CC BY 4.0，保留数据作者署名和来源；下载、命名和参考定义见 [CIP 数据说明](data/cip/README.md)。
- **CMU**：仓库保留少量骨架示例，来源与转换说明见 [公开动作实验](docs/公开动作数据骨架实验.md)。
- **AMASS / SMPL+H**：由使用者自行取得许可并下载，见 [本地使用说明](docs/AMASS下载与交接.md)。
- **密钥**：通过 `DEEPSEEK_API_KEY`、`ZHIPUAI_API_KEY`、`DASHSCOPE_API_KEY` 环境变量提供；`.env.example` 只是示例，程序不会自动加载。显式真实调用会产生费用。
- 项目自身暂未指定开源许可证；第三方数据和软件遵循各自许可。

原始数据、人体模型、训练权重、数据库和本机输出已列入忽略规则。提交前运行 `python tools/check_repository.py` 检查候选文件。

## 目录与文档

```text
metafit_agent/   事件、模型客户端、状态、存储与本机接口
examples/        模拟客户端、数据获取与实验入口
research/        固定实验方案、复现脚本及不含原始数据的结果
tests/           自动化测试
docs/            使用说明、接口契约、实验报告和历史记录
data/            数据来源、许可与本地数据目录
tools/           提交前检查工具
```

- [文档总目录](docs/README.md) · [项目一页通](docs/MetaFit_Agent_项目一页通.md)
- [贡献与提交指南](CONTRIBUTING.md) · [改进记录](CHANGELOG.md)
- [研究评审说明](docs/研究原型评审交付说明.md) · [后续工作](docs/ROADMAP.md)
