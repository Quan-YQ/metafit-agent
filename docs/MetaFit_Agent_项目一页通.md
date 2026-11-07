# MetaFit Agent 项目一页通

> 面向第一次读代码的人；汇总截至 2026-11-07 的本地模拟和公开数据研究。具体行为以代码、测试和接口契约为准。

## 1. 我们到底做了什么？

设想：用户完成一组训练，App 把这组的信息交给 Agent；Agent 结合已有信息，考虑是否给反馈，同时优先处理用户报告的不适。**现在做成的是本机模拟原型**：可以用模拟数据走通输入、检查、安全状态、存储和接口；还没有正式 App、真实用户身份、真实消息投递，也不能提供经过专业验证的健康建议。

这里的 Agent 是一套程序流程，**不是只有一个大模型**。大模型只负责提出候选；接收数据、检查规则、保存记录和安全控制由普通程序负责。

## 2. 一次训练事件怎么走完整流程？

```text
模拟 App/示例客户端 → 本机 HTTP 接口 R5 → 会话与安全检查 R4/R5
   → 事件验证 R1 → SQLite 保存、去重 R5
   → 组织上下文 R4 → 生成候选 R2 → 引用及规则检查 R3
   → 最终安全复核、记录 R5 → 结构化响应给调用方
```

这是正常路径的简图，**不是每次都会走到模型**：

1. App 先创建会话。新会话处于待确认，用户要主动发起并完成安全确认；系统不会自动弹问。跳过或过期也不等于确认无不适。
2. App 提交“完成一组”的事件。待确认时，合法事件可先保存，但不调用模型；后续确认也不会自动重新跑这条事件。
3. 用户明确报告不适时，优先返回固定安全提示；如果报告无法归属会话，提示报告未记录。保存失败也不能伪装成已持久保存。
4. 正常情况下，程序检查事件、组织可用事实，再请求模型给结构化**候选**；程序检查引用和免打扰规则，并在返回前再次核对安全状态。
5. 重发同一事件且内容相同，返回 `duplicate_event`，不重复处理；同一标识对应不同内容则报冲突。接口返回状态、需要的动作和展示信息，候选文字始终不可展示或投递。

**重要区别：**本机 HTTP 服务默认用 `OfflineModel`，只返回固定模拟结果、不调用真实模型。独立研究脚本已测试 DeepSeek 和智谱文本调用，以及 DeepSeek 读取公开骨架图；这不等于 App 接口已接入真实模型。

动作研究是与 App 接口分开的实验：公开数据可生成骨架，视觉模型观察后与几何量对照，再进入模拟候选与引用检查。AMASS 骨架只在本地处理。数据来源、结果和限制见[研究原型评审说明](研究原型评审交付说明.md)。

## 3. 每个阶段负责什么？

| 阶段 | 功能与核心代码 | 为什么要有它 |
|---|---|---|
| R0 工程基础 | [启动入口](../metafit_agent/__main__.py)、[项目配置](../pyproject.toml)、测试环境 | 让项目可安装、运行和自动验证；业务运行依赖为零。 |
| R1 事件 | [training_events.py](../metafit_agent/training_events.py)：检查会话、动作、次数、不适等输入 | 区分格式有效、缺失和错误，避免把无效信息当事实。 |
| R2 决策 | [decision.py](../metafit_agent/decision.py)、[deepseek_client.py](../metafit_agent/deepseek_client.py)：整理事实、最多一次模型调用、解析候选 | 模型可建议 `silent`（不打扰）、`notify`（建议提示）或 `defer`（建议延后）；候选不等于可以发送。 |
| R3 把关 | [decision_review.py](../metafit_agent/decision_review.py)：检查引用范围、免打扰及失败处理 | 引用能找到不代表文字真实或安全；规则失败就保守停止。 |
| R4 上下文和状态 | [context_builder.py](../metafit_agent/context_builder.py)、[interaction_state.py](../metafit_agent/interaction_state.py)、[coordinator.py](../metafit_agent/coordinator.py) | 把当前事件、模拟历史等组织成事实；记录询问和不适状态；协调时优先保护安全。 |
| R5 保存与接口 | [event_store.py](../metafit_agent/event_store.py)、[decision_store.py](../metafit_agent/decision_store.py)、[snapshot_store.py](../metafit_agent/snapshot_store.py)、[safety_store.py](../metafit_agent/safety_store.py)、[simulation_api.py](../metafit_agent/simulation_api.py) | 用 SQLite 保存事件、决策、引用快照和安全状态；提供只监听本机的模拟 App 接口。 |
| 真实 App 联调和评估 | 尚未完成 | 已开展公开数据的研究实验；真实 App、设备和端到端效果仍需双方核对，不能用本地测试代替。 |

**R5 的五个小单元**依次是：①保存事件与防重复；②记录决策；③保存当时引用的事实快照；④保存确认和不适状态；⑤提供本机模拟 HTTP 接口及契约。快照能帮助追溯，不能证明事实或模型文字正确。

## 4. 代码是怎样接起来的？

- `simulation_api.py` 是接口接待员：识别路径、解析请求、调用协调器，再把结果整理成 JSON；接口字段的约定在 [OpenAPI 草案](openapi.r5-draft.yaml)。
- `DurableSafetyCoordinator` 是总协调员。它在原协调器上逐层增加事件保存、决策记录、引用快照和安全状态持久化。Python 中 `class 子类(父类)` 表示子类复用并扩展父类功能；这就是多个 `*_coordinator.py` 同时存在的原因。
- `*_store.py` 负责与本地 SQLite 数据库读写；`demo_*.py` 是单独的演示入口；`tests/` 用自动化用例检查正常、重复和异常分支。
- `def` 定义函数，`class` 定义一类对象，`import` 引入其他文件的功能，`return` 把结果交给调用者。JSON 可以理解成接口传递的字段和值；HTTP 状态码说明请求结果，但不能代替读取安全信息。

**从代码追一条线：**打开 [simulation_api.py 的事件入口](../metafit_agent/simulation_api.py#L228) → [安全协调器](../metafit_agent/safety_coordinator.py#L390) → [基础协调器](../metafit_agent/coordinator.py#L239) → [R2 决策](../metafit_agent/decision.py#L97) / [R3 检查](../metafit_agent/decision_review.py#L62)。遇到保存或重复，再看 [事件存储](../metafit_agent/event_store.py#L173)。代码中旧的独立入口可能绕开新协调器，不能假设所有入口共享同一安全保护。

## 5. 我们是怎么一步步做出来的？

1. 先确定目标和边界：训练事件触发反馈，安全优先；先用模拟数据验证思路。
2. 建立工程和事件结构（R0、R1），让输入能被验证；再做模型候选与规则检查（R2、R3）。
3. 加入上下文、用户主动确认和单协调器保护（R4），随后逐步加上 SQLite 保存、历史、快照和安全状态（R5 前四单元）。
4. 手写 OpenAPI 契约并建立本地模拟接口（R5 第五单元）；经多轮独立评审，修正状态、重复请求及不适报告等问题，当前为 `0.3.0-draft`。
5. 每个单元经过设计、实现、测试和用户复核；评审提出的问题继续修订。**当前 0.3.0 安全修订仅在本地模拟范围有条件通过**，材料可供 App 团队评审，不等于双方已确认或真实联调完成。

## 6. 现在能验证什么，还缺什么？

在项目根目录运行下面两条，只使用模拟数据：

```powershell
.\.venv\Scripts\python.exe -m metafit_agent.demo_training_event examples\set_completed.json
.\.venv\Scripts\python.exe -m pytest -q
```

第一条展示 R1 事件验证；第二条运行自动化测试。测试数量和跳过项会随可选数据、依赖与运行环境变化；研究记录见[研究原型评审说明](研究原型评审交付说明.md)。测试通过只表示覆盖的程序场景符合预期，不代表医疗安全或真实业务效果已验证。

还需要：App 团队确认契约和分工、完整 OpenAPI 3.1 验证、专业审查安全提示和升级流程、真实身份与数据保护、跨进程/设备安全一致性、实际投递，以及真实数据和 App 联调。模拟数据库是明文，**不要放入真实健康数据或密钥**。

想进一步查细节：看[项目首页](../README.md)、[App 对接指南](R5_App对接指南.md)和[当前接口契约](openapi.r5-draft.yaml)。
