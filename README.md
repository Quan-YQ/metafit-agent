# MetaFit Agent｜App 本地模拟评审版

这是一个供 App 与 Agent 团队**阅读、运行和讨论**的 Python 项目。它模拟用户完成一组训练后，Agent 如何接收事件、处理安全确认、组织事实、检查候选决策并返回状态。当前接口契约为 **`0.3.0-draft`**；本地模拟安全修订已获有条件复核通过，接口和分工仍待 App 团队确认。

**使用边界：仅限模拟数据。** 接口只监听运行它的电脑上的 `127.0.0.1`；手机的 `127.0.0.1` 是手机自己。没有真实身份鉴权、跨设备安全一致性和实际消息投递；本地 SQLite 为明文。请勿输入真实健康数据、个人信息或密钥。默认接口使用固定的离线模型，候选内容不可展示或投递。

## 从哪里开始

| 你的目的 | 阅读材料 |
|---|---|
| 快速理解项目和整体流程 | [项目一页通](docs/MetaFit_Agent_项目一页通.md) |
| 作为 App 开发者接入或评审 | [App 接入指南](docs/R5_App对接指南.md) · [开发者速查卡](docs/R5_App_开发者速查卡.md) |
| 查询字段、响应及版本 | [OpenAPI 契约](docs/openapi.r5-draft.yaml) · [版本和兼容规则](docs/CONTRACT_CHANGELOG.md) |
| 与双方讨论安全与分工 | [待确认问题](docs/R5_App_待确认问题清单.md) · [产品安全决策卡](docs/R5_App_产品安全决策卡.md) |
| 无服务时了解返回格式 | [静态响应示例](examples/r5_api_fixtures.json) · [HTTP 请求示例](examples/r5_api_requests.http) |

OpenAPI 是接口字段与响应的唯一契约草案；其他材料提供使用说明。兼容规则和责任分工是建议，尚未经双方确认。

## 目录

```text
MetaFit-Agent-App评审版/
├── README.md           从这里开始
├── pyproject.toml      Python 版本、项目和测试依赖
├── metafit_agent/      程序源代码
├── docs/               接口契约、项目说明与 App 评审材料
├── examples/           模拟请求、响应、客户端与 doctor 自检
└── tests/              自动化测试
```

`.env.example` 是不含密钥的配置示例，程序不会自动读取 `.env`。本评审版没有包含个人开发环境、运行数据库或历史审查文件；历史材料由项目维护方保存。

## 本机运行（Windows PowerShell）

在本文件所在目录打开 PowerShell，先安装 Python 3.12，然后：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m metafit_agent.simulation_api --port 8000
```

安装开发依赖需要可访问 Python 包源；项目本身的运行时依赖为空。服务启动后，在**第二个** PowerShell 窗口、同一个项目目录中运行：

```powershell
.\.venv\Scripts\python.exe examples\r5_api_client.py --port 8000
```

这是完整模拟流程，看到 `duplicate_event` 表示示例走到重复事件步骤。在服务窗口按 `Ctrl+C` 停止。自检、其他系统路径、接口命令和错误处理见[接入指南](docs/R5_App对接指南.md)；macOS/Linux 服务启动尚未在实机验证。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试需要本机回环网络和测试依赖；缺少可选的 PowerShell 7 或 `sh/curl` 时，相应环境测试可能跳过。测试通过不代表真实 App 联调或专业安全审查通过。

## 已知待办

App 团队确认事件来源、接口语义、重试和分工；产品及专业人员审查安全文案、本地兜底和紧急升级；完成完整 OpenAPI 3.1 验证、真实身份及数据保护、跨实例状态一致性和实际投递，再安排真实数据和 App 联调。详细问题见[待确认清单](docs/R5_App_待确认问题清单.md)。
