# 开发者速查卡｜MetaFit 模拟接口

**`0.3.0-draft`｜仅 127.0.0.1、模拟数据、无鉴权、不可投递**  
契约：[OpenAPI](openapi.r5-draft.yaml)；逐步命令与 UX 表：[接入指南](R5_App对接指南.md)。

非 Windows 或没有仓库：按[指南 1.2–1.4](R5_App对接指南.md#12-macoslinux-与项目仓库)选择服务、sh/curl、`.http` 或静态 fixtures 路径。

```powershell
# 窗口 1：项目根目录启动服务；Ctrl+C 停止
.\.venv\Scripts\python.exe -m metafit_agent.simulation_api --port 8000

# 窗口 2：运行已有示例，或连接现有模拟服务自检
.\.venv\Scripts\python.exe examples\r5_api_client.py --port 8000
.\.venv\Scripts\python.exe examples\r5_api_doctor.py --connect 8000 --json
```

`GET /sim/v0/meta` 比对版本；HEAD 不支持健康检查。`--connect` 会在目标模拟库写新记录；仅输出 `event_saved` 不证明物理持久性。`--reset` 只能作用于自启动 doctor 专用库，需输入 `RESET` 二次确认。

| 看到什么 | 下一步 |
|---|---|
| 201 注册 | 查确认状态；重启后 201 未必是全新会话。 |
| 200 `reviewed` | 检查 `required_action`；候选仍不可展示、播报或投递。 |
| 200 `answered` | 核对 `confirmation_state=confirmed`；别只看 200。 |
| 200 `duplicate_event` | 事件已有记录或预占；看 `record_state`，不无限重试或重复调模型。 |
| `pending_confirmation` | 用户主动发起；不要自动弹问或把未知当无不适。 |
| `confirmation_closed` | 回答未必被接受；按经评审流程提供主动新会话。 |
| `safety_pending` | 显示安全提示；`volatile` 表示未确认持久化。 |
| 400 `safety_report_unattributed` | 显示提示；报告没保存，在正确会话重新报告。 |
| 409 `event_conflict` | 核对 ID 与内容，不盲目重试。 |
| 409 `session_exists` | 核对模拟会话并查询状态，不当新建成功。 |
| 400 `invalid_request` / `invalid_event` | 核对输入；遇明确不适先遵循安全提示。 |
| 413 | 请求太大；按上限修正，不原样重试。 |
| 405 / 414 | 明确方法或地址错误，不等同断网；HEAD 无 body。 |
| 404 | 普通会话/事件不存在；可识别的未归属 reported 走 400 安全提示。 |
| 503 / 无响应 | 提示及本地兜底待专业审查；不假定已保存或一定未保存。 |
| `candidate.may_display=false` | 不展示、不播报、不投递候选文字。 |

别用 `localhost` 或跨设备 loopback；别把 200 当“回答已接受”；别存真实数据。`0.2.1→0.2.2` 的安全 404→400 是历史例外；`0.3.0` 调整畸形不适报告的错误码，[版本规则](CONTRACT_CHANGELOG.md)待双方确认。
