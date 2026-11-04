# MetaFit Agent × App 接入指南（内部草案）

**契约：`0.3.0-draft`｜日期：2026-11-03｜状态：待双方评审**

这份指南帮助开发者使用**模拟数据**，并确定 App 收到结果后该做什么。字段定义以 [OpenAPI 契约](openapi.r5-draft.yaml)为准；本指南不复制完整 schema。`0.3.0-draft` 的安全修订已在本地模拟范围有条件通过复核；App 团队尚未确认，材料定稿条件见文末。

**不能用于真实用户或健康数据。** 当前没有真实鉴权、跨进程安全一致性、实际消息投递或经过专业审查的安全文案；`candidate.may_display` 始终为 `false`。所有产品提示、医疗边界及紧急升级路径均是**待产品和专业人员联合审查**的草案。App 团队未确认分工或契约。

## 1. 选择你的接入路径

- Windows + 项目仓库：[1.1 内部工作站](#11-内部-windows-工作站五分钟跑通)。
- macOS/Linux + 项目仓库：[1.2 启动服务](#12-macoslinux-与项目仓库)。
- 无仓库但本机已有可信模拟服务：[1.3 sh/curl 或 .http](#13-已有本机服务使用-shcurl-或-http)。
- 无服务也无仓库：[1.4 静态 fixtures](#14-无服务无仓库使用静态-fixtures)。

### 1.1 内部 Windows 工作站：五分钟跑通

前提：Windows、PowerShell、Python 3.12，项目根目录 `metafit-agent` 已有 `.venv`。此服务只接受本机 `127.0.0.1`；`localhost`、浏览器 Origin、跨设备访问都不受支持。

在项目根目录的第一个 PowerShell 窗口启动服务：

```powershell
.\.venv\Scripts\python.exe -m metafit_agent.simulation_api --port 8000
```

预期出现 `模拟服务: http://127.0.0.1:8000/sim/v0/meta`。关闭服务在该窗口按 `Ctrl+C`。默认模拟数据库 `outputs/r5_simulation/api_demo.db`，明文且可能被 OneDrive 同步；不存真实资料。

在第二个 PowerShell 窗口执行：

```powershell
.\.venv\Scripts\python.exe examples\r5_api_client.py --port 8000
```

预期看到创建、确认、事件和查询步骤，最后出现 `duplicate_event`。脚本的“展示/回答”只是程序模拟，不代表真实用户已经自报。

也可以运行独立自检：

```powershell
.\.venv\Scripts\python.exe examples\r5_api_doctor.py --port 8001
```

自检启动仅绑定本机的临时服务，使用 `outputs/r5_simulation/doctor/doctor.db`，输出每项 `PASS/FAIL` 和失败时的下一步。`--port 0`（默认）由系统分配空闲端口；要验证固定端口可指定 `--port 8001`。`--json` 输出结构化结果；`--connect 8000` 连接现有本机模拟服务，会写入新模拟记录，不能验证对方数据库路径或物理可靠性。仅通过 `/meta` 不能证明服务身份真实。

`--reset` 仅针对 doctor 专用演示库：脚本验证目录标记和文件，要求输入 `RESET` 二次确认，拒绝未知文件或链接；不要让其他程序同时使用该库。它不能与 `--connect` 共用。自检不调用真实模型，不证明真实设备、产品体验或身体安全。

```powershell
.\.venv\Scripts\python.exe examples\r5_api_doctor.py --reset
```

上述命令会先询问；只在确认清理 doctor 专用模拟库时输入 `RESET`，否则取消。完成清理后脚本会重新创建该模拟库并运行自检。

如果异常中断后提示 `.doctor.lock` 已存在，脚本不会自动删除锁。先检查是否仍有 doctor，或其他程序正在使用 `outputs/r5_simulation/doctor/doctor.db`；不能只凭锁文件时间判断。Windows 可先查看相关 Python 进程：

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'r5_api_doctor\.py|simulation_api' } | Select-Object ProcessId, Name
```

上面的筛选不保证找到以其他方式启动的占用者。只有在人工确认没有任何程序使用此专用库、且锁确实遗留后，才检查并手动删除项目内精确路径 `outputs/r5_simulation/doctor/.doctor.lock`；不要递归删除目录或其他数据库。之后重新运行 doctor；若无法确认是否仍在使用，先不要删除锁。

在 macOS/Linux 上可用 `ps -ef | grep '[r]5_api_doctor.py'` 辅助排查；它同样不能证明没有其他程序正在使用该数据库，需人工核对后再处理锁。

### 1.2 macOS/Linux 与项目仓库

如果你在 macOS/Linux 上**已有项目仓库和 Python 3.12**，可在项目根目录运行 `python3.12 -m metafit_agent.simulation_api --port 8000`；该路径不依赖 PowerShell。当前 Windows 环境没有完成 macOS/Linux 实机验证，Python 与 SQLite 环境还需对方确认。

### 1.3 已有本机服务：使用 sh/curl 或 .http

如果你**没有仓库，但本机已有可信的模拟服务**，可以用任何支持 HTTP 的客户端访问下面的 `127.0.0.1:8000`，例如 sh/curl 或 [.http 示例](../examples/r5_api_requests.http)。仅有指南或契约文件**不会自动启动服务**；手机上的 `127.0.0.1` 也不是电脑上的服务。浏览器 Origin 当前不受支持。

在支持 `sh`、`curl` 和 Python 3 的终端中运行下面整段；它先自动检查 HTTP 200、版本 `0.3.x-draft`（当前 `0.3.0-draft`）和三个模拟标志，通过后才会创建模拟数据。使用子 shell 使失败只终止这一段，不能把后半段拆出来单独执行。版本和能力是服务自报，不是身份认证。

<!-- bash-smoke:begin -->
```bash
(
BASE='http://127.0.0.1:8000/sim/v0'
if python3 --version >/dev/null 2>&1; then PYTHON=python3
elif python --version >/dev/null 2>&1; then PYTHON=python
else printf '%s\n' '需要 Python 3；停止。' >&2; exit 1
fi
META_RESPONSE=$(curl --noproxy '*' -sS --max-time 5 -w '\n%{http_code}' "$BASE/meta") || exit 1
if ! printf '%s' "$META_RESPONSE" | "$PYTHON" -c '
import json, re, sys
try:
    body, status = sys.stdin.read().rsplit("\n", 1)
    data = json.loads(body)
    meta, features = data["meta"], data["features"]
    version = meta["contract_version"]
    if (sys.version_info < (3, 0) or status != "200" or not isinstance(version, str)
            or not re.fullmatch(r"0\.3\.[0-9]+-draft", version)
            or meta["simulation"] is not True or features["real_model"] is not False
            or features["delivery"] is not False):
        raise ValueError("invalid simulation environment")
except (ValueError, KeyError, TypeError):
    sys.exit("模拟服务或契约版本不匹配，停止。")
print("PASS: simulation contract and capabilities; service identity is not verified")
'; then exit 1; fi
SID="curl_$(date +%s)_$$"
SESSION_JSON=$(printf '{"user_id":"curl_user","session_id":"%s"}' "$SID")
curl --noproxy '*' -sS -i -H 'Content-Type: application/json' --data "$SESSION_JSON" "$BASE/sessions"
EVENT_JSON=$(printf '{"event":{"schema_version":"0.1","event_id":"curl_event","event_type":"set_completed","user_id":"curl_user","session_id":"%s","set_id":"set_1","exercise_id":"squat","timestamp":"2026-11-03T00:00:00Z","source":"simulation"},"do_not_disturb":false}' "$SID")
curl --noproxy '*' -sS -i -H 'Content-Type: application/json' --data "$EVENT_JSON" "$BASE/sessions/$SID/events"
)
```
<!-- bash-smoke:end -->

预期创建 201，事件返回 200 `pending_confirmation`、`event_saved=true`；这是**待确认保存**，不会调用模型或产生可投递反馈。`curl -i` 会显示 HTTP 错误状态和 body，不能仅凭命令进程退出码当作业务成功。

### 1.4 无服务无仓库：使用静态 fixtures

如果**没有服务和仓库**，可从随材料提供的 [静态 fixtures](../examples/r5_api_fixtures.json) 打开 `duplicate_event`、`safety_report_unattributed` 等完整 JSON 响应，在开发工具中人工 mock；不要求 Python 或运行接口。至少需要把指南、契约和 fixtures 一起提供；如果只收到指南/契约、没有 fixtures，也没有服务，只能审阅契约，不能声称已跑通。不要用虚构 fixture 冒充服务实测。

## 2. 接口地图

所有路径前缀均为 `/sim/v0`，具体请求、字段和状态码见契约。

| 方法及路径 | App 何时使用 | 关键判断 |
|---|---|---|
| `GET /meta` | 启动时检查契约和功能 | 检查版本；只用 GET 探活，不用 HEAD |
| `POST /sessions` | 注册模拟会话 | 201 不一定证明是全新会话；重启恢复语义待确认 |
| `GET /sessions/{id}` | 恢复当前确认界面 | 查看 `confirmation_state`、`question_id`、`question_presented` |
| `POST /sessions/{id}/questions` | 用户主动确认或报告不适 | `open`、`present`、`answer`、`skip` 是不同步骤 |
| `POST /sessions/{id}/events` | 提交完成一组的模拟事件 | 重复同内容 200；冲突 409；等待确认也可能保存 |
| `GET /sessions/{id}/events/{event_id}` | 查询历史处理记录 | 历史 `event_status/event_code` 与当前安全快照是两件事 |

契约中其他标准 HTTP 方法只声明 405/503 错误，**不是可用业务接口**。App 与服务不同设备时，`127.0.0.1` 各自指向自己的设备；本服务不提供跨设备接入方案。

## 3. App 决策表：先看安全，再看 HTTP

这是供讨论的 UX 草案，**不是专业安全结论**。下列文案与是否暂停、求助、升级，须由产品和专业负责人审查。无论成功或错误，先读取响应体的 `required_action` 和 `display`；网络断开或无响应体时，不能假定安全或假定事件未保存。

| 响应或场景 | 建议用户看到什么 | App 应做什么 | 不应做什么 |
|---|---|---|---|
| 201 注册会话 | 会话已注册、需确认状态 | 检查 `confirmation_state` 和 `can_request_confirmation` | 不仅凭 201 假定数据库里是全新会话 |
| 200 `reviewed` | 模拟处理中或不可投递提示 | 检查 `required_action` 与 `candidate.may_display` | 不把 `candidate.text` 当可展示的训练反馈 |
| 200 `answered` | 用户回答已按当前状态处理 | 仍检查 `confirmation_state=confirmed` 和安全状态 | 不只凭 200 判断确认已完成 |
| 200 `duplicate_event` | 这条事件已有记录或已预占 | 检查 `record_state` 与当前安全提示；按双方商定策略处理 | 不重新触发模型、不无限重试；不假定记录一定完成或已投递 |
| 200 `pending_confirmation` | 提示可由用户主动确认 | 根据 `confirmation_state` 决定发起、展示或回答 | 不自动弹问；不把未知当作无不适 |
| 200 `confirmation_closed` | 该会话的确认已结束，可主动开始新会话 | 保留关闭状态；按产品审查后的方式提供新会话入口 | 不把 200 当作回答已接受；不自动重问 |
| 200 `safety_pending` | 显示固定安全提示；具体文案待审 | 保留安全路径；如 `volatile`，提示报告保存未确认并依产品方案处理 | 不显示普通候选；不要求已记录报告者重复报告 |
| 400 `safety_report_unattributed` | 安全提示，并说明报告**未记录** | 不适提示先行；定位正确会话后由用户重新报告 | 不吞掉 400 body；不认为报告已保存 |
| 400 `invalid_request` / `invalid_event` | 请求或事件有误 | 在模拟环境核对输入与契约；明确 reported 的畸形问题请求另走安全提示 | 不盲目重试同一错误输入，不回显真实健康信息 |
| 409 `question_conflict` | 可理解的状态冲突提示 | 读取当前确认快照，必要时重新查询 | 不猜“已经回答”或重建用户答案 |
| 409 `event_conflict` | 事件内容冲突提示 | 停止盲目重试，核对事件标识及内容 | 不改 ID 来掩盖冲突 |
| 409 `session_exists` | 会话已注册 | 核对模拟 ID，按现有会话查询或约定处理 | 不把冲突当作新建成功 |
| 413 `request_too_large` | 请求过大 | 检查大小上限；按约定修正请求 | 不原样反复重试或上传真实数据 |
| 503 `storage_failed` / `safety_state_unknown` / `server_busy` | 显示响应里的安全提示与暂不可用说明 | 保留错误 body；提示下一步并按约定排查 | 不展示候选；不无限自动重试；不声称保存成功或一定失败 |
| 404 `state_missing` / `event_missing` | 普通请求未找到记录 | 查明会话或事件；明确 reported 的不可归属请求按 400 行处理 | 不把普通 404 当作业务成功 |
| 405 / 414 | 方法不支持或地址过长 | 读取 JSON 错误和 405 的 `Allow`，修正客户端；HEAD 按协议无 body | 不把它们当作网络断开或业务成功 |
| 无响应、无法解析 | 连接或响应错误；安全情况下按 App 本地兜底方案 | 记录有限诊断；是否重试待双方确认 | 不假设一定安全或一定未保存；不传真实数据调试 |

`reported` 事件或 `answer=reported` 无法归属到有效会话，或已知会话的明确 `answer=reported` 信封不合格，`0.3.0-draft` 返回 **400 `safety_report_unattributed`**，提示但不归属、不保存。无法解析为 JSON、网络断开或服务过载仍可能无法得到提示；真实 App 必须有经审查的本地安全兜底，不应依赖这项模拟能力。

## 4. 可复制的模拟调用

下面命令按顺序在**项目根目录的第二个窗口**运行。先运行初始化代码一次；它使用 UTF-8，读取错误响应体，并把 HTTP 状态与 body 一起返回。代码仅针对模拟接口。

<!-- smoke:begin -->
```powershell
$base = 'http://127.0.0.1:8000/sim/v0'
Add-Type -AssemblyName System.Net.Http -ErrorAction Stop
function Invoke-MF {
  param([string]$Method, [string]$Path, $Body = $null)
  $handler = [System.Net.Http.HttpClientHandler]::new()
  $handler.UseProxy = $false
  $client = [System.Net.Http.HttpClient]::new($handler)
  $client.Timeout = [TimeSpan]::FromSeconds(5)
  $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($Method), [Uri]"$base$Path")
  $r = $null
  try {
    if ($null -ne $Body) {
      $json = $Body | ConvertTo-Json -Depth 30 -Compress
      $req.Content = [System.Net.Http.StringContent]::new($json, [System.Text.Encoding]::UTF8, 'application/json')
    }
    $r = $client.SendAsync($req).GetAwaiter().GetResult()
    $text = $r.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    return [pscustomobject]@{ http = [int]$r.StatusCode; body = ($text | ConvertFrom-Json) }
  } finally {
    if ($null -ne $r) { $r.Dispose() }
    $req.Dispose(); $client.Dispose()
  }
}
function Test-MFEnvironment($Result) {
  $m = $Result.body.meta; $f = $Result.body.features
  if ($Result.http -ne 200 -or $m.simulation -isnot [bool] -or $m.simulation -ne $true -or
      $f.real_model -isnot [bool] -or $f.real_model -ne $false -or
      $f.delivery -isnot [bool] -or $f.delivery -ne $false) {
    Write-Warning '服务未声明模拟、可能使用真实模型或允许投递；停止模拟接入。'
    return $false
  }
  if ($m.contract_version -notmatch '^0\.3\.[0-9]+-draft$') {
    Write-Warning "客户端支持 0.3.x-draft，服务版本为 $($m.contract_version)；请核对迁移说明。"
    return $false
  }
  return $true
}
$meta = Invoke-MF GET '/meta'
if (-not (Test-MFEnvironment $meta)) { throw '模拟服务或契约版本不匹配，停止。' }
$sid = 'guide_' + [guid]::NewGuid().ToString('N')
$user = 'guide_' + [guid]::NewGuid().ToString('N')
$fixtures = Get-Content .\examples\r5_api_fixtures.json -Raw -Encoding UTF8 | ConvertFrom-Json
```
<!-- smoke:end -->

### 4.1 创建会话

<!-- smoke:begin -->
```powershell
$created = Invoke-MF POST '/sessions' @{ user_id = $user; session_id = $sid }
$created | ConvertTo-Json -Depth 10
```
<!-- smoke:end -->

期望：201，`confirmation_state=not_asked`，`can_request_confirmation=true`。App 提供由用户主动发起确认的入口；不要因创建成功就自动弹窗。fixture：`new_session_pending`。重启恢复可能也为 201，不宜仅凭 201 推断全新。

### 4.2 用户主动确认

<!-- smoke:begin -->
```powershell
$opened = Invoke-MF POST "/sessions/$sid/questions" @{ operation='open'; question_id='q1'; kind='safety_confirmation' }
$shown = Invoke-MF POST "/sessions/$sid/questions" @{ operation='present'; question_id='q1' }
$answered = Invoke-MF POST "/sessions/$sid/questions" @{ operation='answer'; question_id='q1'; answer='none' }
$opened; $shown; $answered
```
<!-- smoke:end -->

期望依次为 `open`、已展示、`confirmed`。`present` 仅在客户端确实向用户展示后发送；`none` 只能来自用户明确回答。不要把 `open` 或未展示跳过当作回答。fixtures：`confirmation_open`、`confirmed`、`confirmation_closed`。

### 4.3 正常事件与查询

<!-- smoke:begin -->
```powershell
$event = Get-Content .\examples\set_completed.json -Raw -Encoding UTF8 | ConvertFrom-Json
$event.user_id = $user; $event.session_id = $sid; $event.event_id = 'guide_event'
$event.discomfort.status = 'none'
$submit = @{ event = $event; do_not_disturb = $false }
$sent = Invoke-MF POST "/sessions/$sid/events" $submit
$queried = Invoke-MF GET "/sessions/$sid/events/guide_event"
$sent; $queried
```
<!-- smoke:end -->

期望首次提交 `event_saved=true`，查询中 `event_status/event_code` 是历史结果。`record_state=decided` 只说明有历史决策记录，可能是 stopped/pending；不表示已投递。不要展示候选。fixture：`reviewed`。

### 4.4 重复提交

<!-- smoke:begin -->
```powershell
$duplicate = Invoke-MF POST "/sessions/$sid/events" $submit
$duplicate
```
<!-- smoke:end -->

期望 200 `duplicate_event`，不再次调用模型。同一组合键改内容会 409 `event_conflict`；不要在冲突时无限重试。fixtures：`duplicate_event`、`event_conflict`。

### 4.5 待确认时提交

<!-- smoke:begin -->
```powershell
$pendingSid = 'pending_' + [guid]::NewGuid().ToString('N')
Invoke-MF POST '/sessions' @{ user_id = $user; session_id = $pendingSid }
$pendingEvent = $event | ConvertTo-Json -Depth 30 | ConvertFrom-Json
$pendingEvent.session_id = $pendingSid; $pendingEvent.event_id = 'pending_event'
$pending = Invoke-MF POST "/sessions/$pendingSid/events" @{ event=$pendingEvent; do_not_disturb=$false }
$pending
```
<!-- smoke:end -->

期望 200 `pending_confirmation`、`event_saved=true`，该事件不调用模型。后续确认也不会自动重跑该事件；不要把这条记录误当作可投递反馈。fixture：`pending_event`。

### 4.6 不适与保存失败

<!-- smoke:begin -->
```powershell
$safeSid = 'safe_' + [guid]::NewGuid().ToString('N')
Invoke-MF POST '/sessions' @{ user_id = $user; session_id = $safeSid }
$safety = Invoke-MF POST "/sessions/$safeSid/questions" @{ operation='answer'; question_id='safety_report'; answer='reported' }
$safety
$fixtures | Where-Object name -eq 'reported_volatile' | Select-Object name, preconditions, response
```
<!-- smoke:end -->

前两条是实际模拟调用，期望 200 `safety_pending`；第三条只是读取**故障注入后的静态 fixture**，不是现场制造数据库故障。`volatile` 表示报告写入未确认；App 不应将其显示为“已安全保存”。fixture：`reported_volatile`。

### 4.7 故障响应（离线 fixture）

<!-- smoke:begin -->
```powershell
$fixtures | Where-Object { $_.name -in @('storage_failed','safety_state_unknown','server_busy') } |
  Select-Object name, preconditions, fault, response
```
<!-- smoke:end -->

这是读取三条现有 fixture；真实 503 需对应故障条件，warning 可能随场景变化。App 必须读取错误 body 中的 `display`，不能只按状态码隐藏安全提示。不要故意破坏现有数据库来演示。

### 4.8 未知会话的不适信号

<!-- smoke:begin -->
```powershell
$ghost = 'ghost_' + [guid]::NewGuid().ToString('N')
$missing = Invoke-MF GET "/sessions/$ghost"
$unknownAnswer = Invoke-MF POST "/sessions/$ghost/questions" @{ operation='answer'; question_id='q1'; answer='reported' }
$ghostEvent = $event | ConvertTo-Json -Depth 30 | ConvertFrom-Json
$ghostEvent.session_id = $ghost; $ghostEvent.discomfort.status = 'reported'
$unknownEvent = Invoke-MF POST "/sessions/$ghost/events" @{ event=$ghostEvent; do_not_disturb=$false }
$badKnownAnswer = Invoke-MF POST "/sessions/$sid/questions" @{ operation='answer'; answer='reported' }
$missing; $unknownAnswer; $unknownEvent; $badKnownAnswer
```
<!-- smoke:end -->

普通查询是 404；另三项是 400 `safety_report_unattributed`，有提示但 `event_saved=false`，不写任何会话。最后一项模拟已存在会话但答案缺少 `question_id`。App 应按经审查的流程提供本地提示，并在正确会话重新报告；不要误认为报告已记录。fixtures：`unknown_session_reported_event`、`unknown_session_reported_answer`、`known_session_malformed_reported_answer`。

### 4.9 版本门禁和不匹配演示

示例客户端固定接受 `0.3.0-draft`；上面初始化代码的门禁示意接受 `0.3.x-draft`，并检查 `simulation=true`、`real_model=false`、`delivery=false`。此前 `0.2.2-draft` 不通过此门禁；迁移原因见[兼容规则](CONTRACT_CHANGELOG.md)。

<!-- smoke:begin -->
```powershell
$meta = Invoke-MF GET '/meta'
$versionOK = Test-MFEnvironment $meta
$fakeMeta = [pscustomobject]@{ http=200; body=[pscustomobject]@{
  meta=[pscustomobject]@{ simulation=$true; contract_version='0.4.0-draft' }
  features=[pscustomobject]@{ real_model=$false; delivery=$false }
} }
$versionBad = Test-MFEnvironment $fakeMeta
$versionOK; $versionBad
```
<!-- smoke:end -->

期望第一项 `True`，第二项 `False` 并出现提示。第二项是本地虚构的版本字符串，不是当前服务响应，因此没有对应服务 fixture；`/meta` 的实际示例以契约文件为准。版本不兼容或模拟标志不符时不要继续普通模拟对接；`/meta` 版本和能力是自报，不是身份认证。连接失败或安全信号的本地 UX 由产品/专业负责人确定。

## 5. 安全、身份和数据边界

- Agent 的字段校验、引用检查及模型结构检查不证明模型文字正确或安全；任何普通候选都不可直接展示、播报或投递。
- “确认无不适”只能来自有效展示后的明确回答，跳过、过期或 unknown 都不解除；重启和新会话按现行策略重新确认，不自动重问。
- 单协调器的状态检查不能约束旧入口、其他进程、设备或网络。真实账号、授权、加密、数据保留、备份与物理擦除未完成。
- 服务和 doctor 数据库均为本地明文模拟文件，工作区可能被 OneDrive 同步；本地其他程序也可访问接口和文件。禁止放入真实健康数据、个人信息和密钥。
- 当前 `0.3.0-draft` 是对 `0.2.2-draft` 的进一步安全修订，本次独立复核在本地模拟范围有条件通过；不代表真实场景或 App 联调通过。

## 6. 建议分工与决策

下表均为 **proposed / 待确认**，非团队决定；不确定的负责人和日期由双方填写。完整问题及影响见[待确认清单](R5_App_待确认问题清单.md)，产品与安全讨论见[决策卡](R5_App_产品安全决策卡.md)。

| 事项 | Agent 侧建议职责 | App 侧建议职责 | 决策人 / 时间 |
|---|---|---|---|
| 事件和指标 | 提供模拟结构、说明缺失与单位边界 | 核对采集、动作理解来源和质量 | 双方待定 |
| 账号与设备 | 说明模拟 ID 不具鉴权 | 提供真实身份和设备绑定方案 | App / 安全负责人待定 |
| 展示与交互 | 提供处理状态及候选限制 | 设计主动确认、跳过、过期和显示状态 | App / 产品待定 |
| 安全与升级 | 说明程序保护边界及未归属报告 | 提出本地兜底和重新报告流程 | **双方 / 产品 / 专业人员联合审查，待定** |
| 数据与删除 | 说明模拟保存与删除局限 | 确定真实保留、删除及备份处理 | 双方 / 合规待定 |
| 版本和错误 | 提供契约、示例、兼容变更记录 | 实现版本门禁、错误提示及重试 | 双方待定 |

## 7. 对接前必须确认的事

1. 正式联调时由谁提供动作完成事件、运动清单、指标单位、质量、时间和重复标识？尚未核实的 Human+ 能力不能作为已实现能力。
2. App 如何在连接失败、报告未保存、会话不可用时向用户提供经专业审查的本地安全提示与重新报告？谁负责紧急升级流程？
3. 真实身份及访问授权、跨进程安全、消息时效和投递前复核如何实现？
4. 谁签署 201/200、`record_state`、超时 400、范围冲突错误码等 P2 语义？何时冻结？
5. 谁运行完整 OpenAPI 3.1 验证器、评审提示文字并确认测试数据/环境？

这些问题没有确定负责人或日期；不应把建议表当成团队承诺。本地模拟复核通过不能替代双方正式联调确认。

### 材料定稿条件

本 Markdown 可供 App 团队评审，双方尚未确认契约和分工。定稿前需完成内容评审、完整 OpenAPI 3.1 验证器验证，以及产品/专业安全审查；macOS/Linux 上的服务启动还需在相应系统实测。用户确认定稿后再导出 PDF，并检查 8–12 页目标与版面；Markdown 本身没有固定页数。

## 8. 相关资料

- [契约和错误示例](openapi.r5-draft.yaml)；[fixtures](../examples/r5_api_fixtures.json)；[版本兼容与变更](CONTRACT_CHANGELOG.md)。
- [项目一页通](MetaFit_Agent_项目一页通.md)；[待确认清单](R5_App_待确认问题清单.md)；[开发者速查卡](R5_App_开发者速查卡.md)。
- 本地模拟复核不代表 App 团队确认或真实场景验收。
