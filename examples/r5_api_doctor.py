"""零依赖本机模拟自检；只写入隔离的模拟数据，不做真实安全判断。"""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import sys
import threading
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metafit_agent.safety_store import SafetySQLiteStore
from metafit_agent.simulation_api import LocalServer, SimulationAPI

DOCTOR_DIR = ROOT / "outputs" / "r5_simulation" / "doctor"
MARKER_TEXT = "MetaFit-Agent isolated doctor database v1\n"
SUPPORTED = "0.3.x（从 0.3.0-draft 起）"


def compatible(version):
    match = re.fullmatch(r"0\.3\.(\d+)-draft", version) if isinstance(version, str) else None
    return bool(match)


def _safe_dir(path):
    """只接受指定目录；拒绝符号链接、联接点和替换的标记。"""
    base = path.parent
    if not base.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("doctor 目录必须位于项目内，不能经链接指向其他位置")
    base.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError("doctor 演示目录不能是符号链接或联接点")
    if path.exists() and (not path.is_dir() or path.resolve().parent != base.resolve()):
        raise ValueError("doctor 演示目录无效")
    marker = path / ".metafit-doctor"
    if path.exists():
        if marker.is_symlink() or not marker.is_file() or marker.read_text(encoding="utf-8") != MARKER_TEXT:
            raise ValueError("已有目录缺少有效 doctor 标记；不会清理或使用")
    else:
        path.mkdir()
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(MARKER_TEXT)
    for name in ("doctor.db", "doctor.db-wal", "doctor.db-shm"):
        target = path / name
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.stat().st_nlink > 1 or target.resolve().parent != path.resolve())):
            raise ValueError("doctor 库文件不能是链接、硬链接或其他目录中的文件")
    return marker


def _acquire_lock(path):
    lock = path / ".doctor.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("doctor 演示库正被使用或留有锁；请确认没有运行的 doctor") from None
    os.close(fd)
    return lock


def _reset(path):
    """仅删除明确枚举的 doctor 库文件；遇到未知文件或链接则停止。"""
    allowed = {".metafit-doctor", ".doctor.lock", "doctor.db", "doctor.db-wal", "doctor.db-shm"}
    entries = list(path.iterdir())
    if any(item.name not in allowed or item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("doctor 目录存在未知文件、链接或子目录；未删除任何文件")
    print("仅清理 doctor 专用演示库，输入 RESET 确认：", file=sys.stderr, flush=True)
    try:
        confirmation = input().strip()
    except EOFError:
        confirmation = ""
    if confirmation != "RESET":
        raise ValueError("未确认 RESET；未删除数据库")
    for name in ("doctor.db", "doctor.db-wal", "doctor.db-shm"):
        target = path / name
        # 重检真实路径，避免误删链接或目录。
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.stat().st_nlink > 1 or target.resolve().parent != path.resolve())):
            raise ValueError("目标不是 doctor 目录内的普通文件；停止清理")
    for name in ("doctor.db", "doctor.db-wal", "doctor.db-shm"):
        target = path / name
        if target.exists():
            target.unlink()


def _call(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    try:
        conn.request(method, "/sim/v0" + path, body=payload,
                     headers={"Content-Type": "application/json"} if payload is not None else {})
        resp = conn.getresponse()
        value = json.loads(resp.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("响应不是 JSON 对象")
        return resp.status, value
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="MetaFit 本机模拟接入自检；不处理真实用户数据。")
    parser.add_argument("--port", type=int, default=0, help="启动自有服务的端口；0 表示系统分配")
    parser.add_argument("--connect", type=int, metavar="PORT", help="连接已有的 127.0.0.1 模拟服务，会写入新的模拟记录")
    parser.add_argument("--reset", action="store_true", help="二次确认后只清理 doctor 专用库，不能与 --connect 同用")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出检查结果")
    args = parser.parse_args(argv)
    checks = []
    server = worker = lock = None

    def mark(name, ok, detail, next_step=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail, "next_step": next_step})
        if not ok:
            raise ValueError(detail)

    try:
        mark("python", sys.version_info[:2] == (3, 12), f"Python {sys.version_info.major}.{sys.version_info.minor}", "使用项目 Python 3.12 环境。")
        if args.connect is not None:
            if args.reset or args.port != 0:
                raise ValueError("--connect 不能与 --reset 或 --port 同用")
            if not 1 <= args.connect <= 65535:
                raise ValueError("连接端口须为 1–65535")
            port = args.connect
            mark("loopback", True, f"仅连接 127.0.0.1:{port}，将写入新的模拟记录")
        else:
            if not 0 <= args.port <= 65535:
                raise ValueError("监听端口须为 0–65535")
            _safe_dir(DOCTOR_DIR)
            lock = _acquire_lock(DOCTOR_DIR)
            if args.reset:
                _reset(DOCTOR_DIR)
            server = LocalServer(args.port, SimulationAPI(SafetySQLiteStore(DOCTOR_DIR / "doctor.db")))
            port = server.server_port
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            mark("loopback", server.server_address[0] == "127.0.0.1", f"自有模拟服务 127.0.0.1:{port}", "检查端口是否被占用。")
        status, meta = _call(port, "GET", "/meta")
        mark("meta", status == 200 and meta.get("meta", {}).get("simulation") is True,
             f"HTTP {status}，simulation={meta.get('meta', {}).get('simulation')}", "确认连接的是本机模拟服务。")
        version = meta.get("meta", {}).get("contract_version")
        mark("version", compatible(version), f"客户端支持 {SUPPORTED}；服务版本 {version}", "请核对契约变更记录并升级/降级；不要静默继续。")
        features, limits = meta.get("features"), meta.get("limits")
        expected = {"safety_confirmation": True, "event_deduplication": True, "real_model": False, "delivery": False}
        mark("capabilities", isinstance(features, dict) and all(features.get(k) is v for k, v in expected.items()) and isinstance(limits, dict) and
             all(type(limits.get(k)) is int and limits[k] > 0 for k in ("max_body_bytes", "events_per_request", "max_concurrent", "read_timeout_seconds", "confirmation_seconds")),
             f"features={features}，limits={limits}", "核对接口版本、功能和限制。")
        sid = "doctor_" + uuid4().hex
        user = "doctor_" + uuid4().hex
        status, created = _call(port, "POST", "/sessions", {"user_id": user, "session_id": sid})
        mark("session", status == 201 and created.get("confirmation_state") == "not_asked",
             f"创建模拟会话 HTTP {status}，状态 {created.get('confirmation_state')}", "检查模拟库可写与会话状态。")
        route = "/sessions/" + sid
        for op, params in [("open", {"kind": "safety_confirmation"}), ("present", {}), ("answer", {"answer": "none"})]:
            status, result = _call(port, "POST", route + "/questions", {"operation": op, "question_id": "doctor_q", **params})
            expected_state = "confirmed" if op == "answer" else "open"
            mark("confirmation_" + op, status == 200 and result.get("candidate", {}).get("may_display") is False and
                 result.get("confirmation_state") == expected_state and
                 (op != "present" or result.get("question_presented") is True),
                 f"{op}: HTTP {status}, state={result.get('confirmation_state')}", "检查安全确认与存储状态。")
        event = json.loads((ROOT / "examples" / "set_completed.json").read_text(encoding="utf-8"))
        event.update(user_id=user, session_id=sid, event_id="doctor_event", discomfort={"status": "none", "source": "self_report"})
        payload = {"event": event, "do_not_disturb": False}
        status, result = _call(port, "POST", route + "/events", payload)
        mark("event_saved", status == 200 and result.get("status") == "reviewed" and result.get("event_saved") is True and result.get("candidate", {}).get("may_display") is False,
             f"事件 HTTP {status}，event_saved={result.get('event_saved')}；仅验证模拟服务记录", "检查存储可写和模拟模型。")
        status, duplicate = _call(port, "POST", route + "/events", payload)
        mark("duplicate", status == 200 and duplicate.get("code") == "duplicate_event",
             f"重复事件 HTTP {status}，code={duplicate.get('code')}", "检查重复提交契约。")
        if args.connect is None:
            mark("database", (DOCTOR_DIR / "doctor.db").is_file(), f"仅自检专用文件：{DOCTOR_DIR / 'doctor.db'}", "不要放入真实用户数据。")
        else:
            mark("database", True, "通过 event_saved 验证服务接受模拟写入；无法验证其文件路径、物理持久性或部署权限。")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if not checks or checks[-1]["status"] != "FAIL":
            checks.append({"check": "execution", "status": "FAIL", "detail": str(exc),
                           "next_step": "检查端口、权限、模拟库和版本；不要使用真实数据。"})
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            if worker is not None:
                worker.join(timeout=5)
        if lock is not None:
            try: lock.unlink()
            except OSError: checks.append({"check": "lock_cleanup", "status": "FAIL", "detail": "自检锁无法移除。", "next_step": "确认没有运行中的 doctor 后手动排查。"})
    result = {"overall": "PASS" if checks and all(c["status"] == "PASS" for c in checks) else "FAIL",
              "simulation": True, "checks": checks}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(f"{item['status']}  {item['check']}: {item['detail']}")
            if item["status"] == "FAIL" and item["next_step"]:
                print("  下一步：" + item["next_step"])
        print("结果：" + result["overall"] + "（仅本机模拟；不代表真实安全、鉴权或投递）")
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
