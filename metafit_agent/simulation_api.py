"""R5 本机模拟 HTTP 接口：无鉴权、无真实模型、无投递，不用于生产。"""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import logging
from pathlib import Path
import re
import socket
import threading
import sys
import time
from urllib.parse import urlsplit

from .event_store import StorageError
from .interaction_state import SAFETY_MESSAGE
from .model_response import ModelResponse
from .safety_coordinator import DurableSafetyCoordinator
from .safety_store import SafetySQLiteStore
from .training_events import validate_training_event

VERSION = "0.3.0-draft"
MAX_BODY = 65536
MAX_CONCURRENT = 4
TIMEOUT_SECONDS = 5
CONFIRMATION_SECONDS = 120
ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
UNKNOWN_MESSAGE = "当前安全状态无法确认，请先停止当前动作，并按需寻求专业帮助。"
PENDING_MESSAGE = "请由用户主动发起确认。"
CLOSED_MESSAGE = "本会话的确认已结束；需要继续时，请主动开始新会话。"
UNATTRIBUTED_MESSAGE = "收到不适信号，但报告信息不完整或无法确认其所属会话，报告未记录。请先停止当前动作，并使用正确会话重新报告不适。"
PUBLIC_CODES = {"pending_confirmation", "confirmation_closed", "duplicate_event", "event_conflict", "session_exists",
                "question_conflict", "invalid_request", "invalid_event", "state_missing", "event_missing",
                "request_too_large", "storage_failed", "safety_state_unknown", "server_busy", "method_not_allowed",
                "uri_too_long", "decision_stopped", "session_invalidated", "confirmation_not_needed",
                "question_already_presented", "question_already_created", "safety_report_unattributed"}
LOGGER = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _response(status="stopped", code=None, action="none", kind="unavailable", text="模拟结果不可投递。"):
    return {"status": status, "code": code, "required_action": action,
            "display": {"kind": kind, "text": text, "expires_at": None},
            "candidate": {"text": None, "may_display": False},
            "meta": {"contract_version": VERSION, "simulation": True},
            "confirmation_state": None, "can_request_confirmation": False, "question_id": None, "question_presented": None,
            "safety_state_durability": None, "event_saved": None, "record_state": None, "event_status": None, "event_code": None,
            "warnings": [], "features": None, "limits": None}


def _error(code, status=400):
    text = {"invalid_request": "请求无效。", "invalid_event": "模拟事件无效。",
            "state_missing": "未找到模拟会话。", "event_missing": "未找到模拟事件。",
            "session_exists": "模拟会话已存在。", "event_conflict": "事件内容冲突。",
            "question_conflict": "确认操作与当前状态冲突。", "request_too_large": "请求体过大。",
            "method_not_allowed": "该路径不支持此方法。", "uri_too_long": "请求地址过长。"}.get(code, "请求无法处理。")
    if status == 503:
        return status, _response(code=code, kind="safety_unknown", text=UNKNOWN_MESSAGE)
    return status, _response(code=code, kind="error", text=text)


def _unattributed_report():
    body = _response(code="safety_report_unattributed", action="report_discomfort", kind="safety_unknown", text=UNATTRIBUTED_MESSAGE)
    body["event_saved"] = False
    body["warnings"] = ["unattributed_discomfort"]
    return 400, body


def _allowed_methods(path):
    if path == "/sim/v0/meta": return ("GET",)
    if path == "/sim/v0/sessions": return ("POST",)
    if re.fullmatch(r"/sim/v0/sessions/[A-Za-z0-9_-]{1,64}", path): return ("GET",)
    if re.fullmatch(r"/sim/v0/sessions/[A-Za-z0-9_-]{1,64}/(?:events|questions)", path): return ("POST",)
    if re.fullmatch(r"/sim/v0/sessions/[A-Za-z0-9_-]{1,64}/events/[A-Za-z0-9_-]{1,64}", path): return ("GET",)
    return ()


def _unattributed_signal(method, parts, data):
    if method != "POST" or len(parts) != 6 or not isinstance(data, dict):
        return False
    event = data.get("event")
    return ((parts[5] == "events" and isinstance(event, dict) and
             isinstance(event.get("discomfort"), dict) and event["discomfort"].get("status") == "reported") or
            (parts[5] == "questions" and data.get("operation") == "answer" and data.get("answer") == "reported"))


class OfflineModel:
    """固定 silent，只验证管线；不联网，也不生成可投递文本。"""
    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, facts):
        with self._lock:
            self.calls += 1
        status = next((fact.value for fact in facts if fact.name == "discomfort_status"), "unknown")
        reason = "不适状态未知，仅依据当前模拟事实。" if status == "unknown" else "仅为离线模拟。"
        content = json.dumps({"suggested_action": "silent", "message": None,
                              "reason": reason, "evidence": []}, ensure_ascii=False)
        return ModelResponse(content, "offline", "offline", None, None, 0, "stop")


class SimulationAPI:
    def __init__(self, store, model=None):
        self.store = store
        self.coordinator = DurableSafetyCoordinator(store)
        self.model = model if model is not None else OfflineModel()

    def _snapshot(self, sid):
        snap = self.coordinator.confirmation_view(sid)
        # 只过期指定的安全确认，GET 不会展示问题或推进其他类型的问题。
        if snap and snap["question_id"] and snap["deadline"] and _now() > snap["deadline"]:
            self.coordinator.apply(sid, "expire_safety_confirmation", question_id=snap["question_id"], now=_iso(_now()), clock_id="api_clock")
            snap = self.coordinator.confirmation_view(sid)
        return snap

    def _state(self, sid, *, status=None, code=None, event_saved=None, record_state=None, warnings=()):
        snap = self._snapshot(sid)
        if snap is None:
            return _error("state_missing", 404)
        view = snap["view"]
        state = snap["confirmation_state"]
        if view.safety_reported:
            body = _response("safety_pending", code, "report_discomfort", "safety_pending", SAFETY_MESSAGE)
            http = 200
        elif state == "unknown":
            http, body = _error("safety_state_unknown", 503)
        elif state == "closed":
            http, body = 200, _response("pending_confirmation", "confirmation_closed", "none", "confirmation_closed", CLOSED_MESSAGE)
        elif view.pending_confirmation:
            http, body = 200, _response("pending_confirmation", code or "pending_confirmation", "confirm", "pending_confirmation", PENDING_MESSAGE)
        else:
            http, body = 200, _response(status or "reviewed", code)
        if body["code"] is not None and body["code"] not in PUBLIC_CODES:
            body["code"] = "decision_stopped"
        body.update(confirmation_state=state, can_request_confirmation=state == "not_asked" and view.pending_confirmation,
                    question_id=snap["question_id"], question_presented=snap["question_presented"],
                    safety_state_durability=view.safety_state_durability, event_saved=event_saved,
                    record_state=record_state, warnings=list(warnings))
        if state == "open":
            body["display"]["text"] = "确认问题已展示，请等待回答。" if snap["question_presented"] else "确认问题已创建，等待用户主动展示。"
        if snap["deadline"]:
            body["display"]["expires_at"] = _iso(snap["deadline"])
        return http, body

    def dispatch(self, method, path, data=None):
        if path == "/sim/v0/meta" and method == "GET":
            body = _response("ready", None)
            body["features"] = {"safety_confirmation": True, "event_deduplication": True, "real_model": False, "delivery": False}
            body["limits"] = {"max_body_bytes": MAX_BODY, "events_per_request": 1, "max_concurrent": MAX_CONCURRENT,
                              "read_timeout_seconds": TIMEOUT_SECONDS, "confirmation_seconds": CONFIRMATION_SECONDS}
            return 200, body
        if path == "/sim/v0/sessions" and method == "POST":
            if not isinstance(data, dict) or set(data) != {"user_id", "session_id"} or not all(isinstance(data[k], str) and ID.fullmatch(data[k]) for k in data):
                return _error("invalid_request")
            result = self.coordinator.register(user_id=data["user_id"], session_id=data["session_id"], clock_id="api_clock", new_session=True)
            if result.code == "session_exists":
                return _error("session_exists", 409)
            if result.code == "scope_mismatch":
                return _error("question_conflict", 409)
            http, body = self._state(data["session_id"], warnings=result.warnings)
            return (201 if http == 200 and body["status"] != "safety_pending" else http), body
        parts = path.split("/")
        if len(parts) < 5 or parts[:4] != ["", "sim", "v0", "sessions"]:
            return _error("state_missing", 404)
        sid = parts[4]
        if not ID.fullmatch(sid):
            return _unattributed_report() if _unattributed_signal(method, parts, data) else _error("state_missing", 404)
        if method == "GET" and len(parts) == 5:
            return self._state(sid)
        session = self.coordinator.confirmation_view(sid)
        if session is None or not session["view"].active:
            # 输入不能归属到可用会话时，只返回不适提示，不创建会话或写入任何记录。
            if _unattributed_signal(method, parts, data):
                return _unattributed_report()
            if session is None:
                return _error("state_missing", 404)
        if method == "POST" and len(parts) == 6 and parts[5] == "questions":
            return self._question(sid, data)
        if method == "POST" and len(parts) == 6 and parts[5] == "events":
            return self._event(sid, data)
        if method == "GET" and len(parts) == 7 and parts[5] == "events" and ID.fullmatch(parts[6]):
            return self._get_event(sid, parts[6])
        return _error("state_missing", 404)

    def _question(self, sid, data):
        # 明确不适即使问题信封不完整也不能退化为没有提示的普通错误。
        reported = isinstance(data, dict) and data.get("operation") == "answer" and data.get("answer") == "reported"
        def invalid():
            return _unattributed_report() if reported else _error("invalid_request")
        if not isinstance(data, dict) or not set(data).issubset({"operation", "question_id", "kind", "answer"}):
            return invalid()
        op, qid = data.get("operation"), data.get("question_id")
        if not isinstance(op, str) or op not in {"open", "present", "answer", "skip"} or not isinstance(qid, str) or not ID.fullmatch(qid):
            return invalid()
        if op == "open" and (data.get("kind") != "safety_confirmation" or "answer" in data):
            return invalid()
        if op == "answer" and (not isinstance(data.get("answer"), str) or data.get("answer") not in {"none", "unknown", "reported"} or "kind" in data):
            return invalid()
        if op in {"present", "skip"} and set(data) != {"operation", "question_id"}:
            return invalid()
        snap = self._snapshot(sid)
        if snap["confirmation_state"] == "unknown" and not (op == "answer" and data.get("answer") == "reported"):
            return self._state(sid)
        now = _now()
        args = {"question_id": qid, "now": _iso(now), "clock_id": "api_clock"}
        if op == "open":
            args.update(kind="safety_confirmation", deadline=_iso(now + timedelta(seconds=CONFIRMATION_SECONDS)))
        if op == "answer":
            args["answer"] = data["answer"]
        result = self.coordinator.apply(sid, {"open": "open_question", "present": "present_question", "answer": "answer_question", "skip": "skip_question"}[op], **args)
        http, body = self._state(sid, status=result.status, code=result.code, warnings=result.warnings)
        if http == 200 and body["status"] != "safety_pending" and result.status not in {"safety_pending", "expired", "skipped", "answered", "created", "presented", "duplicate"} and body["confirmation_state"] != "closed":
            body.update(status="stopped", code="question_conflict")
            body["display"]["text"] = "确认操作与当前状态冲突；请查看当前确认状态。"
            return 409, body
        if http == 200 and result.status == "presented" and body["confirmation_state"] == "open":
            body["display"].update(kind="question", text=result.question_text)
        return http, body

    def _event(self, sid, data):
        event = data.get("event") if isinstance(data, dict) else None
        reported = isinstance(event, dict) and isinstance(event.get("discomfort"), dict) and event["discomfort"].get("status") == "reported"
        if not isinstance(data, dict) or set(data) != {"event", "do_not_disturb"} or not isinstance(data["event"], dict) or type(data["do_not_disturb"]) is not bool:
            return _unattributed_report() if reported else _error("invalid_request")
        snap = self._snapshot(sid)
        event = data["event"]
        # 不匹配的信号给出安全提示，但不能把它归属给 URL 所指的其他用户。
        if reported and (event.get("user_id") != snap["view"].user_id or event.get("session_id") != sid):
            return _unattributed_report()
        if not reported and (validate_training_event(event).event is None or
                             not isinstance(event.get("event_id"), str) or not ID.fullmatch(event["event_id"]) or
                             event.get("schema_version") != "0.1" or event.get("event_type") != "set_completed" or
                             event.get("source") != "simulation"):
            return _error("invalid_event")
        if not reported and snap["confirmation_state"] == "unknown":
            return self._state(sid)
        result = self.coordinator.decide(sid, event, self.model, do_not_disturb=data["do_not_disturb"])
        if result.status == "safety_pending":
            return self._state(sid, status=result.status, code=result.code, event_saved=result.event_saved, warnings=result.warnings)
        current = self.coordinator.confirmation_view(sid)
        if current and current["view"].safety_reported:
            return self._state(sid, event_saved=result.event_saved, warnings=result.warnings)
        if result.storage_status in {"unknown", "saved_state_unconfirmed"} or result.code == "storage_failed":
            http, body = _error("storage_failed", 503)
            body["event_saved"] = result.event_saved
            body["warnings"] = list(result.warnings)
            return http, body
        if result.storage_status == "conflict" or result.code == "event_conflict":
            return _error("event_conflict", 409)
        if result.code == "scope_mismatch":
            return _error("invalid_event")
        if result.storage_status == "invalid_event":
            return _error("invalid_event")
        duplicate = result.storage_status == "existing" or result.code == "duplicate_event"
        record_state = None
        warnings = result.warnings
        if result.event_saved is True:
            try:
                rows = self.store.query(snap["view"].user_id, session_id=sid, event_id=event["event_id"], limit=1)
                record_state = rows[0]["record_state"] if rows else None
            except (StorageError, KeyError, ValueError):
                warnings = tuple(dict.fromkeys((*warnings, "record_state_unavailable")))
        return self._state(sid, status=result.status, code="duplicate_event" if duplicate else result.code,
                           event_saved=result.event_saved, record_state=record_state, warnings=warnings)

    def _get_event(self, sid, eid):
        snap = self._snapshot(sid)
        try:
            record = self.store.get(snap["view"].user_id, sid, eid)
            if record is None:
                return _error("event_missing", 404)
            rows = self.store.query(snap["view"].user_id, session_id=sid, event_id=eid, limit=1)
        except StorageError:
            if snap["view"].safety_reported:
                return self._state(sid, warnings=("storage_failed",))
            return _error("storage_failed", 503)
        row = rows[0] if rows else None
        state = row["record_state"] if row else None
        http, body = self._state(sid, status=row["status"] if row else "stopped", code=row["code"] if row else None,
                                 event_saved=True, record_state=state)
        body["event_status"] = row["status"] if row else None
        body["event_code"] = row["code"] if row else None
        return http, body


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, api):
        self.api = api
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)
        super().__init__(("127.0.0.1", port), Handler)

    def get_request(self):
        conn, addr = super().get_request()
        conn.settimeout(TIMEOUT_SECONDS)
        return conn, addr

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            status, body = _error("server_busy", 503)
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            try:
                # 忙时仅限时读取方法，避免 HEAD 被误发送正文；无法确认方法时保守不发正文。
                deadline = time.monotonic() + 0.15
                method = bytearray()
                complete = False
                while len(method) < 16:
                    remaining_time = deadline - time.monotonic()
                    if remaining_time <= 0:
                        break
                    request.settimeout(remaining_time)
                    try:
                        part = request.recv(1)
                    except socket.timeout:
                        break
                    if not part:
                        break
                    if part == b" ":
                        complete = True
                        break
                    method.extend(part)
                is_head = complete and method == b"HEAD"
                response_body = payload if complete and not is_head else b""
                content_length = len(payload) if is_head else len(response_body)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: " + str(content_length).encode() + b"\r\n\r\n" + response_body)
                # 发送后先半关闭写端，并短暂丢弃已到达的请求数据；避免 Windows
                # 因未读请求体而复位连接，使客户端收不到已发送的 503。
                request.shutdown(socket.SHUT_WR)
                request.settimeout(0.15)
                remaining = MAX_BODY + 8192
                while remaining > 0:
                    chunk = request.recv(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # 异常类型可供排障；不记录请求、地址、事件、数据库路径或堆栈。
        error_type = sys.exc_info()[0]
        LOGGER.error("模拟 HTTP 连接异常：%s", error_type.__name__ if error_type else "unknown")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 禁止日志回显用户输入

    def _send(self, status, body, allow=None):
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if allow:
            self.send_header("Allow", ", ".join(allow))
        self.end_headers()
        if getattr(self, "command", None) != "HEAD":
            self.wfile.write(payload)

    def send_error(self, code, message=None, explain=None):
        # 标准库在解析请求前可能尚未设置版本；仍发送明确的 HTTP JSON 错误。
        if getattr(self, "request_version", "HTTP/0.9") == "HTTP/0.9":
            self.request_version = "HTTP/1.0"
        if code == 414:
            return self._send(*_error("uri_too_long", 414))
        if code == 501:
            return self._method_not_allowed()
        return self._send(*_error("invalid_request", 400))

    def _method_not_allowed(self):
        allowed = _allowed_methods(urlsplit(getattr(self, "path", "")).path)
        if not allowed:
            return self._send(*_error("state_missing", 404))
        return self._send(*_error("method_not_allowed", 405), allow=allowed)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method):
        try:
            # 避免浏览器跨站请求和 DNS 重绑定把无鉴权模拟服务暴露给网页。
            host = self.headers.get("Host", "")
            if (host != "127.0.0.1" and host != f"127.0.0.1:{self.server.server_port}") or self.headers.get("Origin"):
                return self._send(*_error("invalid_request"))
            if urlsplit(self.path).query or urlsplit(self.path).fragment:
                return self._send(*_error("invalid_request"))
            allowed = _allowed_methods(urlsplit(self.path).path)
            if allowed and method not in allowed:
                return self._method_not_allowed()
            data = None
            if method == "POST":
                if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/json":
                    return self._send(*_error("invalid_request"))
                length = self.headers.get("Content-Length")
                if not length or not length.isascii() or not length.isdigit():
                    return self._send(*_error("invalid_request"))
                if int(length) > MAX_BODY:
                    return self._send(*_error("request_too_large", 413))
                raw = self.rfile.read(int(length))
                def pairs(items):
                    result = {}
                    for key, value in items:
                        if key in result:
                            raise ValueError("duplicate field")
                        result[key] = value
                    return result
                def bad_constant(_):
                    raise ValueError("invalid number")
                data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
            self._send(*self.server.api.dispatch(method, urlsplit(self.path).path, data))
        except (ValueError, UnicodeError, RecursionError, socket.timeout):
            try:
                self._send(*_error("invalid_request"))
            except OSError:
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            LOGGER.error("模拟 HTTP 请求异常：%s", type(exc).__name__)
            try:
                self._send(*_error("safety_state_unknown", 503))
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="仅限 127.0.0.1 的 MetaFit 模拟服务，无鉴权，不用于真实数据。")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=Path("outputs/r5_simulation/api_demo.db"))
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    server = LocalServer(args.port, SimulationAPI(SafetySQLiteStore(args.db)))
    print(f"模拟服务: http://127.0.0.1:{server.server_port}/sim/v0/meta", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
