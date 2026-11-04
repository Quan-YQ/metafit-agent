"""只向本机模拟服务发送模拟事件；先按模块说明启动服务。

本示例固定验证 0.3.0-draft，并非支持 0.3.x 的兼容客户端模板。
"""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}/sim/v0"
    sid = "demo_" + uuid4().hex
    event = json.loads((Path(__file__).parent / "set_completed.json").read_text(encoding="utf-8"))
    event.update(session_id=sid, event_id="demo_event", discomfort={"status": "none", "source": "self_report"})

    def call(method, path, body=None):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urlopen(req, timeout=5) as response:
                result = json.load(response)
        except HTTPError as exc:
            print("HTTP 错误", exc.code, exc.read().decode("utf-8", errors="replace"))
            raise SystemExit(1) from None
        if result.get("meta", {}).get("contract_version") != "0.3.0-draft" or result.get("meta", {}).get("simulation") is not True or result.get("candidate", {}).get("may_display") is not False:
            raise SystemExit("契约版本或候选展示边界不匹配，停止演示。")
        print(method, path, "→", result["status"], result["code"], result["required_action"])
        return result

    meta = call("GET", "/meta")
    if (meta["meta"]["contract_version"] != "0.3.0-draft" or meta["meta"].get("simulation") is not True or
            not isinstance(meta.get("features"), dict) or meta["features"].get("real_model") is not False or
            meta["features"].get("delivery") is not False):
        raise SystemExit("契约版本不匹配或服务未声明仅模拟，停止演示。")
    call("POST", "/sessions", {"user_id": event["user_id"], "session_id": sid})
    route = "/sessions/" + sid
    call("GET", route)
    # 这是脚本明确模拟用户操作，不代表真实用户已看到或回答。
    call("POST", route + "/questions", {"operation": "open", "question_id": "q1", "kind": "safety_confirmation"})
    call("POST", route + "/questions", {"operation": "present", "question_id": "q1"})
    call("POST", route + "/questions", {"operation": "answer", "question_id": "q1", "answer": "none"})
    call("POST", route + "/events", {"event": event, "do_not_disturb": False})
    call("GET", route + "/events/demo_event")
    call("POST", route + "/events", {"event": event, "do_not_disturb": False})


if __name__ == "__main__":
    main()
