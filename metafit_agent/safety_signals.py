"""识别模拟输入中的不适信号；该识别不证明来源、身份或内容真实性。"""
from typing import Any


def inspect_discomfort_signal(event: Any) -> tuple[bool, bool]:
    """返回（明确 reported，来源字段是否符合自报格式）；不回显输入。"""
    report = event.get("discomfort") if isinstance(event, dict) else None
    if not isinstance(report, dict):
        return False, False
    status, source = report.get("status"), report.get("source")
    reported = isinstance(status, str) and status.strip() == "reported"
    source_matches = isinstance(source, str) and source.strip() == "self_report"
    return reported, source_matches
