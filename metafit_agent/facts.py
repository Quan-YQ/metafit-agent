"""模拟事实及其单次决策范围，不提供真实身份或来源认证。"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    name: str
    value: str | int | float
    user_id: str
    session_id: str
    event_id: str
    field: str
    origin: str = "event"
    kind: str = "current_event"

    @property
    def local_reference(self) -> tuple[str, str, str, str, str]:
        return (self.user_id, self.session_id, self.event_id, self.origin, self.field)


@dataclass(frozen=True, slots=True)
class DecisionScope:
    user_id: str
    session_id: str
    event_id: str
    allowed_origins: tuple[str, ...] = ("event",)
    allowed_history: tuple[tuple[str, str], ...] = ()  # (event_id, session_id)
