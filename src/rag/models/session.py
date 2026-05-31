from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass(frozen=True)
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


@dataclass(frozen=True)
class Session:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    turns: tuple[ConversationTurn, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def add_turn(self, role: str, content: str) -> Session:
        turn = ConversationTurn(role=role, content=content)
        return Session(
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            turns=(*self.turns, turn),
            created_at=self.created_at,
        )

    @property
    def history_text(self) -> str:
        lines = [f"{t.role}: {t.content}" for t in self.turns]
        return "\n".join(lines)
