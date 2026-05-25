from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
