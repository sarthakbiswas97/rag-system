from __future__ import annotations

import json
import logging
from typing import Any

from redis import Redis

from rag.models.session import ConversationTurn, Session

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 3600  # 1 hour
SESSION_PREFIX = "session:"
MAX_TURNS = 20


def _session_key(session_id: str) -> str:
    return f"{SESSION_PREFIX}{session_id}"


def _session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "tenant_id": session.tenant_id,
        "created_at": session.created_at,
        "turns": [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
            }
            for t in session.turns
        ],
    }


def _dict_to_session(data: dict[str, Any]) -> Session:
    turns = tuple(
        ConversationTurn(
            role=t["role"],
            content=t["content"],
            timestamp=t["timestamp"],
        )
        for t in data.get("turns", [])
    )
    return Session(
        session_id=data["session_id"],
        tenant_id=data.get("tenant_id", ""),
        turns=turns,
        created_at=data.get("created_at", ""),
    )


class SessionStore:
    """Redis-backed conversational session store."""

    def __init__(
        self,
        client: Redis,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_turns: int = MAX_TURNS,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._max_turns = max_turns

    def create(self, tenant_id: str = "") -> Session:
        session = Session(tenant_id=tenant_id)
        self._save(session)

        logger.info(
            "Session created",
            extra={
                "session_id": session.session_id,
                "tenant_id": tenant_id,
            },
        )
        return session

    def get(self, session_id: str) -> Session | None:
        key = _session_key(session_id)
        raw = self._client.get(key)
        if raw is None:
            return None

        data = json.loads(raw)
        return _dict_to_session(data)

    def add_turn(
        self, session_id: str, role: str, content: str
    ) -> Session | None:
        session = self.get(session_id)
        if session is None:
            return None

        updated = session.add_turn(role, content)

        if len(updated.turns) > self._max_turns:
            trimmed_turns = updated.turns[-self._max_turns :]
            updated = Session(
                session_id=updated.session_id,
                tenant_id=updated.tenant_id,
                turns=trimmed_turns,
                created_at=updated.created_at,
            )

        self._save(updated)
        return updated

    def delete(self, session_id: str) -> bool:
        key = _session_key(session_id)
        deleted = self._client.delete(key)
        return deleted > 0

    def _save(self, session: Session) -> None:
        key = _session_key(session.session_id)
        data = json.dumps(_session_to_dict(session))
        self._client.setex(key, self._ttl, data)
