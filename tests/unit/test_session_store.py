from __future__ import annotations

from unittest.mock import MagicMock

from rag.models.session import Session
from rag.session.store import SessionStore, _dict_to_session, _session_to_dict


def _make_mock_redis() -> MagicMock:
    """Mock Redis client with in-memory dict storage."""
    client = MagicMock()
    storage: dict[str, str] = {}

    def mock_setex(key: str, ttl: int, value: str) -> None:
        storage[key] = value

    def mock_get(key: str) -> str | None:
        return storage.get(key)

    def mock_delete(key: str) -> int:
        if key in storage:
            del storage[key]
            return 1
        return 0

    client.setex = mock_setex
    client.get = mock_get
    client.delete = mock_delete
    return client


class TestSessionStore:
    def test_create_session(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        session = store.create(tenant_id="t1")

        assert session.session_id
        assert session.tenant_id == "t1"
        assert session.turns == ()

    def test_get_session(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        created = store.create(tenant_id="t1")

        retrieved = store.get(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.tenant_id == "t1"

    def test_get_nonexistent_returns_none(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        assert store.get("nonexistent") is None

    def test_add_turn(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        session = store.create()

        updated = store.add_turn(session.session_id, "user", "Hello")

        assert updated is not None
        assert len(updated.turns) == 1
        assert updated.turns[0].role == "user"
        assert updated.turns[0].content == "Hello"

    def test_add_multiple_turns(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        session = store.create()

        store.add_turn(session.session_id, "user", "Q1")
        store.add_turn(session.session_id, "assistant", "A1")
        updated = store.add_turn(session.session_id, "user", "Q2")

        assert updated is not None
        assert len(updated.turns) == 3

    def test_add_turn_nonexistent_returns_none(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        result = store.add_turn("nonexistent", "user", "Hello")
        assert result is None

    def test_delete_session(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        session = store.create()

        assert store.delete(session.session_id) is True
        assert store.get(session.session_id) is None

    def test_delete_nonexistent(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        assert store.delete("nonexistent") is False

    def test_max_turns_trims_oldest(self) -> None:
        store = SessionStore(client=_make_mock_redis(), max_turns=3)
        session = store.create()

        store.add_turn(session.session_id, "user", "Q1")
        store.add_turn(session.session_id, "assistant", "A1")
        store.add_turn(session.session_id, "user", "Q2")
        updated = store.add_turn(session.session_id, "assistant", "A2")

        assert updated is not None
        assert len(updated.turns) == 3
        assert updated.turns[0].content == "A1"

    def test_session_persists_across_gets(self) -> None:
        store = SessionStore(client=_make_mock_redis())
        session = store.create(tenant_id="t1")
        store.add_turn(session.session_id, "user", "Hello")

        retrieved = store.get(session.session_id)

        assert retrieved is not None
        assert len(retrieved.turns) == 1
        assert retrieved.tenant_id == "t1"


class TestSessionSerialization:
    def test_roundtrip(self) -> None:
        session = Session(tenant_id="t1")
        session = session.add_turn("user", "Hello")
        session = session.add_turn("assistant", "Hi there")

        data = _session_to_dict(session)
        restored = _dict_to_session(data)

        assert restored.session_id == session.session_id
        assert restored.tenant_id == session.tenant_id
        assert len(restored.turns) == 2
        assert restored.turns[0].role == "user"
        assert restored.turns[1].content == "Hi there"

    def test_empty_session_roundtrip(self) -> None:
        session = Session()
        data = _session_to_dict(session)
        restored = _dict_to_session(data)

        assert restored.session_id == session.session_id
        assert restored.turns == ()


class TestSessionModel:
    def test_add_turn_immutable(self) -> None:
        session = Session()
        updated = session.add_turn("user", "Hello")

        assert session.turns == ()
        assert len(updated.turns) == 1

    def test_history_text(self) -> None:
        session = Session()
        session = session.add_turn("user", "What is X?")
        session = session.add_turn("assistant", "X is Y.")

        assert session.history_text == "user: What is X?\nassistant: X is Y."

    def test_history_text_empty(self) -> None:
        session = Session()
        assert session.history_text == ""
