from __future__ import annotations

import enum
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TenantStatus(enum.StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class FineTuneStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(enum.StrEnum):
    EMBEDDING = "embedding"
    LLM = "llm"


class EventType(enum.StrEnum):
    QUERY = "query"
    INGEST = "ingest"


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), unique=False, nullable=False, default=""
    )
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    status: Mapped[TenantStatus] = mapped_column(
        Enum(TenantStatus), default=TenantStatus.ACTIVE, nullable=False
    )
    embedding_model_version: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(tz=UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
        nullable=False,
    )

    fine_tune_jobs: Mapped[list[FineTuneJob]] = relationship(
        back_populates="tenant", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_tenants_api_key_hash", "api_key_hash"),
        Index("ix_tenants_status", "status"),
    )


class FineTuneJob(Base):
    __tablename__ = "fine_tune_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False
    )
    status: Mapped[FineTuneStatus] = mapped_column(
        Enum(FineTuneStatus), default=FineTuneStatus.PENDING, nullable=False
    )
    model_type: Mapped[ModelType] = mapped_column(Enum(ModelType), nullable=False)
    config: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(tz=UTC), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    tenant: Mapped[Tenant] = relationship(back_populates="fine_tune_jobs")

    __table_args__ = (
        Index("ix_fine_tune_jobs_tenant_id", "tenant_id"),
        Index("ix_fine_tune_jobs_status", "status"),
    )


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False
    )
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    elapsed_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(tz=UTC), nullable=False
    )

    __table_args__ = (
        Index("ix_usage_events_tenant_id", "tenant_id"),
        Index("ix_usage_events_event_type", "event_type"),
        Index("ix_usage_events_created_at", "created_at"),
    )
