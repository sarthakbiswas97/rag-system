from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(default="")

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v.strip()


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v.strip()


class TenantOut(BaseModel):
    id: str
    name: str
    email: str
    status: str
    embedding_model_version: str | None
    created_at: str
    updated_at: str


class CreateTenantResponse(BaseModel):
    tenant: TenantOut
    api_key: str = Field(..., description="Shown once at creation. Store it securely.")


class TenantListResponse(BaseModel):
    tenants: list[TenantOut]
    count: int


class UpdateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v.strip()


class TenantStatsOut(BaseModel):
    tenant_id: str
    chunk_count: int


class UsageEventOut(BaseModel):
    event_type: str
    value: int
    elapsed_ms: float | None
    created_at: str


class UsageSummaryOut(BaseModel):
    tenant_id: str
    period: str
    totals: dict[str, int]
    recent: list[UsageEventOut]
