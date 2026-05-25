from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v.strip()


class TenantOut(BaseModel):
    id: str
    name: str
    status: str
    embedding_model_version: str | None
    created_at: str
    updated_at: str


class CreateTenantResponse(BaseModel):
    tenant: TenantOut
    api_key: str = Field(
        ..., description="Shown once at creation. Store it securely."
    )


class TenantListResponse(BaseModel):
    tenants: list[TenantOut]
    count: int
