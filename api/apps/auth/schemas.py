"""
Auth Pydantic schemas.

Input validation and output serialization for auth routes.
"""

from datetime import datetime
from typing import Optional
import uuid
from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Request Schemas ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Login with email + password."""
    email: EmailStr
    password: str = Field(..., min_length=8)


class RegisterRequest(BaseModel):
    """Register a new employee account."""
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    department: str
    role: str = Field(default="employee")

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        valid = ["Sales", "HR", "Finance", "Operations", "Manufacturing", "IT"]
        if v not in valid:
            raise ValueError(f"Department must be one of: {valid}")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("employee", "manager", "admin"):
            raise ValueError("Role must be employee, manager, or admin")
        return v


class RefreshRequest(BaseModel):
    """Refresh access token using refresh token."""
    refresh_token: str


# ── Response Schemas ──────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    """Public user data (no sensitive fields)."""
    id: uuid.UUID
    email: str
    full_name: str
    department: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenPair(BaseModel):
    """Access + refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    """Login response with tokens and user data."""
    user: UserResponse
    tokens: TokenPair
