"""
Auth ORM model.

User accounts with department-based access control.
"""

import uuid
from typing import Optional
from sqlalchemy import String, Boolean, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from api.db.base_model import BaseModel

VALID_DEPARTMENTS = ["Sales", "HR", "Finance", "Operations", "Manufacturing", "IT"]


class User(BaseModel):
    """
    User account.

    Department field is the security boundary for RAG isolation.
    Only IT users can upload to any department.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(512), nullable=False)
    department: Mapped[str] = mapped_column(
        SAEnum(*VALID_DEPARTMENTS, name="department_enum"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        SAEnum("employee", "manager", "admin", name="role_enum"),
        nullable=False,
        default="employee",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ChannelIdentity(BaseModel):
    """
    Channel identity.

    Maps external Teams/M365 identities to the system.
    """
    __tablename__ = "channel_identities"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
