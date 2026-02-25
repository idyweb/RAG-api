"""
Agent models.
"""

from sqlalchemy import String, Boolean, Integer
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from api.db.base_model import BaseModel


class AgentTool(BaseModel):
    """
    Agent Tool mapping.
    Stores live API integration routes for multi-agent architecture.
    """
    __tablename__ = "agent_tools"

    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True, unique=True)
    description: Mapped[str] = mapped_column(String(1000), nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    allowed_departments: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
