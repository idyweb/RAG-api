"""
rag models.

TODO: Define SQLAlchemy ORM models
All models inherit from BaseModel
"""

from sqlalchemy import String, Text, Float, Integer, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column

from api.db.base_model import BaseModel


class QueryLog(BaseModel):
    """
    Records query details for analytics and monitoring.
    """
    __tablename__ = "query_logs"
    
    query: Mapped[str] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(String(36))
    department: Mapped[str] = mapped_column(String(100), index=True)
    result_count: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    cached: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[str] = mapped_column(String(50))
    
    __table_args__ = (
        Index('idx_querylog_dept_date', 'department', 'created_at'),
    )
