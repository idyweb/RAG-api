"""
Base model with production-grade query patterns.

Algorithm Complexity Standards:
- All queries must document time complexity
- Pagination is mandatory for list queries
- Index usage must be explicit
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, TypeVar
from sqlalchemy import DateTime, select, func, desc, asc
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T", bound="BaseModel")


class Base(DeclarativeBase):
    """Base class for all models"""

    pass


class BaseModel(Base):
    """
    Abstract base model with common fields and optimized CRUD methods.

    All queries enforce:
    - Explicit pagination
    - Index awareness
    - Performance guarantees
    """

    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    is_deleted: Mapped[bool] = mapped_column(default=False, index=True)

    # CREATE OPERATIONS

    @classmethod
    async def create(
        cls: type[T], db: AsyncSession, commit: bool = True, **kwargs
    ) -> T:
        """
        Create new instance and optionally commit.
        """
        instance = cls(**kwargs)
        db.add(instance)

        if commit:
            await db.commit()
            await db.refresh(instance)

        return instance

    @classmethod
    async def create_many(
        cls: type[T], db: AsyncSession, items: List[Dict[str, Any]], commit: bool = True
    ) -> List[T]:
        """
        Bulk create multiple instances.
        """
        instances = [cls(**item) for item in items]
        db.add_all(instances)

        if commit:
            await db.commit()
            for instance in instances:
                await db.refresh(instance)

        return instances

    # READ OPERATIONS

    @classmethod
    async def get_by_id(cls: type[T], db: AsyncSession, id: Any) -> Optional[T]:
        """
        Get single record by primary key.
        """
        return await db.get(cls, id)

    @classmethod
    async def find_one(
        cls: type[T],
        db: AsyncSession,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[T]:
        """
        Get first matching record.
        """
        query = select(cls).where(cls.is_deleted == False)
        if filters:
            query = query.filter_by(**filters)
        if kwargs:
            query = query.filter_by(**kwargs)
        result = await db.execute(query)
        return result.scalars().first()

    @classmethod
    async def find_unique(
        cls: type[T],
        db: AsyncSession,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[T]:
        """
        Get unique record or raise if multiple exist.
        """
        query = select(cls)

        if filters:
            for key, value in filters.items():
                query = query.filter_by(**{key: value})

        if kwargs:
            query = query.filter_by(**kwargs)

        result = await db.execute(query)
        return result.scalars().one_or_none()

    @classmethod
    async def find_many(
        cls: type[T],
        db: AsyncSession,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        **kwargs,
    ) -> List[T]:
        """
        Get paginated list of records.
        """
        limit = min(limit, 1000)
        query = select(cls).where(cls.is_deleted == False).offset(offset).limit(limit)

        # Apply filters
        if filters:
            for key, value in filters.items():
                query = query.filter_by(**{key: value})

        # Apply kwargs filters
        if kwargs:
            query = query.filter_by(**kwargs)

        # Apply ordering
        if order_by and hasattr(cls, order_by):
            column = getattr(cls, order_by)
            query = query.order_by(desc(column) if order_desc else asc(column))
        else:
            query = query.order_by(desc(cls.created_at))

        result = await db.execute(query)
        return list(result.scalars().all())

    async def soft_delete(self, db: AsyncSession, commit: bool = True):
        """Mark record as deleted without removing from DB (Audit-Safe)."""
        self.is_deleted = True
        if commit:
            await db.commit()
            await db.refresh(self)

    @classmethod
    async def count(
        cls: type[T],
        db: AsyncSession,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> int:
        """
        Count matching records.
        """
        query = select(func.count()).select_from(cls)

        if filters:
            for key, value in filters.items():
                query = query.filter_by(**{key: value})

        if kwargs:
            query = query.filter_by(**kwargs)

        result = await db.execute(query)
        return result.scalar_one()

    @classmethod
    async def exists(
        cls: type[T],
        db: AsyncSession,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """
        Check if matching record exists.
        """
        query = select(cls.id)

        if filters:
            for key, value in filters.items():
                query = query.filter_by(**{key: value})

        if kwargs:
            query = query.filter_by(**kwargs)

        exists_query = select(query.exists())
        result = await db.execute(exists_query)
        return result.scalar()

    # UPDATE OPERATIONS

    async def save(self, db: AsyncSession, commit: bool = True) -> T:
        """
        Save changes to existing instance.

        """
        self.updated_at = datetime.now(timezone.utc)
        db.add(self)

        if commit:
            await db.commit()
            await db.refresh(self)

        return self

    @classmethod
    async def update_many(
        cls: type[T],
        db: AsyncSession,
        filters: Dict[str, Any],
        updates: Dict[str, Any],
        commit: bool = True,
    ) -> int:
        """
        Bulk update matching records.
        """
        updates["updated_at"] = datetime.now(timezone.utc)

        query = select(cls)
        for key, value in filters.items():
            query = query.filter_by(**{key: value})

        result = await db.execute(query)
        instances = result.scalars().all()

        for instance in instances:
            for key, value in updates.items():
                setattr(instance, key, value)

        if commit:
            await db.commit()

        return len(instances)

    # DELETE OPERATIONS

    async def delete(self, db: AsyncSession, commit: bool = True) -> None:
        """
        Delete this instance.
        """
        await db.delete(self)

        if commit:
            await db.commit()

    @classmethod
    async def delete_many(
        cls: type[T], db: AsyncSession, filters: Dict[str, Any], commit: bool = True
    ) -> int:
        """
        Bulk delete matching records.

        Algorithm: Batch DELETE
        """
        query = select(cls)
        for key, value in filters.items():
            query = query.filter_by(**{key: value})

        result = await db.execute(query)
        instances = result.scalars().all()

        for instance in instances:
            await db.delete(instance)

        if commit:
            await db.commit()

        return len(instances)

    # PAGINATION HELPERS

    @classmethod
    async def paginate(
        cls: type[T],
        db: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get paginated results with metadata.

        """
        # Enforce limits
        per_page = min(per_page, 100)
        page = max(page, 1)

        # Calculate offset
        offset = (page - 1) * per_page

        # Get total count
        total = await cls.count(db, filters=filters, **kwargs)

        # Get items
        items = await cls.find_many(
            db,
            filters=filters,
            limit=per_page,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc,
            **kwargs,
        )

        # Calculate pages
        pages = (total + per_page - 1) // per_page

        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
            "has_next": page < pages,
            "has_prev": page > 1,
        }