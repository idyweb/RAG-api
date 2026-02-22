"""
Session shim.

`api.apps.rag.routers` and other modules import `get_session` from here.
Re-exports `get_db` from the canonical database.py.
"""

from api.db.database import get_db as get_session  # noqa: F401

__all__ = ["get_session"]
