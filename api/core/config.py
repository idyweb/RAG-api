"""
Config shim.

security.py and other core modules import from `api.core.config`.
This file re-exports from the canonical settings location.
"""

from api.config.settings import settings  # noqa: F401

__all__ = ["settings"]
