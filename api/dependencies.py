"""FastAPI dependency injection for store and orchestrator."""

from typing import Optional
from storage.postgres import PostgresStore

_store_instance: Optional[PostgresStore] = None
_orchestrator_instance = None


def get_store() -> PostgresStore:
    """Return the global PostgresStore instance (set during lifespan)."""
    if _store_instance is None:
        raise RuntimeError("Store not initialized")
    return _store_instance


def get_orchestrator():
    """Return the global orchestrator instance (optional, set during lifespan)."""
    return _orchestrator_instance


def init_store(store: PostgresStore):
    global _store_instance
    _store_instance = store


def init_orchestrator(orch):
    global _orchestrator_instance
    _orchestrator_instance = orch
