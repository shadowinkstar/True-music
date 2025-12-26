from typing import Optional

from .config import AppConfig

_config: Optional[AppConfig] = None
_clip_manager = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_clip_manager(manager) -> None:
    global _clip_manager
    _clip_manager = manager


def get_clip_manager():
    return _clip_manager


def require_clip_manager():
    manager = get_clip_manager()
    if manager is None:
        raise RuntimeError("Clip manager is not initialized. Call app main first.")
    return manager
