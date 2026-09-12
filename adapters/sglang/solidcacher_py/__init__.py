"""solidcacher_py — ctypes binding + high-level wrapper for solidcacher."""

from ._binding import (
    ERROR_NAMES,
    KV_MAX_GROUPS,
    KV_MAX_LAYERS_CAP,
    KV_PAGE_SIZE,
    KV_TOKENS_PER_GROUP,
    err_name,
    load_library,
)
from .cache import CacheKey, GetResult, STAT_KEYS, Solidcacher, SolidcacherError

__all__ = [
    "Solidcacher",
    "SolidcacherError",
    "GetResult",
    "CacheKey",
    "STAT_KEYS",
    "err_name",
    "ERROR_NAMES",
    "load_library",
    "KV_TOKENS_PER_GROUP",
    "KV_PAGE_SIZE",
    "KV_MAX_LAYERS_CAP",
    "KV_MAX_GROUPS",
]
