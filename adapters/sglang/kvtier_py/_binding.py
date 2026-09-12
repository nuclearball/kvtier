"""ctypes binding for kvtier (libkvtier).

Mirrors the public structs and prototypes of kvtier/include/kvtier.h
(pinned to the layout as of 2026-09).  The shared library is located via
(in order of precedence):

  1. explicit ``path`` argument to :func:`load_library`
  2. ``$KVTier_LIBRARY`` environment variable
  3. ``<repo>/build/libkvtier.dylib`` / ``.so`` (build with CMake, see
     the repository README)
  4. system search paths via ``ctypes.util.find_library``

Struct layouts must match the C definitions exactly; ctypes uses the same
natural alignment rules as the platform C compiler.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# constants (kvtier.h)
# ---------------------------------------------------------------------------

KV_TOKENS_PER_GROUP = 32
KV_PAGE_SIZE = 4096
KV_MAX_LAYERS_CAP = 256
KV_MAX_GROUPS = 256  # private constant in cache.c (put/get hard limit)

KV_EOK = 0
KV_ENOMEM = -1
KV_EEXIST = -2
KV_ENOENT = -3
KV_EBUSY = -4
KV_EREAD = -5
KV_EWRITE = -6
KV_ECRC = -7
KV_EFULL = -8
KV_EIO = -9
KV_ETIMEDOUT = -10
KV_EINVAL = -11
KV_EVICTED = -12

ERROR_NAMES = {
    KV_EOK: "KV_EOK",
    KV_ENOMEM: "KV_ENOMEM",
    KV_EEXIST: "KV_EEXIST",
    KV_ENOENT: "KV_ENOENT",
    KV_EBUSY: "KV_EBUSY",
    KV_EREAD: "KV_EREAD",
    KV_EWRITE: "KV_EWRITE",
    KV_ECRC: "KV_ECRC",
    KV_EFULL: "KV_EFULL",
    KV_EIO: "KV_EIO",
    KV_ETIMEDOUT: "KV_ETIMEDOUT",
    KV_EINVAL: "KV_EINVAL",
    KV_EVICTED: "KV_EVICTED",
}


def err_name(rc: int) -> str:
    return ERROR_NAMES.get(rc, f"KV_E{rc}")


# ---------------------------------------------------------------------------
# structs
# ---------------------------------------------------------------------------


try:  # package import
    from ._config_fields import CONFIG_FIELDS, CONFIG_ABI_VERSION
except ImportError:  # direct/script import
    from _config_fields import CONFIG_FIELDS, CONFIG_ABI_VERSION


class KVConfig(ctypes.Structure):
    """Mirror of ``struct kv_config`` (kvtier.h).

    The field list is generated from kvtier/tools/config_schema.json by
    ``python3 tools/gen_config.py`` (see ``_config_fields.py``); do not edit by
    hand.
    """

    _fields_ = CONFIG_FIELDS


class KVDataRef(ctypes.Structure):
    """Mirror of ``struct kv_data_ref`` — one layer's payload slice."""

    _fields_ = [
        ("base", ctypes.c_void_p),
        ("off", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
    ]


class KVRecordDesc(ctypes.Structure):
    """One parsed record inside a cache_get_result."""

    _fields_ = [
        ("layer_id", ctypes.c_uint16),
        ("off", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
    ]


class CacheGetResult(ctypes.Structure):
    """Mirror of ``struct cache_get_result``."""

    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("buf_len", ctypes.c_uint32),
        ("n_records", ctypes.c_uint16),
        ("recs", KVRecordDesc * KV_MAX_LAYERS_CAP),
        ("ver", ctypes.c_uint32),
        ("prefix_id", ctypes.c_uint64),
        ("group_idx", ctypes.c_uint32),
    ]


# void (*ack)(void *user, int rc)
AckFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)

# ---------------------------------------------------------------------------
# library loading
# ---------------------------------------------------------------------------


def _candidate_paths() -> list[str]:
    env = os.environ.get("KVTier_LIBRARY")
    if env:
        return [env]
    here = Path(__file__).resolve()
    # walk up to the repo root from adapters/<name>/kvtier_py/ and look
    # for the kvtier tree (CMake build in build/)
    cands: list[str] = []
    for parent in here.parents:
        for name in ("kvtier", ""):
            sc = (parent / name) if name else parent
            if (sc / "include" / "kvtier.h").exists():
                cands += [
                    str(sc / "build" / n)
                    for n in ("libkvtier.dylib", "libkvtier.so")
                ]
                cands += [
                    str(sc / n)
                    for n in ("libkvtier.dylib", "libkvtier.so")
                ]
                break
        if cands:
            break
    found = ctypes.util.find_library("kvtier")
    if found:
        cands.append(found)
    return cands


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load libkvtier and wire up all prototypes.  Raises OSError if absent."""
    candidates = [path] if path else _candidate_paths()
    lib = None
    last_err: Exception | None = None
    for cand in candidates:
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError as e:  # noqa: PERF203
            last_err = e
    if lib is None:
        raise OSError(
            "libkvtier shared library not found "
            f"(tried: {candidates}); build it with CMake (see the repository README)"
        ) from last_err

    lib.kv_config_default.argtypes = [ctypes.POINTER(KVConfig)]
    lib.kv_config_default.restype = None

    lib.cache_open.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),            # cache_t **out
        ctypes.POINTER(ctypes.c_char_p),            # const char *const *uris
        ctypes.c_int,                               # n_devs
        ctypes.POINTER(KVConfig),                   # cfg
    ]
    lib.cache_open.restype = ctypes.c_int

    lib.cache_close.argtypes = [ctypes.c_void_p]
    lib.cache_close.restype = None

    lib.cache_put.argtypes = [
        ctypes.c_void_p,                            # cache_t *
        ctypes.c_uint64,                            # prefix_id
        ctypes.c_uint32,                            # group_idx
        ctypes.POINTER(ctypes.c_uint32),            # tokens
        ctypes.c_uint32,                            # n_tokens_total
        ctypes.c_uint32,                            # expire_ts
        ctypes.c_uint16,                            # n_layers
        ctypes.POINTER(KVDataRef),                  # recs
        AckFn,                                      # ack
        ctypes.c_void_p,                            # user
    ]
    lib.cache_put.restype = ctypes.c_int

    lib.cache_get.argtypes = [
        ctypes.c_void_p,                            # cache_t *
        ctypes.c_uint64,                            # prefix_id
        ctypes.POINTER(ctypes.c_uint32),            # tokens
        ctypes.c_uint32,                            # n_tokens_total
        ctypes.POINTER(CacheGetResult),             # out
    ]
    lib.cache_get.restype = ctypes.c_int

    lib.cache_result_free.argtypes = [ctypes.POINTER(CacheGetResult)]
    lib.cache_result_free.restype = None

    lib.cache_evict.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.cache_evict.restype = ctypes.c_int

    lib.cache_stats.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.cache_stats.restype = ctypes.c_uint64

    lib.cache_dev_count.argtypes = [ctypes.c_void_p]
    lib.cache_dev_count.restype = ctypes.c_int

    return lib
