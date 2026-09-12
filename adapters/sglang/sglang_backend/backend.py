"""SolidcacherStorage — solidcacher-backed sglang HiCache storage shim.

Implements the sglang ``HiCacheStorage`` v1 contract (pinned to sglang
0.5.18, python/sglang/srt/mem_cache/hicache_storage.py):

    set(key, value)                 -> bool      one flat data page per key
    get(key, target_location)       -> tensor|bytes|None
    exists(key)                     -> bool
    batch_exists(keys)              -> int       longest consecutive prefix
    batch_set(keys, values)         -> bool
    batch_get(keys, target_locs)    -> list
    clear() / get_stats()

sglang itself is imported lazily and is optional: the class is duck-type
compatible on its own, and :func:`as_hicache_storage` mixes in the real ABC
when the package is installed (required only for plugging into
``--hicache-storage-backend``).

Design notes
------------
* sglang keys are opaque per-page strings (content hashes) — they are mapped
  through :class:`~sglang_backend.keycodec.BlobCodec` into a single-group
  solidcacher address.  Longest-prefix semantics live in sglang's caller
  (batch_exists counts consecutive hits); solidcacher only supplies exact
  per-key hit/miss.
* Eviction is intentionally NOT implemented: the HiCacheStorage contract has
  no delete — space reclamation is the storage's job, which is exactly what
  solidcacher's generational GC does.
* Writes wait for the solidcacher ack (durable + published), mirroring
  HiCacheFile's synchronous os.replace().
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from solidcacher_py.cache import Solidcacher, SolidcacherError

from .keycodec import BlobCodec

log = logging.getLogger("sglang_backend.solidcacher")


def _writable_byte_view(obj):
    """Best-effort writable 1-D uint8 view of a tensor-like page."""
    if isinstance(obj, bytearray):
        return obj
    if hasattr(obj, "numpy"):  # torch tensor
        arr = obj.detach().cpu().contiguous().numpy()
    elif hasattr(obj, "dtype") and hasattr(obj, "view"):  # numpy array
        arr = obj
        if not arr.flags.c_contiguous:
            import numpy as np

            arr = np.ascontiguousarray(arr)
    else:
        return bytearray(obj)  # generic fallback (copies)
    if arr.dtype.itemsize != 1:
        arr = arr.view("B")
    return arr.reshape(-1)


class SolidcacherStorage:
    """Duck-typed sglang HiCacheStorage backend backed by solidcacher."""

    sglang_version_pinned = "0.5.18"

    def __init__(
        self,
        dev_uris,
        lib_path: str | None = None,
        storage_config=None,
        codec=None,
        **cache_config,
    ):
        self.cache = Solidcacher(dev_uris, lib_path=lib_path, **cache_config)
        self.codec = codec if codec is not None else BlobCodec()
        self.dev_uris = list(self.cache.dev_uris)
        self.key_suffix = self._rank_suffix(storage_config)

    # -- key helpers ---------------------------------------------------------

    @staticmethod
    def _rank_suffix(storage_config) -> str:
        """Mirror HiCacheFile's per-rank/model key namespacing (TP shards KV,
        so different ranks must not collide on the same page key)."""
        if storage_config is None:
            return ""
        parts = []
        if getattr(storage_config, "model_name", None):
            parts.append(str(storage_config.model_name).replace("/", "-"))
        if not getattr(storage_config, "is_mla_model", False):
            parts.append(f"{storage_config.tp_rank}_{storage_config.tp_size}")
        if getattr(storage_config, "pp_size", 1) > 1:
            parts.append(f"{storage_config.pp_size}_{storage_config.pp_rank}")
        return "_" + "-".join(parts) if parts else ""

    def _skey(self, key: str) -> str:
        return key + self.key_suffix

    # -- v1 contract ----------------------------------------------------------

    def set(self, key: str, value=None, target_location=None, target_sizes=None) -> bool:
        key = self._skey(key)
        ck = self.codec.encode(key)
        if self.cache.exists(ck.prefix_id, ck.tokens):
            return True  # fast path, mirrors HiCacheFile.set
        if value is not None:
            payload = value
        elif target_location is not None:
            payload = _writable_byte_view(target_location)
        else:
            raise ValueError("set() needs value or target_location")
        try:
            rc = self.cache.put_sync(ck.prefix_id, ck.tokens, [payload])
        except (SolidcacherError, TypeError, BufferError) as e:
            log.error("solidcacher put failed for %s: %s", key, e)
            return False
        if rc != 0:
            log.error("solidcacher put rc=%d for %s", rc, key)
            return False
        return True

    def get(self, key: str, target_location=None, target_sizes=None):
        key = self._skey(key)
        ck = self.codec.encode(key)
        res = self.cache.get(ck.prefix_id, ck.tokens)
        if res is None:
            return None
        data = res.layer_data(0)
        if target_location is None:
            return data
        view = _writable_byte_view(target_location)
        expected = len(view)
        if expected != len(data):
            log.error(
                "page size mismatch for %s: target=%d stored=%d",
                key, expected, len(data),
            )
            return None
        view[:] = data  # copy into the caller's page
        return target_location

    def exists(self, key: str) -> bool:
        ck = self.codec.encode(self._skey(key))
        return self.cache.exists(ck.prefix_id, ck.tokens)

    def lookup_rc(self, key: str) -> int:
        """Raw solidcacher rc for a key (KV_ENOENT vs KV_EVICTED vs EOK)."""
        ck = self.codec.encode(self._skey(key))
        return self.cache.lookup_rc(ck.prefix_id, ck.tokens)

    def batch_exists(self, keys, extra_info=None) -> int:
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        if values is None and target_locations is None:
            values = [None] * len(keys)
        for i, key in enumerate(keys):
            value = values[i] if values is not None else None
            loc = target_locations[i] if target_locations is not None else None
            if not self.set(key, value=value, target_location=loc):
                return False
        return True

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        if target_locations is None:
            return [self.get(k) for k in keys]
        return [self.get(k, loc) for k, loc in zip(keys, target_locations)]

    # -- lifecycle / diagnostics ---------------------------------------------

    @classmethod
    def from_config(cls, storage_config, extra_kwargs=None):
        """sglang dynamic-backend constructor convention.

        StorageBackendFactory._create_dynamic_backend instantiates external
        backends as ``backend_class(storage_config, kwargs)`` — this
        classmethod matches that call shape and sources runtime settings
        from (in order): extra_kwargs dict, storage_config.extra_config,
        environment.
        """
        extra = dict(extra_kwargs or {})
        cfg_extra = dict(getattr(storage_config, "extra_config", None) or {})
        merged = {**cfg_extra, **extra}

        dev_uris = merged.pop("dev_uris", None)
        if dev_uris is None:
            env = os.environ.get("SOLIDCACHER_DEV_URIS", "")
            dev_uris = [u for u in env.split(",") if u]
        if not dev_uris:
            raise ValueError(
                "solidcacher dev_uris not configured: set "
                "SOLIDCACHER_DEV_URIS or pass dev_uris via "
                "hicache_storage_backend_extra_config"
            )
        lib_path = merged.pop("lib_path", None) or os.environ.get("KVCACHE_LIBRARY")

        # remaining keys: keep the ones that are kv_config fields, ignore the
        # rest (e.g. sglang's own interface_v1 flag) with a debug note
        from solidcacher_py._binding import KVConfig

        known = {f[0] for f in KVConfig._fields_}
        ignored = [k for k in merged if k not in known]
        if ignored:
            log.debug("ignoring non-kv_config extra keys: %s", ignored)
        cache_cfg = {k: merged[k] for k in merged if k in known}

        return cls(
            dev_uris,
            lib_path=lib_path,
            storage_config=storage_config,
            **cache_cfg,
        )

    def clear(self):
        """Wipe the cache: close, delete device files, reopen (validation)."""
        cfg = dict(self.cache.config)
        uris = self.dev_uris
        self.cache.close()
        for uri in uris + [u + ".ckpt" for u in uris]:
            try:
                os.remove(uri)
            except FileNotFoundError:
                pass
        self.cache = Solidcacher(uris, **cfg)

    def get_stats(self):
        st = self.cache.stats()
        return {
            "solidcacher": st,
            "key_suffix": self.key_suffix,
            "sglang_version_pinned": self.sglang_version_pinned,
        }

    def close(self):
        self.cache.close()


def as_hicache_storage(dev_uris=None, **kwargs):
    """Return a SolidcacherStorage instance mixed with sglang's ABC when the
    sglang package is importable; a plain duck-typed instance otherwise."""
    try:
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
    except Exception as e:  # noqa: BLE001 — sglang optional
        log.info("sglang not importable (%s); returning duck-typed backend", e)
        return SolidcacherStorage(dev_uris, **kwargs)

    class SolidcacherHiCacheStorage(HiCacheStorage, SolidcacherStorage):
        """Real HiCacheStorage subclass (usable via storage-backend factory)."""

        def __init__(self, dev_uris, storage_config=None, **kw):
            SolidcacherStorage.__init__(
                self, dev_uris, storage_config=storage_config, **kw
            )

    return SolidcacherHiCacheStorage(dev_uris, **kwargs)
