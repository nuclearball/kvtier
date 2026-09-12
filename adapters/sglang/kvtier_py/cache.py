"""High-level Python wrapper over kvtier's C API.

Semantics notes (verified against kvtier/src/cache.c):

* ``cache_put`` is **asynchronous** and does NOT copy the payload for the
  single/replica paths — it stores pointers and the writer thread reads them
  later.  This wrapper therefore keeps every payload buffer alive until the
  ack callback fires (see :class:`_PendingPut`).
* The ack fires exactly once, from an internal writer thread, after the
  chunk is durable and the radix leaf published.
* ``cache_get`` is synchronous and all-or-nothing per (prefix_id, group):
  ``KV_EOK`` on hit, ``KV_ENOENT`` if never stored/expired, ``KV_EVICTED``
  if the key existed but was discarded by GC/evict (tombstone lookup).
"""

from __future__ import annotations

import ctypes
import logging
import threading
from dataclasses import dataclass

import sys

from ._binding import (
    KV_EOK,
    KVConfig,
    KVDataRef,
    CacheGetResult,
    AckFn,
    err_name,
    load_library,
)

log = logging.getLogger("kvtier")


@dataclass(frozen=True)
class CacheKey:
    """Complete kvtier address: prefix_id + token path + group depth.

    Accepted directly by :meth:`Kvtier.put` / ``get`` / ``exists`` /
    ``lookup_rc`` / ``evict`` — preferred over passing the fields separately
    (it is surprisingly easy to forget ``group_idx`` and silently collapse
    every page of a sequence onto group 0).
    """

    prefix_id: int
    tokens: tuple  # full token list covering groups 0..group_idx
    group_idx: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def _split(self):
        return self.prefix_id, list(self.tokens), self.group_idx

STAT_KEYS = (
    "puts",
    "hits",
    "misses",
    "batches",
    "bytes_written",
    "bytes_migrated",
    "rotations",
    "drops",
    "leaves",
    "live_bytes",
    "capacity_bytes",
    "generations",
    "gc_triggers",
    "gc_evicted_leaves",
    "journal_dropped",
    "ver",
)

_CONFIG_FIELDS = {f[0] for f in KVConfig._fields_}


def _norm_rc(rc: int) -> int:
    """Normalize a kvtier return code to the kvtier.h enum.

    The C code returns error codes as POSITIVE magnitudes (``-KV_ENOENT``
    with ``KV_ENOENT = -3`` => ``3``), while the header documents the enum as
    negative.  Normalize to the negative form so Python code can compare
    against KV_ENOENT/KV_EVICTED/... directly.
    """
    return -rc if rc > 0 else rc


# ---------------------------------------------------------------------------
# payload handling (zero-copy where possible)
# ---------------------------------------------------------------------------


def _payload_view(obj):
    """Normalize *obj* to a 1-D bytes-like view.

    Zero-copy for bytes / bytearray / memoryview / numpy arrays / torch CPU
    tensors; single copy fallback for anything else.
    """
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, bytearray):
        return obj
    if isinstance(obj, memoryview):
        return obj if obj.format == "B" and obj.ndim == 1 else obj.cast("B")

    if hasattr(obj, "numpy"):  # torch tensor
        t = obj.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        arr = t.contiguous().numpy()
        return _np_bytes_view(arr)

    np = sys.modules.get("numpy")
    if np is not None and isinstance(obj, np.ndarray):
        return _np_bytes_view(obj)

    return memoryview(bytes(obj))  # generic fallback (copies)


def _np_bytes_view(arr):
    np = sys.modules.get("numpy")
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.view(np.uint8)
    return arr.reshape(-1)


def _ptr_for(view):
    """Return (void*, keepalive) for a 1-D bytes-like view."""
    n = len(view)
    if isinstance(view, bytes):
        return ctypes_cast_bytes(view), view
    if isinstance(view, memoryview) and view.readonly:
        data = view.tobytes()
        return ctypes_cast_bytes(data), data
    buf = (ctypes.c_char * n).from_buffer(view)
    return ctypes.cast(buf, ctypes.c_void_p), buf


def ctypes_cast_bytes(data: bytes):
    return ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p)


# ---------------------------------------------------------------------------
# pending put bookkeeping (buffer lifetime)
# ---------------------------------------------------------------------------

_LIVE_PUTS: dict[int, "_PendingPut"] = {}
_LIVE_PUTS_LOCK = threading.Lock()


class _PendingPut:
    """Everything that must outlive ``cache_put`` until the ack fires."""

    __slots__ = (
        "ack_fn",
        "user_ack",
        "user_data",
        "token_arr",
        "ref_arr",
        "ptrs",
        "event",
        "rc",
    )

    def __init__(self):
        self.ack_fn = None
        self.user_ack = None
        self.user_data = None
        self.token_arr = None
        self.ref_arr = None
        self.ptrs = []
        self.event = None
        self.rc = None


def _ack_trampoline(user_ptr, rc):
    """Runs on the kvtier writer thread."""
    if user_ptr is None:
        return
    pending = _LIVE_PUTS.pop(int(user_ptr), None)
    if pending is None:
        return
    rc = _norm_rc(rc)
    pending.rc = rc
    if pending.event is not None:
        pending.event.set()
    if pending.user_ack is not None:
        try:
            pending.user_ack(pending.user_data, rc)
        except Exception:  # noqa: BLE001 — never propagate into C
            log.exception("user ack callback raised")
    del pending  # release payload refs after the callback


_ACK_TRAMPOLINE = AckFn(_ack_trampoline)
# process-lifetime keepalive for the trampoline (C holds a raw fn pointer)
_ACK_TRAMPOLINE_KEEPALIVE = _ACK_TRAMPOLINE


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class GetResult:
    """Copy of a cache_get_result; safe to use after cache_result_free."""

    data: bytes
    ver: int
    prefix_id: int
    group_idx: int
    records: list  # list[(layer_id, off, len)]

    def layer_data(self, layer_id: int) -> bytes:
        for lid, off, ln in self.records:
            if lid == layer_id:
                return self.data[off : off + ln]
        raise KeyError(f"layer {layer_id} not present (records={self.records})")


__all__ = [
    "Kvtier",
    "KvtierError",
    "GetResult",
    "STAT_KEYS",
]


class KvtierError(Exception):
    def __init__(self, op: str, rc: int):
        self.op = op
        self.rc = rc
        super().__init__(f"{op} failed: {err_name(rc)} ({rc})")


# ---------------------------------------------------------------------------
# main wrapper
# ---------------------------------------------------------------------------


class Kvtier:
    """ owns one cache_t instance.  Not fork/thread-shared beyond the C
    library's own internal threading."""

    def __init__(self, dev_uris, lib_path: str | None = None, **config):
        self.lib = load_library(lib_path)
        if isinstance(dev_uris, (str, bytes)):
            dev_uris = [dev_uris]
        self.dev_uris = [str(u) for u in dev_uris]
        if not self.dev_uris:
            raise ValueError("at least one device URI is required")

        unknown = set(config) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"unknown kv_config fields: {sorted(unknown)}")

        cfg = KVConfig()
        self.lib.kv_config_default(ctypes.byref(cfg))
        for key, val in config.items():
            setattr(cfg, key, val)
        self.config = config

        n = len(self.dev_uris)
        uris = (ctypes.c_char_p * n)(*[u.encode() for u in self.dev_uris])
        self._uris_keepalive = uris
        handle = ctypes.c_void_p()
        rc = self.lib.cache_open(
            ctypes.byref(handle),
            ctypes.cast(uris, ctypes.POINTER(ctypes.c_char_p)),
            n,
            ctypes.byref(cfg),
        )
        if rc != KV_EOK:
            raise KvtierError("cache_open", _norm_rc(rc))
        self._handle = handle

    # -- lifecycle ---------------------------------------------------------

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self.lib.cache_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- put ----------------------------------------------------------------

    def put(
        self,
        prefix_id,
        tokens=None,
        records=None,
        group_idx: int = 0,
        expire_ts: int = 0,
        on_ack=None,
        user=None,
    ) -> int:
        """Async put of one group.

        With a :class:`CacheKey`::

            put(ck, [payload0, payload1, ...])

        With explicit fields::

            put(prefix_id, tokens_list, [payloads], group_idx=g)

        tokens:      sequence of uint32 covering groups 0..group_idx
                     (>= (group_idx+1)*32 tokens, last group may be partial)
        records:     iterable of per-layer bytes-like payloads (n_layers)
        on_ack:      optional callback (user, rc) fired from a writer thread
        returns:     enqueue rc (KV_EOK == accepted); final status via ack
        """
        if self._handle is None:
            raise KvtierError("cache_put", -111)  # closed

        if isinstance(prefix_id, CacheKey):
            ck = prefix_id
            if tokens is not None and records is None:
                records = tokens  # put(ck, [records]) shorthand
            pid, toks, g = ck._split()
        else:
            if tokens is None:
                raise ValueError("tokens required")
            pid, toks, g = prefix_id, list(tokens), group_idx
        if records is None:
            raise ValueError("records required")

        views = [_payload_view(r) for r in records]
        pending = _PendingPut()
        pending.user_ack = on_ack
        pending.user_data = user

        n_tokens = len(toks)
        token_arr = (ctypes.c_uint32 * max(n_tokens, 1))(*toks)
        pending.token_arr = token_arr

        ref_arr = (KVDataRef * len(views))()
        for i, view in enumerate(views):
            ptr, keepalive = _ptr_for(view)
            pending.ptrs.append(keepalive)  # hold buffer alive until ack
            ref_arr[i].base = ptr
            ref_arr[i].off = 0
            ref_arr[i].len = len(view)
        pending.ref_arr = ref_arr

        with _LIVE_PUTS_LOCK:
            key = id(pending)
            _LIVE_PUTS[key] = pending

        rc = self.lib.cache_put(
            self._handle,
            ctypes.c_uint64(pid),
            ctypes.c_uint32(g),
            token_arr,
            ctypes.c_uint32(n_tokens),
            ctypes.c_uint32(expire_ts),
            ctypes.c_uint16(len(views)),
            ref_arr,
            _ACK_TRAMPOLINE,
            ctypes.c_void_p(key),
        )
        if rc != KV_EOK:
            # not enqueued: trampoline will never fire — reclaim now
            with _LIVE_PUTS_LOCK:
                _LIVE_PUTS.pop(key, None)
        return _norm_rc(rc)

    def put_sync(self, *args, timeout: float | None = None, **kwargs) -> int:
        """Put and block until the ack fires; returns the final rc
        (normalized: 0 == KV_EOK, negative == kvtier.h error enum)."""
        ev = threading.Event()
        holder: dict[str, int] = {}

        def _ack(_user, rc):
            holder["rc"] = _norm_rc(rc)
            ev.set()

        # make sure the C-level ack has a pending entry even though our
        # python ack closes over the event
        rc = self.put(*args, on_ack=_ack, user=None, **kwargs)
        if rc != KV_EOK:
            return rc
        if not ev.wait(timeout):
            raise TimeoutError("put ack did not arrive in time")
        return holder.get("rc", -1)

    # -- get ----------------------------------------------------------------

    def _resolve(self, prefix_id, tokens, n_tokens, group_idx=0):
        if isinstance(prefix_id, CacheKey):
            pid, toks, g = prefix_id._split()
            return pid, toks, len(toks)
        if tokens is None:
            raise ValueError("tokens required when prefix_id is not a CacheKey")
        toks = list(tokens)
        return prefix_id, toks, len(toks) if n_tokens is None else n_tokens

    def get(self, prefix_id, tokens=None, n_tokens: int | None = None):
        """Sync get of the deepest group.  Returns GetResult or None on miss."""
        if self._handle is None:
            raise KvtierError("cache_get", -111)
        pid, toks, n = self._resolve(prefix_id, tokens, n_tokens)
        token_arr = (ctypes.c_uint32 * max(n, 1))(*toks)
        res = CacheGetResult()
        rc = self.lib.cache_get(
            self._handle,
            ctypes.c_uint64(pid),
            token_arr,
            ctypes.c_uint32(n),
            ctypes.byref(res),
        )
        if rc != KV_EOK:
            return None
        data = ctypes.string_at(res.buf, res.buf_len)
        records = [
            (res.recs[i].layer_id, res.recs[i].off, res.recs[i].len)
            for i in range(res.n_records)
        ]
        out = GetResult(
            data=data,
            ver=res.ver,
            prefix_id=res.prefix_id,
            group_idx=res.group_idx,
            records=records,
        )
        self.lib.cache_result_free(ctypes.byref(res))
        return out

    def lookup_rc(self, prefix_id, tokens=None, n_tokens: int | None = None) -> int:
        """Raw cache_get rc (normalized): KV_EOK / KV_ENOENT / KV_EVICTED / ..."""
        if self._handle is None:
            raise KvtierError("cache_get", -111)
        pid, toks, n = self._resolve(prefix_id, tokens, n_tokens)
        token_arr = (ctypes.c_uint32 * max(n, 1))(*toks)
        res = CacheGetResult()
        rc = self.lib.cache_get(
            self._handle,
            ctypes.c_uint64(pid),
            token_arr,
            ctypes.c_uint32(n),
            ctypes.byref(res),
        )
        if rc == KV_EOK:
            self.lib.cache_result_free(ctypes.byref(res))
        return _norm_rc(rc)

    def exists(self, prefix_id, tokens=None, n_tokens: int | None = None) -> bool:
        return self.lookup_rc(prefix_id, tokens, n_tokens) == KV_EOK

    # -- evict / stats --------------------------------------------------------

    def evict(self, prefix_id, tokens=None, n_tokens: int | None = None) -> int:
        pid, toks, n = self._resolve(prefix_id, tokens, n_tokens)
        token_arr = (ctypes.c_uint32 * max(n, 1))(*toks)
        rc = self.lib.cache_evict(
            self._handle,
            ctypes.c_uint64(pid),
            token_arr,
            ctypes.c_uint32(n),
        )
        return _norm_rc(rc)

    def stats(self, key: str | None = None):
        if key is not None:
            return self.lib.cache_stats(self._handle, key.encode())
        return {k: self.lib.cache_stats(self._handle, k.encode()) for k in STAT_KEYS}

    def dev_count(self) -> int:
        return self.lib.cache_dev_count(self._handle)
