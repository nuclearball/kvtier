"""Binding-level correctness: put/get roundtrip, layers, stripe, evict, stats."""

from __future__ import annotations

import threading

import pytest

from conftest import make_cache
from kvtier_py._binding import (
    KV_ECRC,
    KV_ENOENT,
    KV_EVICTED,
    KV_TOKENS_PER_GROUP,
    err_name,
)


def tokens_for(seq: int, n: int = KV_TOKENS_PER_GROUP):
    return [(seq * 7919 + i * 104729 + i) % 0x7FFFFFFF for i in range(n)]


def test_roundtrip_single_layer(cache):
    toks = tokens_for(1)
    payload = bytes(range(256)) * 4  # 1024 B
    assert cache.put_sync(0xA1, toks, [payload]) == 0

    res = cache.get(0xA1, toks)
    assert res is not None
    assert res.group_idx == 0
    assert res.records == [(0, 0, 1024)]  # 方案 B: pure payload, no inline header
    assert res.layer_data(0) == payload

    assert cache.lookup_rc(0xB2, toks) == KV_ENOENT


def test_roundtrip_multi_layer(cache):
    """sglang shape: one page = n_layers per-layer records."""
    toks = tokens_for(2)
    n_layers = 32
    payloads = [bytes([l]) * (256 + l) for l in range(n_layers)]
    assert cache.put_sync(0xA2, toks, payloads) == 0

    res = cache.get(0xA2, toks)
    assert res is not None
    assert len(res.records) == n_layers
    for l, (lid, off, ln) in enumerate(res.records):
        assert lid == l
        assert ln == len(payloads[l])
        assert res.layer_data(l) == payloads[l]


def test_stripe_split_path(tmp_path):
    """Payload > stripe_threshold (configured small) must take the stripe
    write path and still round-trip byte-identically."""
    c = make_cache(tmp_path, batch_min_bytes=4096, stripe_threshold=8192)
    try:
        toks = tokens_for(3)
        payload = bytes((i * 31) & 0xFF for i in range(64 * 1024))  # 64 KiB
        assert c.put_sync(0xA3, toks, [payload]) == 0
        res = c.get(0xA3, toks)
        assert res is not None
        assert res.layer_data(0) == payload
    finally:
        c.close()


def test_evict_then_tombstone(cache):
    toks = tokens_for(4)
    assert cache.put_sync(0xA4, toks, [b"x" * 512]) == 0
    assert cache.exists(0xA4, toks)
    assert cache.evict(0xA4, toks) == 0
    # key existed and was evicted → tombstone must surface KV_EVICTED
    assert cache.lookup_rc(0xA4, toks) == KV_EVICTED
    # never-existed key stays KV_ENOENT
    assert cache.lookup_rc(0xA5, tokens_for(5)) == KV_ENOENT


def test_overwrite_same_key(cache):
    """put to the same address twice: latest version wins on get."""
    toks = tokens_for(6)
    assert cache.put_sync(0xA6, toks, [b"first" * 64]) == 0
    assert cache.put_sync(0xA6, toks, [b"second" * 64]) == 0
    res = cache.get(0xA6, toks)
    assert res is not None
    assert res.layer_data(0).rstrip(b"\x00").startswith(b"second")


def test_sync_ack_thread(cache):
    """on_ack must fire exactly once, from a non-main thread."""
    toks = tokens_for(7)
    done = threading.Event()
    seen = []

    def ack(user, rc):
        seen.append((user, rc, threading.current_thread() is threading.main_thread()))
        done.set()

    assert cache.put(0xA7, toks, [b"ack" * 32], on_ack=ack, user="u") == 0
    assert done.wait(10)
    assert seen == [("u", 0, False)]


def test_partial_group_tokens(cache):
    """Last group may hold < 32 tokens (n_tokens_total < 32)."""
    toks = tokens_for(8, 17)
    assert cache.put_sync(0xA8, toks, [b"tail" * 8]) == 0
    res = cache.get(0xA8, toks)
    assert res is not None and res.group_idx == 0


def test_stats_counters(cache):
    toks = tokens_for(9)
    before = cache.stats()
    cache.put_sync(0xA9, toks, [b"s" * 128])
    cache.get(0xA9, toks)
    cache.get(0xA9, toks)  # second read = another hit
    cache.get(0xB9, toks)  # miss
    after = cache.stats()
    assert after["puts"] == before["puts"] + 1
    assert after["hits"] == before["hits"] + 2
    assert after["misses"] == before["misses"] + 1
    assert after["live_bytes"] >= 128
    assert err_name(0) == "KV_EOK" and err_name(KV_ECRC) == "KV_ECRC"


def test_double_close_is_safe(cache):
    cache.close()
    cache.close()  # idempotent


def test_unknown_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown kv_config"):
        make_cache(tmp_path, no_such_field=1)
