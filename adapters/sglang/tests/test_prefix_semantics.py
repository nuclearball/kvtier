"""Validation core: kvtier's radix multi-group semantics as exercised by
sglang-shaped access patterns (TokenCodec content addressing).

Covers:
* multi-group radix chains (page k == radix depth k+1)
* longest-consecutive-prefix batch_exists (sglang's hit contract)
* holes break the prefix at exactly the right page
* cross-sequence prefix sharing (same first-page tokens → same radix path)
* KV_EVICTED vs KV_ENOENT distinction per page
* KV_MAX_GROUPS bound surfaces as an error, not corruption
"""

from __future__ import annotations

import pytest

from conftest import make_cache  # noqa: F401
from sglang_backend.keycodec import TokenCodec
from kvtier_py._binding import (
    KV_ENOENT,
    KV_EVICTED,
    KV_MAX_GROUPS,
    KV_TOKENS_PER_GROUP,
)

PAGE = KV_TOKENS_PER_GROUP


def stream(n_tokens: int, seed: int = 0) -> list[int]:
    return [(seed * 2654435761 + i * 40503 + i * i) % 0x7FFFFFFF for i in range(n_tokens)]


@pytest.fixture()
def codec():
    return TokenCodec()


def test_multi_group_chain(cache, codec):
    """10-page sequence: each page lands at increasing radix depth."""
    toks = stream(10 * PAGE)
    payloads = {p: bytes([p]) * 2048 for p in range(10)}
    for p in range(10):
        ck = codec.encode(toks, p)
        assert ck.group_idx == p
        assert cache.put_sync(ck, [payloads[p]]) == 0

    for p in range(10):
        ck = codec.encode(toks, p)
        res = cache.get(ck)
        assert res is not None, f"page {p} missing"
        assert res.group_idx == p
        assert res.layer_data(0) == payloads[p]


def test_batch_exists_longest_prefix(cache, codec):
    """sglang batch_exists contract: consecutive-prefix hit count."""
    toks = stream(10 * PAGE)
    for p in range(10):
        ck = codec.encode(toks, p)
        cache.put_sync(ck, [bytes([p]) * 256])

    def page_key(p):
        return codec.encode(toks, p)

    assert _longest_prefix(cache, codec, toks, 10) == 10

    # punch a hole at page 4 (evict it) → prefix stops at 4
    ck4 = page_key(4)
    assert cache.evict(ck4) == 0
    assert _longest_prefix(cache, codec, toks, 10) == 4
    # tombstone: page 4 reports KV_EVICTED, page 5..9 still KV_EOK
    assert cache.lookup_rc(ck4) == KV_EVICTED
    ck7 = page_key(7)
    assert cache.lookup_rc(ck7) == 0


def _longest_prefix(cache, codec, toks, n_pages) -> int:
    for p in range(n_pages):
        ck = codec.encode(toks, p)
        if cache.lookup_rc(ck.prefix_id, ck.tokens) != 0:
            return p
    return n_pages


def test_shorter_query_is_exact_match(cache, codec):
    """kvtier is exact per (prefix_id, depth): a depth whose leaf was
    never written misses even when deeper pages exist above it, and a
    truncated token path never reaches deeper leaves."""
    toks = stream(6 * PAGE)
    # write pages 0,1,2 then jump to 5 — depths 4,5 have NO leaf
    for p in (0, 1, 2, 5):
        ck = codec.encode(toks, p)
        cache.put_sync(ck, [bytes([p]) * 128])

    # written pages hit
    for p in (0, 1, 2, 5):
        assert cache.exists(codec.encode(toks, p)), f"page {p} should hit"
    # page 3/4: internal nodes exist on page 5's path but carry no leaf
    assert cache.lookup_rc(codec.encode(toks, 3)) == KV_ENOENT
    assert cache.lookup_rc(codec.encode(toks, 4)) == KV_ENOENT
    # truncating page 5's path to depth 4 lands on the same leafless node
    ck5 = codec.encode(toks, 5)
    assert cache.lookup_rc(ck5.prefix_id, ck5.tokens[: 4 * PAGE]) == KV_ENOENT


def test_prefix_sharing_across_sequences(cache, codec):
    """Two sequences sharing the first 4 pages: both map onto the same radix
    path for pages 0..3 (content-addressed prefix_id), diverge after."""
    shared = stream(4 * PAGE, seed=7)
    tail_a = stream(2 * PAGE, seed=11)
    tail_b = stream(2 * PAGE, seed=13)

    seq_a = shared + tail_a
    seq_b = shared + tail_b

    payloads = {}
    for name, seq in (("a", seq_a), ("b", seq_b)):
        for p in range(6):
            ck = codec.encode(seq, p)
            payloads[(name, p)] = bytes([ord(name[0]), p]) * 256
            rc = cache.put_sync(ck, [payloads[(name, p)]])
            assert rc == 0, f"{name} page {p}: rc={rc}"

    # shared pages resolve to the SAME address for both sequences
    # (address = content hash of the prefix through that page)
    for p in range(4):
        assert codec.encode(seq_a, p) == codec.encode(seq_b, p)
    # divergent pages: different prefix content → different address
    ck_a4 = codec.encode(seq_a, 4)
    ck_b4 = codec.encode(seq_b, 4)
    assert ck_a4.prefix_id != ck_b4.prefix_id
    assert ck_a4 != ck_b4

    # DEDUP SEMANTICS: shared pages are one physical copy — b's re-put of the
    # shared prefix overwrote a's (same address, higher ver).  Both sequences
    # now read b's bytes for pages 0..3, while divergent pages stay isolated.
    for p in range(4):
        res_a = cache.get(codec.encode(seq_a, p))
        res_b = cache.get(codec.encode(seq_b, p))
        assert res_a is not None and res_b is not None
        assert res_a.layer_data(0) == payloads[("b", p)]
        assert res_b.layer_data(0) == payloads[("b", p)]
    for name, seq in (("a", seq_a), ("b", seq_b)):
        for p in (4, 5):
            res = cache.get(codec.encode(seq, p))
            assert res is not None
            assert res.layer_data(0) == payloads[(name, p)]


def test_shared_prefix_evict_affects_both(cache, codec):
    """Content addressing means evicting a shared page evicts it for every
    sequence that maps onto it — sglang-side LRU must treat the storage as
    shared-namespace (this is the price of dedup)."""
    shared = stream(3 * PAGE, seed=21)
    ck1 = codec.encode(shared, 1)
    cache.put_sync(ck1, [b"shared" * 64])

    assert cache.evict(ck1) == 0
    assert cache.lookup_rc(ck1) == KV_EVICTED


def test_max_groups_bound(tmp_path, codec):
    """page_idx >= KV_MAX_GROUPS must be rejected by the codec before it can
    create a bogus address (kvtier hard-caps at 256 groups)."""
    with pytest.raises(ValueError, match="KV_MAX_GROUPS"):
        codec.encode(stream(KV_MAX_GROUPS * PAGE), 256)


def test_page_size_must_match_group_size():
    from kvtier_py._binding import KV_TOKENS_PER_GROUP as G

    TokenCodec(page_size=G)
    with pytest.raises(ValueError, match="KV_TOKENS_PER_GROUP"):
        TokenCodec(page_size=16)
