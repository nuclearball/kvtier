"""Key codecs: sglang page keys → solidcacher (prefix_id, tokens, group_idx).

solidcacher addresses data by (prefix_id, per-32-token-group hashes) routed
through a radix tree.  A codec converts sglang's page-granular string keys
into that address space.  Two flavours:

* :class:`BlobCodec` — hash the key string into prefix_id, derive 32
  synthetic key tokens from it, always group_idx=0.  Works with real sglang
  (whose keys are opaque content hashes).  This is the production-shape
  mapping; identical in spirit to the llama adapter.

* :class:`TokenCodec` — content-addressed multi-group layout for validation:
  prefix_id is derived from the *first* page's tokens so sequences that share
  a prefix share the radix path, group_idx == page index, and the key tokens
  are the real token stream.  sglang's storage interface cannot hand back
  tokens (keys are irreversible hashes), so this codec is driven directly by
  tests/demo to exercise solidcacher's radix multi-group path.
"""

from __future__ import annotations

from solidcacher_py._binding import KV_MAX_GROUPS, KV_TOKENS_PER_GROUP
from solidcacher_py.cache import CacheKey

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3

_SILLY_MASK = 0x7FFFFFFF


def _fnv1a(data: bytes, seed: int) -> int:
    h = _FNV_OFFSET ^ seed
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def mix64(x: int) -> int:
    """splitmix64 finalizer (same as solidcacher kv_hash_mix64)."""
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return x


def hash_key_string(key: str) -> int:
    """prefix_id for a page key: FNV-1a over utf-8, splitmix64 finalized."""
    return mix64(_fnv1a(key.encode("utf-8"), 0))


def synthetic_tokens(seed: int, n: int = KV_TOKENS_PER_GROUP) -> list[int]:
    """Deterministic key tokens derived from a seed (LCG, as the llama
    adapter's make_tokens).  Only used where real tokens are unavailable."""
    out = []
    for i in range(n):
        v = (seed * 2654435761 + i * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        out.append((v >> 16) & _SILLY_MASK)
    return out


class BlobCodec:
    """sglang page key → single-group blob address (production shape)."""

    tokens_per_key = KV_TOKENS_PER_GROUP

    def encode(self, key: str) -> CacheKey:
        pid = hash_key_string(key)
        return CacheKey(pid, tuple(synthetic_tokens(pid)), 0)


class TokenCodec:
    """Real-token content address (validation shape).

    Page k of a token stream is addressed by::

        prefix_id = content-hash(tokens[0:(k+1)*32])
        group_idx = k

    VALIDATED SEMANTICS: solidcacher's logical address is
    ``(prefix_id, group_idx)`` — the writer's publish path looks up the
    side table by exactly that pair and skips the token-hash path entirely
    (writer.c side_get shortcut), so two token streams that share a
    prefix_id COLLAPSE onto the same group index even if their tokens
    differ.  Encoding the full prefix content into prefix_id makes every
    distinct prefix content a distinct address:

    * identical page content across sequences → identical (prefix_id,
      group_idx) → natural dedup (last write wins, one physical copy)
    * divergent pages → different prefix_id → full isolation

    This mirrors sglang's own page keys (content hashes of the token
    prefix) and RadixAttention's content addressing.
    """

    def __init__(self, page_size: int = KV_TOKENS_PER_GROUP):
        if page_size != KV_TOKENS_PER_GROUP:
            # solidcacher's group size is a compile-time constant; a native
            # integration would pin sglang's --page-size to 32.
            raise ValueError(
                f"page_size must equal KV_TOKENS_PER_GROUP={KV_TOKENS_PER_GROUP}"
            )
        self.page_size = page_size

    def max_pages(self) -> int:
        return KV_MAX_GROUPS

    @staticmethod
    def prefix_id_of(tokens) -> int:
        """Content hash over a whole token prefix."""
        h = _fnv1a(
            b"".join(int(t).to_bytes(4, "little") for t in tokens), 0
        )
        return mix64(h)

    def encode(self, tokens, page_idx: int) -> CacheKey:
        if page_idx >= KV_MAX_GROUPS:
            raise ValueError(
                f"page_idx {page_idx} exceeds solidcacher KV_MAX_GROUPS={KV_MAX_GROUPS}"
            )
        tokens = list(tokens)
        n_needed = (page_idx + 1) * self.page_size
        if len(tokens) < n_needed:
            raise ValueError(
                f"need >= {n_needed} tokens for page {page_idx}, got {len(tokens)}"
            )
        prefix = tokens[:n_needed]
        pid = self.prefix_id_of(prefix)
        return CacheKey(pid, tuple(prefix), page_idx)
