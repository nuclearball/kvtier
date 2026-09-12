"""sglang HiCache storage backend shim on top of kvtier.

Pin: sglang 0.5.18 (HiCacheStorage v1 contract).
"""

from .backend import KvtierStorage, as_hicache_storage
from .keycodec import BlobCodec, CacheKey, TokenCodec, hash_key_string

__all__ = [
    "KvtierStorage",
    "as_hicache_storage",
    "BlobCodec",
    "TokenCodec",
    "CacheKey",
    "hash_key_string",
]
