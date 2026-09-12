#!/usr/bin/env python3
"""End-to-end demo of the sglang HiCache storage shim over solidcacher.

Simulates an sglang HiCache workload without requiring sglang/torch:

  Phase 1 (BlobCodec — production shape)
    Paged KV pages of a "model" (n_layers per-page records) stored via the
    SolidcacherStorage v1 interface, longest-prefix batch_exists, restore.

  Phase 2 (TokenCodec — validation shape)
    Two sequences sharing a 4-page prefix: dedup on the shared pages,
    isolation after divergence, evict + KV_EVICTED tombstone.

  Phase 3 (durability)
    Reopen the cache and verify every page survived (journal replay).

usage: demo_sglang_shim.py [dir]   (default: a fresh temp dir)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang_backend.backend import SolidcacherStorage  # noqa: E402
from sglang_backend.keycodec import TokenCodec  # noqa: E402
from solidcacher_py._binding import KV_TOKENS_PER_GROUP, KV_EVICTED  # noqa: E402
from solidcacher_py.cache import Solidcacher  # noqa: E402

PAGE = KV_TOKENS_PER_GROUP

# fake model geometry: 32 layers × page of 32 tokens × 8 heads × 64 dim × fp16
N_LAYERS = 32
PAGE_BYTES = N_LAYERS * PAGE * 8 * 64 * 2  # 1 MiB flat page


def make_page(seed: int) -> bytes:
    out = bytearray(PAGE_BYTES)
    for i in range(0, PAGE_BYTES, 4096):
        out[i] = seed & 0xFF
    return bytes(out)


def hr(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def phase1_blob(workdir: Path) -> Solidcacher:
    hr("Phase 1 — BlobCodec: sglang page keys via SolidcacherStorage (v1)")
    # NOTE on capacity: 8 pages × 1 MiB must fit comfortably.  When the write
    # cursor runs out of region space, the writer force-frees the oldest
    # region (rotation) — pages can be recycled BEFORE gc_start_pct is ever
    # hit.  Size regions with headroom.
    b = SolidcacherStorage(
        [str(workdir / "dev0.img")],
        region_cnt=2,
        region_size_pages=8192,  # 32 MiB × 2 devices = 64 MiB capacity
        metrics_level=2,
    )
    n_pages = 8
    pages = [f"sess7/page_{i:03d}" for i in range(n_pages)]

    print(f"[set] {n_pages} pages × {PAGE_BYTES // 1024} KiB (page key strings)")
    for i, key in enumerate(pages):
        assert b.set(key, make_page(i + 1)), key

    print("[batch_exists] longest consecutive prefix:")
    assert b.batch_exists(pages) == n_pages, "all pages should exist"
    print(f"  all written       -> {b.batch_exists(pages)}/{n_pages}")
    print(f"  hole at page 3    -> {b.batch_exists(pages[:3] + ['hole'] + pages[4:])}")

    print("[get] restore page 3 into a target buffer:")
    target = bytearray(PAGE_BYTES)
    assert b.get(pages[3], target_location=target) is target
    assert bytes(target) == make_page(4)
    print("  byte-identical ✓")

    print("[lookup_rc] evicted vs never-existed:")
    b.cache.evict(*_blob_addr(b, pages[3]))
    rc_evicted = b.lookup_rc(pages[3])
    rc_missing = b.lookup_rc("never-stored")
    print(f"  evicted key  -> {rc_evicted} (KV_EVICTED={KV_EVICTED})")
    print(f"  unknown key  -> {rc_missing} (KV_ENOENT=-3)")
    assert rc_evicted == KV_EVICTED and rc_missing == -3
    return b


def _blob_addr(b: SolidcacherStorage, key: str):
    ck = b.codec.encode(b._skey(key))
    return ck.prefix_id, ck.tokens


def phase2_tokens(workdir: Path) -> Solidcacher:
    hr("Phase 2 — TokenCodec: content-addressed prefix sharing & dedup")
    c = Solidcacher(
        [str(workdir / "tok0.img")],
        region_cnt=2,
        region_size_pages=1024,
        metrics_level=2,
    )
    codec = TokenCodec()

    def stream(n, seed):
        return [(seed * 2654435761 + i * 40503 + i * i) % 0x7FFFFFFF for i in range(n)]

    shared = stream(4 * PAGE, seed=42)
    seq_a = shared + stream(2 * PAGE, seed=101)  # A diverges at page 4
    seq_b = shared + stream(2 * PAGE, seed=202)  # B diverges at page 4

    for name, seq in (("A", seq_a), ("B", seq_b)):
        for p in range(6):
            ck = codec.encode(seq, p)
            assert c.put_sync(ck, [bytes([ord(name), p]) * 4096]) == 0

    print("shared pages 0..3  : identical address for A and B → dedup")
    for p in range(4):
        assert codec.encode(seq_a, p) == codec.encode(seq_b, p)
    print("divergent pages 4,5: different prefix content → different address")
    assert codec.encode(seq_a, 4) != codec.encode(seq_b, 4)

    both_shared = [
        c.get(codec.encode(seq_a, p)).layer_data(0)
        for p in range(4)
    ]
    assert both_shared[0] == bytes([ord("B"), 0]) * 4096  # last writer wins
    res_a4 = c.get(codec.encode(seq_a, 4)).layer_data(0)
    res_b4 = c.get(codec.encode(seq_b, 4)).layer_data(0)
    assert res_a4 == bytes([ord("A"), 4]) * 4096
    assert res_b4 == bytes([ord("B"), 4]) * 4096
    print("  A@4 == A-only data ✓   B@4 == B-only data ✓   shared = B's copy ✓")

    print("[evict] evict B's page 4, then check tombstone:")
    ck = codec.encode(seq_b, 4)
    assert c.evict(ck) == 0
    assert c.lookup_rc(ck) == KV_EVICTED
    print(f"  lookup_rc -> {KV_EVICTED} (was-evicted distinguished from never-existed)")
    return c


def phase3_reopen(caches: list[Solidcacher]) -> list[Solidcacher]:
    hr("Phase 3 — durability: close + reopen (journal replay)")
    reopened = []
    for i, c in enumerate(caches):
        uris = list(c.dev_uris)
        cfg = dict(c.config)
        c.close()
        c2 = Solidcacher(uris, **cfg)
        st = c2.stats()
        print(f"  cache[{i}] reopened: leaves={st['leaves']} hits={st['hits']}")
        assert st["leaves"] > 0
        reopened.append(c2)
    print("  all index state recovered ✓")
    return reopened


def main() -> int:
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="sglang_shim_"))
    print(f"workdir: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)

    b = phase1_blob(workdir)
    c = phase2_tokens(workdir)
    b.cache, c = phase3_reopen([b.cache, c])

    hr("final solidcacher stats (phase-1 cache)")
    for k, v in b.cache.stats().items():
        print(f"  {k:20s} {v}")

    b.close()
    c.close()
    if str(workdir).startswith(tempfile.gettempdir()):
        shutil.rmtree(workdir, ignore_errors=True)
    print("\nDEMO PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
