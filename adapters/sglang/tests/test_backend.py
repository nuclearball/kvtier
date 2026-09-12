"""Backend shim: sglang HiCacheStorage v1 semantics (BlobCodec path)."""

from __future__ import annotations

import pytest

from conftest import make_cache  # noqa: F401 — ensures sys.path
from sglang_backend.backend import SolidcacherStorage, as_hicache_storage
from sglang_backend.keycodec import BlobCodec
from solidcacher_py._binding import KV_ENOENT, KV_EVICTED


@pytest.fixture()
def backend(tmp_path):
    uris = [str(tmp_path / f"dev{i}.img") for i in range(2)]
    b = SolidcacherStorage(
        uris, region_cnt=2, region_size_pages=1024, metrics_level=2
    )
    yield b
    b.close()


PAGES = [f"page_{i:03d}" for i in range(8)]


def test_set_get_roundtrip_bytes(backend):
    payload = bytes(range(256)) * 16  # 4096 B flat page
    assert backend.set(PAGES[0], payload) is True
    out = backend.get(PAGES[0])
    assert out == payload


def test_get_into_target_location(backend):
    payload = bytes(64) + b"abc" + bytes(61)
    assert backend.set(PAGES[1], payload) is True

    target = bytearray(128)
    ret = backend.get(PAGES[1], target_location=target)
    assert ret is target
    assert bytes(target) == payload


def test_get_missing_returns_none(backend):
    assert backend.get("nope") is None


def test_set_twice_is_idempotent(backend):
    assert backend.set(PAGES[2], b"v1" * 32) is True
    # second set: fast path skips rewrite — data stays v1
    assert backend.set(PAGES[2], b"v2" * 32) is True
    assert backend.get(PAGES[2]) == b"v1" * 32


def test_exists_and_batch_exists(backend):
    for i, key in enumerate(PAGES[:5]):
        assert backend.set(key, bytes([i]) * 128)

    assert backend.exists(PAGES[0])
    assert not backend.exists("missing")

    # longest consecutive prefix from the start
    assert backend.batch_exists(PAGES[:5]) == 5
    assert backend.batch_exists(PAGES) == 5  # stops at first missing
    # hole in the middle
    assert backend.batch_exists([PAGES[0], PAGES[1], "hole", PAGES[3]]) == 2
    assert backend.batch_exists([]) == 0


def test_batch_set_get(backend):
    values = [bytes([i]) * 100 for i in range(4)]
    assert backend.batch_set(PAGES[:4], values) is True
    assert backend.batch_get(PAGES[:4]) == values

    targets = [bytearray(100) for _ in range(4)]
    rets = backend.batch_get(PAGES[:4], target_locations=targets)
    assert all(ret is t for ret, t in zip(rets, targets))
    assert [bytes(t) for t in targets] == values


def test_lookup_rc_distinguishes_evicted(backend):
    assert backend.set(PAGES[6], b"e" * 128)
    assert backend.lookup_rc(PAGES[6]) == 0
    ck = BlobCodec().encode(backend._skey(PAGES[6]))
    assert backend.cache.evict(ck) == 0
    assert backend.lookup_rc(PAGES[6]) == KV_EVICTED
    assert backend.lookup_rc("never") == KV_ENOENT


def test_clear_wipes_and_reopens(backend):
    assert backend.set(PAGES[7], b"gone" * 16)
    backend.clear()
    assert backend.get(PAGES[7]) is None
    # and the reopened instance still works
    assert backend.set(PAGES[7], b"fresh" * 16)
    assert backend.get(PAGES[7]) == b"fresh" * 16


def test_get_stats_shape(backend):
    assert backend.set("statpage", b"x" * 128)
    st = backend.get_stats()
    assert "solidcacher" in st
    assert st["solidcacher"]["puts"] >= 1
    assert st["sglang_version_pinned"] == "0.5.18"


def test_rank_suffix_namespacing(tmp_path):
    class Cfg:
        tp_rank, tp_size = 0, 2
        pp_rank, pp_size = 0, 1
        is_mla_model = False
        model_name = "test-model"

    uris = [str(tmp_path / "dev0.img")]
    b = SolidcacherStorage(uris, storage_config=Cfg(), region_cnt=2, region_size_pages=1024)
    b0 = SolidcacherStorage(uris, storage_config=None, region_cnt=2, region_size_pages=1024)
    try:
        assert b0.key_suffix == ""
        assert "test-model" in b.key_suffix and "0_2" in b.key_suffix
        # rank-suffixed key is a DIFFERENT address from the unsuffixed one
        from sglang_backend.keycodec import hash_key_string

        assert hash_key_string(b._skey("k")) != hash_key_string(b0._skey("k"))
        assert b.set("k", b"ranked" * 8)
        assert b.get("k") == b"ranked" * 8
        # unsuffixed namespace on the same storage has nothing at "k"
        assert b0.exists("k") is False
    finally:
        b.close()
        b0.close()


def test_dynamic_backend_constructor_convention(tmp_path):
    """sglang's StorageBackendFactory._create_dynamic_backend calls
    backend_class(storage_config, kwargs) — from_config must match that
    shape and source settings from extra_kwargs / extra_config / env."""

    class Cfg:
        tp_rank, tp_size = 1, 4
        pp_rank, pp_size = 0, 1
        is_mla_model = False
        model_name = "dyn-model"
        extra_config = {"interface_v1": 1, "metrics_level": 2}

    uris = [str(tmp_path / "dev0.img"), str(tmp_path / "dev1.img")]
    # 1) extra_kwargs wins over extra_config; interface_v1 must be ignored
    b = SolidcacherStorage.from_config(
        Cfg(), {"dev_uris": uris, "region_cnt": 2, "region_size_pages": 1024}
    )
    try:
        assert "dyn-model" in b.key_suffix and "1_4" in b.key_suffix
        assert b.cache.config.get("region_size_pages") == 1024
        assert b.set("dyn", b"d" * 64) and b.get("dyn") == b"d" * 64
    finally:
        b.close()

    # 2) settings via storage_config.extra_config only
    Cfg.extra_config = {"interface_v1": 1, "dev_uris": uris, "metrics_level": 2}
    b2 = SolidcacherStorage.from_config(Cfg())
    try:
        assert b2.exists("dyn") is False  # fresh reopen of same files: index rebuilt
        assert b2.set("dyn2", b"e" * 64)
    finally:
        b2.close()

    # 3) no uris anywhere -> clear error
    class NoCfg(Cfg):
        extra_config = {"interface_v1": 1}

    with pytest.raises(ValueError, match="dev_uris"):
        SolidcacherStorage.from_config(NoCfg())


def test_as_hicache_storage_factory(tmp_path):
    obj = as_hicache_storage(
        [str(tmp_path / "dev0.img")], region_cnt=2, region_size_pages=1024
    )
    try:
        # sglang not installed in this env → duck-typed instance
        assert hasattr(obj, "set") and hasattr(obj, "batch_exists")
        assert obj.set("k", b"x" * 64)
        assert obj.get("k") == b"x" * 64
    finally:
        obj.close()
