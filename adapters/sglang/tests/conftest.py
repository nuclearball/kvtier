"""Shared pytest fixtures: temp device files + small-geometry cache."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kvtier_py.cache import Kvtier  # noqa: E402

# small geometry keeps device files tiny:
#   pages = KV_REGION_BASE_PAGE(2) + region_cnt*region_size_pages + journal(512)
SMALL_CFG = dict(
    region_cnt=2,
    region_size_pages=1024,  # 4 MiB per region
    metrics_level=2,
)


def make_cache(tmp_path, n_devs=1, **overrides):
    uris = [str(tmp_path / f"dev{i}.img") for i in range(n_devs)]
    cfg = dict(SMALL_CFG)
    cfg.update(overrides)
    return Kvtier(uris, **cfg)


@pytest.fixture()
def cache(tmp_path):
    c = make_cache(tmp_path)
    yield c
    c.close()


@pytest.fixture()
def cache2(tmp_path):
    c = make_cache(tmp_path, n_devs=2)
    yield c
    c.close()
