"""Durability & recovery: journal replay on reopen, and SIGKILL crash
recovery of the writer path (child process killed without close())."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import make_cache
from solidcacher_py._binding import KV_TOKENS_PER_GROUP

REPO = Path(__file__).resolve().parents[1]

CHILD_SRC = r'''
import os, sys, time, signal
sys.path.insert(0, {repo!r})
from solidcacher_py.cache import Solidcacher

uris = {uris!r}
c = Solidcacher(uris, **{cfg!r})
for k in range(8):
    toks = [(k * 65537 + i) % 0x7FFFFFFF for i in range({page})]
    rc = c.put_sync(0xC000 + k, toks, [bytes([k]) * 512])
    assert rc == 0, rc
# NO close(): the child is SIGKILLed with in-flight state; the journal +
# superblock double-write must still let replay recover every acked put.
os.kill(os.getpid(), signal.SIGKILL)
'''


def test_reopen_preserves_data(tmp_path):
    uris = [str(tmp_path / "dev0.img")]
    cfg = dict(region_cnt=2, region_size_pages=1024)

    c = make_cache(tmp_path)
    toks = [(7 * 65537 + i) % 0x7FFFFFFF for i in range(KV_TOKENS_PER_GROUP)]
    assert c.put_sync(0xC0DE, toks, [b"durable" * 64]) == 0
    c.close()

    from solidcacher_py.cache import Solidcacher

    c2 = Solidcacher(uris, **cfg)
    try:
        res = c2.get(0xC0DE, toks)
        assert res is not None
        assert res.layer_data(0) == b"durable" * 64
    finally:
        c2.close()


def test_sigkill_crash_recovery(tmp_path):
    uris = [str(tmp_path / "dev0.img")]
    cfg = dict(region_cnt=2, region_size_pages=1024)
    src = CHILD_SRC.format(repo=str(REPO), uris=uris, cfg=cfg, page=KV_TOKENS_PER_GROUP)

    proc = subprocess.Popen(
        [sys.executable, "-c", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # wait for the child to finish its puts (it kills itself when done)
    proc.wait(60)
    assert proc.returncode == -signal.SIGKILL, (proc.returncode, proc.stderr.read())

    from solidcacher_py.cache import Solidcacher

    c = Solidcacher(uris, **cfg)
    try:
        found = 0
        for k in range(8):
            toks = [(k * 65537 + i) % 0x7FFFFFFF for i in range(KV_TOKENS_PER_GROUP)]
            res = c.get(0xC000 + k, toks)
            if res is not None:
                assert res.layer_data(0) == bytes([k]) * 512
                found += 1
        # every put was acked before the kill, so replay must find all 8
        assert found == 8, f"recovered {found}/8 acked puts after SIGKILL"
    finally:
        c.close()
