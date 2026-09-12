/* DRAM read cache tests (dram-cache-design.md):
 *   - arena backend probe + degrade chain (HUGETLB -> THP -> plain)
 *   - size-class alloc/free, arena exhaustion -> external fallback
 *   - SIEVE eviction (budget bound, hand wrap, stale ver replace)
 *   - cache-level coherence: put overwrite invalidates, evict, TTL
 */
#include "dram.hpp"
#include "solidcacher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace sc;

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

#define KEY1 0xDEAD0001ull
#define KEY2 0xDEAD0002ull

static int put_entry(dram_cache *d, uint64_t pid, uint32_t ver,
                     uint32_t val, uint32_t buf_len) {
    uint8_t buf[8192];
    memset(buf, (int)(val & 0xFF), buf_len * 2);
    dram_rec recs[2] = {
        { .layer_id = 0, .off = 0, .len = buf_len },
        { .layer_id = 1, .off = buf_len, .len = buf_len },
    };
    /* buf holds 2 records back to back */
    return dram_put(d, pid, 0, ver, 0, buf, buf_len * 2, recs, 2);
}

static bool get_matches(dram_cache *d, uint64_t pid, uint32_t ver,
                        uint32_t val) {
    uint8_t *buf = NULL;
    uint32_t buf_len = 0;
    dram_rec recs[16];
    uint16_t n = 0;
    int rc = dram_get(d, pid, 0, ver, &buf, &buf_len, recs, 16, &n);
    if (rc != KV_EOK)
        return false;
    bool ok = buf_len == 8192 && n == 2 && recs[0].off == 0 &&
              recs[1].off == 4096;
    for (uint32_t i = 0; ok && i < buf_len; i++)
        ok = buf[i] == (uint8_t)(val & 0xFF);
    aligned_free(buf);
    return ok;
}

/* ---------------- unit: alloc backends + basic roundtrip --------------- */

static void test_backend_probe(void) {
    dram_cache *d = NULL;
    /* tiny budget: exercises the whole map path cheaply */
    CHECK(dram_init(&d, 64 * 1024, 32 * 1024, KV_DRAM_BACKEND_AUTO) == KV_EOK);
    CHECK(d != NULL);
    CHECK(d->backend >= KV_DRAM_BACKEND_PLAIN &&
          d->backend <= KV_DRAM_BACKEND_HUGETLB);
    /* 2MiB alignment when THP/HUGETLB was selected */
    if (d->backend != KV_DRAM_BACKEND_PLAIN)
        CHECK((reinterpret_cast<uint64_t>(d->arena) &
               ((2ull * 1024 * 1024) - 1)) == 0);
    dram_destroy(d);

    /* explicit unavailable backend must degrade, not fail */
    d = NULL;
    CHECK(dram_init(&d, 64 * 1024, 32 * 1024, KV_DRAM_BACKEND_HUGETLB) ==
          KV_EOK);
    CHECK(d != NULL);
    dram_destroy(d);

    /* invalid backend */
    d = NULL;
    CHECK(dram_init(&d, 64 * 1024, 32 * 1024, 99) == -KV_EINVAL);
    CHECK(d == NULL);
}

static void test_roundtrip_invalidate(void) {
    dram_cache *d = NULL;
    CHECK(dram_init(&d, 256 * 1024, 64 * 1024, KV_DRAM_BACKEND_AUTO) ==
          KV_EOK);

    CHECK(put_entry(d, KEY1, 5, 0xA5, 4096) == KV_EOK);
    CHECK(get_matches(d, KEY1, 5, 0xA5));
    /* ver mismatch -> miss */
    CHECK(!get_matches(d, KEY1, 4, 0xA5));
    CHECK(dram_stat(d, "dram_hits") == 1);
    CHECK(dram_stat(d, "dram_misses") == 1);
    CHECK(dram_stat(d, "dram_entries") == 1);

    /* ver replace */
    CHECK(put_entry(d, KEY1, 6, 0x5A, 4096) == KV_EOK);
    CHECK(dram_stat(d, "dram_entries") == 1);
    CHECK(get_matches(d, KEY1, 6, 0x5A));
    CHECK(!get_matches(d, KEY1, 5, 0xA5));

    /* stale backfill (older ver) is ignored */
    CHECK(put_entry(d, KEY1, 5, 0x11, 4096) == KV_EOK);
    CHECK(get_matches(d, KEY1, 6, 0x5A));

    /* invalidate */
    dram_invalidate(d, KEY1, 0);
    CHECK(dram_stat(d, "dram_entries") == 0);
    CHECK(dram_stat(d, "dram_bytes") == 0);
    CHECK(!get_matches(d, KEY1, 6, 0x5A));
    /* idempotent */
    dram_invalidate(d, KEY1, 0);
    dram_destroy(d);
}

/* ---------------- unit: SIEVE eviction + arena exhaustion -------------- */

static void test_sieve_eviction(void) {
    dram_cache *d = NULL;
    /* 8 entries of 16KB class fit in 128KB budget (128/16 = 8) */
    CHECK(dram_init(&d, 128 * 1024, 32 * 1024, KV_DRAM_BACKEND_PLAIN) ==
          KV_EOK);

    for (uint32_t p = 1; p <= 20; p++)
        CHECK(put_entry(d, p, 1, p, 4096) == KV_EOK);
    /* budget enforced (bytes counts entry_total ~= 8272, not class size) */
    CHECK(dram_stat(d, "dram_bytes") <= 128 * 1024);
    CHECK(dram_stat(d, "dram_entries") <= 15);
    CHECK(dram_stat(d, "dram_evictions") >= 5);

    /* newest entries must survive, oldest evicted (SIEVE is FIFO-ish) */
    CHECK(get_matches(d, 20, 1, 20));
    CHECK(!get_matches(d, 1, 1, 1));
    CHECK(dram_stat(d, "dram_entries") <= 15);

    /* get marks visited: re-getting survivors protects them */
    CHECK(get_matches(d, 20, 1, 20));
    CHECK(get_matches(d, 19, 1, 19));
    for (uint32_t p = 30; p <= 32; p++)
        CHECK(put_entry(d, p, 1, p, 4096) == KV_EOK);
    CHECK(get_matches(d, 20, 1, 20) || get_matches(d, 19, 1, 19));

    dram_destroy(d);
}

static void test_external_fallback(void) {
    dram_cache *d = NULL;
    /* entry total > 8M class cap -> external exact-size malloc */
    CHECK(dram_init(&d, 16 * 1024 * 1024, 12 * 1024 * 1024,
                    KV_DRAM_BACKEND_PLAIN) == KV_EOK);
    uint32_t half = 5 * 1024 * 1024;   /* total 10MB+hdr: > 8M class cap,
                                        * < 12M entry_max -> external */
    uint8_t *buf = (uint8_t *)malloc(half * 2);
    CHECK(buf != NULL);
    memset(buf, 0x77, half * 2);
    dram_rec recs[2] = {
        { .layer_id = 0, .off = 0, .len = half },
        { .layer_id = 1, .off = half, .len = half },
    };
    CHECK(dram_put(d, KEY1, 0, 1, 0, buf, half * 2, recs, 2) == KV_EOK);
    uint8_t *out = NULL;
    uint32_t out_len = 0;
    dram_rec orecs[4];
    uint16_t on = 0;
    CHECK(dram_get(d, KEY1, 0, 1, &out, &out_len, orecs, 4, &on) == KV_EOK);
    CHECK(out != NULL);
    CHECK(out_len == half * 2 && on == 2);
    CHECK(out[0] == 0x77 && out[out_len - 1] == 0x77);
    aligned_free(out);
    dram_invalidate(d, KEY1, 0);
    CHECK(dram_stat(d, "dram_entries") == 0);
    free(buf);
    dram_destroy(d);
}

/* ---------------- cache-level coherence -------------------------------- */

#define DEV0 "/tmp/kvdram_dev0.img"

static volatile int acked = -999;
static void on_ack(void *u, int rc) { (void)u; acked = rc; }

static int put_sync(cache_t *c, uint64_t prefix, uint8_t val, uint32_t len) {
    acked = -999;
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
    uint8_t payload[4096];
    memset(payload, val, sizeof(payload));
    kv_data_ref recs[2] = {
        { .base = payload, .off = 0, .len = len },
        { .base = payload, .off = 0, .len = len },
    };
    int rc = cache_put(c, prefix, 0, tokens, 32, 0, 2, recs, on_ack, NULL);
    if (rc != KV_EOK) return rc;
    for (int i = 0; i < 2000000 && acked == -999; i++) usleep(100);
    return acked;
}

static bool get_val(cache_t *c, uint64_t prefix, uint8_t val) {
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
    cache_get_result res;
    int rc = cache_get(c, prefix, tokens, 32, &res);
    if (rc != KV_EOK)
        return false;
    bool ok = res.n_records == 2 && res.buf_len >= 1 &&
              ((uint8_t *)res.buf)[res.recs[0].off] == val &&
              ((uint8_t *)res.buf)[res.recs[1].off] == val;
    cache_result_free(&res);
    return ok;
}

static void test_cache_coherence(void) {
    unlink(DEV0); unlink(DEV0 ".ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 2;
    cfg.region_size_pages = 512;
    cfg.dram_cache_bytes = 256 * 1024;
    cfg.dram_entry_max_bytes = 64 * 1024;
    const char *uris[] = { DEV0 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 1, &cfg) == KV_EOK);
    CHECK(c != NULL);

    /* miss -> IO -> backfill; second get must be a DRAM hit */
    CHECK(put_sync(c, 100, 0x42, 1024) == KV_EOK);
    CHECK(get_val(c, 100, 0x42));               /* miss + backfill */
    uint64_t hits0 = cache_stats(c, "dram_hits");
    CHECK(get_val(c, 100, 0x42));               /* dram hit */
    CHECK(cache_stats(c, "dram_hits") == hits0 + 1);
    CHECK(get_val(c, 100, 0x42));               /* dram hit again */

    /* overwrite: new ver must invalidate the cached entry */
    CHECK(put_sync(c, 100, 0x99, 1024) == KV_EOK);
    CHECK(get_val(c, 100, 0x99));
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((100 * 2654435761u) ^ (uint64_t)i);
    cache_get_result res;
    CHECK(cache_get(c, 100, tokens, 32, &res) == KV_EOK);
    CHECK(((uint8_t *)res.buf)[res.recs[0].off] == 0x99);
    cache_result_free(&res);

    /* evict removes the entry too */
    CHECK(cache_evict(c, 100, tokens, 32) == KV_EOK);
    CHECK(cache_get(c, 100, tokens, 32, &res) == -KV_EVICTED);

    /* TTL expiry: never cached, never hit */
    acked = -999;
    {
        for (int i = 0; i < 32; i++)
            tokens[i] = (uint32_t)((200 * 2654435761u) ^ (uint64_t)i);
        uint8_t payload[4096];
        memset(payload, 0x7E, sizeof(payload));
        kv_data_ref recs[1] = {
            { .base = payload, .off = 0, .len = 1024 },
        };
        CHECK(cache_put(c, 200, 0, tokens, 32,
                        (uint32_t)now_s() - 10, 1, recs,
                        on_ack, NULL) == KV_EOK);
        for (int i = 0; i < 2000000 && acked == -999; i++) usleep(100);
        CHECK(acked == KV_EOK);
    }
    CHECK(cache_get(c, 200, tokens, 32, &res) == -KV_ENOENT);

    /* disabled tier: dram stats all zero, cache still works */
    unlink(DEV0); unlink(DEV0 ".ckpt");
    cfg.dram_cache_bytes = 0;
    cache_t *c2 = NULL;
    CHECK(cache_open(&c2, uris, 1, &cfg) == KV_EOK);
    CHECK(put_sync(c2, 300, 0x11, 1024) == KV_EOK);
    CHECK(get_val(c2, 300, 0x11));
    CHECK(cache_stats(c2, "dram_hits") == 0);
    CHECK(cache_stats(c2, "dram_backend") == 0);
    cache_close(c2);

    cache_close(c);
    unlink(DEV0); unlink(DEV0 ".ckpt");
}

int main(void) {
    test_backend_probe();
    test_roundtrip_invalidate();
    test_sieve_eviction();
    test_external_fallback();
    test_cache_coherence();
    if (failures == 0) {
        printf("test_dram: OK\n");
        return 0;
    }
    printf("test_dram: %d failures\n", failures);
    return 1;
}
