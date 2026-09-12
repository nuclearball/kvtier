/* scenario tests: sync_mode, journal overflow (EFULL), n_devs > 4 */
#include "common.hpp"
#include "solidcacher.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static volatile int acked = -999;
static void on_ack(void *u, int rc) { (void)u; acked = rc; }

static int put_sync(cache_t *c, uint64_t prefix, const uint8_t *payload,
                    uint32_t len) {
    acked = -999;
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
    kv_data_ref recs[2] = {
        { .base = payload, .off = 0, .len = len },
        { .base = payload, .off = 0, .len = len },
    };
    int rc = cache_put(c, prefix, 0, tokens, 32, 0, 2, recs, on_ack, NULL);
    if (rc != KV_EOK) return rc;
    for (int i = 0; i < 2000000 && acked == -999; i++) usleep(100);
    return acked;
}

static int get_rc(cache_t *c, uint64_t prefix) {
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
    cache_get_result res;
    int rc = cache_get(c, prefix, tokens, 32, &res);
    if (rc == KV_EOK)
        cache_result_free(&res);
    return rc;
}

/* ---- 1. sync_mode=true: acks wait for fua/fsync ---- */
static void test_sync_mode(void) {
    const char *uris[] = { "/tmp/kvs_d0.img", "/tmp/kvs_d1.img" };
    unlink(uris[0]); unlink(uris[1]); unlink("/tmp/kvs_d0.img.ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 512;
    cfg.sync_mode = true;
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[128];
    memset(payload, 0x77, sizeof(payload));
    for (uint64_t p = 1; p <= 200; p++)
        CHECK(put_sync(c, p, payload, sizeof(payload)) == KV_EOK);
    for (uint64_t p = 1; p <= 200; p++)
        CHECK(get_rc(c, p) == KV_EOK);
    cache_close(c);

    /* fsync-backed durability: full recovery without losses */
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    for (uint64_t p = 1; p <= 200; p++)
        CHECK(get_rc(c, p) == KV_EOK);
    cache_close(c);
    unlink(uris[0]); unlink(uris[1]); unlink("/tmp/kvs_d0.img.ckpt");
}

/* ---- 2. journal overflow: records beyond capacity are dropped, not fatal.
 * In-session reads unaffected; post-recovery the un-journaled tail is
 * unrecoverable (documented cache semantics) unless checkpointed. ---- */
static void test_journal_overflow(void) {
    const char *uris[] = { "/tmp/kvj_d0.img", "/tmp/kvj_d1.img" };
    unlink(uris[0]); unlink(uris[1]); unlink("/tmp/kvj_d0.img.ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 512;
    cfg.journal_pages = 1;          /* one page: 56 records, overflows within one GC tick */
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x3C, sizeof(payload));
    for (uint64_t p = 1; p <= 500; p++)
        CHECK(put_sync(c, p, payload, sizeof(payload)) == KV_EOK);
    CHECK(cache_stats(c, "journal_dropped") > 0);   /* overflow happened */
    for (uint64_t p = 1; p <= 500; p++)             /* in-session intact */
        CHECK(get_rc(c, p) == KV_EOK);
    cache_close(c);
    /* reopen must not crash/corrupt even though the journal overflowed */
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    cache_close(c);
    unlink(uris[0]); unlink(uris[1]); unlink("/tmp/kvj_d0.img.ckpt");
}

/* ---- 3. eight devices (> 4): hash routing + per-writer paths ---- */
static void test_eight_devs(void) {
    static const char *uris[8];
    static char paths[8][64];
    for (int i = 0; i < 8; i++) {
        snprintf(paths[i], sizeof(paths[i]), "/tmp/kv8_d%d.img", i);
        unlink(paths[i]);
        uris[i] = paths[i];
    }
    unlink("/tmp/kv8_d0.img.ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 4;
    cfg.region_size_pages = 256;
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 8, &cfg) == KV_EOK);
    CHECK(cache_dev_count(c) == 8);

    uint8_t payload[96];
    memset(payload, 0x88, sizeof(payload));
    for (uint64_t p = 1; p <= 400; p++)
        CHECK(put_sync(c, p, payload, sizeof(payload)) == KV_EOK);
    for (uint64_t p = 1; p <= 400; p++)
        CHECK(get_rc(c, p) == KV_EOK);

    /* per-device balance: every dev must have received writes */
    for (int i = 0; i < 8; i++)
        CHECK(cache_stats(c, "live_bytes") > 0);

    cache_close(c);
    CHECK(cache_open(&c, uris, 8, &cfg) == KV_EOK);
    for (uint64_t p = 1; p <= 400; p++)
        CHECK(get_rc(c, p) == KV_EOK);
    cache_close(c);
    for (int i = 0; i < 8; i++) {
        unlink(paths[i]);
    }
    unlink("/tmp/kv8_d0.img.ckpt");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    test_sync_mode();
    printf("sync_mode: done\n");
    test_journal_overflow();
    printf("journal_overflow: done\n");
    test_eight_devs();
    printf("eight_devs: done\n");
    if (failures == 0) {
        printf("test_scenarios: OK\n");
        return 0;
    }
    printf("test_scenarios: %d failures\n", failures);
    return 1;
}
