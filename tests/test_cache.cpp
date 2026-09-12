#include "common.hpp"
#include "cache.hpp"
#include "hash.hpp"
#include "radix.hpp"
#include "device.hpp"
#include "kvtier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/stat.h>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

#define DEV0 "/tmp/kv_dev0.img"
#define DEV1 "/tmp/kv_dev1.img"

/* ---------- helpers ---------- */

static void fill_tokens(uint32_t *tok, int n, uint64_t seed) {
    for (int i = 0; i < n; i++)
        tok[i] = (uint32_t)((seed * 2654435761u) ^ (uint64_t)i);
}

static int ack_count;
static void ack_cb(void *user, int rc) {
    (void)user;
    ack_count = rc;
}

/* synchronous put that waits for the ack */
static int put_sync(cache_t *c, uint64_t prefix, uint32_t n_tokens_total,
                    uint32_t expire_ts, const uint8_t *payload,
                    uint32_t payload_len) {
    ack_count = -999;
    uint32_t tokens[128];
    fill_tokens(tokens, n_tokens_total, prefix);
    struct kv_data_ref recs[2] = {
        { .base = payload, .off = 0, .len = payload_len },
        { .base = payload, .off = 0, .len = payload_len },
    };
    uint32_t groups = (n_tokens_total + KV_TOKENS_PER_GROUP - 1) /
                      KV_TOKENS_PER_GROUP;
    int rc = cache_put(c, prefix, groups - 1, tokens, n_tokens_total,
                       expire_ts, 2, recs, ack_cb, NULL);
    if (rc != KV_EOK)
        return rc;
    for (int i = 0; i < 500000 && ack_count == -999; i++)
        usleep(100);
    return ack_count;
}

static int get_verify(cache_t *c, uint64_t prefix, uint32_t n_tokens_total,
                      const uint8_t *expect, uint32_t expect_len) {
    uint32_t tokens[128];
    fill_tokens(tokens, n_tokens_total, prefix);
    struct cache_get_result res;
    int rc = cache_get(c, prefix, tokens, n_tokens_total, &res);
    if (rc != KV_EOK)
        return rc;
    /* find layer 0 */
    for (int i = 0; i < res.n_records; i++) {
        if (res.recs[i].layer_id == 0) {
            int ok = (res.recs[i].len == expect_len) &&
                     memcmp((const uint8_t *)res.buf + res.recs[i].off,
                            expect, expect_len) == 0;
            cache_result_free(&res);
            return ok ? KV_EOK : -99;
        }
    }
    cache_result_free(&res);
    return -98;
}

/* ---------- tests ---------- */

/* put with an arbitrary number of layers (each layer = same payload) */
static int put_layers(cache_t *c, uint64_t prefix, uint32_t n_tokens_total,
                      uint32_t expire_ts, uint16_t n_layers,
                      const uint8_t *payload, uint32_t layer_len) {
    ack_count = -999;
    uint32_t tokens[128];
    fill_tokens(tokens, n_tokens_total, prefix);
    struct kv_data_ref recs[KV_MAX_LAYERS_CAP];
    for (uint16_t i = 0; i < n_layers; i++)
        recs[i] = kv_data_ref{
            .base = payload, .off = 0, .len = layer_len };
    uint32_t groups = (n_tokens_total + KV_TOKENS_PER_GROUP - 1) /
                      KV_TOKENS_PER_GROUP;
    int rc = cache_put(c, prefix, groups - 1, tokens, n_tokens_total,
                       expire_ts, n_layers, recs, ack_cb, NULL);
    if (rc != KV_EOK)
        return rc;
    for (int i = 0; i < 500000 && ack_count == -999; i++)
        usleep(100);
    return ack_count;
}

/* get and verify every layer of a multi-layer put */
static int get_verify_layers(cache_t *c, uint64_t prefix,
                             uint32_t n_tokens_total, const uint8_t *expect,
                             uint32_t layer_len, uint16_t expect_layers) {
    uint32_t tokens[128];
    fill_tokens(tokens, n_tokens_total, prefix);
    struct cache_get_result res;
    int rc = cache_get(c, prefix, tokens, n_tokens_total, &res);
    if (rc != KV_EOK)
        return rc;
    if (res.n_records != expect_layers) {
        cache_result_free(&res);
        return -97;
    }
    bool seen[KV_MAX_LAYERS_CAP];
    memset(seen, 0, sizeof(seen));
    int ok = 1;
    for (int i = 0; i < res.n_records; i++) {
        uint16_t lid = res.recs[i].layer_id;
        if (lid >= expect_layers || seen[lid] ||
            res.recs[i].len != layer_len ||
            memcmp((const uint8_t *)res.buf + res.recs[i].off,
                   expect, layer_len) != 0) {
            ok = 0;
            break;
        }
        seen[lid] = true;
    }
    cache_result_free(&res);
    return ok ? KV_EOK : -96;
}

/* runtime max_layers: reject above the limit, accept at the limit */
static void test_max_layers_limit(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;

    const char *uris[] = { DEV0, DEV1 };

    /* config validation */
    struct kv_config bad = cfg;
    bad.max_layers = KV_MAX_LAYERS_CAP + 1;
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &bad) ==
          -KV_EINVAL);
    bad = cfg;
    bad.batch_min_bytes = KV_PAGE_SIZE + 512;   /* not 4K aligned */
    CHECK(cache_open(&c, uris, 2, &bad) ==
          -KV_EINVAL);
    bad = cfg;
    bad.batch_min_bytes = (uint32_t)bad.stripe_threshold + KV_PAGE_SIZE;
    CHECK(cache_open(&c, uris, 2, &bad) ==
          -KV_EINVAL);

    cfg.max_layers = 8;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[1024];
    for (size_t i = 0; i < sizeof(payload); i++)
        payload[i] = (uint8_t)(i * 7);
    CHECK(put_layers(c, 1, 32, 0, 9, payload, sizeof(payload)) == -KV_EINVAL);
    CHECK(put_layers(c, 1, 32, 0, 8, payload, sizeof(payload)) == KV_EOK);
    CHECK(get_verify_layers(c, 1, 32, payload, sizeof(payload), 8) == KV_EOK);
    cache_close(c);
}

/* 128 layers roundtrip + persistence across clean close/reopen */
static void test_layers_128(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[1024];
    for (size_t i = 0; i < sizeof(payload); i++)
        payload[i] = (uint8_t)(i ^ 0x5A);
    CHECK(put_layers(c, 1, 32, 0, 128, payload, sizeof(payload)) == KV_EOK);
    CHECK(get_verify_layers(c, 1, 32, payload, sizeof(payload), 128) ==
          KV_EOK);
    cache_close(c);

    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    CHECK(get_verify_layers(c, 1, 32, payload, sizeof(payload), 128) ==
          KV_EOK);
    cache_close(c);
}

/* small batch_min flushes (nearly) every put as its own batch */
static void test_batch_min(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;
    cfg.batch_min_bytes = KV_PAGE_SIZE;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x3C, sizeof(payload));
    for (uint64_t p = 1; p <= 8; p++)
        CHECK(put_sync(c, p, 32, 0, payload, sizeof(payload)) == KV_EOK);
    CHECK(cache_stats(c, "batches") >= 8);
    for (uint64_t p = 1; p <= 8; p++)
        CHECK(get_verify(c, p, 32, payload, sizeof(payload)) == KV_EOK);
    cache_close(c);
}

/* stripe_threshold above the compile-time default must not overflow the
 * (now dynamically sized) batch staging buffer */
static void test_big_stripe_threshold(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;           /* 16 MiB per region */
    cfg.epoch_secs = 3600;
    cfg.stripe_threshold = 2ull * 1024 * 1024;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint32_t len = 1536 * 1024;             /* 1.5 MiB, non-striped */
    uint8_t *payload = static_cast<uint8_t *>(malloc(len));
    CHECK(payload != NULL);
    for (uint32_t i = 0; i < len; i++)
        payload[i] = (uint8_t)(i >> 4);
    CHECK(put_layers(c, 1, 32, 0, 1, payload, len) == KV_EOK);
    CHECK(get_verify_layers(c, 1, 32, payload, len, 1) == KV_EOK);
    cache_close(c);

    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    CHECK(get_verify_layers(c, 1, 32, payload, len, 1) == KV_EOK);
    cache_close(c);
    free(payload);
}


static void test_put_get_evict(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;   /* 16 MiB per region */
    cfg.epoch_secs = 3600;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    CHECK(cache_dev_count(c) == 2);

    uint8_t payload[64];
    memset(payload, 0xAB, sizeof(payload));
    CHECK(put_sync(c, 1, 32, 0, payload, sizeof(payload)) == KV_EOK);
    CHECK(get_verify(c, 1, 32, payload, sizeof(payload)) == KV_EOK);
    CHECK(cache_stats(c, "hits") == 1);

    /* miss */
    uint32_t tokens[32];
    fill_tokens(tokens, 32, 999);
    struct cache_get_result res;
    CHECK(cache_get(c, 999, tokens, 32, &res) == -KV_ENOENT);

    /* overwrite with newer payload */
    uint8_t payload2[64];
    memset(payload2, 0xCD, sizeof(payload2));
    CHECK(put_sync(c, 1, 32, 0, payload2, sizeof(payload2)) == KV_EOK);
    CHECK(get_verify(c, 1, 32, payload2, sizeof(payload2)) == KV_EOK);

    /* evict then miss: evicted keys report KV_EVICTED (tombstone), while
     * never-existing keys report KV_ENOENT */
    fill_tokens(tokens, 32, 1);
    CHECK(cache_evict(c, 1, tokens, 32) == KV_EOK);
    CHECK(cache_get(c, 1, tokens, 32, &res) == -KV_EVICTED);

    cache_close(c);
}

static void test_ttl(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;
    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[32];
    memset(payload, 0x11, sizeof(payload));
    /* expire 2 seconds from now; sleep past it (time has 1s resolution) */
    uint32_t now = (uint32_t)time(NULL);
    CHECK(put_sync(c, 2, 32, now + 1, payload, sizeof(payload)) == KV_EOK);
    CHECK(get_verify(c, 2, 32, payload, sizeof(payload)) == KV_EOK);
    usleep(2500 * 1000);
    uint32_t tokens[32];
    fill_tokens(tokens, 32, 2);
    struct cache_get_result res;
    /* either expired or the GC TTL pass dropped it -> must be a miss */
    CHECK(cache_get(c, 2, tokens, 32, &res) == -KV_ENOENT);

    cache_close(c);
}

static void test_recovery(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    /* write several prefixes */
    for (uint64_t p = 1; p <= 20; p++) {
        uint8_t payload[32];
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        CHECK(put_sync(c, p, 32, 0, payload, sizeof(payload)) == KV_EOK);
    }
    cache_close(c);   /* clean close */

    /* reopen */
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    for (uint64_t p = 1; p <= 20; p++) {
        uint8_t payload[32];
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        CHECK(get_verify(c, p, 32, payload, sizeof(payload)) == KV_EOK);
    }
    cache_close(c);
}

/* crash simulation: fork a child that writes and _exit(2) without close */
static void test_crash_recovery(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;

    const char *uris[] = { DEV0, DEV1 };

    /* first format */
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    cache_close(c);

    pid_t pid = fork();
    if (pid == 0) {
        /* child: reopen, write, then die without close */
        cache_t *cc = NULL;
        if (cache_open(&cc, uris, 2, &cfg) != KV_EOK)
            _exit(3);
        for (uint64_t p = 100; p <= 130; p++) {
            uint8_t payload[32];
            memset(payload, (int)(p & 0xFF), sizeof(payload));
            if (put_sync(cc, p, 32, 0, payload, sizeof(payload)) != KV_EOK)
                _exit(4);
        }
        _exit(0);   /* deliberate crash: no cache_close */
    }
    int status;
    CHECK(waitpid(pid, &status, 0) == pid);

    /* parent reopens and verifies */
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    for (uint64_t p = 100; p <= 130; p++) {
        uint8_t payload[32];
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        CHECK(get_verify(c, p, 32, payload, sizeof(payload)) == KV_EOK);
    }
    cache_close(c);
}

/* forced rotation + gc: small regions force many rotations */static void test_rotation_gc(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 4;
    cfg.region_size_pages = 128;   /* 512 KiB regions -> frequent rotation */
    cfg.epoch_secs = 3600;
    cfg.gc_high_water_pct = 30;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    /* write enough distinct prefixes to force many rotations */
    for (uint64_t p = 1000; p <= 1500; p++) {
        uint8_t payload[64];
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        CHECK(put_sync(c, p, 32, 0, payload, sizeof(payload)) == KV_EOK);
    }
    CHECK(cache_stats(c, "rotations") >= 2);

    /* many should still be readable (zero-copy GC keeps them) */
    int found = 0;
    for (uint64_t p = 1000; p <= 1500; p++) {
        uint8_t payload[64];
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        if (get_verify(c, p, 32, payload, sizeof(payload)) == KV_EOK)
            found++;
    }
    CHECK(found > 0);
    cache_close(c);
}

/* stripe with > 32 parts (stream > 8MiB): regression for BUG-002.
 * verifies EVERY layer (deep layers used to be silently lost) + recovery. */
static void test_stripe_many_parts(void) {
    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;           /* 96 MiB per dev */
    cfg.epoch_secs = 3600;

    cache_t *c = NULL;
    const char *uris[] = { DEV0, DEV1 };
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint32_t len = 65536;                   /* 126 layers -> 32 parts */
    uint8_t *payload = static_cast<uint8_t *>(malloc(len));
    CHECK(payload != NULL);
    for (uint32_t i = 0; i < len; i++)
        payload[i] = (uint8_t)(i * 3);
    CHECK(put_layers(c, 1, 32, 0, 126, payload, len) == KV_EOK);
    CHECK(get_verify_layers(c, 1, 32, payload, len, 126) == KV_EOK);

    len = 69632;                            /* 126 layers -> 35 parts (was broken) */
    free(payload);
    payload = static_cast<uint8_t *>(malloc(len));
    CHECK(payload != NULL);
    for (uint32_t i = 0; i < len; i++)
        payload[i] = (uint8_t)(i >> 2);
    CHECK(put_layers(c, 2, 32, 0, 126, payload, len) == KV_EOK);
    CHECK(get_verify_layers(c, 2, 32, payload, len, 126) == KV_EOK);
    cache_close(c);

    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    CHECK(get_verify_layers(c, 2, 32, payload, len, 126) == KV_EOK);
    cache_close(c);
    free(payload);
}

int main(void) {
    unlink(DEV0); unlink(DEV1);
    unlink("/tmp/kv_dev0.img.ckpt");
    test_put_get_evict();
    test_max_layers_limit();
    test_layers_128();
    test_batch_min();
    test_big_stripe_threshold();
    test_stripe_many_parts();
    test_ttl();
    test_recovery();
    test_crash_recovery();
    test_rotation_gc();
    unlink(DEV0); unlink(DEV1);
    unlink("/tmp/kv_dev0.img.ckpt");
    if (failures == 0) {
        printf("test_cache: OK\n");
        return 0;
    }
    printf("test_cache: %d failures\n", failures);
    return 1;
}
