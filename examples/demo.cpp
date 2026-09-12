/*
 * minimal demo: put/get a few prefixes against a 2-device file-backed cache,
 * print stats, close, reopen, verify recovery.
 */
#include "kvtier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEV0 "/tmp/kvcache_dev0.img"
#define DEV1 "/tmp/kvcache_dev1.img"

static volatile int acked = -999;
static void on_ack(void *user, int rc) { (void)user; acked = rc; }

static void fill_tokens(uint32_t *tok, int n, uint64_t seed) {
    for (int i = 0; i < n; i++)
        tok[i] = (uint32_t)((seed * 2654435761u) ^ (uint64_t)i);
}

static int put_sync(cache_t *c, uint64_t prefix, uint32_t ntokens,
                    const uint8_t *payload, size_t len) {
    acked = -999;
    uint32_t tokens[128];
    fill_tokens(tokens, ntokens, prefix);
    struct kv_data_ref recs[2] = {
        { .base = payload, .off = 0, .len = (uint32_t)len },
        { .base = payload, .off = 0, .len = (uint32_t)len },
    };
    uint32_t groups = (ntokens + KV_TOKENS_PER_GROUP - 1) / KV_TOKENS_PER_GROUP;
    int rc = cache_put(c, prefix, groups - 1, tokens, ntokens, 0, 2, recs,
                       on_ack, NULL);
    while (acked == -999)
        usleep(100);
    return rc;
}

static int get_verify(cache_t *c, uint64_t prefix, uint32_t ntokens,
                      const uint8_t *expect, size_t len) {
    uint32_t tokens[128];
    fill_tokens(tokens, ntokens, prefix);
    struct cache_get_result res;
    int rc = cache_get(c, prefix, tokens, ntokens, &res);
    if (rc != KV_EOK)
        return rc;
    int ok = 0;
    for (int i = 0; i < res.n_records; i++) {
        if (res.recs[i].layer_id == 0 &&
            res.recs[i].len == len &&
            memcmp((const uint8_t *)res.buf + res.recs[i].off, expect, len) == 0)
            ok = 1;
    }
    cache_result_free(&res);
    return ok ? KV_EOK : -1;
}

int main(void) {
    unlink(DEV0); unlink(DEV1);
    unlink("/tmp/kvcache_dev0.img.ckpt");

    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;

    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    if (cache_open(&c, uris, 2, &cfg) != KV_EOK) {
        fprintf(stderr, "open failed\n");
        return 1;
    }

    uint8_t payload[64];
    for (uint64_t p = 1; p <= 100; p++) {
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        if (put_sync(c, p, 32, payload, sizeof(payload)) != KV_EOK)
            fprintf(stderr, "put %llu failed\n", (unsigned long long)p);
    }
    int hits = 0;
    for (uint64_t p = 1; p <= 100; p++) {
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        if (get_verify(c, p, 32, payload, sizeof(payload)) == KV_EOK)
            hits++;
    }
    printf("session: hits=%d/100  leaves=%llu  writes=%lluKiB\n",
           hits, (unsigned long long)cache_stats(c, "leaves"),
           (unsigned long long)cache_stats(c, "bytes_written") / 1024);
    cache_close(c);

    /* reopen and re-verify (journal replay + checkpoint) */
    if (cache_open(&c, uris, 2, &cfg) != KV_EOK) {
        fprintf(stderr, "reopen failed\n");
        return 1;
    }
    int recovered = 0;
    for (uint64_t p = 1; p <= 100; p++) {
        memset(payload, (int)(p & 0xFF), sizeof(payload));
        if (get_verify(c, p, 32, payload, sizeof(payload)) == KV_EOK)
            recovered++;
    }
    printf("recovered after reopen: %d/100\n", recovered);
    cache_close(c);
    unlink(DEV0); unlink(DEV1);
    unlink("/tmp/kvcache_dev0.img.ckpt");
    return (hits == 100 && recovered == 100) ? 0 : 1;
}
