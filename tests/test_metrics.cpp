#include "solidcacher.h"
#include "metrics/metrics.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace sc;

#ifndef KV_METRICS_ON
#define KV_METRICS_ON 1
#endif

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

#define DEV0 "/tmp/kvm_dev0.img"
#define DEV1 "/tmp/kvm_dev1.img"

static volatile int acked = -999;
static void on_ack(void *u, int rc) { (void)u; acked = rc; }

static void test_hist(void) {
    kv_hist h;
    kv_hist_init(&h);
    CHECK(kv_hist_count(&h) == 0);
    /* deterministic samples: 900, 1100, 3000, 5000 ns */
    uint64_t samples[] = { 900, 1100, 3000, 5000 };
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); i++)
        kv_hist_sample(&h, samples[i]);
    CHECK(kv_hist_count(&h) == 4);
    CHECK(kv_hist_avg(&h) == 2500);
    CHECK(kv_hist_max(&h) == 5000);
    /* buckets are log2: verify raw bucket placement */
    CHECK(h.count[9].load() == 1);    /* 900  -> [512,1024)   */
    CHECK(h.count[10].load() == 1);   /* 1100 -> [1024,2048)  */
    CHECK(h.count[11].load() == 1);   /* 3000 -> [2048,4096)  */
    CHECK(h.count[12].load() == 1);   /* 5000 -> [4096,8192)  */
    /* percentile = upper bucket bound of the ceil-target bucket */
    CHECK(kv_hist_percentile(&h, 25) == 1024);
    CHECK(kv_hist_percentile(&h, 50) == 2048);
    CHECK(kv_hist_percentile(&h, 100) == 8192);
    /* merge adds counts */
    kv_hist h2;
    kv_hist_init(&h2);
    kv_hist_sample(&h2, 900);
    kv_hist_merge(&h, &h2);
    CHECK(kv_hist_count(&h) == 5);
    CHECK(h.count[9].load() == 2);
}

static int put_sync(cache_t *c, uint64_t prefix, uint32_t ntok,
                    const uint8_t *payload, uint32_t len) {
    acked = -999;
    uint32_t tokens[64];
    for (uint32_t i = 0; i < ntok; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
    kv_data_ref recs[2] = {
        { .base = payload, .off = 0, .len = len },
        { .base = payload, .off = 0, .len = len },
    };
    uint32_t groups = (ntok + 31) / 32;
    int rc = cache_put(c, prefix, groups - 1, tokens, ntok, 0, 2, recs,
                       on_ack, NULL);
    if (rc != KV_EOK) return rc;
    for (int i = 0; i < 2000000 && acked == -999; i++) usleep(200);
    return acked;
}

static void test_snapshot_counters(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    CHECK(cfg.metrics_level == KV_MLEVEL_BASIC);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x5A, sizeof(payload));
    for (uint64_t p = 1; p <= 10; p++)
        CHECK(put_sync(c, p, 32, payload, sizeof(payload)) == KV_EOK);

    kv_metrics_snap s;
    CHECK(kv_metrics_snapshot(c, &s) == KV_EOK);
    CHECK(s.puts == 10);
    CHECK(s.puts_err == 0);
    /* 10 puts x 2 layers x 64B */
    CHECK(s.payload_bytes == 10 * 2 * 64);
    CHECK(s.n_devs == 2);
    /* WAF: data written (pages incl. headers) over unique payload */
    CHECK(s.waf >= 1.0);
    /* per-device writes sum to the total data-plane writes */
    uint64_t sum = s.dev[0].bytes_written + s.dev[1].bytes_written;
    CHECK(sum == s.bytes_written);

    /* gets: 10 hits + 3 misses */
    for (uint64_t p = 1; p <= 10; p++) {
        uint32_t tokens[32];
        for (int i = 0; i < 32; i++)
            tokens[i] = (uint32_t)((p * 2654435761u) ^ (uint64_t)i);
        cache_get_result res;
        if (cache_get(c, p, tokens, 32, &res) == KV_EOK)
            cache_result_free(&res);
    }
    for (uint64_t p = 100; p <= 102; p++) {
        uint32_t tokens[32];
        for (int i = 0; i < 32; i++)
            tokens[i] = (uint32_t)((p * 2654435761u) ^ (uint64_t)i);
        cache_get_result res;
        CHECK(cache_get(c, p, tokens, 32, &res) == -KV_ENOENT);
    }
    CHECK(kv_metrics_snapshot(c, &s) == KV_EOK);
    CHECK(s.gets_hit == 10 && s.gets_miss == 3);
    CHECK(s.hit_ratio > 0.76 && s.hit_ratio < 0.77);

    /* csv exporter emits the stable key set */
    FILE *fp = fopen("/tmp/kvm_metrics.csv", "w+");
    CHECK(fp != NULL);
    kv_metrics_csv(c, fp);
    fflush(fp);
    rewind(fp);
    char line[128];
    int seen_puts = 0, seen_waf = 0, seen_p50 = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "puts,", 5) == 0) seen_puts = 1;
        if (strncmp(line, "waf,", 4) == 0) seen_waf = 1;
        if (strncmp(line, "put_ack_p50_ns,", 15) == 0) seen_p50 = 1;
    }
    CHECK(seen_puts && seen_waf && seen_p50);
    fclose(fp);

    kv_metrics_dump(c, stderr);
    cache_close(c);
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
}

static void test_full_level_histogram(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.metrics_level = KV_MLEVEL_FULL;
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x33, sizeof(payload));
    for (uint64_t p = 1; p <= 5; p++)
        CHECK(put_sync(c, p, 32, payload, sizeof(payload)) == KV_EOK);

    kv_metrics_snap s;
    CHECK(kv_metrics_snapshot(c, &s) == KV_EOK);
    /* one put_ack sample per put */
    CHECK(kv_hist_count(&s.h_put_ack) == 5);
    CHECK(kv_hist_max(&s.h_put_ack) > 0);
    CHECK(kv_hist_avg(&s.h_put_ack) > 0);

    cache_close(c);
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
}

int main(void) {
    test_hist();
#if KV_METRICS_ON
    test_snapshot_counters();
#else
    printf("test_metrics: counter tests skipped (KV_METRICS_DISABLE)\n");
#endif
    test_full_level_histogram();
    if (failures == 0) {
        printf("test_metrics: OK\n");
        return 0;
    }
    printf("test_metrics: %d failures\n", failures);
    return 1;
}
