/* test_layout: model-aware layout via kv_mp_config_values -> kv_config */
#include "common.hpp"
#include "kvtier.h"
#include "model_profile.h"

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

#define DEV0 "/tmp/kv_layout_d0.img"
#define DEV1 "/tmp/kv_layout_d1.img"
#define PAGE 4096

static volatile int ack_count = -999;
static void ack_cb(void *user, int rc) { (void)user; ack_count = rc; }

static void fill_tokens(uint32_t *t, uint32_t n, uint64_t prefix) {
    for (uint32_t i = 0; i < n; i++)
        t[i] = (uint32_t)(prefix * 131 + i * 7 + 1);
}

static int put_layers(cache_t *c, uint64_t prefix, uint16_t n_layers,
                      const uint8_t *layer, uint32_t layer_len) {
    ack_count = -999;
    uint32_t tokens[64];
    fill_tokens(tokens, 32, prefix);
    kv_data_ref recs[KV_MAX_LAYERS_CAP];
    for (uint16_t i = 0; i < n_layers; i++)
        recs[i] = kv_data_ref{ .base = layer, .off = 0,
                               .len = layer_len };
    int rc = cache_put(c, prefix, 0, tokens, 32, 0, n_layers, recs,
                       ack_cb, NULL);
    if (rc != KV_EOK)
        return rc;
    for (int i = 0; i < 2000000 && ack_count == -999; i++)
        usleep(100);
    return ack_count;
}

static int get_layers(cache_t *c, uint64_t prefix, uint16_t expect_layers,
                      const uint8_t *layer, uint32_t layer_len) {
    uint32_t tokens[64];
    fill_tokens(tokens, 32, prefix);
    cache_get_result res;
    int rc = cache_get(c, prefix, tokens, 32, &res);
    if (rc != KV_EOK)
        return rc;
    if (res.n_records != expect_layers) {
        cache_result_free(&res);
        return -97;
    }
    for (int i = 0; i < res.n_records; i++) {
        if (res.recs[i].len != layer_len ||
            memcmp((const uint8_t *)res.buf + res.recs[i].off, layer,
                   layer_len) != 0) {
            cache_result_free(&res);
            return -96;
        }
    }
    cache_result_free(&res);
    return KV_EOK;
}

/* DeepSeek MLA: 1152 B/token/层, 32-token group = 36864 B/层 (9 页恰好对齐).
 * stripe_unit 必须既能整除 36864 又能整除 4096 -> lcm = 36864；取 256KiB
 * 的上整倍数 = 294912 (8 x 36864)。 */
static void test_mla_layout(void) {
    unlink(DEV0); unlink(DEV1);
    unlink(DEV0 ".ckpt");

    const kv_model_profile *mp = kv_mp_by_name("deepseek-v3");
    CHECK(mp != NULL);

    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;
    cfg.dram_cache_bytes = 0;
    struct kv_mp_config_values v;
    CHECK(kv_mp_config_values(mp, PAGE, &v) == KV_MP_OK);
    cfg.n_groups_tokens = v.n_groups_tokens;
    cfg.max_layers = v.max_layers;
    cfg.stripe_unit = v.stripe_unit;
    cfg.stripe_threshold = v.stripe_threshold;
    cfg.batch_min_bytes = v.batch_min_bytes;
    cfg.region_align_bytes = v.region_align_bytes;
    cfg.dram_entry_max_bytes = v.dram_entry_max_bytes;

    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    if (!c) return;

    CHECK(cache_stats(c, "stripe_unit") == 294912ull);
    CHECK(cache_stats(c, "stripe_threshold") == 262144ull);
    CHECK(cache_stats(c, "batch_min_bytes") == 262144ull);

    /* 61 层 x 36864 B = 2.14 MiB/put -> 条带 (8 片) */
    const uint16_t n_layers = 61;
    const uint32_t layer_len = 1152 * 32;   /* 36864 */
    uint8_t *layer = (uint8_t *)malloc(layer_len);
    CHECK(layer != NULL);
    if (!layer) { cache_close(c); return; }
    for (uint32_t i = 0; i < layer_len; i++)
        layer[i] = (uint8_t)(i * 31 + 7);

    for (uint64_t p = 1; p <= 6; p++) {
        layer[0] = (uint8_t)p;
        CHECK(put_layers(c, p, n_layers, layer, layer_len) == KV_EOK);
    }
    for (uint64_t p = 1; p <= 6; p++) {
        layer[0] = (uint8_t)p;
        CHECK(get_layers(c, p, n_layers, layer, layer_len) == KV_EOK);
    }
    cache_close(c);
    free(layer);

    /* recovery: reopen with the same profile */
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    if (!c) return;
    layer = (uint8_t *)malloc(layer_len);
    for (uint32_t i = 0; i < layer_len; i++)
        layer[i] = (uint8_t)(i * 31 + 7);
    for (uint64_t p = 1; p <= 6; p++) {
        layer[0] = (uint8_t)p;
        CHECK(get_layers(c, p, n_layers, layer, layer_len) == KV_EOK);
    }
    cache_close(c);
    free(layer);
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
}

/* legacy path (model == NULL) keeps the compile-time constants */
static void test_legacy_default(void) {
    unlink(DEV0);
    unlink(DEV0 ".ckpt");
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 6;
    cfg.region_size_pages = 4096;
    cfg.epoch_secs = 3600;
    cfg.dram_cache_bytes = 0;

    const char *uris[] = { DEV0 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 1, &cfg) == KV_EOK);
    if (!c) return;
    CHECK(cache_stats(c, "stripe_unit") == KV_STRIPE_UNIT);
    CHECK(cache_stats(c, "stripe_threshold") == KV_STRIPE_THRESHOLD);
    cache_close(c);
    unlink(DEV0); unlink(DEV0 ".ckpt");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    test_mla_layout();
    test_legacy_default();
    if (failures == 0) {
        printf("test_layout: OK\n");
        return 0;
    }
    printf("test_layout: %d failures\n", failures);
    return 1;
}
