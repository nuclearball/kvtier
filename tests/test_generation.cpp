/* generation GC tests (design: 世代划分 / FRU 最老世代丢弃 / 水位滞回 /
 * 不存在的字符串返回不存在, 被丢弃的返回 KV_EVICTED) */
/* white-box access to internals (Makefile compiles with -Isrc) */
#include "solidcacher.h"
#include "common.hpp"
#include "cache.hpp"
#include "radix.hpp"
#include "hot.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mutex>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

#define DEV0 "/tmp/kvg_dev0.img"
#define DEV1 "/tmp/kvg_dev1.img"

static volatile int acked = -999;
static void on_ack(void *u, int rc) { (void)u; acked = rc; }

static void key_tokens(uint64_t prefix, uint32_t *tokens) {
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
}

static int put_sync_exp(cache_t *c, uint64_t prefix, const uint8_t *payload,
                        uint32_t len, uint32_t expire_ts) {
    acked = -999;
    uint32_t tokens[32];
    key_tokens(prefix, tokens);
    struct kv_data_ref recs[2] = {
        { .base = payload, .off = 0, .len = len },
        { .base = payload, .off = 0, .len = len },
    };
    int rc = cache_put(c, prefix, 0, tokens, 32, expire_ts, 2, recs,
                       on_ack, NULL);
    if (rc != KV_EOK) return rc;
    for (int i = 0; i < 2000000 && acked == -999; i++) usleep(100);
    return acked;
}

static int put_sync(cache_t *c, uint64_t prefix, const uint8_t *payload,
                    uint32_t len) {
    return put_sync_exp(c, prefix, payload, len, 0);
}

static int get_rc(cache_t *c, uint64_t prefix) {
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
    struct cache_get_result res;
    int rc = cache_get(c, prefix, tokens, 32, &res);
    if (rc == KV_EOK)
        cache_result_free(&res);
    return rc;
}

/* base config: 2 devices x 6 regions x 2MiB = 24MiB.
 * 1KB payload takes one 4K page/chunk -> 6144 keys capacity. */
static void base_cfg(struct kv_config *cfg) {
    kv_config_default(cfg);
    cfg->region_cnt = 6;
    cfg->region_size_pages = 512;
    cfg->epoch_secs = 3600;
    cfg->gc_start_pct = 80;
    cfg->gc_stop_pct = 60;
    cfg->gc_step_gens = 1;
    cfg->min_gen_age_secs = 0;
}

/* ---------------- test 1: watermark + FRU + hysteresis ---------------- */
static void test_fru_hysteresis(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    const uint32_t vlen = 1024;
    const uint64_t n_keys = 5700;   /* 93% of capacity */
    uint8_t *payload = (uint8_t *)malloc(vlen);

    int wfail = 0;
    for (uint64_t p = 1; p <= n_keys; p++) {
        memset(payload, (int)(p & 0xFF), vlen);
        if (put_sync(c, p, payload, vlen) != KV_EOK) wfail++;
    }
    CHECK(wfail == 0);

    uint64_t live = cache_stats(c, "live_bytes");
    uint64_t cap = cache_stats(c, "capacity_bytes");
    printf("[fru] after write: live=%lluMiB/%lluMiB (%.0f%%) leaves=%llu\n",
           (unsigned long long)(live >> 20), (unsigned long long)(cap >> 20),
           100.0 * (double)live / (double)cap,
           (unsigned long long)cache_stats(c, "leaves"));
    CHECK(live * 100 > cap * 80);
    /* nonexistent string must report not-exists (never existed) */
    CHECK(get_rc(c, n_keys + 12345) == -KV_ENOENT);

    sleep(5);   /* GC ticks (2s gate): drain below the stop watermark */

    live = cache_stats(c, "live_bytes");
    uint64_t leaves1 = cache_stats(c, "leaves");
    printf("[fru] after GC  : live=%lluMiB (%.0f%%) leaves=%llu "
           "gc_evicted=%llu triggers=%llu\n",
           (unsigned long long)(live >> 20),
           100.0 * (double)live / (double)cap,
           (unsigned long long)leaves1,
           (unsigned long long)cache_stats(c, "gc_evicted_leaves"),
           (unsigned long long)cache_stats(c, "gc_triggers"));
    CHECK(cache_stats(c, "gc_triggers") >= 1);
    CHECK(cache_stats(c, "gc_evicted_leaves") > 0);
    CHECK(leaves1 < (uint64_t)cache_stats(c, "leaves") + leaves1); /* noop guard */
    CHECK(live * 100 <= cap * 60);          /* hysteresis: below stop line */

    /* FRU: oldest generations evicted -> KV_EVICTED (not ENOENT) */
    printf("[fru] get(1)=%d get(100)=%d\n", get_rc(c, 1), get_rc(c, 100));
    CHECK(get_rc(c, 1) == -KV_EVICTED);
    CHECK(get_rc(c, 100) == -KV_EVICTED);
    /* newest generation survives */
    memset(payload, (int)(n_keys & 0xFF), vlen);
    CHECK(get_rc(c, n_keys) == KV_EOK);
    memset(payload, (int)((n_keys - 500) & 0xFF), vlen);
    CHECK(get_rc(c, n_keys - 500) == KV_EOK);
    /* never-existing still ENOENT */
    CHECK(get_rc(c, n_keys + 99999) == -KV_ENOENT);

    /* recovery: evicted stay gone (ENOENT after reopen, tomb is runtime) */
    cache_close(c);
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);
    CHECK(get_rc(c, 1) == -KV_ENOENT);
    memset(payload, (int)(n_keys & 0xFF), vlen);
    CHECK(get_rc(c, n_keys) == KV_EOK);
    cache_close(c);
    free(payload);
}

/* --------- test 2: min_gen_age protects freshly written generations ---- */
static void test_min_gen_age(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    cfg.min_gen_age_secs = 3600;    /* nothing is old enough to evict */
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    const uint32_t vlen = 1024;
    const uint64_t n_keys = 5700;
    uint8_t *payload = (uint8_t *)malloc(vlen);
    for (uint64_t p = 1; p <= n_keys; p++) {
        memset(payload, (int)(p & 0xFF), vlen);
        put_sync(c, p, payload, vlen);
    }
    sleep(4);
    printf("[age] live=%lluMiB leaves=%llu gc_evicted=%llu\n",
           (unsigned long long)(cache_stats(c, "live_bytes") >> 20),
           (unsigned long long)cache_stats(c, "leaves"),
           (unsigned long long)cache_stats(c, "gc_evicted_leaves"));
    CHECK(cache_stats(c, "gc_evicted_leaves") == 0);   /* nothing dropped */
    memset(payload, (int)(1 & 0xFF), vlen);
    CHECK(get_rc(c, 1) == KV_EOK);                     /* oldest intact */
    cache_close(c);
    free(payload);
}

/* ------------------- test 3: manual evict -> KV_EVICTED --------------- */
static void test_evict_tombstone(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x11, sizeof(payload));
    CHECK(put_sync(c, 7, payload, sizeof(payload)) == KV_EOK);
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++)
        tokens[i] = (uint32_t)((7 * 2654435761u) ^ (uint64_t)i);
    CHECK(cache_evict(c, 7, tokens, 32) == KV_EOK);
    CHECK(get_rc(c, 7) == -KV_EVICTED);         /* evicted, not ENOENT */
    CHECK(get_rc(c, 8) == -KV_ENOENT);          /* never existed */
    cache_close(c);
}

/* ------- test 4: A13 - post-recovery writes survive generational GC ---- */
static void test_recovery_write_gc(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    const uint32_t vlen = 1024;
    uint8_t *payload = (uint8_t *)malloc(vlen);
    /* pre-recovery: 4000 keys = 16.4MiB = 71% */
    for (uint64_t p = 1; p <= 4000; p++) {
        memset(payload, (int)(p & 0xFF), vlen);
        put_sync(c, p, payload, vlen);
    }
    cache_close(c);

    /* recovery: every writer must resume on a NEW-epoch OPEN region */
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    /* post-recovery: 1000 new keys push live to ~85% -> GC drains */
    for (uint64_t p = 60001; p <= 61000; p++) {
        memset(payload, (int)(p & 0xFF), vlen);
        CHECK(put_sync(c, p, payload, vlen) == KV_EOK);
    }
    sleep(5);
    printf("[a13] live=%lluMiB gc_evicted=%llu\n",
           (unsigned long long)(cache_stats(c, "live_bytes") >> 20),
           (unsigned long long)cache_stats(c, "gc_evicted_leaves"));
    /* A13 symptom: writer1's fresh data lived in an old-epoch FROZEN
     * region and was discarded by the very first GC round */
    int lost = 0;
    for (uint64_t p = 60001; p <= 61000; p++)
        if (get_rc(c, p) != KV_EOK) lost++;
    CHECK(lost == 0);
    CHECK(cache_stats(c, "gc_evicted_leaves") > 0);   /* GC did drain */
    CHECK(get_rc(c, 1) == -KV_EVICTED || get_rc(c, 1) == -KV_ENOENT);
    cache_close(c);
    free(payload);
}

/* poll until cache_stats(c, "leaves") reaches target (GC tick ~2s jitter) */
static uint64_t wait_leaves(cache_t *c, uint64_t target, int timeout_s) {
    for (int i = 0; i < timeout_s * 2; i++) {
        if (cache_stats(c, "leaves") == target) break;
        usleep(500 * 1000);
    }
    return cache_stats(c, "leaves");
}

/* ---- test 5: TTL expiry pass (gc.c TTL sweep) ----
 * expired unpinned leaf is dropped by the TTL pass; an expired leaf
 * pinned by a reader survives the pass and is dropped only after unpin */
static void test_ttl_expiry(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    cfg.gc_start_pct = 200;         /* keep generational GC out */
    cfg.dram_cache_bytes = 0;
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x22, sizeof(payload));
    uint32_t exp = (uint32_t)now_s() + 2;
    CHECK(put_sync_exp(c, 101, payload, sizeof(payload), exp) == KV_EOK);
    CHECK(put_sync_exp(c, 102, payload, sizeof(payload), exp) == KV_EOK);
    uint64_t leaves0 = cache_stats(c, "leaves");
    CHECK(leaves0 == 2);
    CHECK(get_rc(c, 101) == KV_EOK);        /* not expired yet */

    /* pin key 102's leaf so the TTL pass must skip it */
    struct radix_node *pin = side_get(&c->side, 102, 0);
    CHECK(pin && pin->leaf);
    pin->leaf->refcnt.fetch_add(1);

    sleep(4);   /* expire (2s) + at least one GC tick (TTL gate = 1s) */

    /* both report ENOENT: cache_get checks expire_ts before touching data */
    CHECK(get_rc(c, 101) == -KV_ENOENT);
    CHECK(get_rc(c, 102) == -KV_ENOENT);
    /* unpinned leaf swept, pinned leaf still in the index */
    uint64_t leaves1 = wait_leaves(c, leaves0 - 1, 8);
    printf("[ttl] leaves=%llu (was %llu, target %llu)\n",
           (unsigned long long)leaves1,
           (unsigned long long)leaves0,
           (unsigned long long)(leaves0 - 1));
    CHECK(leaves1 == leaves0 - 1);

    /* unpin -> next tick sweeps it too */
    pin->leaf->refcnt.fetch_sub(1);
    CHECK(wait_leaves(c, leaves0 - 2, 8) == leaves0 - 2);
    cache_close(c);
}

/* ---- test 6: reader pin (refcnt) ----
 * pinned leaf: evict -> KV_EBUSY, data stays readable; GC that drops the
 * oldest generation keeps the pinned key alive until it is unpinned */
static void test_reader_pin(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    cfg.gc_start_pct = 50;
    cfg.gc_stop_pct = 40;
    cfg.dram_cache_bytes = 0;
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    const uint32_t vlen = 1024;
    const uint64_t n_keys = 3300;   /* ~55% of 24MiB: over the 50% start */
    uint8_t *payload = (uint8_t *)malloc(vlen);

    /* write key 1 and pin it BEFORE the cache crosses the GC watermark */
    memset(payload, 0x33, vlen);
    CHECK(put_sync(c, 1, payload, vlen) == KV_EOK);
    struct radix_node *pin = side_get(&c->side, 1, 0);
    CHECK(pin && pin->leaf);
    pin->leaf->refcnt.fetch_add(1);

    int wfail = 0;
    for (uint64_t p = 2; p <= n_keys; p++) {
        memset(payload, (int)(p & 0xFF), vlen);
        if (put_sync(c, p, payload, vlen) != KV_EOK) wfail++;
    }
    CHECK(wfail == 0);

    sleep(5);   /* GC ticks: trigger + drain */
    printf("[pin] triggers=%llu evicted=%llu live=%lluMiB\n",
           (unsigned long long)cache_stats(c, "gc_triggers"),
           (unsigned long long)cache_stats(c, "gc_evicted_leaves"),
           (unsigned long long)(cache_stats(c, "live_bytes") >> 20));
    CHECK(cache_stats(c, "gc_triggers") >= 1);
    CHECK(cache_stats(c, "gc_evicted_leaves") > 0);
    /* pinned key survives the GC rounds that dropped its generation */
    memset(payload, 0x33, vlen);
    CHECK(get_rc(c, 1) == KV_EOK);

    /* unpin; the GC watermark may already be satisfied (hysteresis), so
     * evict deterministically: evict while pinned -> KV_EBUSY (deferred),
     * after unpin evict succeeds and the key reads back as KV_EVICTED */
    uint32_t tokens[32];
    key_tokens(1, tokens);
    CHECK(cache_evict(c, 1, tokens, 32) == -KV_EBUSY);
    CHECK(get_rc(c, 1) == KV_EOK);
    pin->leaf->refcnt.fetch_sub(1);
    /* tiny race: a draining GC tick may drop the key between unpin and
     * evict (cache_drop_region inserts a tomb for prefix <= 100), so
     * accept either EOK or ENOENT -- the get must read EVICTED either way */
    int erc = cache_evict(c, 1, tokens, 32);
    CHECK(erc == KV_EOK || erc == -KV_ENOENT);
    CHECK(get_rc(c, 1) == -KV_EVICTED);
    cache_close(c);
    free(payload);
}

/* ---- test 7: epoch expiry trims empty FROZEN regions +
 *            writer-side cache_force_free_region ---- */
static void test_epoch_trim_and_force_free(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    cfg.epoch_secs = 1;             /* regions expire almost immediately */
    cfg.gc_start_pct = 200;         /* keep generational GC out */
    cfg.dram_cache_bytes = 0;
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    const uint32_t vlen = 1024;
    uint8_t *payload = (uint8_t *)malloc(vlen);
    /* fill until a rotation freezes a region (512 x 4K pages per region) */
    uint64_t p;
    for (p = 1; p <= 1200 && cache_stats(c, "rotations") == 0; p++) {
        memset(payload, (int)(p & 0xFF), vlen);
        CHECK(put_sync(c, p, payload, vlen) == KV_EOK);
    }
    CHECK(cache_stats(c, "rotations") >= 1);
    uint64_t n_keys = p - 1;

    /* white-box: some FROZEN region with live data must exist now */
    int frozen_dev = -1;
    for (int d = 0; d < c->n_devs && frozen_dev < 0; d++)
        for (uint32_t i = 0; i < c->rms[d].region_cnt; i++)
            if (c->rms[d].rg[i].state == KV_RG_FROZEN &&
                c->rms[d].rg[i].live_bytes > 0) {
                frozen_dev = d; break;
            }
    CHECK(frozen_dev >= 0);

    /* writer-side force-free: drops the least-live FROZEN region and
     * returns it to FREE even though it still has live data */
    int freed;
    c->rms[frozen_dev].rotate_mtx.lock();
    freed = cache_force_free_region(c, &c->writers[frozen_dev], -1);
    c->rms[frozen_dev].rotate_mtx.unlock();
    printf("[ffr] force_free dev=%d idx=%d\n", frozen_dev, freed);
    CHECK(freed >= 0);
    CHECK(c->rms[frozen_dev].rg[freed].state == KV_RG_FREE);

    /* evict everything -> the remaining FROZEN regions go zero-live */
    uint32_t tokens[32];
    for (uint64_t k = 1; k <= n_keys; k++) {
        key_tokens(k, tokens);
        cache_evict(c, k, tokens, 32);
    }

    /* epoch pass (1s expiry, 2s tick): empty FROZEN regions are trimmed */
    sleep(5);
    int stale = 0;
    for (int d = 0; d < c->n_devs; d++)
        for (uint32_t i = 0; i < c->rms[d].region_cnt; i++)
            if (c->rms[d].rg[i].state == KV_RG_FROZEN &&
                c->rms[d].rg[i].live_bytes == 0 &&
                c->rms[d].rg[i].close_ts != 0)
                stale++;
    printf("[epoch] stale empty frozen regions: %d\n", stale);
    CHECK(stale == 0);
    cache_close(c);
    free(payload);
}

/* ---- test 8: hotness decay (gc tick decays radix hit counters) ---- */
static void test_hot_decay(void) {
    unlink(DEV0); unlink(DEV1); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    cfg.hot_hit_threshold = 4;
    cfg.hot_decay_ms = 300;
    cfg.gc_start_pct = 200;
    cfg.dram_cache_bytes = 0;
    const char *uris[] = { DEV0, DEV1 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 2, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x44, sizeof(payload));
    CHECK(put_sync(c, 55, payload, sizeof(payload)) == KV_EOK);

    for (int i = 0; i < 10; i++)
        CHECK(get_rc(c, 55) == KV_EOK);
    struct radix_node *node = side_get(&c->side, 55, 0);
    CHECK(node);
    unsigned hot_before = node->hit_count.load();
    printf("[hot] hit_count before decay: %u\n", hot_before);
    CHECK(hot_before >= 4);

    sleep(3);   /* >= 1 GC tick; decay window 300ms */

    unsigned hot_after = node->hit_count.load();
    printf("[hot] hit_count after decay:  %u\n", hot_after);
    CHECK(hot_after < hot_before);
    cache_close(c);
}

/* ---- test 9: journal fill triggers checkpoint on the GC tick ---- */
static void test_checkpoint_on_journal_fill(void) {
    unlink(DEV0); unlink(DEV0 ".ckpt");
    struct kv_config cfg;
    base_cfg(&cfg);
    cfg.region_cnt = 6;
    cfg.journal_pages = 2;          /* tiny journal: wraps quickly */
    cfg.gc_start_pct = 200;
    cfg.dram_cache_bytes = 0;
    const char *uris[] = { DEV0 };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 1, &cfg) == KV_EOK);

    uint8_t payload[64];
    memset(payload, 0x55, sizeof(payload));
    for (uint64_t p = 1; p <= 200; p++)
        CHECK(put_sync(c, p, payload, sizeof(payload)) == KV_EOK);

    struct journal *j = &c->journals[0];
    uint32_t page_before = j->cur_page;
    uint64_t dropped_before = cache_stats(c, "journal_dropped");
    printf("[ckpt] before: cur_page=%u cur_off=%u dropped=%llu\n",
           j->cur_page, j->cur_off,
           (unsigned long long)dropped_before);
    /* journal really filled: on page 1, near its end, or already dropped */
    CHECK(page_before >= 1 ||
          j->cur_off > (uint32_t)(KV_PAGE_SIZE * 4 / 5) ||
          dropped_before > 0);
    CHECK(get_rc(c, 1) == KV_EOK);

    sleep(4);   /* checkpoint gate is 2s on the GC tick */

    printf("[ckpt] after : cur_page=%u cur_off=%u\n", j->cur_page, j->cur_off);
    CHECK(j->cur_page == 0);        /* journal_reset ran */
    CHECK(get_rc(c, 1) == KV_EOK);  /* index intact across the checkpoint */
    cache_close(c);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    test_fru_hysteresis();
    test_min_gen_age();
    test_evict_tombstone();
    test_recovery_write_gc();
    test_ttl_expiry();
    test_reader_pin();
    test_epoch_trim_and_force_free();
    test_hot_decay();
    test_checkpoint_on_journal_fill();
    if (failures == 0) {
        printf("test_generation: OK\n");
        return 0;
    }
    printf("test_generation: %d failures\n", failures);
    return 1;
}
