/*
 * bench: parameterized load generator + metrics exporter.
 *
 * usage: bench [model] [n_ops] [value_bytes] [n_devs] [out.csv] [dir] [layers] [batch_min_kb] [region_mb] [kv_model]
 *   model: u = uniform random | z = zipf(0.99) | s = sequential
 *   layers: KV layers per group (default 2); batch_min_kb: 0 = default
 *   kv_model: optional profile name (model_profile.h), e.g. deepseek-v3
 *   region_mb: per-region size in MiB (0/omitted = 32MiB default)
 *
 * phases: write n_ops keys -> read them back -> reopen and re-verify
 * (recovery).  metrics_level=FULL; CSV written at each phase boundary.
 */
#include "solidcacher.h"
#include "model_profile.h"
#include "metrics/metrics.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

using namespace sc;

#define MAX_DEVS 8
#define TOKENS   32

static const char *g_uris[MAX_DEVS];

static volatile int acked = -999;
static void on_ack(void *u, int rc) { (void)u; acked = rc; }

/* xorshift64* PRNG + zipf inverted-CDF sampler */
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static double rng_uniform(void) {
    return (double)(rng_next() >> 11) / 9007199254740992.0;
}

/* zipf theta=0.99 over [1, n]: inverse CDF via harmonic sum + binary search */
static uint64_t zipf_pick(uint64_t n) {
    double u = rng_uniform();
    static double *cdf = NULL;
    static uint64_t cdf_n = 0;
    if (cdf_n != n) {
        free(cdf);
        cdf = static_cast<double *>(malloc((size_t)n * sizeof(double)));
        double sum = 0;
        for (uint64_t i = 1; i <= n; i++) {
            sum += 1.0 / (double)i;
            cdf[i - 1] = sum;
        }
        for (uint64_t i = 0; i < n; i++) cdf[i] /= sum;
        cdf_n = n;
    }
    uint64_t lo = 0, hi = n - 1;
    while (lo < hi) {
        uint64_t mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo + 1;
}

static void fill_tokens(uint32_t *tok, uint64_t prefix) {
    for (int i = 0; i < TOKENS; i++)
        tok[i] = (uint32_t)((prefix * 2654435761u) ^ (uint64_t)i);
}

static int put_sync(cache_t *c, uint64_t prefix, const uint8_t *payload,
                    uint32_t len, uint16_t n_layers) {
    acked = -999;
    uint32_t tokens[TOKENS];
    fill_tokens(tokens, prefix);
    struct kv_data_ref recs[KV_MAX_LAYERS_CAP];
    for (uint16_t i = 0; i < n_layers; i++) {
        recs[i].base = payload;
        recs[i].off = 0;
        recs[i].len = len;
    }
    int rc = cache_put(c, prefix, 0, tokens, TOKENS, 0, n_layers, recs,
                       on_ack, NULL);
    if (rc != KV_EOK) return rc;
    while (acked == -999) usleep(100);
    return acked;
}

static int get_verify(cache_t *c, uint64_t prefix, const uint8_t *expect,
                      uint32_t len) {
    uint32_t tokens[TOKENS];
    fill_tokens(tokens, prefix);
    struct cache_get_result res;
    int rc = cache_get(c, prefix, tokens, TOKENS, &res);
    if (rc != KV_EOK) return rc;
    int ok = 0;
    for (int i = 0; i < res.n_records; i++)
        if (res.recs[i].layer_id == 0 && res.recs[i].len == len &&
            memcmp((const uint8_t *)res.buf + res.recs[i].off, expect, len) == 0)
            ok = 1;
    cache_result_free(&res);
    return ok ? KV_EOK : -1;
}

static double bench_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void export_csv(cache_t *c, FILE *fp, const char *phase,
                       const char *model, double secs,
                       const struct kv_metrics_snap *base) {
    struct kv_metrics_snap s;
    kv_metrics_snapshot(c, &s);
    uint64_t d_puts = s.puts - (base ? base->puts : 0);
    uint64_t d_wbytes = s.bytes_written - (base ? base->bytes_written : 0);
    uint64_t d_gets = (s.gets_hit + s.gets_miss) -
                      (base ? base->gets_hit + base->gets_miss : 0);
    fprintf(fp, "phase,%s,%s\n", model, phase);
    fprintf(fp, "phase_wall_s,%s,%.3f\n", model, secs);
    fprintf(fp, "put_iops,%s,%.1f\n", model, secs > 0 ? d_puts / secs : 0);
    fprintf(fp, "w_mibps,%s,%.1f\n", model,
            secs > 0 ? d_wbytes / 1048576.0 / secs : 0);
    fprintf(fp, "get_iops,%s,%.1f\n", model, secs > 0 ? d_gets / secs : 0);
    /* one-line summary row of the interesting counters */
    fprintf(fp, "counters,%s,puts=%llu puts_err=%llu hit=%llu miss=%llu "
                "batches=%llu waf=%.3f waf_total=%.3f leaves=%llu rotations=%llu "
                "journal_dropped=%llu io_errs=%llu qmax0=%llu qmax1=%llu\n",
            model,
            (unsigned long long)s.puts, (unsigned long long)s.puts_err,
            (unsigned long long)s.gets_hit, (unsigned long long)s.gets_miss,
            (unsigned long long)s.batches, s.waf, s.waf_total,
            (unsigned long long)s.leaves,
            (unsigned long long)s.rotations,
            (unsigned long long)s.journal_dropped,
            (unsigned long long)s.io_errs_total,
            (unsigned long long)s.dev[0].q_max,
            (unsigned long long)(s.n_devs > 1 ? s.dev[1].q_max : 0));
    fprintf(fp, "put_ack_p50_ns,%s,%llu\n", model,
            (unsigned long long)kv_hist_percentile(&s.h_put_ack, 50));
    fprintf(fp, "put_ack_p99_ns,%s,%llu\n", model,
            (unsigned long long)kv_hist_percentile(&s.h_put_ack, 99));
    fprintf(fp, "get_p50_ns,%s,%llu\n", model,
            (unsigned long long)kv_hist_percentile(&s.h_get, 50));
    fprintf(fp, "get_p99_ns,%s,%llu\n", model,
            (unsigned long long)kv_hist_percentile(&s.h_get, 99));
    fprintf(fp, "batch_io_p50_ns,%s,%llu\n", model,
            (unsigned long long)kv_hist_percentile(&s.h_batch_io, 50));
    fflush(fp);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "u";
    uint64_t n_ops    = argc > 2 ? strtoull(argv[2], NULL, 10) : 1000;
    uint32_t vlen     = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;
    int n_devs        = argc > 4 ? atoi(argv[4]) : 4;
    const char *csv   = argc > 5 ? argv[5] : "/tmp/bench_metrics.csv";
    const char *dir   = argc > 6 ? argv[6] : "/tmp";   /* device file dir */
    uint16_t n_layers = argc > 7 ? (uint16_t)atoi(argv[7]) : 2;
    uint32_t batch_kb = argc > 8 ? (uint32_t)atoi(argv[8]) : 0;
    uint32_t region_mb = argc > 9 ? (uint32_t)atoi(argv[9]) : 0;
    const char *kv_model = argc > 10 ? argv[10] : NULL;
    const struct kv_model_profile *mp =
        kv_model ? kv_mp_by_name(kv_model) : NULL;
    if (kv_model && !mp) {
        fprintf(stderr, "unknown kv_model '%s'\n", kv_model);
        return 2;
    }

    if (n_devs < 1) n_devs = 1;
    if (n_devs > MAX_DEVS) n_devs = MAX_DEVS;
    for (int i = 0; i < n_devs; i++) {
        static char paths[MAX_DEVS][512];
        snprintf(paths[i], sizeof(paths[i]), "%s/bench_d%d.img", dir, i);
        unlink(paths[i]);
        g_uris[i] = paths[i];
    }
    { char ck[600]; snprintf(ck, sizeof(ck), "%s/bench_d0.img.ckpt", dir);
      unlink(ck); }

    struct kv_config cfg;
    kv_config_default(&cfg);
    cfg.metrics_level = KV_MLEVEL_FULL;
    cfg.region_cnt = 8;
    cfg.region_size_pages = 8192;   /* 32MiB regions */
    if (region_mb)
        cfg.region_size_pages = (uint64_t)region_mb * 1024 * 1024 / KV_PAGE_SIZE;
    cfg.epoch_secs = 3600;
    if (mp) {
        struct kv_mp_config_values v;
        if (kv_mp_config_values(mp, KV_PAGE_SIZE, &v) != KV_MP_OK)
            return 2;
        cfg.n_groups_tokens   = v.n_groups_tokens;
        cfg.max_layers        = v.max_layers;
        cfg.stripe_unit       = v.stripe_unit;
        cfg.stripe_threshold  = v.stripe_threshold;
        cfg.batch_min_bytes   = v.batch_min_bytes;
        cfg.region_align_bytes = v.region_align_bytes;
        cfg.dram_entry_max_bytes = v.dram_entry_max_bytes;
    }
    cfg.max_layers = n_layers;
    if (batch_kb) cfg.batch_min_bytes = batch_kb * 1024;

    FILE *fp = fopen(csv, "w+");
    if (!fp) { fprintf(stderr, "cannot open %s\n", csv); return 2; }

    uint8_t *payload = static_cast<uint8_t *>(malloc(vlen));
    memset(payload, 0x7E, vlen);

    /* ---- phase 1: write ---- */
    cache_t *c = NULL;
    if (cache_open(&c, g_uris, n_devs, &cfg) != KV_EOK) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    uint64_t *keys = static_cast<uint64_t *>(malloc((size_t)n_ops * sizeof(uint64_t)));
    struct kv_metrics_snap base;
    kv_metrics_snapshot(c, &base);

    double t0 = bench_now_s();
    int wfail = 0;
    for (uint64_t i = 0; i < n_ops; i++) {
        uint64_t prefix;
        if (model[0] == 'z')      prefix = zipf_pick(n_ops);
        else if (model[0] == 's') prefix = i + 1;
        else                      prefix = (rng_next() % n_ops) + 1;
        keys[i] = prefix;
        payload[0] = (uint8_t)prefix;
        payload[1] = (uint8_t)(prefix >> 8);
        if (put_sync(c, prefix, payload, vlen, n_layers) != KV_EOK) wfail++;
    }
    double wsec = bench_now_s() - t0;
    printf("[%s] write: %llu ops, %d failed, %.1f ops/s\n", model,
           (unsigned long long)n_ops, wfail, wsec > 0 ? n_ops / wsec : 0);
    export_csv(c, fp, "write", model, wsec, &base);

    /* ---- phase 2: read (same key stream: mixed hit/miss like the write) ---- */
    struct kv_metrics_snap base2;
    kv_metrics_snapshot(c, &base2);
    t0 = bench_now_s();
    int miss = 0;
    for (uint64_t i = 0; i < n_ops; i++) {
        uint64_t prefix = keys[i];
        payload[0] = (uint8_t)prefix;
        payload[1] = (uint8_t)(prefix >> 8);
        if (get_verify(c, prefix, payload, vlen) != KV_EOK) miss++;
    }
    double rsec = bench_now_s() - t0;
    printf("[%s] read : %llu ops, %d miss/fail, %.1f ops/s\n", model,
           (unsigned long long)n_ops, miss, rsec > 0 ? n_ops / rsec : 0);
    export_csv(c, fp, "read", model, rsec, &base2);

    kv_metrics_dump(c, stderr);
    cache_close(c);

    /* ---- phase 3: reopen (recovery): verify every written key ---- */
    t0 = bench_now_s();
    if (cache_open(&c, g_uris, n_devs, &cfg) != KV_EOK) {
        fprintf(stderr, "reopen failed\n");
        return 1;
    }
    double osec = bench_now_s() - t0;
    int lost = 0;
    for (uint64_t i = 0; i < n_ops; i++) {
        uint64_t prefix = keys[i];
        payload[0] = (uint8_t)prefix;
        payload[1] = (uint8_t)(prefix >> 8);
        if (get_verify(c, prefix, payload, vlen) != KV_EOK) lost++;
    }
    printf("[%s] recov: %.2fs open+verify, %llu/%llu readable\n", model, osec,
           (unsigned long long)(n_ops - lost), (unsigned long long)n_ops);
    export_csv(c, fp, "recovery", model, osec, NULL);
    fprintf(fp, "recovery_readable,%s,%llu\n", model,
            (unsigned long long)(n_ops - lost));
    cache_close(c);
    fclose(fp);

    free(keys);
    free(payload);
    for (int i = 0; i < n_devs; i++) {
        unlink(g_uris[i]);
    }
    { char ck[600]; snprintf(ck, sizeof(ck), "%s/bench_d0.img.ckpt", dir);
      unlink(ck); }
    return (wfail || lost) ? 1 : 0;
}
