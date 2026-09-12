#ifndef SC_METRICS_HPP
#define SC_METRICS_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace sc {

enum { KV_MLEVEL_OFF = 0, KV_MLEVEL_BASIC = 1, KV_MLEVEL_FULL = 2 };
inline constexpr int KV_METRICS_MAX_DEVS = 16;
inline constexpr int KV_HIST_BUCKETS = 32;

struct kv_hist {
    std::atomic<uint64_t> count[KV_HIST_BUCKETS];
    std::atomic<uint64_t> n{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> min_ns{0};
    std::atomic<uint64_t> max_ns{0};

    kv_hist() {
        for (auto &c : count)
            c.store(0);
    }
};

void kv_hist_clear(kv_hist *h);
void kv_hist_init(kv_hist *h);
void kv_hist_sample(kv_hist *h, uint64_t ns);
void kv_hist_merge(kv_hist *dst, const kv_hist *src);
uint64_t kv_hist_percentile(const kv_hist *h, unsigned pct);
uint64_t kv_hist_avg(const kv_hist *h);
uint64_t kv_hist_max(const kv_hist *h);
uint64_t kv_hist_count(const kv_hist *h);

struct kv_metrics {
    kv_hist h_put_ack;
    kv_hist h_get;
};

struct kv_metrics_snap {
    uint64_t wall_ns = 0;

    uint64_t puts = 0, puts_err = 0;
    uint64_t gets_hit = 0, gets_miss = 0;
    uint64_t batches = 0;
    uint64_t bytes_written = 0;
    uint64_t bytes_read = 0;

    int n_devs = 0;
    struct {
        uint64_t bytes_written;
        uint64_t stripe_parts;
        uint64_t q_backlog;
        uint64_t q_max;
        uint64_t io_errs;
    } dev[KV_METRICS_MAX_DEVS];

    uint64_t payload_bytes = 0;
    uint64_t meta_bytes_written = 0;
    uint64_t bytes_migrated = 0;
    uint64_t rotations = 0;
    uint64_t dfree_depth = 0;
    uint64_t drops = 0;

    uint64_t leaves = 0;
    uint64_t journal_dropped = 0;
    uint64_t journal_fill_pct = 0;
    uint64_t ver_counter = 0;
    uint64_t replica_puts = 0;

    uint64_t io_errs_total = 0;

    double hit_ratio = 0.0;
    double waf = 0.0;
    double waf_total = 0.0;
    double avg_batch_kib = 0.0;

    kv_hist h_put_ack;
    kv_hist h_batch_io;
    kv_hist h_get;
};

} // namespace sc

struct cache;

namespace sc {

int kv_metrics_snapshot(cache *c, kv_metrics_snap *out);
void kv_metrics_dump(cache *c, FILE *fp);
void kv_metrics_csv(cache *c, FILE *fp);

} // namespace sc

#endif
