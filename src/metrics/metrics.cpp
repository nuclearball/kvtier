#include "metrics/metrics.hpp"
#include "cache.hpp"
#include "common.hpp"

namespace sc {

void kv_hist_clear(kv_hist *h) {
    if (!h)
        return;
    for (auto &c : h->count)
        c.store(0);
    h->n.store(0);
    h->total_ns.store(0);
    h->min_ns.store(0);
    h->max_ns.store(0);
}

void kv_hist_init(kv_hist *h) { kv_hist_clear(h); }

void kv_hist_sample(kv_hist *h, uint64_t ns) {
    if (!h)
        return;
    uint32_t b = ns ? 63 - static_cast<uint32_t>(__builtin_clzll(ns)) : 0;
    if (b >= KV_HIST_BUCKETS)
        b = KV_HIST_BUCKETS - 1;
    h->count[b].fetch_add(1);
    h->n.fetch_add(1);
    h->total_ns.fetch_add(ns);
    uint64_t cur = h->min_ns.load();
    while ((cur == 0 || ns < cur) && !h->min_ns.compare_exchange_weak(cur, ns))
        ;
    cur = h->max_ns.load();
    while (ns > cur && !h->max_ns.compare_exchange_weak(cur, ns))
        ;
}

void kv_hist_merge(kv_hist *dst, const kv_hist *src) {
    if (!dst || !src)
        return;
    for (int i = 0; i < KV_HIST_BUCKETS; i++)
        dst->count[i].fetch_add(src->count[i].load());
    dst->n.fetch_add(src->n.load());
    dst->total_ns.fetch_add(src->total_ns.load());
    uint64_t s = src->min_ns.load();
    if (s) {
        uint64_t d = dst->min_ns.load();
        while ((d == 0 || s < d) && !dst->min_ns.compare_exchange_weak(d, s))
            ;
    }
    uint64_t sm = src->max_ns.load();
    if (sm) {
        uint64_t d = dst->max_ns.load();
        while (sm > d && !dst->max_ns.compare_exchange_weak(d, sm))
            ;
    }
}

uint64_t kv_hist_percentile(const kv_hist *h, unsigned pct) {
    if (!h || pct > 100)
        return 0;
    uint64_t n = h->n.load();
    if (!n)
        return 0;
    uint64_t target = (n * pct + 99) / 100;
    uint64_t acc = 0;
    for (int i = 0; i < KV_HIST_BUCKETS; i++) {
        acc += h->count[i].load();
        if (acc >= target)
            return i + 1 < KV_HIST_BUCKETS ? (1ull << (i + 1)) : (1ull << i);
    }
    return 0;
}

uint64_t kv_hist_avg(const kv_hist *h) {
    if (!h)
        return 0;
    uint64_t n = h->n.load();
    return n ? h->total_ns.load() / n : 0;
}

uint64_t kv_hist_max(const kv_hist *h) { return h ? h->max_ns.load() : 0; }

uint64_t kv_hist_count(const kv_hist *h) { return h ? h->n.load() : 0; }

static void snap_reset(kv_metrics_snap *out) {
    for (auto &d : out->dev)
        d = {};
    kv_hist_clear(&out->h_put_ack);
    kv_hist_clear(&out->h_batch_io);
    kv_hist_clear(&out->h_get);
}

int kv_metrics_snapshot(struct cache *c, kv_metrics_snap *out) {
    if (!c || !out)
        return -KV_EINVAL;
    snap_reset(out);
    out->wall_ns = now_ns();
    out->n_devs =
        c->n_devs > KV_METRICS_MAX_DEVS ? KV_METRICS_MAX_DEVS : c->n_devs;

    out->puts = c->stat_put_reqs.load();
    out->puts_err = c->stat_puts_err.load();
    out->gets_hit = c->stat_get_hits.load();
    out->gets_miss = c->stat_get_misses.load();
    out->batches = c->stat_batches.load();
    out->bytes_written = c->stat_bytes_written.load();
    out->payload_bytes = c->stat_payload_bytes.load();
    out->bytes_migrated = c->stat_bytes_migrated.load();
    out->rotations = c->stat_rotations.load();
    out->drops = c->stat_drops.load();
    out->replica_puts = c->stat_puts_replica.load();
    out->leaves = c->radix.n_leaves.load();
    out->ver_counter = c->ver_counter.load();
    out->dfree_depth = c->dfree.items.size();

    uint64_t errs = 0;
    for (int i = 0; i < out->n_devs; i++) {
        shard_writer *w = &c->writers[i];
        out->dev[i].bytes_written = w->stat_bytes_written.load();
        out->dev[i].stripe_parts = w->stat_stripe_parts.load();
        out->dev[i].q_max = w->stat_q_max.load();
        out->dev[i].q_backlog = w->q.tail.load() - w->q.head.load();
        if (c->journals[i].dev) {
            out->journal_dropped += c->journals[i].dropped;
            out->meta_bytes_written += c->journals[i].bytes_written;
            uint64_t fill = c->journals[i].journal_pages
                                ? static_cast<uint64_t>(c->journals[i].cur_page) *
                                      100 / c->journals[i].journal_pages
                                : 0;
            if (fill > out->journal_fill_pct)
                out->journal_fill_pct = fill;
        }
        if (c->devs[i]->fd >= 0) {
            out->dev[i].io_errs = c->devs[i]->stats.io_errs;
            errs += c->devs[i]->stats.io_errs;
        }
    }
    out->io_errs_total = errs;

    out->hit_ratio = (out->gets_hit + out->gets_miss)
                         ? static_cast<double>(out->gets_hit) /
                               static_cast<double>(out->gets_hit + out->gets_miss)
                         : 0.0;
    out->waf = out->payload_bytes ? static_cast<double>(out->bytes_written) /
                                        static_cast<double>(out->payload_bytes)
                                  : 0.0;
    out->waf_total =
        out->payload_bytes
            ? static_cast<double>(out->bytes_written + out->meta_bytes_written) /
                  static_cast<double>(out->payload_bytes)
            : 0.0;
    out->avg_batch_kib = out->batches
                             ? static_cast<double>(out->bytes_written) /
                                   static_cast<double>(out->batches) / 1024.0
                             : 0.0;

    kv_hist_merge(&out->h_put_ack, &c->m.h_put_ack);
    kv_hist_merge(&out->h_get, &c->m.h_get);
    for (int i = 0; i < out->n_devs; i++)
        kv_hist_merge(&out->h_batch_io, &c->writers[i].h_batch);
    return KV_EOK;
}

static void fmt_ns(char *buf, size_t sz, uint64_t ns) {
    if (ns >= 1000000000ull)
        std::snprintf(buf, sz, "%.2fs", static_cast<double>(ns) / 1e9);
    else if (ns >= 1000000ull)
        std::snprintf(buf, sz, "%.2fms", static_cast<double>(ns) / 1e6);
    else if (ns >= 1000ull)
        std::snprintf(buf, sz, "%.2fus", static_cast<double>(ns) / 1e3);
    else
        std::snprintf(buf, sz, "%lluns", static_cast<unsigned long long>(ns));
}

static void dump_hist(FILE *fp, const char *name, const kv_hist *h) {
    uint64_t n = h->n.load();
    if (!n) {
        std::fprintf(fp, "  hist %-10s (no samples)\n", name);
        return;
    }
    char b1[24], b2[24], b3[24];
    fmt_ns(b1, sizeof(b1), kv_hist_percentile(h, 50));
    fmt_ns(b2, sizeof(b2), kv_hist_percentile(h, 99));
    fmt_ns(b3, sizeof(b3), h->max_ns.load());
    std::fprintf(fp, "  hist %-10s n=%llu avg=%lluns P50=%s P99=%s max=%s\n",
                 name, static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(kv_hist_avg(h)), b1, b2, b3);
    for (int i = 0; i < KV_HIST_BUCKETS; i++) {
        uint64_t cnt = h->count[i].load();
        if (!cnt)
            continue;
        double pct = 100.0 * static_cast<double>(cnt) / static_cast<double>(n);
        std::fprintf(fp, "    2^%-2d~2^%-2d ns %12llu %5.1f%% ", i, i + 1,
                     static_cast<unsigned long long>(cnt), pct);
        int bar = static_cast<int>(pct / 2.0 + 0.5);
        for (int k = 0; k < bar; k++)
            fputc('#', fp);
        fputc('\n', fp);
    }
}

void kv_metrics_dump(struct cache *c, FILE *fp) {
    kv_metrics_snap s;
    if (kv_metrics_snapshot(c, &s) != KV_EOK)
        return;
    std::fprintf(fp, "==== kvcache metrics ====\n");
    std::fprintf(
        fp, "puts=%llu (err=%llu)  gets hit=%llu miss=%llu (ratio=%.2f%%)\n",
        static_cast<unsigned long long>(s.puts),
        static_cast<unsigned long long>(s.puts_err),
        static_cast<unsigned long long>(s.gets_hit),
        static_cast<unsigned long long>(s.gets_miss), s.hit_ratio * 100.0);
    std::fprintf(fp,
                 "batches=%llu avg=%.1fKiB  bytes_written=%lluMiB "
                 "payload=%lluMiB WAF=%.3f (total=%.3f)\n",
                 static_cast<unsigned long long>(s.batches), s.avg_batch_kib,
                 static_cast<unsigned long long>(s.bytes_written >> 20),
                 static_cast<unsigned long long>(s.payload_bytes >> 20), s.waf,
                 s.waf_total);
    std::fprintf(fp, "meta_written=%lluKiB journal_fill=%llu%%\n",
                 static_cast<unsigned long long>(s.meta_bytes_written >> 10),
                 static_cast<unsigned long long>(s.journal_fill_pct));
    std::fprintf(fp,
                 "leaves=%llu rotations=%llu migrated=%lluMiB dfree=%llu "
                 "replica_puts=%llu\n",
                 static_cast<unsigned long long>(s.leaves),
                 static_cast<unsigned long long>(s.rotations),
                 static_cast<unsigned long long>(s.bytes_migrated >> 20),
                 static_cast<unsigned long long>(s.dfree_depth),
                 static_cast<unsigned long long>(s.replica_puts));
    std::fprintf(fp, "journal_dropped=%llu io_errs=%llu drops=%llu\n",
                 static_cast<unsigned long long>(s.journal_dropped),
                 static_cast<unsigned long long>(s.io_errs_total),
                 static_cast<unsigned long long>(s.drops));
    if (s.journal_dropped)
        std::fprintf(fp,
                     "  !! ALARM: journal metadata overflow: %llu record(s) "
                     "dropped before a checkpoint; crash may lose them\n",
                     static_cast<unsigned long long>(s.journal_dropped));
    for (int i = 0; i < s.n_devs; i++)
        std::fprintf(fp,
                     "  dev%d: wbytes=%lluMiB stripes=%llu q_backlog=%llu "
                     "q_max=%llu io_errs=%llu\n",
                     i,
                     static_cast<unsigned long long>(s.dev[i].bytes_written >> 20),
                     static_cast<unsigned long long>(s.dev[i].stripe_parts),
                     static_cast<unsigned long long>(s.dev[i].q_backlog),
                     static_cast<unsigned long long>(s.dev[i].q_max),
                     static_cast<unsigned long long>(s.dev[i].io_errs));
    dump_hist(fp, "put_ack", &s.h_put_ack);
    dump_hist(fp, "get", &s.h_get);
    dump_hist(fp, "batch_io", &s.h_batch_io);
}

void kv_metrics_csv(struct cache *c, FILE *fp) {
    kv_metrics_snap s;
    if (kv_metrics_snapshot(c, &s) != KV_EOK)
        return;
    std::fprintf(fp, "metric,value\n");
    std::fprintf(fp, "puts,%llu\n", static_cast<unsigned long long>(s.puts));
    std::fprintf(fp, "puts_err,%llu\n",
                 static_cast<unsigned long long>(s.puts_err));
    std::fprintf(fp, "gets_hit,%llu\n",
                 static_cast<unsigned long long>(s.gets_hit));
    std::fprintf(fp, "gets_miss,%llu\n",
                 static_cast<unsigned long long>(s.gets_miss));
    std::fprintf(fp, "hit_ratio,%.6f\n", s.hit_ratio);
    std::fprintf(fp, "batches,%llu\n", static_cast<unsigned long long>(s.batches));
    std::fprintf(fp, "bytes_written,%llu\n",
                 static_cast<unsigned long long>(s.bytes_written));
    std::fprintf(fp, "bytes_read,%llu\n",
                 static_cast<unsigned long long>(s.bytes_read));
    std::fprintf(fp, "payload_bytes,%llu\n",
                 static_cast<unsigned long long>(s.payload_bytes));
    std::fprintf(fp, "waf,%.6f\n", s.waf);
    std::fprintf(fp, "meta_bytes_written,%llu\n",
                 static_cast<unsigned long long>(s.meta_bytes_written));
    std::fprintf(fp, "waf_total,%.6f\n", s.waf_total);
    std::fprintf(fp, "journal_fill_pct,%llu\n",
                 static_cast<unsigned long long>(s.journal_fill_pct));
    std::fprintf(fp, "avg_batch_bytes,%.1f\n", s.avg_batch_kib * 1024.0);
    std::fprintf(fp, "bytes_migrated,%llu\n",
                 static_cast<unsigned long long>(s.bytes_migrated));
    std::fprintf(fp, "rotations,%llu\n",
                 static_cast<unsigned long long>(s.rotations));
    std::fprintf(fp, "dfree_depth,%llu\n",
                 static_cast<unsigned long long>(s.dfree_depth));
    std::fprintf(fp, "journal_dropped,%llu\n",
                 static_cast<unsigned long long>(s.journal_dropped));
    std::fprintf(fp, "io_errs,%llu\n",
                 static_cast<unsigned long long>(s.io_errs_total));
    std::fprintf(fp, "drops,%llu\n", static_cast<unsigned long long>(s.drops));
    std::fprintf(fp, "leaves,%llu\n", static_cast<unsigned long long>(s.leaves));
    std::fprintf(fp, "ver_counter,%llu\n",
                 static_cast<unsigned long long>(s.ver_counter));
    std::fprintf(fp, "replica_puts,%llu\n",
                 static_cast<unsigned long long>(s.replica_puts));
    for (int i = 0; i < s.n_devs; i++) {
        std::fprintf(fp, "dev%d_bytes_written,%llu\n", i,
                     static_cast<unsigned long long>(s.dev[i].bytes_written));
        std::fprintf(fp, "dev%d_stripe_parts,%llu\n", i,
                     static_cast<unsigned long long>(s.dev[i].stripe_parts));
        std::fprintf(fp, "dev%d_q_max,%llu\n", i,
                     static_cast<unsigned long long>(s.dev[i].q_max));
        std::fprintf(fp, "dev%d_q_backlog,%llu\n", i,
                     static_cast<unsigned long long>(s.dev[i].q_backlog));
        std::fprintf(fp, "dev%d_io_errs,%llu\n", i,
                     static_cast<unsigned long long>(s.dev[i].io_errs));
    }
    std::fprintf(fp, "put_ack_n,%llu\n",
                 static_cast<unsigned long long>(kv_hist_count(&s.h_put_ack)));
    std::fprintf(fp, "put_ack_p50_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_put_ack, 50)));
    std::fprintf(fp, "put_ack_p90_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_put_ack, 90)));
    std::fprintf(fp, "put_ack_p99_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_put_ack, 99)));
    std::fprintf(fp, "put_ack_avg_ns,%llu\n",
                 static_cast<unsigned long long>(kv_hist_avg(&s.h_put_ack)));
    std::fprintf(fp, "put_ack_max_ns,%llu\n",
                 static_cast<unsigned long long>(kv_hist_max(&s.h_put_ack)));
    std::fprintf(fp, "get_n,%llu\n",
                 static_cast<unsigned long long>(kv_hist_count(&s.h_get)));
    std::fprintf(fp, "get_p50_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_get, 50)));
    std::fprintf(fp, "get_p99_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_get, 99)));
    std::fprintf(fp, "get_avg_ns,%llu\n",
                 static_cast<unsigned long long>(kv_hist_avg(&s.h_get)));
    std::fprintf(fp, "batch_io_n,%llu\n",
                 static_cast<unsigned long long>(kv_hist_count(&s.h_batch_io)));
    std::fprintf(fp, "batch_io_p50_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_batch_io, 50)));
    std::fprintf(fp, "batch_io_p99_ns,%llu\n",
                 static_cast<unsigned long long>(
                     kv_hist_percentile(&s.h_batch_io, 99)));
    std::fprintf(fp, "batch_io_avg_ns,%llu\n",
                 static_cast<unsigned long long>(kv_hist_avg(&s.h_batch_io)));
}

} // namespace sc
