#ifndef SC_WRITER_HPP
#define SC_WRITER_HPP

#include "common.hpp"
#include "device.hpp"
#include "io.hpp"
#include "journal.hpp"
#include "metrics/metrics.hpp"
#include "region.hpp"

#include <atomic>
#include <thread>

struct cache;

namespace sc {

struct shard_writer;

struct kv_mpsc_slot {
    std::atomic<uint64_t> seq{0};
    void *ptr = nullptr;
};

struct kv_mpsc_ring {
    kv_mpsc_slot *slots = nullptr;
    uint64_t mask = 0;
    std::atomic<uint64_t> tail{0};
    std::atomic<uint64_t> head{0};
    int init = 0;
};

int mpsc_init(kv_mpsc_ring *q, uint32_t power);
void mpsc_destroy(kv_mpsc_ring *q);
int mpsc_push(kv_mpsc_ring *q, void *ptr);
void *mpsc_pop(kv_mpsc_ring *q);

struct put_ctx {
    std::atomic<int> refs{0};
    std::atomic<int> err{KV_EOK};
    void (*ack)(void *, int) = nullptr;
    void *user = nullptr;
    void *staging = nullptr;
    cache *c = nullptr;
    uint64_t t0_ns = 0;
};

void put_ctx_init(put_ctx *pc, void (*ack)(void *, int), void *user);
void put_ctx_unref(put_ctx *pc);

struct put_req {
    uint64_t prefix_id = 0;
    uint32_t group_idx = 0;
    uint32_t n_tokens = 0;
    uint32_t n_tokens_total = 0;
    uint32_t expire_ts = 0;
    uint32_t ver = 0;
    uint16_t n_recs = 0;
    uint16_t flags = 0;
    uint32_t stripe_idx = 0, stripe_cnt = 0;
    uint32_t stripe_total = 0;
    kv_data_ref recs[KV_MAX_LAYERS_CAP] = {};
    kv_data_ref slice{};
    uint32_t *tokens = nullptr;
    uint64_t *hashes = nullptr;
    void *own_buf = nullptr;
    put_ctx *ctx = nullptr;
    uint32_t mig_ver = 0;
    kv_addr mig_old;
    int gc_req = 0;
    uint64_t page_no = 0;
    uint32_t npages = 0;
    uint32_t crc = 0;
    uint32_t data_len = 0;
    kv_shape *shape = nullptr;
    uint32_t slot = 0;
    put_req *next = nullptr;
};

inline constexpr int KV_BATCH_MAX_REQS = 512;
inline constexpr uint64_t KV_BATCH_BUF_SIZE =
    KV_STRIPE_THRESHOLD + 4 * KV_PAGE_SIZE;

struct batch_buf {
    uint8_t *buf = nullptr;
    uint32_t used = 0;
    uint64_t start_page = 0;
    uint32_t npages = 0;
    uint16_t n_reqs = 0;
    put_req *reqs[KV_BATCH_MAX_REQS] = {};
    IoCtx ictx;
    std::atomic<int> busy{0};
    uint64_t last_page = 0;
    int region = 0;
    shard_writer *w = nullptr;
    uint64_t submit_ns = 0;
};

struct shard_writer {
    cache *c = nullptr;
    int dev_id = 0;
    Device *dev = nullptr;
    RegionMgr *rm = nullptr;
    journal *jour = nullptr;
    struct write_cursor {
        int region = 0;
        uint64_t next_page = 0;
    } cursor;
    kv_mpsc_ring q;
    batch_buf batch[2];
    int cur_buf = 0;
    IoRing iour;
    std::thread tid;
    int stop = 0;
    int started = 0;
    uint64_t since_wm = 0;
    uint64_t confirmed_page = 0;
    std::atomic<uint64_t> stat_bytes_written{0};
    std::atomic<uint64_t> stat_stripe_parts{0};
    std::atomic<uint64_t> stat_q_max{0};
    kv_hist h_batch;
};

int shard_writer_start(shard_writer *w);
void shard_writer_stop(shard_writer *w);
int shard_writer_enqueue(shard_writer *w, put_req *r);

} // namespace sc

#endif
