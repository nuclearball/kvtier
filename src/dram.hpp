#ifndef SC_DRAM_HPP
#define SC_DRAM_HPP

#include "common.hpp"

#include <atomic>
#include <mutex>

namespace sc {

#define KV_DRAM_DEFAULT_BYTES (256ull * 1024 * 1024)
#define KV_DRAM_DEFAULT_ENTRY_MAX (8ull * 1024 * 1024)

#define KV_DRAM_BACKEND_AUTO 0
#define KV_DRAM_BACKEND_PLAIN 1
#define KV_DRAM_BACKEND_THP 2
#define KV_DRAM_BACKEND_HUGETLB 3
#define KV_DRAM_BACKEND_MAX 3

#define KV_DRAM_N_SHARDS 64
#define KV_DRAM_SHARD_BUCKETS 256
#define KV_DRAM_N_CLASSES 6

struct dram_rec {
    uint16_t layer_id;
    uint32_t off;
    uint32_t len;
};

struct dram_entry {
    uint64_t prefix_id;
    uint32_t group_idx;
    uint32_t ver;
    uint32_t expire_ts;
    uint32_t buf_len;
    uint16_t n_records;
    uint8_t external;
    uint8_t visited;
    uint8_t size_class;
    uint8_t _pad;
    struct dram_entry *hnext;
    struct dram_entry *sprev, *snext;
};

inline dram_rec *dram_entry_recs(dram_entry *e) {
    return reinterpret_cast<dram_rec *>(e + 1);
}
inline uint8_t *dram_entry_buf(dram_entry *e) {
    return reinterpret_cast<uint8_t *>(dram_entry_recs(e) + e->n_records);
}

struct dram_shard {
    std::mutex mtx;
    uint64_t budget = 0;
    dram_entry *buckets[KV_DRAM_SHARD_BUCKETS] = {};
    dram_entry *s_head = nullptr;
    dram_entry *s_tail = nullptr;
    dram_entry *s_hand = nullptr;
    uint64_t bytes = 0;
};

struct dram_cache {
    uint64_t budget = 0;
    uint64_t shard_budget = 0;
    uint64_t entry_max = 0;
    uint8_t backend = 0;
    int n_shards = 0;
    void *mmap_base = nullptr;
    uint64_t mmap_len = 0;
    void *arena = nullptr;
    uint64_t bump = 0;
    void *freelists[KV_DRAM_N_CLASSES] = {};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
    uint64_t entries = 0;
    uint64_t bytes = 0;
    dram_shard shards[KV_DRAM_N_SHARDS];
};

int dram_init(dram_cache **out, uint64_t budget, uint64_t entry_max,
              uint8_t backend_req);
void dram_destroy(dram_cache *d);

int dram_get(dram_cache *d, uint64_t prefix_id, uint32_t group_idx,
             uint32_t expect_ver, uint8_t **buf, uint32_t *buf_len,
             dram_rec *recs, uint16_t rec_cap, uint16_t *n_recs);

int dram_put(dram_cache *d, uint64_t prefix_id, uint32_t group_idx, uint32_t ver,
             uint32_t expire_ts, const uint8_t *buf, uint32_t buf_len,
             const dram_rec *recs, uint16_t n_recs);

void dram_invalidate(dram_cache *d, uint64_t prefix_id, uint32_t group_idx);

uint64_t dram_stat(dram_cache *d, const char *key);

} // namespace sc

#endif
