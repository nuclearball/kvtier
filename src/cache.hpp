#ifndef SC_CACHE_HPP
#define SC_CACHE_HPP

#include "common.hpp"
#include "device.hpp"
#include "dram.hpp"
#include "io.hpp"
#include "journal.hpp"
#include "metrics/metrics.hpp"
#include "radix.hpp"
#include "region.hpp"
#include "writer.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace sc {

struct kv_cuckoo;

inline constexpr uint32_t KV_SIDE_INIT_CAP = 4096;

struct kv_layout {
    uint32_t page_size = 0;
    uint32_t alignment = 0;
    uint64_t stripe_unit = 0;
    uint64_t stripe_threshold = 0;
    uint64_t batch_min_bytes = 0;
    uint64_t region_align_bytes = 0;
    uint64_t dram_entry_max_bytes = 0;
};

struct side_slot {
    uint64_t prefix_id = 0;
    uint32_t group_idx = 0;
    radix_node *node = nullptr;
    uint8_t used = 0;
};

struct side_tbl {
    std::vector<side_slot> slots;
    uint32_t cap = 0;
    uint32_t count = 0;
};

void side_init(side_tbl *t);
void side_destroy(side_tbl *t);
int side_put(side_tbl *t, uint64_t prefix_id, uint32_t group_idx,
             radix_node *node);
radix_node *side_get(const side_tbl *t, uint64_t prefix_id,
                     uint32_t group_idx);

struct leaf_dfree {
    std::mutex mtx;
    std::vector<radix_leaf *> items;
};
void dfree_push(leaf_dfree *d, radix_leaf *leaf);
void dfree_reap(leaf_dfree *d);
void dfree_destroy(leaf_dfree *d);

struct gc_ctx {
    cache *c = nullptr;
    std::thread tid;
    std::atomic<int> stop{0};
};

} // namespace sc

struct cache {
    kv_config cfg;
    int n_devs = 0;
    std::vector<std::unique_ptr<sc::Device>> devs;
    std::unique_ptr<sc::RegionMgr[]> rms;
    std::unique_ptr<sc::journal[]> journals;
    std::unique_ptr<sc::shard_writer[]> writers;
    sc::radix_tree radix;
    sc::side_tbl side;
    sc::shape_pool shapes;
    sc::kv_layout layout;
    std::shared_mutex radix_lock;
    std::mutex ckpt_mtx;
    std::atomic<uint64_t> ver_counter{0};
    sc::gc_ctx gc;
    char ckpt_path[1024] = {};
    sc::leaf_dfree dfree;
    std::atomic<int> stop{0};
    int replaying = 0;
    std::atomic<int> ckpt_wanted{0};
    uint64_t last_decay_ms = 0;
    uint64_t last_ttl_ms = 0;
    uint64_t last_ckpt_ms = 0;
    std::atomic<uint64_t> stat_put_reqs{0};
    std::atomic<uint64_t> stat_get_hits{0};
    std::atomic<uint64_t> stat_get_misses{0};
    std::atomic<uint64_t> stat_batches{0};
    std::atomic<uint64_t> stat_bytes_written{0};
    std::atomic<uint64_t> stat_meta_bytes_written{0};
    std::atomic<uint64_t> stat_bytes_migrated{0};
    std::atomic<uint64_t> stat_rotations{0};
    std::atomic<uint64_t> stat_drops{0};
    std::atomic<uint64_t> stat_puts_err{0};
    std::atomic<uint64_t> stat_payload_bytes{0};
    std::atomic<uint64_t> stat_puts_replica{0};
    std::atomic<uint64_t> stat_gc_triggers{0};
    std::atomic<uint64_t> stat_gc_evicted_leaves{0};
    int gc_draining = 0;
    std::unique_ptr<uint64_t[]> tomb_keys;
    uint32_t tomb_mask = 0;
    sc::dram_cache *dram = nullptr;
#ifdef KV_USE_CUCKOO_TOMB
    sc::kv_cuckoo *tomb_cf = nullptr;
#endif
    sc::kv_metrics m;
};

namespace sc {

inline int region_of_page(const RegionMgr *rm, uint64_t page) {
    if (page < kRegionBasePage)
        return -1;
    return static_cast<int>((page - kRegionBasePage) / rm->region_size_pages);
}

void region_live_add(cache *c, uint32_t dev_id, uint64_t page_no,
                     uint32_t len_pages);
void region_live_sub(cache *c, uint32_t dev_id, uint64_t page_no,
                     uint32_t len_pages);

void leaf_publish(cache *c, shard_writer *w, radix_node *node,
                  uint64_t prefix_id, uint32_t group_idx, uint32_t ver,
                  const kv_addr *addr, uint32_t slot, uint32_t flags,
                  uint32_t stripe_idx, uint32_t stripe_cnt,
                  uint32_t stripe_total, uint32_t data_len, kv_shape *shape,
                  const uint64_t *path, uint32_t path_len);

void leaf_drop(cache *c, radix_node *node);
void trim_region(cache *c, int dev_id, int idx);
int cache_drop_region(cache *c, shard_writer *w, int region_idx);
int gc_migrate_region(cache *c, int dev_id, int region_idx);
void cache_recompute_live(cache *c);
void cache_signal_stop(cache *c);
int cache_checkpoint(cache *c);
void cache_fire_ack(void (*ack)(void *, int), void *user, int rc);
int cache_force_free_region(cache *c, shard_writer *w, int avoid);

// internal API surface (implementation in cache.cpp)
int cache_open_impl(cache **out, const char *const *dev_uris, int n_devs,
                    const kv_config *cfg);
void cache_close_impl(cache *c);

} // namespace sc

#endif
