#ifndef SC_RADIX_HPP
#define SC_RADIX_HPP

#include "common.hpp"

#include <atomic>
#include <mutex>
#include <vector>

namespace sc {

inline constexpr int KV_INLINE_SLOTS = 8;
inline constexpr uint16_t KV_EDGE_F_HOT = (1u << 0);
inline constexpr uint16_t KV_EDGE_F_DELETED = (1u << 1);

#pragma pack(push, 1)
struct kv_addr {
    uint32_t dev_id;
    uint64_t page_no;
    uint32_t page_off;
    uint32_t len_pages;
    uint64_t last_write_ts;
    uint32_t expire_ts;
    uint32_t crc;
};
static_assert(sizeof(kv_addr) == 36, "kv_addr must be 36B");
#pragma pack(pop)

struct kv_stripe_part {
    uint32_t dev_id;
    uint64_t page_no;
    uint32_t len_pages;
};

struct kv_stripe_addr {
    uint32_t n_parts;
    uint32_t total_len;
};

inline kv_stripe_part *stripe_parts(kv_stripe_addr *s) {
    return reinterpret_cast<kv_stripe_part *>(s + 1);
}
inline const kv_stripe_part *stripe_parts(const kv_stripe_addr *s) {
    return reinterpret_cast<const kv_stripe_part *>(s + 1);
}

struct kv_shape {
    uint32_t id;
    uint16_t n_recs;
};

inline uint32_t *shape_lens(kv_shape *s) {
    return reinterpret_cast<uint32_t *>(s + 1);
}
inline const uint32_t *shape_lens(const kv_shape *s) {
    return reinterpret_cast<const uint32_t *>(s + 1);
}

struct shape_pool {
    std::mutex mtx;
    std::vector<kv_shape *> items;
};

kv_shape *shape_pool_intern(shape_pool *sp, const uint32_t *lens,
                            uint16_t n_recs);
void shape_pool_destroy(shape_pool *sp);

struct radix_leaf {
    std::atomic<unsigned> refcnt{0};
    std::atomic<unsigned> ver{0};
    uint64_t prefix_id = 0;
    kv_addr addr[KV_REPLICA_CNT];
    kv_stripe_addr *stripe = nullptr;
    kv_shape *shape = nullptr;
};

struct radix_node;

struct radix_edge {
    uint64_t child_hash = 0;
    radix_node *child = nullptr;
    uint16_t flags = 0;
};

struct radix_ovf {
    uint32_t cap = 0;
    uint32_t used = 0;
    uint32_t n_del = 0;
    radix_edge *slots = nullptr;
};

struct radix_node {
    std::atomic<unsigned> hit_count{0};
    uint32_t depth = 0;
    uint16_t n_edges = 0;
    uint16_t ovf_flag = 0;
    radix_edge slots[KV_INLINE_SLOTS];
    radix_ovf *ovf = nullptr;
    radix_node *parent = nullptr;
    radix_leaf *leaf = nullptr;
};

struct radix_tree {
    radix_node *root = nullptr;
    std::atomic<unsigned> n_leaves{0};
};

void radix_init(radix_tree *t);
void radix_destroy(radix_tree *t);

radix_node *radix_walk(radix_tree *t, const uint64_t *hashes, int n,
                       int create);

void radix_hit_ascend(radix_node *node, unsigned inc);
void radix_hit_decay(radix_node *node);

using radix_visit_fn = int (*)(radix_node *node, void *arg);
int radix_visit(radix_tree *t, radix_visit_fn fn, void *arg);

using radix_leaf_fn = int (*)(radix_leaf *leaf, void *arg);
int radix_visit_leaves(radix_tree *t, radix_leaf_fn fn, void *arg);

} // namespace sc

#endif
