#ifndef SC_JOURNAL_HPP
#define SC_JOURNAL_HPP

#include "common.hpp"
#include "device.hpp"
#include "radix.hpp"

#include <cstdio>
#include <mutex>

namespace sc {

inline constexpr uint32_t KV_JOURNAL_REC_SIZE = 64;
inline constexpr uint32_t KV_JOURNAL_RECS_PER_PAGE =
    KV_PAGE_SIZE / KV_JOURNAL_REC_SIZE;

inline constexpr uint32_t KV_JFLAG_SLOT1 = (1u << 9);
inline constexpr uint32_t KV_JFLAG_UNIFORM = (1u << 10);

#pragma pack(push, 1)

struct journal_rec {
    uint8_t op;
    uint8_t _pad;
    uint16_t dev_id;
    uint64_t prefix_id;
    uint32_t group_idx;
    uint32_t ver;
    struct {
        uint32_t dev;
        uint64_t page;
        uint32_t off;
        uint32_t len;
    } addr;
    uint32_t stripe_idx;
    uint32_t stripe_cnt;
    uint32_t flags;
    uint32_t expire_ts;
    uint32_t crc32;
    uint16_t path_len;
    uint16_t n_recs;
};
static_assert(sizeof(journal_rec) == 64, "journal rec must be 64B");

struct ckpt_hdr {
    uint32_t magic;
    uint32_t version;
    uint32_t n_devs;
    uint32_t region_cnt;
    uint64_t region_size_pages;
    uint64_t n_recs;
    uint64_t ver_counter;
    uint32_t crc32;
    uint32_t pad;
};
static_assert(sizeof(ckpt_hdr) == 48, "ckpt hdr must be 48B");

inline constexpr uint8_t CKPT_REC_INTERNAL = 1;
inline constexpr uint8_t CKPT_REC_LEAF = 2;
inline constexpr uint8_t CKPT_REC_LEAF_STRIPE = 3;
inline constexpr uint32_t KV_SHAPE_NONE = 0xFFFFFFFFu;

struct ckpt_node_rec {
    uint8_t type;
    uint8_t _pad[3];
    uint32_t depth;
    uint64_t child_hash;
    uint64_t prefix_id;
    uint32_t group_idx;
    uint32_t ver;
    uint32_t expire_ts;
    uint64_t last_write_ts;
    kv_addr addr[KV_REPLICA_CNT];
    uint32_t n_parts;
    uint32_t total_len;
    uint32_t shape_id;
    uint32_t crc32;
};
static_assert(sizeof(ckpt_node_rec) == 132, "ckpt node rec must be 132B");

#pragma pack(pop)

struct journal {
    Device *dev = nullptr;
    uint64_t journal_page = 0;
    uint32_t journal_pages = 0;
    uint32_t cur_page = 0;
    uint32_t cur_off = 0;
    uint8_t *stage = nullptr;
    uint32_t dropped = 0;
    uint64_t bytes_written = 0;
    std::mutex mtx;
};

int journal_init(journal *j, Device *dev, uint64_t journal_page,
                 uint32_t journal_pages);
void journal_destroy(journal *j);

int journal_put(journal *j, uint64_t prefix_id, uint32_t group_idx,
                uint32_t ver, const kv_addr *addr, uint32_t expire_ts,
                uint32_t stripe_idx, uint32_t stripe_cnt, uint32_t flags,
                const uint32_t *rec_lens, uint16_t n_recs, uint32_t data_len,
                const uint64_t *path, uint32_t path_len);
int journal_del(journal *j, uint64_t prefix_id, uint32_t group_idx);
int journal_trim(journal *j, uint32_t region_idx, uint64_t epoch);

using journal_replay_fn = int (*)(const journal_rec *rec, uint32_t data_crc,
                                  const uint32_t *lens, uint16_t n_lens,
                                  uint32_t data_len, const uint64_t *path,
                                  void *arg);
int journal_replay(journal *j, journal_replay_fn cb, void *arg);
int journal_reset(journal *j);
int journal_flush(journal *j);

struct ckpt_leaf_snapshot {
    uint64_t prefix_id = 0;
    uint32_t group_idx = 0;
    uint32_t ver = 0;
    uint32_t expire_ts = 0;
    uint64_t last_write_ts = 0;
    kv_addr addr[KV_REPLICA_CNT];
    uint32_t n_parts = 0;
    uint32_t total_len = 0;
    struct part {
        uint32_t dev_id;
        uint64_t page_no;
        uint32_t len_pages;
    } *parts = nullptr;
};

using ckpt_leaf_apply = void (*)(radix_node *node,
                                 const ckpt_leaf_snapshot *snap, void *arg);

int ckpt_write(const char *path, radix_tree *t, uint64_t ver_counter,
               uint32_t n_devs, uint32_t region_cnt, uint64_t region_size_pages,
               shape_pool *shapes);
int ckpt_load(const char *path, radix_tree *t, uint64_t *ver_counter,
              uint32_t n_devs, uint32_t region_cnt, uint64_t region_size_pages,
              shape_pool *shapes, ckpt_leaf_apply apply, void *arg);

} // namespace sc

#endif
