#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* constants (design-glm-5.3-lld.md §1)                               */
/* ------------------------------------------------------------------ */
#define KV_TOKENS_PER_GROUP   32        /* tokens per group          */
#define KV_PAGE_SIZE          4096      /* NVMe base page            */
#define KV_CHUNK_HDR_SIZE     64        /* chunk header (with pad)   */
#define KV_WATERMARK_INTERVAL 16        /* watermark flush interval  */
#define KV_BATCH_MIN_BYTES    (256*1024)/* group-commit batch floor  */
#define KV_STRIPE_UNIT        (256*1024)/* stripe unit for big chunks*/
#define KV_REPLICA_CNT        2         /* hot-prefix replica count  */
#define KV_MAX_LAYERS_CAP     256       /* compile-time array bound  */
#define KV_MAX_LAYERS         KV_MAX_LAYERS_CAP /* deprecated alias  */
#define KV_STRIPE_THRESHOLD   (1024*1024) /* stripes when > 1MiB     */
#define KV_MAX_STRIPE_PARTS   1024      /* hard cap: put beyond -> EINVAL */

/* DRAM read cache defaults (dram-cache-design.md) */
#define KV_DRAM_DEFAULT_BYTES     (256ull * 1024 * 1024)
#define KV_DRAM_DEFAULT_ENTRY_MAX (8ull * 1024 * 1024)
#define KV_DRAM_BACKEND_AUTO      0
#define KV_DRAM_BACKEND_PLAIN     1
#define KV_DRAM_BACKEND_THP       2
#define KV_DRAM_BACKEND_HUGETLB   3
#define KV_DRAM_BACKEND_MAX       3

/* disk magics */
#define KV_SB_MAGIC  0x3143564Bu /* 'KVC1' */
#define KV_RG_MAGIC  0x4752564Bu /* 'KVRG' */
#define KV_CK_MAGIC  0x4B43564Bu /* 'KVCK' */
#define KV_CP_MAGIC  0x504B435Bu /* 'KCP1' checkpoint */

/* region states */
#define KV_RG_FREE     0
#define KV_RG_OPEN     1
#define KV_RG_FROZEN   2
#define KV_RG_CLEANING 3

/* chunk header flags */
#define KV_CHUNK_F_REPLICA  (1u << 0)
#define KV_CHUNK_F_STRIPE   (1u << 1)
#define KV_CHUNK_F_HOT      (1u << 2)

/* journal opcodes */
#define KV_JOP_PUT  1
#define KV_JOP_DEL  2
#define KV_JOP_TRIM 3

/* radix node kinds */
#define KV_RADIX_INTERNAL 1
#define KV_RADIX_LEAF     2

/* sentinel device id for "no second replica" */
#define KV_INVALID_DEV UINT32_MAX

/* error codes */
enum {
    KV_EOK       = 0,
    KV_ENOMEM    = -1,
    KV_EEXIST    = -2,
    KV_ENOENT    = -3,   /* cache miss             */
    KV_EBUSY     = -4,   /* refcnt > 0, deferred   */
    KV_EREAD     = -5,
    KV_EWRITE    = -6,
    KV_ECRC      = -7,
    KV_EFULL     = -8,
    KV_EIO       = -9,
    KV_ETIMEDOUT = -10,
    KV_EINVAL    = -11,
    KV_EVICTED   = -12,  /* key existed but was discarded by GC/evict    */
};

/* ------------------------------------------------------------------ */
/* configuration                                                      */
/* ------------------------------------------------------------------ */
/* Configuration struct.  Field layout is generated from
 * tools/config_schema.json (run `make gen-config`); do not edit by hand.
 * `stripe_unit` / `region_align_bytes` are generic layout knobs: the
 * model_profile module (optional, adapter-side) computes them for a model,
 * but the core library stays model-agnostic.  0 => core default. */
struct kv_config {
#include "kv_config_gen.h"
};
#include "kv_config_defaults.h"

void kv_config_default(struct kv_config *cfg);

/* ------------------------------------------------------------------ */
/* public API                                                         */
/* ------------------------------------------------------------------ */
typedef struct cache cache_t;

int  cache_open(cache_t **out, const char *const *dev_uris, int n_devs,
                const struct kv_config *cfg);
void cache_close(cache_t *c);

/* Per-layer data reference for one group of tokens (K‖V, opaque). */
struct kv_data_ref {
    const void *base;   /* caller buffer                                  */
    uint32_t off;       /* byte offset inside base                        */
    uint32_t len;       /* number of bytes                                */
};

/* Async put of a single 32-token group.
 *
 * tokens[] holds n_tokens_total tokens that cover groups 0..group_idx
 * (>= (group_idx+1)*32 tokens when the final group is full).  The cache
 * copies everything it needs; `ack` fires exactly once, from an internal
 * thread, after the chunk(s) are durable (and published). */
int cache_put(cache_t *c, uint64_t prefix_id, uint32_t group_idx,
              const uint32_t *tokens, uint32_t n_tokens_total,
              uint32_t expire_ts, uint16_t n_layers,
              const struct kv_data_ref *recs,
              void (*ack)(void *user, int rc), void *user);

/* Synchronous get of the deepest group for the given prefix string. */
struct cache_get_result {
    void      *buf;        /* malloc'd record stream (caller frees)        */
    uint32_t   buf_len;    /* total bytes of record stream                 */
    uint16_t   n_records;  /* number of KV records in buf                  */
    struct {
        uint16_t layer_id;
        uint32_t off;
        uint32_t len;
    } recs[KV_MAX_LAYERS_CAP];
    uint32_t   ver;
    uint64_t   prefix_id;
    uint32_t   group_idx;
};
int  cache_get(cache_t *c, uint64_t prefix_id, const uint32_t *tokens,
               uint32_t n_tokens_total, struct cache_get_result *out);
void cache_result_free(struct cache_get_result *res);

/* Evict the deepest group; refcnt>0 chunks are deferred (KV_EBUSY). */
int  cache_evict(cache_t *c, uint64_t prefix_id, const uint32_t *tokens,
                 uint32_t n_tokens_total);

/* Diagnostics (used by the integration tests). */
uint64_t cache_stats(cache_t *c, const char *key);
int      cache_dev_count(cache_t *c);

#ifdef __cplusplus
}
#endif

#endif /* KV_CACHE_H */
