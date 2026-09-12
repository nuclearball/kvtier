/**
 * llama-kvcache.h — llama.cpp KV cache ↔ SSD KV cache adapter
 *
 * Uses llama.cpp's own serialization (llama_state_seq_get_data / set_data)
 * to dump/restore per-sequence KV cache into/from a kvtier SSD cache.
 *
 * This avoids depending on internal KV layout (varies by model, quantization,
 * n_stream, v_trans, etc.) and works with any architecture llama.cpp supports.
 *
 * Usage:
 *   1. llama_kvcache_ctx_init(&ctx, llama_ctx, cache_devs, n_devs, &cfg)
 *   2. llama_kvcache_save(&ctx, seq_id)          — async, fires ack
 *   3. llama_kvcache_restore(&ctx, seq_id, dest) — sync, returns token count
 *   4. llama_kvcache_ctx_destroy(&ctx)
 */
#ifndef LLAMA_KVCACHE_H
#define LLAMA_KVCACHE_H

#include <llama.h>
#include <kvtier.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_kvcache_ctx;

/**
 * Callback fired when an async save completes.
 * rc: KV_EOK on success, negative error code on failure.
 */
typedef void (*llama_kvcache_ack_fn)(void *user, int rc);

/**
 * Per-sequence metadata stored alongside the KV state.
 * This is the token list that was active when the KV was saved,
 * needed for session restore.
 */
struct llama_kvcache_seq_info {
    llama_seq_id   seq_id;
    llama_token   *tokens;
    size_t         n_tokens;
};

/**
 * Configuration for the adapter.
 */
struct llama_kvcache_cfg {
    uint32_t  tokens_per_group;   /* kvtier group size (0 = 32)       */
    uint32_t  region_cnt;         /* SSD regions (0 = default 6)           */
    uint64_t  region_size_pages;  /* 0 = auto                              */
    uint32_t  epoch_secs;         /* 0 = default 1800                      */
    uint32_t  gc_start_pct;       /* GC trigger threshold (0 = 80)         */
    uint32_t  gc_stop_pct;        /* GC stop threshold (0 = 70)            */
    bool      sync_mode;          /* fua/fsync on put ack                  */
    uint32_t  metrics_level;      /* 0=off, 1=counts, 2=histograms         */
};

void llama_kvcache_cfg_default(struct llama_kvcache_cfg *cfg);

/**
 * Opaque adapter context.
 */
struct llama_kvcache_ctx {
    struct llama_context *llama_ctx;  /* owned llama context                */
    cache_t           *cache;         /* kvtier instance                */
    struct kv_config   cache_cfg;     /* kvtier config snapshot         */
    int                n_devs;        /* number of SSD devices               */
    char             **dev_uris;      /* device URIs (copies)                */
    uint32_t           group_size;    /* tokens per group (32)               */
    /* model geometry (cached from llama model params)                       */
    uint32_t           n_layers;      /* transformer layers                  */
    uint32_t           n_embd;        /* embedding dimension (K or V)        */
    enum llama_vocab_type vocab_type; /* for diagnostics                     */
};

/**
 * Initialize the adapter.
 *
 * Opens a kvtier instance backed by the given device URIs,
 * and caches model geometry from the llama_context.
 *
 * Returns 0 on success, negative error code on failure.
 */
int llama_kvcache_ctx_init(struct llama_kvcache_ctx *ctx,
                           struct llama_context *llama_ctx,
                           const char *const *dev_uris, int n_devs,
                           const struct llama_kvcache_cfg *cfg);

/**
 * Destroy the adapter and release all resources.
 */
void llama_kvcache_ctx_destroy(struct llama_kvcache_ctx *ctx);

/**
 * Save a sequence's KV cache to SSD (async).
 *
 * Serializes the KV state via llama_state_seq_get_data() and stores
 * the opaque blob into the kvtier. Also saves the token list
 * so the sequence can be restored later.
 *
 * seq_id:       the llama sequence to save
 * tokens/n_tok: the token list active in this sequence (copied internally)
 * ack/user:     completion callback (fired from kvtier writer thread)
 *
 * Returns KV_EOK on enqueue, negative on error.
 */
int llama_kvcache_save(struct llama_kvcache_ctx *ctx,
                       llama_seq_id seq_id,
                       const llama_token *tokens, size_t n_tok,
                       llama_kvcache_ack_fn ack, void *user);

/**
 * Restore a sequence's KV cache from SSD (sync).
 *
 * Reads the saved state from kvtier and feeds it into the
 * destination sequence via llama_state_seq_set_data().
 *
 * seq_id:         source key (the seq_id that was saved)
 * dest_seq_id:    destination sequence in the llama context
 * tokens_out:     output buffer for saved tokens (caller frees)
 * n_tok_capacity: capacity of tokens_out
 * n_tok_out:      actual number of tokens written
 *
 * Returns positive byte count on success, 0 on miss, negative on error.
 */
size_t llama_kvcache_restore(struct llama_kvcache_ctx *ctx,
                             llama_seq_id seq_id,
                             llama_seq_id dest_seq_id,
                             llama_token *tokens_out,
                             size_t n_tok_capacity,
                             size_t *n_tok_out);

/**
 * Check if a sequence exists in the SSD cache.
 * Returns true if saved, false otherwise.
 */
bool llama_kvcache_has(struct llama_kvcache_ctx *ctx, llama_seq_id seq_id);

/**
 * Evict a sequence from the SSD cache.
 */
int llama_kvcache_evict(struct llama_kvcache_ctx *ctx, llama_seq_id seq_id);

/**
 * Print cache statistics to stderr.
 */
void llama_kvcache_stats(struct llama_kvcache_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_KVCACHE_H */
