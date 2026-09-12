/**
 * llama-kvcache.cpp — llama.cpp KV cache ↔ SSD adapter implementation
 */
#include "llama-kvcache.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

[[maybe_unused]] static uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* generate a kvtier prefix_id from seq_id + token hash */
static uint64_t make_prefix_id(llama_seq_id seq_id,
                               const uint32_t *tokens, size_t n_tok) {
    /* FNV-1a over the token stream, mixed with seq_id */
    uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t)seq_id;
    for (size_t i = 0; i < n_tok; i++) {
        uint32_t t = (uint32_t)tokens[i];
        h ^= (uint64_t)t;
        h *= 0x100000001b3ull;
    }
    /* splitmix64 finalizer */
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    h ^= h >> 31;
    return h;
}

/* generate deterministic tokens from seq_id (for kvtier key) */
static void make_tokens(llama_seq_id seq_id, uint32_t *tokens, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        /* linear congruential generator seeded by seq_id */
        uint64_t v = (uint64_t)seq_id * 2654435761ull + (uint64_t)i * 1103515245ull + 12345ull;
        tokens[i] = (uint32_t)(v >> 16) & 0x7FFFFFFF;
    }
}

/* kvtier ack callback wrapper */
struct save_ack_ctx {
    llama_kvcache_ack_fn user_ack;
    void *user_data;
    uint8_t *state_buf;
    struct llama_kvcache_seq_info *info;
};

static void save_ack_trampoline(void *user, int rc) {
    struct save_ack_ctx *ctx = (struct save_ack_ctx *)user;
    if (ctx->user_ack)
        ctx->user_ack(ctx->user_data, rc);
    std::free(ctx->state_buf);
    std::free(ctx->info->tokens);
    std::free(ctx->info);
    std::free(ctx);
}

/* ------------------------------------------------------------------ */
/* config                                                              */
/* ------------------------------------------------------------------ */

void llama_kvcache_cfg_default(struct llama_kvcache_cfg *cfg) {
    std::memset(cfg, 0, sizeof(*cfg));
    cfg->tokens_per_group  = KV_TOKENS_PER_GROUP;
    cfg->region_cnt        = 6;
    cfg->region_size_pages  = 0;  /* auto */
    cfg->epoch_secs        = 1800;
    cfg->gc_start_pct      = 80;
    cfg->gc_stop_pct       = 70;
    cfg->sync_mode         = false;
    cfg->metrics_level     = 0;
}

/* ------------------------------------------------------------------ */
/* ctx lifecycle                                                       */
/* ------------------------------------------------------------------ */

int llama_kvcache_ctx_init(struct llama_kvcache_ctx *ctx,
                           struct llama_context *llama_ctx,
                           const char *const *dev_uris, int n_devs,
                           const struct llama_kvcache_cfg *cfg) {
    if (!ctx || !llama_ctx || !dev_uris || n_devs <= 0)
        return -KV_EINVAL;

    std::memset(ctx, 0, sizeof(*ctx));
    ctx->llama_ctx  = llama_ctx;
    ctx->n_devs     = n_devs;
    ctx->group_size = cfg && cfg->tokens_per_group ? cfg->tokens_per_group
                                                   : KV_TOKENS_PER_GROUP;

    /* copy device URIs */
    ctx->dev_uris = static_cast<char **>(std::calloc((size_t)n_devs, sizeof(char *)));
    if (!ctx->dev_uris) return -KV_ENOMEM;
    for (int i = 0; i < n_devs; i++) {
        ctx->dev_uris[i] = strdup(dev_uris[i]);
        if (!ctx->dev_uris[i]) {
            for (int j = 0; j < i; j++) std::free(ctx->dev_uris[j]);
            std::free(ctx->dev_uris);
            return -KV_ENOMEM;
        }
    }

    /* configure kvtier */
    struct kv_config sccfg;
    kv_config_default(&sccfg);
    if (cfg) {
        if (cfg->region_cnt)        sccfg.region_cnt        = cfg->region_cnt;
        if (cfg->region_size_pages) sccfg.region_size_pages = cfg->region_size_pages;
        if (cfg->epoch_secs)        sccfg.epoch_secs        = cfg->epoch_secs;
        if (cfg->gc_start_pct)      sccfg.gc_start_pct      = cfg->gc_start_pct;
        if (cfg->gc_stop_pct)       sccfg.gc_stop_pct       = cfg->gc_stop_pct;
        sccfg.sync_mode = cfg->sync_mode;
        sccfg.metrics_level = cfg->metrics_level;
    }
    sccfg.max_layers = 1;  /* we store the entire state as one opaque layer */
    std::memcpy(&ctx->cache_cfg, &sccfg, sizeof(sccfg));

    /* open kvtier */
    int rc = cache_open(&ctx->cache, dev_uris, n_devs, &sccfg);
    if (rc != KV_EOK) {
        for (int i = 0; i < n_devs; i++) std::free(ctx->dev_uris[i]);
        std::free(ctx->dev_uris);
        return rc;
    }

    return KV_EOK;
}

void llama_kvcache_ctx_destroy(struct llama_kvcache_ctx *ctx) {
    if (!ctx) return;
    if (ctx->cache) cache_close(ctx->cache);
    if (ctx->dev_uris) {
        for (int i = 0; i < ctx->n_devs; i++)
            std::free(ctx->dev_uris[i]);
        std::free(ctx->dev_uris);
    }
    std::memset(ctx, 0, sizeof(*ctx));
}

/* ------------------------------------------------------------------ */
/* save (async)                                                        */
/* ------------------------------------------------------------------ */

int llama_kvcache_save(struct llama_kvcache_ctx *ctx,
                       llama_seq_id seq_id,
                       const llama_token *tokens, size_t n_tok,
                       llama_kvcache_ack_fn ack, void *user) {
    if (!ctx || !ctx->cache) return -KV_EINVAL;

    /* 1. serialize KV state from llama */
    size_t state_size = llama_state_seq_get_size(ctx->llama_ctx, seq_id);
    if (state_size == 0) return -KV_ENOENT;

    uint8_t *state_buf = static_cast<uint8_t *>(std::malloc(state_size));
    if (!state_buf) return -KV_ENOMEM;

    size_t got = llama_state_seq_get_data(ctx->llama_ctx, state_buf,
                                          state_size, seq_id);
    if (got == 0) {
        std::free(state_buf);
        return -KV_EREAD;
    }

    /* 2. copy token list */
    llama_token *tok_copy = NULL;
    if (tokens && n_tok > 0) {
        tok_copy = static_cast<llama_token *>(std::malloc(n_tok * sizeof(llama_token)));
        if (!tok_copy) { std::free(state_buf); return -KV_ENOMEM; }
        std::memcpy(tok_copy, tokens, n_tok * sizeof(llama_token));
    }

    /* 3. build kvtier key — MUST match restore/has, which derive the
     *    prefix_id from make_tokens(seq_id), not from the user token list */
    uint32_t sc_tokens[KV_TOKENS_PER_GROUP];
    make_tokens(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);
    uint64_t prefix_id = make_prefix_id(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);

    /* 4. store as a single opaque KV layer (the entire serialized state) */
    struct kv_data_ref recs[1];
    recs[0] = kv_data_ref{
        .base = state_buf,
        .off  = 0,
        .len  = (uint32_t)state_size,
    };

    /* 5. wrap user callback to free buffers */
    struct save_ack_ctx *actx = static_cast<struct save_ack_ctx *>(std::calloc(1, sizeof(*actx)));
    if (!actx) { std::free(state_buf); std::free(tok_copy); return -KV_ENOMEM; }
    actx->user_ack  = ack;
    actx->user_data = user;
    actx->state_buf = state_buf;
    actx->info      = static_cast<struct llama_kvcache_seq_info *>(
        std::calloc(1, sizeof(struct llama_kvcache_seq_info)));
    if (!actx->info) { std::free(state_buf); std::free(tok_copy); std::free(actx); return -KV_ENOMEM; }
    actx->info->seq_id   = seq_id;
    actx->info->tokens   = tok_copy;
    actx->info->n_tokens = n_tok;

    /* 6. single group (group_idx = 0), single record (the whole state) */
    int rc = cache_put(ctx->cache, prefix_id, 0,
                       sc_tokens, KV_TOKENS_PER_GROUP,
                       0,          /* no expiry */
                       1,          /* n_layers = 1 opaque blob */
                       recs,
                       save_ack_trampoline, actx);

    if (rc != KV_EOK) {
        std::free(state_buf);
        std::free(tok_copy);
        std::free(actx->info);
        std::free(actx);
    }
    return rc;
}

/* ------------------------------------------------------------------ */
/* restore (sync)                                                      */
/* ------------------------------------------------------------------ */

size_t llama_kvcache_restore(struct llama_kvcache_ctx *ctx,
                             llama_seq_id seq_id,
                             llama_seq_id dest_seq_id,
                             llama_token *tokens_out,
                             size_t n_tok_capacity,
                             size_t *n_tok_out) {
    if (!ctx || !ctx->cache) return 0;
    (void)tokens_out;
    (void)n_tok_capacity;
    if (n_tok_out) *n_tok_out = 0;

    /*
     * We need the original tokens to reconstruct the prefix_id.
     * Since we can't enumerate all possible token lists, we rely on
     * the caller having saved with known tokens.  For the common case
     * of "restore what was just saved", the caller should have the
     * token list from the original request.
     *
     * Alternative: we can also do a scan-based lookup (see below).
     * For now, we use a brute-force token generation from seq_id
     * as a fallback.
     */
    uint32_t sc_tokens[KV_TOKENS_PER_GROUP];
    make_tokens(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);

    uint64_t prefix_id = make_prefix_id(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);

    /* read from kvtier */
    struct cache_get_result res;
    int rc = cache_get(ctx->cache, prefix_id, sc_tokens,
                       KV_TOKENS_PER_GROUP, &res);
    if (rc != KV_EOK) return 0;

    /* locate the layer-0 record (the opaque state blob) in the stream */
    const uint8_t *blob = NULL;
    uint32_t blob_len = 0;
    for (int i = 0; i < res.n_records; i++) {
        if (res.recs[i].layer_id == 0) {
            blob     = (const uint8_t *)res.buf + res.recs[i].off;
            blob_len = res.recs[i].len;
            break;
        }
    }
    if (!blob) {
        cache_result_free(&res);
        return 0;
    }

    /* restore into llama context */
    size_t restored = llama_state_seq_set_data(
        ctx->llama_ctx,
        blob,
        blob_len,
        dest_seq_id);

    cache_result_free(&res);

    if (n_tok_out && restored > 0)
        *n_tok_out = restored;

    return restored;
}

/* ------------------------------------------------------------------ */
/* check / evict                                                       */
/* ------------------------------------------------------------------ */

bool llama_kvcache_has(struct llama_kvcache_ctx *ctx, llama_seq_id seq_id) {
    if (!ctx || !ctx->cache) return false;

    uint32_t sc_tokens[KV_TOKENS_PER_GROUP];
    make_tokens(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);
    uint64_t prefix_id = make_prefix_id(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);

    struct cache_get_result res;
    int rc = cache_get(ctx->cache, prefix_id, sc_tokens,
                       KV_TOKENS_PER_GROUP, &res);
    if (rc == KV_EOK) {
        cache_result_free(&res);
        return true;
    }
    return false;
}

int llama_kvcache_evict(struct llama_kvcache_ctx *ctx, llama_seq_id seq_id) {
    if (!ctx || !ctx->cache) return -KV_EINVAL;

    uint32_t sc_tokens[KV_TOKENS_PER_GROUP];
    make_tokens(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);
    uint64_t prefix_id = make_prefix_id(seq_id, sc_tokens, KV_TOKENS_PER_GROUP);

    return cache_evict(ctx->cache, prefix_id, sc_tokens, KV_TOKENS_PER_GROUP);
}

/* ------------------------------------------------------------------ */
/* diagnostics                                                         */
/* ------------------------------------------------------------------ */

void llama_kvcache_stats(struct llama_kvcache_ctx *ctx) {
    if (!ctx || !ctx->cache) return;
    std::fprintf(stderr, "[llama-kvcache] puts=%lu hits=%lu misses=%lu "
                    "batches=%lu bytes_written=%lu bytes_migrated=%lu "
                    "rotations=%lu drops=%lu errors=%lu\n",
            (unsigned long)cache_stats(ctx->cache, "put_reqs"),
            (unsigned long)cache_stats(ctx->cache, "get_hits"),
            (unsigned long)cache_stats(ctx->cache, "get_misses"),
            (unsigned long)cache_stats(ctx->cache, "batches"),
            (unsigned long)cache_stats(ctx->cache, "bytes_written"),
            (unsigned long)cache_stats(ctx->cache, "bytes_migrated"),
            (unsigned long)cache_stats(ctx->cache, "rotations"),
            (unsigned long)cache_stats(ctx->cache, "drops"),
            (unsigned long)cache_stats(ctx->cache, "put_errs"));
}
