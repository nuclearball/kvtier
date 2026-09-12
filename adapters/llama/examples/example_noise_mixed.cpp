/**
 * example_noise_mixed.cpp — llama.cpp KV save/restore with background noise IO
 *
 * Runs two independent kvtier instances in one process:
 *   App 1 (measured):  llama adapter save/restore cycles, verified for
 *                      byte/token correctness each round.
 *   App 2 (noise):     another cache_t on its own device files, continuously
 *                      doing cache_put + cache_get of random keys.
 *
 * Both instances allocate their own device files; when those files live on
 * the same physical disk they contend at the device level.
 *
 * Usage:
 *   ./example_noise_mixed -m <model.gguf> [-n ngl] [-r rounds]
 *       [--noise 0|1] [--noise-ops N] [--noise-bytes B] [--noise-devs D]
 *       [--dir DIR] [--csv PATH]
 */
#include "llama-kvcache.h"
#include "kvtier.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <pthread.h>
#include <atomic>
#include <ctime>

#define CTX_SIZE     512
#define N_GEN        16
#define N_EXT        8   /* tokens generated beyond the cached region
                            for end-to-end output comparison */

static int g_ctx = CTX_SIZE;
#define NOISE_TOKENS 32

/* ------------------------------ timing ------------------------------ */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* ------------------------------ noise app --------------------------- */
struct noise_stats {
    std::atomic<long> puts, put_errs, gets, get_errs, verify_errs;
};

struct noise_ctx {
    cache_t            *cache;
    uint64_t            n_keys;
    uint32_t            payload_len;
    int                 rate_limit;   /* max puts+gets per sec, 0 = unlim */
    std::atomic<int>    stop;
    struct noise_stats  stats;
    uint8_t            *payload;
    uint32_t           *key_tokens;   /* tokens per pre-filled key */
    uint64_t            rng;
};

static volatile int noise_ack_rc = -999;
static void noise_ack_cb(void *u, int rc) { (void)u; noise_ack_rc = rc; }

static uint64_t noise_rng_next(struct noise_ctx *n) {
    uint64_t x = n->rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    n->rng = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static void noise_fill_tokens(struct noise_ctx *n, uint64_t key,
                              uint32_t *tok) {
    (void)n;
    for (int i = 0; i < NOISE_TOKENS; i++)
        tok[i] = (uint32_t)((key * 2654435761u) ^ ((uint64_t)i << 13));
}

static int noise_put_sync(struct noise_ctx *n, uint64_t key) {
    uint32_t tokens[NOISE_TOKENS];
    noise_fill_tokens(n, key, tokens);
    struct kv_data_ref recs[1] = {
        { .base = n->payload, .off = 0, .len = n->payload_len } };
    noise_ack_rc = -999;
    int rc = cache_put(n->cache, key, 0, tokens, NOISE_TOKENS, 0, 1, recs,
                       noise_ack_cb, NULL);
    if (rc != KV_EOK) return rc;
    while (noise_ack_rc == -999) usleep(100);
    return noise_ack_rc;
}

static int noise_get_verify(struct noise_ctx *n, uint64_t key) {
    uint32_t tokens[NOISE_TOKENS];
    noise_fill_tokens(n, key, tokens);
    struct cache_get_result res;
    int rc = cache_get(n->cache, key, tokens, NOISE_TOKENS, &res);
    if (rc != KV_EOK) return rc;
    int ok = 0;
    for (int i = 0; i < res.n_records; i++)
        if (res.recs[i].layer_id == 0 && res.recs[i].len == n->payload_len &&
            memcmp((const uint8_t *)res.buf + res.recs[i].off,
                   n->payload, n->payload_len) == 0)
            ok = 1;
    cache_result_free(&res);
    return ok ? KV_EOK : -1;
}

static void *noise_thread(void *arg) {
    struct noise_ctx *n = (struct noise_ctx *)arg;
    uint64_t iter = 0;
    double t_start = now_ms();
    double next_slot = t_start;

    while (!n->stop.load()) {
        /* rate limiting: one put + one get per iteration slot */
        if (n->rate_limit > 0) {
            double slot_ms = 2000.0 / (double)n->rate_limit; /* 2 ops/iter */
            double now = now_ms();
            if (now < next_slot) {
                usleep((useconds_t)((next_slot - now) * 1000.0));
            } else if (now - next_slot > 1000.0) {
                next_slot = now;   /* fell behind, reset */
            }
            next_slot += slot_ms;
        }

        uint64_t wkey = noise_rng_next(n) % n->n_keys;
        if (noise_put_sync(n, wkey) == KV_EOK)
            n->stats.puts.fetch_add(1);
        else
            n->stats.put_errs.fetch_add(1);

        uint64_t rkey = noise_rng_next(n) % n->n_keys;
        int rc = noise_get_verify(n, rkey);
        if (rc == KV_EOK)
            n->stats.gets.fetch_add(1);
        else if (rc == KV_ENOENT || rc == KV_EVICTED)
            ; /* raced with GC rotation: acceptable for noise load */
        else if (rc < 0)
            n->stats.get_errs.fetch_add(1);
        else
            n->stats.verify_errs.fetch_add(1);

        iter++;
    }
    (void)t_start;
    (void)iter;
    return NULL;
}

/* --------------------------- measured app --------------------------- */
struct ack_state { volatile int done; volatile int rc; };
static void save_ack_cb(void *user, int rc) {
    struct ack_state *s = (struct ack_state *)user;
    s->rc = rc;
    s->done = 1;
}

static int generate_tokens(struct llama_context *lctx, llama_token *tokens,
                           int n_tokens, int n_past) {
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = n_past + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }
    int rc = llama_decode(lctx, batch);
    llama_batch_free(batch);
    return rc;
}

static llama_token sample_next(struct llama_context *lctx,
                               const struct llama_vocab *vocab) {
    int n_vocab = llama_vocab_n_tokens(vocab);
    float *logits = llama_get_logits(lctx);
    llama_token best = 0;
    float best_val = -1e9f;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] > best_val) { best_val = logits[i]; best = (llama_token)i; }
    return best;
}

static void create_device_file(const char *path, size_t mb) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, (long)mb * 1024 * 1024 - 1, SEEK_SET);
    fputc(0, f);
    fclose(f);
}

/* ------------------------------- main ------------------------------- */
int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *dir         = "/tmp";
    const char *csv_path    = NULL;
    int   ngl               = 0;
    int   rounds            = 20;
    int   noise_on          = 1;
    int   noise_rate        = 0;      /* ops/sec, 0 = unlim */
    uint32_t noise_bytes    = 8192;
    int   noise_devs        = 2;

    static const char usage[] =
        "Usage: %s -m <model.gguf> [-n ngl] [-r rounds] [--ctx N]\n"
        "    [--noise 0|1] [--noise-ops ops_per_sec] [--noise-bytes B]\n"
        "    [--noise-devs D] [--dir DIR] [--csv PATH]\n";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) g_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise") && i + 1 < argc) noise_on = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise-ops") && i + 1 < argc) noise_rate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise-bytes") && i + 1 < argc) noise_bytes = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise-devs") && i + 1 < argc) noise_devs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else { fprintf(stderr, usage, argv[0]); return 1; }
    }
    if (!model_path) { fprintf(stderr, usage, argv[0]); return 1; }
    if (noise_devs < 1) noise_devs = 1;
    if (noise_devs > 8) noise_devs = 8;
    if (noise_bytes < 512) noise_bytes = 512;

    /* -------- device files (each app allocates its own) -------- */
    static char kv_dev[512], noise_devs_s[8][512];
    char *noise_uris[8];
    snprintf(kv_dev, sizeof(kv_dev), "%s/noise_mixed_kv.img", dir);
    for (int i = 0; i < noise_devs; i++) {
        snprintf(noise_devs_s[i], sizeof(noise_devs_s[i]),
                 "%s/noise_mixed_bg%d.img", dir, i);
        noise_uris[i] = noise_devs_s[i];
    }
    create_device_file(kv_dev, 1024);
    for (int i = 0; i < noise_devs; i++)
        create_device_file(noise_devs_s[i], 512);
    { char ck[600]; snprintf(ck, sizeof(ck), "%s.ckpt", kv_dev); unlink(ck); }

    /* -------- noise instance (App 2) -------- */
    struct noise_ctx noise{};
    noise.rng = 0x9e3779b97f4a7c15ull;
    noise.n_keys = 512;
    noise.payload_len = noise_bytes;
    noise.rate_limit = noise_rate;
    noise.stop.store(0);
    noise.stats.puts.store(0);
    noise.stats.put_errs.store(0);
    noise.stats.gets.store(0);
    noise.stats.get_errs.store(0);
    noise.stats.verify_errs.store(0);
    noise.payload = (uint8_t *)malloc(noise_bytes);
    memset(noise.payload, 0x5A, noise_bytes);

    int noise_started = 0;
    pthread_t noise_tid;
    if (noise_on) {
        struct kv_config ncfg;
        kv_config_default(&ncfg);
        ncfg.metrics_level = 1;
        ncfg.max_layers = 1;
        if (cache_open(&noise.cache, (const char *const *)noise_uris,
                       noise_devs, &ncfg) != KV_EOK) {
            fprintf(stderr, "Error: noise cache_open failed\n");
            return 1;
        }
        if (pthread_create(&noise_tid, NULL, noise_thread, &noise) != 0) {
            fprintf(stderr, "Error: pthread_create failed\n");
            return 1;
        }
        noise_started = 1;
        fprintf(stderr, "Noise app started: %d devs x 512MiB, %uB payload, "
                        "%s rate\n", noise_devs, noise_bytes,
                        noise_rate > 0 ? "limited" : "unlimited");
    }

    /* -------- llama app (App 1) -------- */
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = g_ctx;
    cparams.n_batch  = 512;
    struct llama_context *lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    struct llama_kvcache_cfg kvcfg;
    llama_kvcache_cfg_default(&kvcfg);
    kvcfg.metrics_level = 1;

    struct llama_kvcache_ctx kvctx;
    const char *kv_devs[] = { kv_dev };
    if (llama_kvcache_ctx_init(&kvctx, lctx, kv_devs, 1, &kvcfg)
            != KV_EOK) {
        fprintf(stderr, "Error: adapter init failed\n");
        return 1;
    }

    const char *prompt = "Hello, world!";
    llama_token prompt_tokens[256];
    int n_tok = llama_tokenize(vocab, prompt, strlen(prompt),
                               prompt_tokens, 256, true, true);
    if (n_tok <= 0) {
        fprintf(stderr, "Error: tokenization failed\n");
        return 1;
    }

    FILE *csv = csv_path ? fopen(csv_path, "w") : NULL;
    if (csv_path && !csv) {
        fprintf(stderr, "Warning: cannot open %s\n", csv_path);
        csv_path = NULL;
    }
    if (csv)
        fprintf(csv, "round,noise,save_ms,restore_ms,bytes,n_tokens,verify,ext_output\n");

    /* ------------------------------ rounds ------------------------------ */
    int failures = 0;
    double save_ms_sum = 0, restore_ms_sum = 0;

    for (int round = 0; round < rounds; round++) {
        /* 1. prefill + generate */
        llama_memory_clear(llama_get_memory(lctx), true);
        if (generate_tokens(lctx, prompt_tokens, n_tok, 0) != 0) {
            fprintf(stderr, "round %d: prefill failed\n", round);
            failures++;
            break;
        }
        llama_token generated[N_GEN];
        llama_token cur = sample_next(lctx, vocab);
        generated[0] = cur;
        for (int i = 1; i < N_GEN; i++) {
            if (generate_tokens(lctx, &cur, 1, n_tok + i - 1) != 0) break;
            cur = sample_next(lctx, vocab);
            generated[i] = cur;
        }

        size_t total_tok = (size_t)(n_tok + N_GEN);
        llama_token all_tokens[512];
        if (total_tok > 512) total_tok = 512;
        memcpy(all_tokens, prompt_tokens, (size_t)n_tok * sizeof(llama_token));
        memcpy(all_tokens + n_tok, generated,
               (total_tok - (size_t)n_tok) * sizeof(llama_token));

        /* 2. save (async, wait ack) */
        struct ack_state ack = { .done = 0, .rc = 0 };
        double t0 = now_ms();
        if (llama_kvcache_save(&kvctx, 0, all_tokens, total_tok,
                               save_ack_cb, &ack) != KV_EOK) {
            fprintf(stderr, "round %d: save enqueue failed\n", round);
            failures++;
            break;
        }
        while (!ack.done) usleep(200);
        double save_ms = now_ms() - t0;
        if (ack.rc != KV_EOK) {
            fprintf(stderr, "round %d: save failed rc=%d\n", round, ack.rc);
            failures++;
            break;
        }

        /* 2b. reference output: continue N_EXT tokens on the LIVE state
         *     (save snapshot already taken, so this does not affect it) */
        llama_token ext_ref[N_EXT];
        if (generate_tokens(lctx, &generated[N_GEN - 1], 1,
                            n_tok + N_GEN - 1) != 0) {
            fprintf(stderr, "round %d: ext decode failed\n", round);
            failures++;
            break;
        }
        ext_ref[0] = sample_next(lctx, vocab);
        for (int i = 1; i < N_EXT; i++) {
            if (generate_tokens(lctx, &ext_ref[i - 1], 1,
                                n_tok + N_GEN - 1 + i) != 0) break;
            ext_ref[i] = sample_next(lctx, vocab);
        }
        fprintf(stderr, "round %d: saved\n", round);

        /* 3. clear + restore (clear can be disabled to test fork behavior) */
        if (getenv("KV_NO_CLEAR") == NULL)
            llama_memory_clear(llama_get_memory(lctx), true);
        llama_token restored_tokens[512];
        size_t n_restored = 0;
        t0 = now_ms();
        size_t bytes = llama_kvcache_restore(&kvctx, 0, 0,
                                             restored_tokens, 512,
                                             &n_restored);
        double restore_ms = now_ms() - t0;
        if (bytes == 0) {
            fprintf(stderr, "round %d: restore miss (post has=%d)\n", round,
                    llama_kvcache_has(&kvctx, 0));
            failures++;
            break;
        }

        /* 4. verify: replay generated tokens over the restored cache and
         *    check each sampled next-token matches */
        int match = 1;
        for (int i = 0; i < N_GEN - 1 && match; i++) {
            llama_token cur = generated[i];
            if (generate_tokens(lctx, &cur, 1, n_tok + i) != 0) break;
            if (sample_next(lctx, vocab) != generated[i + 1]) {
                fprintf(stderr, "Mismatch at pos %d\n", i);
                match = 0;
            }
        }

        /* 4b. output-layer verify: continue N_EXT tokens beyond the cached
         *     region after restore; must equal the live-state reference */
        int ext_ok = match;
        if (match) {
            llama_token ext_out[N_EXT];
            memset(ext_out, 0xFF, sizeof(ext_out));
            if (generate_tokens(lctx, &generated[N_GEN - 1], 1,
                                n_tok + N_GEN - 1) != 0) {
                match = 0;
            } else {
                ext_out[0] = sample_next(lctx, vocab);
                for (int i = 1; i < N_EXT; i++) {
                    if (generate_tokens(lctx, &ext_out[i - 1], 1,
                                        n_tok + N_GEN - 1 + i) != 0) break;
                    ext_out[i] = sample_next(lctx, vocab);
                }
                for (int i = 0; i < N_EXT; i++) {
                    if (ext_out[i] != ext_ref[i]) {
                        fprintf(stderr,
                                "Output mismatch at ext pos %d: "
                                "expected %d, got %d\n",
                                i, ext_ref[i], ext_out[i]);
                        ext_ok = 0;
                        break;
                    }
                }
            }
        }
        if (!ext_ok) match = 0;
        if (!match) failures++;

        save_ms_sum += save_ms;
        restore_ms_sum += restore_ms;
        fprintf(stderr, "round %3d: save %8.2fms  restore %8.2fms  "
                        "%zuB %zu tok  verify=%s ext_output=%s\n",
                round, save_ms, restore_ms, bytes, n_restored,
                match ? "OK" : "FAIL", ext_ok ? "OK" : "FAIL");
        if (csv)
            fprintf(csv, "%d,%d,%.3f,%.3f,%zu,%zu,%s,%s\n",
                    round, noise_on, save_ms, restore_ms, bytes, n_restored,
                    match ? "OK" : "FAIL", ext_ok ? "OK" : "FAIL");
    }

    /* ----------------------------- teardown ----------------------------- */
    if (noise_started) {
        noise.stop.store(1);
        pthread_join(noise_tid, NULL);
        double puts = noise.stats.puts.load();
        double gets = noise.stats.gets.load();
        fprintf(stderr, "\nNoise stats: puts=%ld (err=%ld) gets=%ld "
                        "(err=%ld verify_err=%ld)\n",
                noise.stats.puts.load(),
                noise.stats.put_errs.load(),
                noise.stats.gets.load(),
                noise.stats.get_errs.load(),
                noise.stats.verify_errs.load());
        if (csv)
            fprintf(csv, "# noise puts=%ld put_errs=%ld gets=%ld get_errs=%ld "
                         "verify_errs=%ld\n",
                    noise.stats.puts.load(),
                    noise.stats.put_errs.load(),
                    noise.stats.gets.load(),
                    noise.stats.get_errs.load(),
                    noise.stats.verify_errs.load());
        (void)puts; (void)gets;
    }

    if (rounds > 0)
        fprintf(stderr, "\nSummary: rounds=%d failures=%d "
                        "avg_save=%.2fms avg_restore=%.2fms\n",
                rounds, failures,
                save_ms_sum / (double)(rounds > 0 ? rounds : 1),
                restore_ms_sum / (double)(rounds > 0 ? rounds : 1));

    llama_kvcache_ctx_destroy(&kvctx);
    llama_free(lctx);
    llama_model_free(model);
    if (noise_started) cache_close(noise.cache);
    free(noise.payload);
    if (csv) fclose(csv);
    unlink(kv_dev);
    { char ck[600]; snprintf(ck, sizeof(ck), "%s.ckpt", kv_dev); unlink(ck); }

    return failures ? 1 : 0;
}
