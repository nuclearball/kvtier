/**
 * example_save_restore.cpp — demonstrate saving and restoring llama KV cache to SSD
 *
 * Usage:
 *   ./example_save_restore -m <model.gguf> -ngl 99
 */
#include "llama-kvcache.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <getopt.h>

#define SSD_DEV  "/tmp/kv_dev0.img"
#define CTX_SIZE 2048
#define N_GEN    32

struct ack_state { volatile int done; volatile int rc; };

static void save_ack_cb(void *user, int rc) {
    struct ack_state *s = (struct ack_state *)user;
    s->rc   = rc;
    s->done = 1;
}

static void create_test_device(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 256 * 1024 * 1024 - 1, SEEK_SET);
    fputc(0, f);
    fclose(f);
}

static int generate_tokens(struct llama_context *lctx, llama_token *tokens,
                           int n_tokens, int n_past) {
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_tokens - 1) ? 1 : 0;
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
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = (llama_token)i;
        }
    }
    return best;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int ngl = 0;

    int opt;
    while ((opt = getopt(argc, argv, "m:n:h")) != -1) {
        switch (opt) {
        case 'm': model_path = optarg; break;
        case 'n': ngl = atoi(optarg); break;
        default:
            fprintf(stderr, "Usage: %s -m <model.gguf> [-n <ngl>]\n", argv[0]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "Error: model path required\n");
        return 1;
    }

    create_test_device(SSD_DEV);

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = CTX_SIZE;
    cparams.n_batch = 512;
    struct llama_context *lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "Model loaded: %s\n", model_path);
    fprintf(stderr, "Context size: %d\n", CTX_SIZE);

    /* --- Phase 1: Generate some tokens --- */
    const char *prompt = "Hello, world!";
    llama_token tokens[256];
    int n_tok = llama_tokenize(vocab, prompt, strlen(prompt),
                               tokens, 256, true, true);
    if (n_tok < 0) {
        fprintf(stderr, "Error: tokenization failed\n");
        llama_free(lctx); llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "\n--- Phase 1: Prefill + generate %d tokens ---\n", N_GEN);

    if (generate_tokens(lctx, tokens, n_tok, 0) != 0) {
        fprintf(stderr, "Error: prefill failed\n");
        llama_free(lctx); llama_model_free(model);
        return 1;
    }

    llama_token generated[N_GEN];
    llama_token cur = sample_next(lctx, vocab);
    generated[0] = cur;
    for (int i = 1; i < N_GEN; i++) {
        if (generate_tokens(lctx, &cur, 1, n_tok + i - 1) != 0) break;
        cur = sample_next(lctx, vocab);
        generated[i] = cur;
    }

    fprintf(stderr, "Generated first tokens: [");
    for (int i = 0; i < N_GEN && i < 8; i++)
        fprintf(stderr, "%d%s", generated[i], i < 7 ? ", " : "");
    fprintf(stderr, "]\n");

    /* --- Phase 2: Save KV cache to SSD --- */
    fprintf(stderr, "\n--- Phase 2: Save KV cache to SSD ---\n");

    size_t total_tok = (size_t)(n_tok + N_GEN);
    llama_token *all_tokens = (llama_token *)malloc(total_tok * sizeof(llama_token));
    memcpy(all_tokens, tokens, n_tok * sizeof(llama_token));
    memcpy(all_tokens + n_tok, generated, N_GEN * sizeof(llama_token));

    struct llama_kvcache_cfg kvcfg;
    llama_kvcache_cfg_default(&kvcfg);
    kvcfg.metrics_level = 1;

    struct llama_kvcache_ctx kvctx;
    const char *kv_devs[] = { SSD_DEV };
    int rc = llama_kvcache_ctx_init(&kvctx, lctx, kv_devs, 1, &kvcfg);
    if (rc != KV_EOK) {
        fprintf(stderr, "Error: adapter init failed (%d)\n", rc);
        free(all_tokens);
        llama_free(lctx); llama_model_free(model);
        return 1;
    }

    struct ack_state ack = { .done = 0, .rc = 0 };
    rc = llama_kvcache_save(&kvctx, 0, all_tokens, total_tok,
                            save_ack_cb, &ack);
    if (rc != KV_EOK) {
        fprintf(stderr, "Error: save enqueue failed (%d)\n", rc);
        llama_kvcache_ctx_destroy(&kvctx);
        free(all_tokens);
        llama_free(lctx); llama_model_free(model);
        return 1;
    }

    while (!ack.done) usleep(1000);
    if (ack.rc != KV_EOK) {
        fprintf(stderr, "Error: save failed (%d)\n", ack.rc);
        llama_kvcache_ctx_destroy(&kvctx);
        free(all_tokens);
        llama_free(lctx); llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "KV cache saved to SSD\n");
    llama_kvcache_stats(&kvctx);

    /* --- Phase 3: Clear and restore --- */
    fprintf(stderr, "\n--- Phase 3: Clear KV cache and restore from SSD ---\n");
    llama_memory_clear(llama_get_memory(lctx), true);
    fprintf(stderr, "KV cache cleared\n");

    llama_token restored_tokens[256];
    size_t n_restored = 0;
    size_t bytes = llama_kvcache_restore(&kvctx, 0, 0,
                                         restored_tokens, 256, &n_restored);
    if (bytes == 0) {
        fprintf(stderr, "Error: restore failed (cache miss)\n");
        llama_kvcache_ctx_destroy(&kvctx);
        free(all_tokens);
        llama_free(lctx); llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "Restored %zu bytes, %zu tokens\n", bytes, n_restored);

    /* --- Phase 4: Verify --- */
    fprintf(stderr, "\n--- Phase 4: Verify restored cache ---\n");

    llama_token verify[N_GEN];
    cur = sample_next(lctx, vocab);
    verify[0] = cur;
    for (int i = 1; i < N_GEN; i++) {
        if (generate_tokens(lctx, &cur, 1, n_tok + i - 1) != 0) break;
        cur = sample_next(lctx, vocab);
        verify[i] = cur;
    }

    int match = 1;
    for (int i = 0; i < N_GEN; i++) {
        if (generated[i] != verify[i]) {
            match = 0;
            fprintf(stderr, "Mismatch at pos %d: expected %d, got %d\n",
                    i, generated[i], verify[i]);
            break;
        }
    }
    fprintf(stderr, "Verification: %s\n", match ? "PASS" : "FAIL");
    llama_kvcache_stats(&kvctx);

    llama_kvcache_ctx_destroy(&kvctx);
    free(all_tokens);
    llama_free(lctx);
    llama_model_free(model);
    unlink(SSD_DEV);

    return match ? 0 : 1;
}
