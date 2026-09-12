#include "common.hpp"
#include "model_profile.h"
#include <cstdio>
#include <cstring>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

#define PAGE 4096

static void test_capacity(void) {
    const kv_model_profile *ll =
        kv_mp_builtin(KV_MODEL_LLAMA3_1_8B);
    CHECK(ll != NULL);
    CHECK(kv_mp_bytes_per_token_layer(ll) == 4096u);      /* 2*8*128*2 */
    CHECK(kv_mp_bytes_per_token(ll) == 128ull * 1024);   /* 32 层 */
    CHECK(kv_mp_group_bytes(ll, 32) == 4ull * 1024 * 1024); /* 32 token */

    const kv_model_profile *ds =
        kv_mp_builtin(KV_MODEL_DEEPSEEK_V3);
    CHECK(ds != NULL);
    CHECK(kv_mp_bytes_per_token_layer(ds) ==
          (512u + 64u) * 2u);                            /* MLA latent */
    CHECK(kv_mp_bytes_per_token(ds) == 70272u);          /* 61 层 */
    CHECK(kv_mp_group_bytes(ds, 32) == 61ull * 1152 * 32);
    /* DeepSeek 32-token group 恰为整数页（1152*32 = 9 pages） */
    CHECK(kv_mp_group_bytes(ds, 32) % PAGE == 0);

    const kv_model_profile *q7 = kv_mp_builtin(KV_MODEL_QWEN2_5_7B);
    CHECK(kv_mp_bytes_per_token_layer(q7) == 2048u);
    CHECK(kv_mp_bytes_per_token(q7) == 56ull * 1024);

    const kv_model_profile *glm = kv_mp_builtin(KV_MODEL_GLM4_9B);
    CHECK(kv_mp_bytes_per_token_layer(glm) == 1024u);
    CHECK(kv_mp_bytes_per_token(glm) == 40ull * 1024);

    const kv_model_profile *falcon = kv_mp_builtin(KV_MODEL_FALCON_7B);
    CHECK(kv_mp_bytes_per_token_layer(falcon) == 256u);
    CHECK(kv_mp_bytes_per_token(falcon) == 8ull * 1024);

    /* context 容量 */
    CHECK(kv_mp_context_bytes(ll, 8192) == 1ull * 1024 * 1024 * 1024);
}

static void test_swa_bound(void) {
    kv_model_profile p = *kv_mp_builtin(KV_MODEL_MISTRAL_7B);
    p.kind = KV_ATTN_SWA;
    p.window = 4096;
    uint64_t bounded = kv_mp_context_bytes(&p, 100000);
    uint64_t full = kv_mp_context_bytes(&p, 4096);
    CHECK(bounded == full);
}

static void test_ssm_zero(void) {
    kv_model_profile p;
    memset(&p, 0, sizeof(p));
    p.kind = KV_ATTN_SSM;
    p.n_layers = 32;
    CHECK(kv_mp_bytes_per_token(&p) == 0);
    CHECK(kv_mp_context_bytes(&p, 1000000) == 0);
}

static void check_tuning(const kv_model_profile *p, const char *tag) {
    kv_model_tuning t;
    CHECK(kv_mp_tune(p, PAGE, 32ull * 1024 * 1024, 256 * 1024,
                     256 * 1024, &t) == KV_MP_OK);
    uint64_t layer_bytes = kv_mp_group_layer_bytes(p, 32);
    printf("  [%s] group=%llu align=%u unit=%llu thr=%llu batch=%llu\n",
           tag, (unsigned long long)t.group_bytes, t.alignment,
           (unsigned long long)t.stripe_unit,
           (unsigned long long)t.stripe_threshold,
           (unsigned long long)t.batch_min_bytes);
    CHECK(t.group_bytes > 0);
    CHECK(t.alignment && (t.alignment & (t.alignment - 1)) == 0);
    CHECK(t.alignment <= PAGE);
    CHECK(t.stripe_unit % PAGE == 0);
    CHECK(layer_bytes && t.stripe_unit % layer_bytes == 0);
    CHECK(t.stripe_threshold >= PAGE);
    CHECK(t.batch_min_bytes % PAGE == 0);
    CHECK(t.batch_min_bytes <= t.stripe_threshold);
    CHECK(t.dram_entry_max_bytes >= t.group_bytes);
    CHECK(t.region_align_bytes % t.group_bytes == 0);
}

static void test_tuning(void) {
    check_tuning(kv_mp_builtin(KV_MODEL_LLAMA3_1_8B), "llama8b");
    check_tuning(kv_mp_builtin(KV_MODEL_LLAMA3_1_405B), "llama405b");
    check_tuning(kv_mp_builtin(KV_MODEL_QWEN2_5_7B), "qwen7b");
    check_tuning(kv_mp_builtin(KV_MODEL_DEEPSEEK_V3), "dsv3-mla");
    check_tuning(kv_mp_builtin(KV_MODEL_GLM4_9B), "glm4-9b");
}

static void test_names(void) {
    CHECK(kv_mp_by_name("deepseek-v3") ==
          kv_mp_builtin(KV_MODEL_DEEPSEEK_V3));
    CHECK(kv_mp_by_name("nope") == NULL);
}

static void test_hf_config_parse(void) {
    /* Llama-3.1-8B-like */
    const char *llama =
        "{\"num_hidden_layers\":32,\"num_attention_heads\":32,"
        "\"num_key_value_heads\":8,\"hidden_size\":4096,"
        "\"torch_dtype\":\"bfloat16\",\"model_type\":\"llama\"}";
    kv_model_profile p;
    CHECK(kv_mp_from_hf_config_json(llama, &p) == KV_MP_OK);
    CHECK(p.kind == KV_ATTN_GQA);
    CHECK(p.n_layers == 32 && p.n_kv_heads == 8 && p.head_dim == 128);
    CHECK(p.dtype_bytes == 2);
    CHECK(kv_mp_bytes_per_token_layer(&p) == 4096u);
    CHECK(kv_mp_bytes_per_token(&p) == 128ull * 1024);

    /* DeepSeek-V3-like MLA (nested text_config, no head_dim) */
    const char *ds =
        "{\"text_config\":{\"num_hidden_layers\":61,"
        "\"num_attention_heads\":128,\"kv_lora_rank\":512,"
        "\"qk_rope_head_dim\":64,\"qk_nope_head_dim\":128,"
        "\"v_head_dim\":128,\"hidden_size\":7168},"
        "\"torch_dtype\":\"bfloat16\",\"model_type\":\"deepseek_v3\"}";
    CHECK(kv_mp_from_hf_config_json(ds, &p) == KV_MP_OK);
    CHECK(p.kind == KV_ATTN_MLA);
    CHECK(p.n_layers == 61 && p.latent_dim == 512 && p.rope_dim == 64);
    CHECK(p.head_dim == 128);                     /* qk_nope_head_dim */
    CHECK(kv_mp_bytes_per_token_layer(&p) == 1152u);
    CHECK(kv_mp_bytes_per_token(&p) == 70272u);

    /* GQA with 2 KV groups (GLM-4 style) */
    const char *gqa2 =
        "{\"num_hidden_layers\":40,\"num_attention_heads\":32,"
        "\"num_key_value_heads\":2,\"head_dim\":128}";
    CHECK(kv_mp_from_hf_config_json(gqa2, &p) == KV_MP_OK);
    CHECK(p.kind == KV_ATTN_GQA);
    CHECK(kv_mp_bytes_per_token_layer(&p) == 2u * 2u * 128u * 2u);

    /* missing heads -> invalid */
    kv_model_profile bad;
    CHECK(kv_mp_from_hf_config_json("{\"num_hidden_layers\":4}",
                                    &bad) != KV_MP_OK);
}

int main(void) {
    test_capacity();
    test_swa_bound();
    test_ssm_zero();
    test_tuning();
    test_names();
    test_hf_config_parse();
    if (failures == 0) {
        printf("test_model_profile: OK\n");
        return 0;
    }
    printf("test_model_profile: %d failures\n", failures);
    return 1;
}
