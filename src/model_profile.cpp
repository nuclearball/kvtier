#include "model_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a ? a : 1;
}

static uint64_t lcm_u64(uint64_t a, uint64_t b) {
    if (!a || !b)
        return 0;
    return a / gcd_u64(a, b) * b;
}

static uint64_t mp_align_up(uint64_t v, uint64_t a) {
    if (!a)
        return v;
    return (v + a - 1) / a * a;
}

static uint64_t mp_align_down(uint64_t v, uint64_t a) {
    if (!a)
        return v;
    return v / a * a;
}

static uint32_t pow2_floor(uint64_t v) {
    uint32_t p = 1;
    while (static_cast<uint64_t>(p) * 2 <= v)
        p *= 2;
    return p;
}

uint32_t kv_mp_kv_layers(const struct kv_model_profile *p) {
    if (!p)
        return 0;
    if (p->kind == KV_ATTN_SSM)
        return 0;
    if (p->kind == KV_ATTN_HYBRID)
        return p->attn_layers ? p->attn_layers : p->n_layers;
    return p->n_layers;
}

uint32_t kv_mp_bytes_per_token_layer(const struct kv_model_profile *p) {
    if (!p)
        return 0;
    uint32_t d = p->dtype_bytes ? p->dtype_bytes : 2;

    switch (p->kind) {
    case KV_ATTN_SSM:
        return 0;
    case KV_ATTN_MLA:
        return static_cast<uint32_t>(
            (static_cast<uint64_t>(p->latent_dim) + p->rope_dim) * d);
    case KV_ATTN_MQA:
        return static_cast<uint32_t>(2ull * 1 * p->head_dim * d);
    case KV_ATTN_MHA:
    case KV_ATTN_GQA:
    case KV_ATTN_SWA:
    case KV_ATTN_HYBRID:
    default:
        if (p->kind == KV_ATTN_SPARSE && p->latent_dim)
            return static_cast<uint32_t>(
                (static_cast<uint64_t>(p->latent_dim) + p->rope_dim) * d);
        return static_cast<uint32_t>(
            static_cast<uint64_t>(2) * p->n_kv_heads * p->head_dim * d);
    }
}

uint64_t kv_mp_bytes_per_token(const struct kv_model_profile *p) {
    return static_cast<uint64_t>(kv_mp_bytes_per_token_layer(p)) *
           kv_mp_kv_layers(p);
}

uint64_t kv_mp_group_layer_bytes(const struct kv_model_profile *p,
                                 uint32_t tokens_in_group) {
    if (!p)
        return 0;
    if (tokens_in_group == 0)
        tokens_in_group = p->tokens_per_group ? p->tokens_per_group : 32;
    return static_cast<uint64_t>(kv_mp_bytes_per_token_layer(p)) *
           tokens_in_group;
}

uint64_t kv_mp_group_bytes(const struct kv_model_profile *p,
                           uint32_t tokens_in_group) {
    return kv_mp_group_layer_bytes(p, tokens_in_group) * kv_mp_kv_layers(p);
}

uint64_t kv_mp_context_bytes(const struct kv_model_profile *p,
                             uint64_t tokens) {
    if (!p || p->kind == KV_ATTN_SSM)
        return 0;
    uint64_t per_tok = kv_mp_bytes_per_token(p);
    if (p->kind == KV_ATTN_SWA && p->window && tokens > p->window)
        tokens = p->window;
    return per_tok * tokens;
}

int kv_mp_tune(const struct kv_model_profile *p, uint32_t page_size,
               uint64_t region_size_bytes, uint64_t target_stripe,
               uint64_t target_batch, struct kv_model_tuning *out) {
    if (!p || !out || page_size == 0)
        return KV_MP_EINVAL;

    uint32_t tpg = p->tokens_per_group ? p->tokens_per_group : 32;
    uint64_t layer_bytes = kv_mp_group_layer_bytes(p, tpg);
    uint64_t group_bytes = layer_bytes * kv_mp_kv_layers(p);
    if (group_bytes == 0)
        return KV_MP_EINVAL;

    if (target_stripe == 0)
        target_stripe = 256 * 1024;
    if (target_batch == 0)
        target_batch = 256 * 1024;

    std::memset(out, 0, sizeof(*out));
    out->page_size = page_size;
    out->group_bytes = group_bytes;

    uint64_t ga = gcd_u64(group_bytes, page_size);
    out->alignment = pow2_floor(ga);
    if (out->alignment == 0)
        out->alignment = 1;

    uint64_t unit_gran =
        lcm_u64(page_size, layer_bytes ? layer_bytes : page_size);
    if (unit_gran == 0)
        unit_gran = page_size;
    out->stripe_unit = mp_align_up(target_stripe, unit_gran);

    if (group_bytes > target_stripe) {
        out->stripe_threshold = mp_align_up(target_stripe, page_size);
    } else {
        out->stripe_threshold = mp_align_down(target_stripe, group_bytes);
        if (out->stripe_threshold == 0)
            out->stripe_threshold = group_bytes;
    }

    uint64_t want =
        mp_align_up(target_batch > group_bytes ? target_batch : group_bytes,
                    page_size);
    if (group_bytes <= out->stripe_threshold)
        want = mp_align_up(want, group_bytes);
    if (want > out->stripe_threshold)
        want = mp_align_down(out->stripe_threshold, page_size);
    if (want == 0)
        want = page_size;
    out->batch_min_bytes = want;

    out->region_align_bytes = group_bytes;
    if (region_size_bytes) {
        uint64_t aligned = mp_align_down(region_size_bytes, group_bytes);
        if (aligned)
            out->region_align_bytes = aligned;
    }

    out->dram_entry_max_bytes = mp_align_up(group_bytes, page_size);
    return KV_MP_OK;
}

int kv_mp_config_values(const struct kv_model_profile *p, uint32_t page_size,
                        struct kv_mp_config_values *out) {
    if (!p || !out)
        return KV_MP_EINVAL;
    struct kv_model_tuning t;
    int rc = kv_mp_tune(p, page_size, 0, 0, 0, &t);
    if (rc != KV_MP_OK)
        return rc;
    std::memset(out, 0, sizeof(*out));
    out->n_groups_tokens = p->tokens_per_group ? p->tokens_per_group : 32;
    out->max_layers = kv_mp_kv_layers(p);
    out->stripe_unit = t.stripe_unit;
    out->stripe_threshold = t.stripe_threshold;
    out->batch_min_bytes = t.batch_min_bytes;
    out->region_align_bytes = t.group_bytes;
    out->dram_entry_max_bytes = t.dram_entry_max_bytes;
    return KV_MP_OK;
}

#define P(name_, kind_, layers_, qh_, kvh_, hd_, dt_, lat_, rope_, win_, attn_) \
    { name_, kind_, layers_, qh_, kvh_, hd_, dt_, lat_, rope_, win_, attn_, 32 }

static const struct kv_model_profile presets[KV_MODEL__COUNT] = {
    { "unknown", KV_ATTN_MHA, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    P("llama-3.1-8b", KV_ATTN_GQA, 32, 32, 8, 128, 2, 0, 0, 0, 0),
    P("llama-3.1-70b", KV_ATTN_GQA, 80, 64, 8, 128, 2, 0, 0, 0, 0),
    P("llama-3.1-405b", KV_ATTN_GQA, 126, 128, 8, 128, 2, 0, 0, 0, 0),
    P("qwen2.5-1.5b", KV_ATTN_GQA, 28, 12, 2, 128, 2, 0, 0, 0, 0),
    P("qwen2.5-7b", KV_ATTN_GQA, 28, 28, 4, 128, 2, 0, 0, 0, 0),
    P("qwen2.5-72b", KV_ATTN_GQA, 80, 64, 8, 128, 2, 0, 0, 0, 0),
    P("mistral-7b", KV_ATTN_GQA, 32, 32, 8, 128, 2, 0, 0, 0, 0),
    P("gemma-2-9b", KV_ATTN_GQA, 42, 16, 8, 256, 2, 0, 0, 0, 0),
    P("glm-4-9b", KV_ATTN_GQA, 40, 32, 2, 128, 2, 0, 0, 0, 0),
    P("deepseek-v3", KV_ATTN_MLA, 61, 128, 0, 128, 2, 512, 64, 0, 0),
    P("falcon-7b", KV_ATTN_MQA, 32, 71, 1, 64, 2, 0, 0, 0, 0),
};

#undef P

const struct kv_model_profile *kv_mp_builtin(enum kv_model_id id) {
    if (id <= KV_MODEL_UNKNOWN || id >= KV_MODEL__COUNT)
        return nullptr;
    return &presets[id];
}

const struct kv_model_profile *kv_mp_by_name(const char *name) {
    if (!name)
        return nullptr;
    for (int i = 1; i < KV_MODEL__COUNT; i++)
        if (std::strncmp(presets[i].name, name, KV_MP_NAME_MAX) == 0)
            return &presets[i];
    return nullptr;
}

static const char *jfind(const char *j, const char *key) {
    size_t kl = std::strlen(key);
    if (!j || !kl)
        return nullptr;
    for (const char *p = j; (p = std::strchr(p, '"')) != nullptr; p++) {
        if (std::strncmp(p + 1, key, kl) == 0 && p[1 + kl] == '"') {
            const char *q = p + 1 + kl + 1;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')
                q++;
            if (*q != ':')
                continue;
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')
                q++;
            return q;
        }
    }
    return nullptr;
}

static int jnum(const char *j, const char *key, long long *out) {
    const char *v = jfind(j, key);
    if (!v)
        return -1;
    char *end = nullptr;
    long long x = std::strtoll(v, &end, 10);
    if (end == v)
        return -1;
    *out = x;
    return 0;
}

static int jstr(const char *j, const char *key, char *out, size_t cap) {
    const char *v = jfind(j, key);
    if (!v || *v != '"')
        return -1;
    v++;
    size_t i = 0;
    while (*v && *v != '"') {
        if (i + 1 < cap)
            out[i++] = *v;
        v++;
    }
    out[i] = 0;
    return 0;
}

static uint16_t dtype_bytes_from_str(const char *s) {
    if (!s || !*s)
        return 2;
    if (std::strstr(s, "8") || std::strstr(s, "fp8") ||
        std::strstr(s, "float8"))
        return 1;
    if (std::strstr(s, "32"))
        return 4;
    return 2;
}

int kv_mp_from_hf_config_json(const char *json, struct kv_model_profile *out) {
    if (!json || !out)
        return KV_MP_EINVAL;
    std::memset(out, 0, sizeof(*out));

    long long n_layers = 0, n_q = 0, n_kv = 0, head_dim = 0, hidden = 0;
    long long lora = 0, rope = 0, nope = 0;
    if (jnum(json, "num_hidden_layers", &n_layers) != 0 || n_layers <= 0)
        return KV_MP_EINVAL;
    if (jnum(json, "num_attention_heads", &n_q) != 0 || n_q <= 0)
        return KV_MP_EINVAL;
    (void)jnum(json, "num_key_value_heads", &n_kv);
    (void)jnum(json, "head_dim", &head_dim);
    (void)jnum(json, "hidden_size", &hidden);
    (void)jnum(json, "kv_lora_rank", &lora);
    (void)jnum(json, "qk_rope_head_dim", &rope);
    (void)jnum(json, "qk_nope_head_dim", &nope);
    if (nope > 0)
        head_dim = nope;

    if (n_kv <= 0)
        n_kv = n_q;
    if (head_dim <= 0 && hidden > 0)
        head_dim = hidden / n_q;
    if (head_dim <= 0)
        head_dim = 128;

    char dt[32] = {0};
    if (jstr(json, "torch_dtype", dt, sizeof(dt)) != 0 &&
        jstr(json, "dtype", dt, sizeof(dt)) != 0)
        dt[0] = 0;

    out->n_layers = static_cast<uint16_t>(n_layers);
    out->n_q_heads = static_cast<uint16_t>(n_q);
    out->n_kv_heads = static_cast<uint16_t>(n_kv);
    out->head_dim = static_cast<uint16_t>(head_dim);
    out->dtype_bytes = dtype_bytes_from_str(dt);
    out->latent_dim = static_cast<uint32_t>(lora);
    out->rope_dim = static_cast<uint32_t>(rope);
    out->tokens_per_group = 32;

    if (lora > 0)
        out->kind = KV_ATTN_MLA;
    else if (n_kv == 1)
        out->kind = KV_ATTN_MQA;
    else if (n_kv == n_q)
        out->kind = KV_ATTN_MHA;
    else
        out->kind = KV_ATTN_GQA;

    char name[KV_MP_NAME_MAX] = {0};
    if (jstr(json, "_name_or_path", name, sizeof(name)) != 0 &&
        jstr(json, "model_type", name, sizeof(name)) != 0)
        std::snprintf(name, sizeof(name), "custom");
    std::snprintf(out->name, sizeof(out->name), "%s", name);
    return KV_MP_OK;
}

int kv_mp_from_hf_config_file(const char *path, struct kv_model_profile *out) {
    if (!path || !out)
        return KV_MP_EINVAL;
    FILE *fp = std::fopen(path, "rb");
    if (!fp)
        return KV_MP_EINVAL;
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return KV_MP_EINVAL;
    }
    long sz = std::ftell(fp);
    if (sz <= 0 || sz > (1 << 20)) {
        std::fclose(fp);
        return KV_MP_EINVAL;
    }
    std::rewind(fp);
    char *buf = static_cast<char *>(std::malloc(static_cast<size_t>(sz) + 1));
    if (!buf) {
        std::fclose(fp);
        return KV_MP_EINVAL;
    }
    size_t rd = std::fread(buf, 1, static_cast<size_t>(sz), fp);
    std::fclose(fp);
    buf[rd] = 0;
    int rc = kv_mp_from_hf_config_json(buf, out);
    std::free(buf);
    return rc;
}
