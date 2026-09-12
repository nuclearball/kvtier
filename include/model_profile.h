#ifndef KV_MODEL_PROFILE_H
#define KV_MODEL_PROFILE_H

/* ------------------------------------------------------------------ */
/* model_profile: model-aware KV layout tuning (independent module)    */
/*                                                                     */
/* 上层（adapter / 应用）传入具体模型的注意力画像，本模块负责：          */
/*   1) 计算每 token / 每 group 的 KV 字节数（MHA/MQA/GQA/MLA/SSM…）    */
/*   2) 推导写路径对齐参数（stripe_unit / batch_min / region 对齐等）    */
/*                                                                     */
/* 本模块只依赖 stdint，不依赖 cache_t / device，可被任意适配层使用。    */
/* 用法：填 profile -> kv_mp_tune() -> 把结果写进 kv_config。            */
/* ------------------------------------------------------------------ */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KV_MP_NAME_MAX 32

enum kv_attn_kind {
    KV_ATTN_MHA = 0,   /* 每头独立 K/V                                  */
    KV_ATTN_MQA,       /* 所有头共享 1 份 K/V                           */
    KV_ATTN_GQA,       /* 每组 Q 头共享 K/V                             */
    KV_ATTN_MLA,       /* 低秩 latent + 解耦 RoPE（与头数无关）          */
    KV_ATTN_SWA,       /* GQA/MHA + 滑动窗口（有界缓存）                 */
    KV_ATTN_SSM,       /* 线性/SSM：无 per-token KV                     */
    KV_ATTN_HYBRID,    /* 部分层 attention + 部分层 SSM                 */
    KV_ATTN_SPARSE,    /* 全量 KV，但按 top-k 稀疏读（NSA/DSA）          */
};

struct kv_model_profile {
    char     name[KV_MP_NAME_MAX];
    enum kv_attn_kind kind;

    uint16_t n_layers;        /* 总层数                                      */
    uint16_t n_q_heads;       /* query 头数                                  */
    uint16_t n_kv_heads;      /* MHA/MQA/GQA/SWA：KV 头数（MQA=1）           */
    uint16_t head_dim;
    uint16_t dtype_bytes;     /* fp16/bf16=2, fp8=1                          */

    uint32_t latent_dim;      /* MLA: kv_lora_rank (d_c)                     */
    uint32_t rope_dim;        /* MLA: qk_rope_head_dim (d_rope)              */

    uint32_t window;          /* SWA: 保留 token 数；0 = 全量                 */
    uint16_t attn_layers;     /* HYBRID: attention 层数；0 => n_layers        */
    uint16_t tokens_per_group;/* KV 分组 token 数（默认 32）                  */
};

/* ---- 容量计算（字节） ---- */

/* 每 token 每层的 KV 字节数（已含 K 和 V） */
uint32_t kv_mp_bytes_per_token_layer(const struct kv_model_profile *p);

/* attention 层数（HYBRID 取 attn_layers，其余 n_layers） */
uint32_t kv_mp_kv_layers(const struct kv_model_profile *p);

/* 每 token 全模型 KV 字节数（= 每 token/层 × attention 层数）；
 * SSM 层不计入 per-token cache。 */
uint64_t kv_mp_bytes_per_token(const struct kv_model_profile *p);

/* 一个 group（tokens_in_group 个 token）每层的 payload 字节数 */
uint64_t kv_mp_group_layer_bytes(const struct kv_model_profile *p,
                                 uint32_t tokens_in_group);

/* 一个 group 的全模型 payload 字节数 */
uint64_t kv_mp_group_bytes(const struct kv_model_profile *p,
                           uint32_t tokens_in_group);

/* 给定上下文长度，全模型 KV 总量（SSM 恒为 0 增长） */
uint64_t kv_mp_context_bytes(const struct kv_model_profile *p, uint64_t tokens);

/* ---- 写路径推导 ---- */

struct kv_model_tuning {
    uint32_t page_size;          /* 输入：逻辑页大小（通常 4096）          */
    uint32_t alignment;          /* 该模型建议的数据对齐（2 的幂，<=page）  */
    uint64_t group_bytes;        /* 满 group payload                       */
    uint64_t stripe_unit;        /* 条带单元：page 与 layer 的最小公倍数的整数倍 */
    uint64_t stripe_threshold;   /* 触发条带的阈值（group 对齐，尽量）      */
    uint64_t batch_min_bytes;    /* 攒批下限（page 对齐且不超过阈值）       */
    uint64_t region_align_bytes; /* region 容量建议按此对齐                */
    uint64_t dram_entry_max_bytes; /* DRAM 缓存单条上限（>= 单 group）     */
};

/* 由 profile 推导写路径参数。
 *   page_size        逻辑页（通常 KV_PAGE_SIZE=4096）
 *   region_size_bytes 当前 region 容量（0 = 忽略）
 *   target_stripe     stripe_unit 的目标量级（0 => 256KiB）
 *   target_batch      batch_min 的目标量级（0 => 256KiB）
 * 返回 KV_MP_OK(=0) 或负错误码。 */
#define KV_MP_OK        0
#define KV_MP_EINVAL   -1
int kv_mp_tune(const struct kv_model_profile *p, uint32_t page_size,
               uint64_t region_size_bytes, uint64_t target_stripe,
               uint64_t target_batch, struct kv_model_tuning *out);

/* Values to copy into a core `struct kv_config` (model-agnostic core: it just
 * consumes the numbers).  This module never includes core headers. */
struct kv_mp_config_values {
    uint32_t n_groups_tokens;
    uint32_t max_layers;
    uint64_t stripe_unit;
    uint64_t stripe_threshold;
    uint64_t batch_min_bytes;
    uint64_t region_align_bytes;
    uint64_t dram_entry_max_bytes;
};
int kv_mp_config_values(const struct kv_model_profile *p, uint32_t page_size,
                        struct kv_mp_config_values *out);

/* ---- 内置预设（数值为公开 config 近似值，精确值以模型 config.json 为准） ---- */

enum kv_model_id {
    KV_MODEL_UNKNOWN = 0,
    KV_MODEL_LLAMA3_1_8B,
    KV_MODEL_LLAMA3_1_70B,
    KV_MODEL_LLAMA3_1_405B,
    KV_MODEL_QWEN2_5_1_5B,
    KV_MODEL_QWEN2_5_7B,
    KV_MODEL_QWEN2_5_72B,
    KV_MODEL_MISTRAL_7B,
    KV_MODEL_GEMMA2_9B,
    KV_MODEL_GLM4_9B,
    KV_MODEL_DEEPSEEK_V3,
    KV_MODEL_FALCON_7B,
    KV_MODEL__COUNT
};

const struct kv_model_profile *kv_mp_builtin(enum kv_model_id id);
const struct kv_model_profile *kv_mp_by_name(const char *name);

/* ---- optional: derive a profile from a model config ------------------ */

/* Parse a HuggingFace `config.json` text (no file I/O) into `out`.
 * Recognized keys: num_hidden_layers, num_attention_heads,
 * num_key_value_heads, head_dim (fallback hidden_size/heads), torch_dtype/
 * dtype, and MLA fields (kv_lora_rank / qk_rope_head_dim).  The first
 * occurrence of a key at any nesting depth is used (handles text_config).
 * Returns KV_MP_OK or a negative error.  Name comes from _name_or_path /
 * model_type, else "custom". */
int kv_mp_from_hf_config_json(const char *json, struct kv_model_profile *out);

/* Convenience wrapper: read the file then call the above. */
int kv_mp_from_hf_config_file(const char *path, struct kv_model_profile *out);

#ifdef __cplusplus
}
#endif

#endif /* KV_MODEL_PROFILE_H */
