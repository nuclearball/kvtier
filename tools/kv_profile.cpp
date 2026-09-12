/* kv-profile: standalone, adapter-side helper that maps a model profile to
 * generic `kv_config` layout values.  NOT linked into libkvtier.
 *
 * usage:
 *   kv-profile <model-name> [page_size]
 *   kv-profile --config <config.json> [page_size]   (HuggingFace config.json)
 *   kv-profile --list
 *
 * Output is shell/JSON-friendly "key=value" lines. */
#include "model_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int emit(const struct kv_model_profile *p, uint32_t page) {
    struct kv_mp_config_values v;
    if (kv_mp_config_values(p, page, &v) != KV_MP_OK) {
        fprintf(stderr, "tuning failed\n");
        return 1;
    }
    printf("model=%s\n", p->name);
    printf("page_size=%u\n", page);
    printf("bytes_per_token_layer=%u\n", kv_mp_bytes_per_token_layer(p));
    printf("bytes_per_token=%llu\n",
           (unsigned long long)kv_mp_bytes_per_token(p));
    printf("group_bytes=%llu\n",
           (unsigned long long)kv_mp_group_bytes(p, v.n_groups_tokens));
    printf("n_groups_tokens=%u\n", v.n_groups_tokens);
    printf("max_layers=%u\n", v.max_layers);
    printf("stripe_unit=%llu\n", (unsigned long long)v.stripe_unit);
    printf("stripe_threshold=%llu\n", (unsigned long long)v.stripe_threshold);
    printf("batch_min_bytes=%llu\n", (unsigned long long)v.batch_min_bytes);
    printf("region_align_bytes=%llu\n",
           (unsigned long long)v.region_align_bytes);
    printf("dram_entry_max_bytes=%llu\n",
           (unsigned long long)v.dram_entry_max_bytes);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--list") == 0) {
        printf("models:\n");
        for (int i = 1; i < KV_MODEL__COUNT; i++) {
            const struct kv_model_profile *p =
                kv_mp_builtin((enum kv_model_id)i);
            printf("  %s\n", p->name);
        }
        return argc < 2 ? 2 : 0;
    }

    uint32_t page = 4096;
    struct kv_model_profile parsed;

    if (strcmp(argv[1], "--config") == 0) {
        if (argc < 3) {
            fprintf(stderr, "usage: %s --config <config.json> [page_size]\n",
                    argv[0]);
            return 2;
        }
        if (argc > 3)
            page = (uint32_t)atoi(argv[3]);
        if (kv_mp_from_hf_config_file(argv[2], &parsed) != KV_MP_OK) {
            fprintf(stderr, "cannot parse config '%s'\n", argv[2]);
            return 2;
        }
        if (page == 0) page = 4096;
        return emit(&parsed, page);
    }

    const struct kv_model_profile *p = kv_mp_by_name(argv[1]);
    if (!p) {
        fprintf(stderr, "unknown model '%s' (try --list or --config)\n", argv[1]);
        return 2;
    }
    if (argc > 2)
        page = (uint32_t)atoi(argv[2]);
    if (page == 0) page = 4096;
    return emit(p, page);
}
