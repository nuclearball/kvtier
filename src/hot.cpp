#include "hot.hpp"
#include "hash.hpp"

namespace sc {

bool hot_check_path(const radix_tree *t, const uint64_t *hashes, int depth,
                    uint32_t threshold) {
    radix_node *cur = t->root;
    radix_node *parent = nullptr;
    for (int k = 0; k < depth; k++) {
        radix_node *child = nullptr;
        if (!cur)
            return false;
        for (int i = 0; i < KV_INLINE_SLOTS; i++) {
            if (cur->slots[i].child_hash == hashes[k] &&
                !(cur->slots[i].flags & KV_EDGE_F_DELETED)) {
                child = cur->slots[i].child;
                break;
            }
        }
        if (!child && cur->ovf) {
            uint32_t mask = cur->ovf->cap - 1;
            for (uint32_t i = hash_mix64(hashes[k]) & mask;; i = (i + 1) & mask) {
                radix_edge *e = &cur->ovf->slots[i];
                if (e->child_hash == 0)
                    break;
                if (e->child_hash == hashes[k] &&
                    !(e->flags & KV_EDGE_F_DELETED)) {
                    child = e->child;
                    break;
                }
            }
        }
        if (!child)
            return false;
        parent = cur;
        cur = child;
    }
    if (cur && cur->hit_count.load() > threshold)
        return true;
    if (parent && parent->hit_count.load() > threshold)
        return true;
    return false;
}

void hot_decay(radix_tree *t) { radix_hit_decay(t->root); }

} // namespace sc
