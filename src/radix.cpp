#include "radix.hpp"
#include "hash.hpp"

#include <cstdlib>

namespace sc {

static radix_node *node_new(uint32_t depth) {
    auto *n = static_cast<radix_node *>(std::calloc(1, sizeof(radix_node)));
    if (!n)
        return nullptr;
    n->depth = depth;
    return n;
}

void radix_init(radix_tree *t) {
    t->root = nullptr;
    t->n_leaves.store(0);
    t->root = node_new(0);
    if (!t->root)
        fatal("radix_init: OOM");
    t->n_leaves.store(0);
}

static radix_node *edge_find(radix_node *n, uint64_t h) {
    for (int i = 0; i < KV_INLINE_SLOTS; i++) {
        if (n->slots[i].child_hash == h &&
            !(n->slots[i].flags & KV_EDGE_F_DELETED))
            return n->slots[i].child;
    }
    return nullptr;
}

static radix_node *edge_find_ovf(radix_node *n, uint64_t h) {
    if (!n->ovf)
        return nullptr;
    uint32_t mask = n->ovf->cap - 1;
    for (uint32_t i = hash_mix64(h) & mask;; i = (i + 1) & mask) {
        radix_edge *e = &n->ovf->slots[i];
        if (e->child_hash == 0)
            return nullptr;
        if (e->child_hash == h && !(e->flags & KV_EDGE_F_DELETED))
            return e->child;
    }
}

static radix_node *edge_get(radix_node *n, uint64_t h) {
    if (n->ovf_flag)
        return edge_find_ovf(n, h);
    return edge_find(n, h);
}

static int ovf_grow(radix_node *n) {
    radix_ovf *o = n->ovf;
    uint32_t new_cap = o->cap ? o->cap * 2 : 16;
    auto *slots =
        static_cast<radix_edge *>(std::calloc(new_cap, sizeof(radix_edge)));
    if (!slots)
        return -KV_ENOMEM;
    uint32_t mask = new_cap - 1;
    for (uint32_t i = 0; i < o->cap; i++) {
        radix_edge *e = &o->slots[i];
        if (e->child_hash == 0 || (e->flags & KV_EDGE_F_DELETED))
            continue;
        for (uint32_t j = hash_mix64(e->child_hash) & mask;; j = (j + 1) & mask) {
            if (slots[j].child_hash == 0) {
                slots[j] = *e;
                break;
            }
        }
    }
    std::free(o->slots);
    o->slots = slots;
    o->cap = new_cap;
    o->n_del = 0;
    return KV_EOK;
}

static int ovf_migrate(radix_node *n) {
    auto *o = static_cast<radix_ovf *>(std::calloc(1, sizeof(radix_ovf)));
    if (!o)
        return -KV_ENOMEM;
    n->ovf = o;
    if (ovf_grow(n) != KV_EOK)
        return -KV_ENOMEM;
    for (int i = 0; i < KV_INLINE_SLOTS; i++) {
        radix_edge *e = &n->slots[i];
        if (e->child_hash == 0 || (e->flags & KV_EDGE_F_DELETED))
            continue;
        uint32_t mask = o->cap - 1;
        for (uint32_t j = hash_mix64(e->child_hash) & mask;; j = (j + 1) & mask) {
            if (o->slots[j].child_hash == 0) {
                o->slots[j] = *e;
                o->used++;
                break;
            }
        }
    }
    std::memset(n->slots, 0, sizeof(n->slots));
    n->ovf_flag = 1;
    return KV_EOK;
}

static int edge_add(radix_node *n, uint64_t h, radix_node *child) {
    if (!n->ovf_flag) {
        for (int i = 0; i < KV_INLINE_SLOTS; i++) {
            if (n->slots[i].child_hash == 0) {
                n->slots[i].child_hash = h;
                n->slots[i].child = child;
                n->slots[i].flags = 0;
                n->n_edges++;
                return KV_EOK;
            }
        }
        if (ovf_migrate(n) != KV_EOK)
            return -KV_ENOMEM;
    }
    radix_ovf *o = n->ovf;
    if ((o->used + o->n_del + 1) > o->cap * 7 / 10)
        ovf_grow(n);
    uint32_t mask = o->cap - 1;
    for (uint32_t i = hash_mix64(h) & mask;; i = (i + 1) & mask) {
        radix_edge *e = &o->slots[i];
        if (e->child_hash == 0 || (e->flags & KV_EDGE_F_DELETED)) {
            if (e->child_hash == 0)
                o->used++;
            else
                o->n_del--;
            e->child_hash = h;
            e->child = child;
            e->flags = 0;
            n->n_edges++;
            return KV_EOK;
        }
    }
}

radix_node *radix_walk(radix_tree *t, const uint64_t *hashes, int n,
                       int create) {
    radix_node *cur = t->root;
    for (int k = 0; k < n; k++) {
        radix_node *child = edge_get(cur, hashes[k]);
        if (!child) {
            if (!create)
                return nullptr;
            child = node_new(static_cast<uint32_t>(k + 1));
            if (!child)
                return nullptr;
            child->parent = cur;
            if (edge_add(cur, hashes[k], child) != KV_EOK) {
                std::free(child);
                return nullptr;
            }
        }
        cur = child;
    }
    return cur;
}

void radix_hit_ascend(radix_node *node, unsigned inc) {
    radix_node *cur = node;
    while (cur) {
        cur->hit_count.fetch_add(inc, std::memory_order_relaxed);
        cur = cur->parent;
    }
}

void radix_hit_decay(radix_node *node) {
    if (!node)
        return;
    node->hit_count.store(node->hit_count.load() >> 1);
    for (int i = 0; i < KV_INLINE_SLOTS; i++) {
        radix_edge *e = &node->slots[i];
        if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED))
            radix_hit_decay(e->child);
    }
    if (node->ovf) {
        for (uint32_t i = 0; i < node->ovf->cap; i++) {
            radix_edge *e = &node->ovf->slots[i];
            if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED))
                radix_hit_decay(e->child);
        }
    }
}

int radix_visit(radix_tree *t, radix_visit_fn fn, void *arg) {
    int rc;
    uint32_t cap = 256;
    radix_node **stk =
        static_cast<radix_node **>(std::malloc(cap * sizeof(radix_node *)));
    if (!stk)
        return -KV_ENOMEM;
    uint32_t top = 0;
    stk[top++] = t->root;
    while (top) {
        radix_node *cur = stk[--top];
        if ((rc = fn(cur, arg)) != 0) {
            std::free(stk);
            return rc;
        }
        uint32_t need = static_cast<uint32_t>(KV_INLINE_SLOTS) +
                        (cur->ovf ? cur->ovf->cap : 0);
        if (top + need > cap) {
            while (top + need > cap)
                cap *= 2;
            auto **ns = static_cast<radix_node **>(
                std::realloc(stk, cap * sizeof(radix_node *)));
            if (!ns) {
                std::free(stk);
                return -KV_ENOMEM;
            }
            stk = ns;
        }
        for (int i = 0; i < KV_INLINE_SLOTS; i++) {
            radix_edge *e = &cur->slots[i];
            if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED))
                stk[top++] = e->child;
        }
        if (cur->ovf) {
            for (uint32_t i = 0; i < cur->ovf->cap; i++) {
                radix_edge *e = &cur->ovf->slots[i];
                if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED))
                    stk[top++] = e->child;
            }
        }
    }
    std::free(stk);
    return 0;
}

struct leaf_visitor {
    radix_leaf_fn fn;
    void *arg;
};

static int leaf_visit_cb(radix_node *node, void *arg) {
    auto *lv = static_cast<leaf_visitor *>(arg);
    if (node->leaf)
        return lv->fn(node->leaf, lv->arg);
    return 0;
}

int radix_visit_leaves(radix_tree *t, radix_leaf_fn fn, void *arg) {
    leaf_visitor lv{fn, arg};
    return radix_visit(t, leaf_visit_cb, &lv);
}

static void node_free_rec(radix_node *n) {
    for (int i = 0; i < KV_INLINE_SLOTS; i++) {
        radix_edge *e = &n->slots[i];
        if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED))
            node_free_rec(e->child);
    }
    if (n->ovf) {
        for (uint32_t i = 0; i < n->ovf->cap; i++) {
            radix_edge *e = &n->ovf->slots[i];
            if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED))
                node_free_rec(e->child);
        }
        std::free(n->ovf->slots);
        std::free(n->ovf);
    }
    if (n->leaf) {
        std::free(n->leaf->stripe);
        std::free(n->leaf);
    }
    std::free(n);
}

void radix_destroy(radix_tree *t) {
    if (t->root)
        node_free_rec(t->root);
    t->root = nullptr;
}

} // namespace sc
