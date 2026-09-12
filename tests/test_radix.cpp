#include "common.hpp"
#include "radix.hpp"
#include "hash.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

struct ctr { int n; };

static void make_hashes(uint64_t *h, int n, uint64_t seed) {
    for (int i = 0; i < n; i++)
        h[i] = hash_mix64(seed + i);
}

static int leaf_ver(struct radix_node *node) {
    return node && node->leaf ? (int)node->leaf->ver.load() : -1;
}

static void test_basic(void) {
    struct radix_tree t;
    radix_init(&t);
    uint64_t h[3];
    make_hashes(h, 3, 100);

    struct radix_node *n1 = radix_walk(&t, h, 1, 1);
    CHECK(n1 != NULL);
    struct radix_leaf *lf = (struct radix_leaf *)calloc(1, sizeof(*lf));
    lf->ver.store(1);
    n1->leaf = lf;
    t.n_leaves.fetch_add(1);

    struct radix_node *g1 = radix_walk(&t, h, 1, 0);
    CHECK(g1 == n1);
    CHECK(leaf_ver(g1) == 1);

    /* miss on different hash */
    uint64_t h2[1] = { hash_mix64(999) };
    CHECK(radix_walk(&t, h2, 1, 0) == NULL);

    radix_destroy(&t);
}

static void test_deep(void) {
    struct radix_tree t;
    radix_init(&t);
    uint64_t h[3];
    make_hashes(h, 3, 7);

    /* 3-group string */
    struct radix_node *n = radix_walk(&t, h, 3, 1);
    CHECK(n != NULL);
    struct radix_leaf *lf = (struct radix_leaf *)calloc(1, sizeof(*lf));
    lf->ver.store(5);
    n->leaf = lf;
    t.n_leaves.fetch_add(1);

    /* prefix of length 2 must still be found as an internal node */
    struct radix_node *p = radix_walk(&t, h, 2, 0);
    CHECK(p != NULL && p->depth == 2);
    CHECK(radix_walk(&t, h, 3, 0) == n);
    radix_destroy(&t);
}

static void test_overflow(void) {
    struct radix_tree t;
    radix_init(&t);
    /* insert 64 leaves with distinct hashes (forces ovf migration) */
    for (int i = 0; i < 64; i++) {
        uint64_t h[1] = { hash_mix64(1000 + i) };
        struct radix_node *n = radix_walk(&t, h, 1, 1);
        struct radix_leaf *lf = (struct radix_leaf *)calloc(1, sizeof(*lf));
        lf->ver.store((unsigned)i + 1);
        n->leaf = lf;
        t.n_leaves.fetch_add(1);
    }
    for (int i = 0; i < 64; i++) {
        uint64_t h[1] = { hash_mix64(1000 + i) };
        struct radix_node *n = radix_walk(&t, h, 1, 0);
        CHECK(n && leaf_ver(n) == i + 1);
    }
    CHECK(t.n_leaves.load() == 64);
    radix_destroy(&t);
}

static int leaf_ctr(struct radix_leaf *lf, void *arg) {
    (void)lf;
    ((struct ctr *)arg)->n++;
    return 0;
}

static void test_visit_leaves(void) {
    struct radix_tree t;
    radix_init(&t);
    for (int i = 0; i < 10; i++) {
        uint64_t h[1] = { hash_mix64(2000 + i) };
        struct radix_node *n = radix_walk(&t, h, 1, 1);
        struct radix_leaf *lf = (struct radix_leaf *)calloc(1, sizeof(*lf));
        lf->ver.store((unsigned)i + 1);
        n->leaf = lf;
        t.n_leaves.fetch_add(1);
    }
    /* count via custom fn */
    struct ctr c = { 0 };
    int rc = radix_visit_leaves(&t, leaf_ctr, &c);
    CHECK(rc == 0 && c.n == 10);
    radix_destroy(&t);
}

/* hit ascend: the increment must reach the whole parent chain */
static void test_hit_ascend(void) {
    struct radix_tree t;
    radix_init(&t);
    uint64_t h[3];
    make_hashes(h, 3, 42);

    struct radix_node *n = radix_walk(&t, h, 3, 1);
    CHECK(n != NULL);
    radix_hit_ascend(n, 3);
    CHECK(n->hit_count.load() == 3);
    CHECK(t.root->hit_count.load() == 3);
    CHECK(t.root->slots[0].child->hit_count.load() == 3);
    CHECK(t.root->slots[0].child->slots[0].child->hit_count.load() == 3);
    radix_destroy(&t);
}

/* decay halves every node in the tree, ovf children included, until 0 */
static void test_hit_decay(void) {
    struct radix_tree t;
    radix_init(&t);
    /* 20 children under one depth-1 node: ovf migration + decay on ovf */
    uint64_t h0[1] = { hash_mix64(777) };
    struct radix_node *p = radix_walk(&t, h0, 1, 1);
    CHECK(p != NULL);
    for (int i = 0; i < 20; i++) {
        uint64_t hi[1] = { hash_mix64(7000 + i) };
        uint64_t path[2] = { h0[0], hi[0] };
        struct radix_node *c = radix_walk(&t, path, 2, 1);
        CHECK(c != NULL);
    }
    CHECK(p->ovf_flag == 1);            /* 20 > KV_INLINE_SLOTS */
    CHECK(p->n_edges == 20);

    radix_hit_ascend(p, 64);
    CHECK(p->hit_count.load() == 64);
    CHECK(t.root->hit_count.load() == 64);

    radix_hit_decay(t.root);
    CHECK(p->hit_count.load() == 32);
    CHECK(t.root->hit_count.load() == 32);
    /* ovf children were never hit; decay must not corrupt them (0>>1=0) */
    int dec = 0;
    for (uint32_t i = 0; i < p->ovf->cap; i++) {
        struct radix_edge *e = &p->ovf->slots[i];
        if (e->child_hash) {
            CHECK(e->child->hit_count.load() == 0);
            dec++;
        }
    }
    CHECK(dec == 20);
    radix_hit_decay(t.root);
    radix_hit_decay(t.root);
    radix_hit_decay(t.root);
    radix_hit_decay(t.root);
    CHECK(p->hit_count.load() == 2);   /* 64 >> 4 */
    radix_destroy(&t);
}

/* visit: early stop via non-zero fn return + empty tree (root only) */
struct stop_ctr { int n; int stop_at; };

static int visit_stop(struct radix_node *node, void *arg) {
    (void)node;
    struct stop_ctr *c = (struct stop_ctr *)arg;
    if (++c->n == c->stop_at) return 1;   /* arbitrary non-zero rc */
    return 0;
}

static int visit_count(struct radix_node *node, void *arg) {
    (void)node;
    (*(int *)arg)++;
    return 0;
}

static void test_visit_early_stop(void) {
    struct radix_tree t;
    radix_init(&t);
    for (int i = 0; i < 5; i++) {
        uint64_t h[1] = { hash_mix64(8000 + i) };
        radix_walk(&t, h, 1, 1);
    }
    struct stop_ctr sc = { 0, 3 };
    CHECK(radix_visit(&t, visit_stop, &sc) == 1);
    CHECK(sc.n == 3);

    /* empty tree: exactly one node (root) visited, rc 0 */
    struct radix_tree t2;
    radix_init(&t2);
    int n = 0;
    CHECK(radix_visit(&t2, visit_count, &n) == 0);
    CHECK(n == 1);
    radix_destroy(&t2);
    radix_destroy(&t);
}

/* wide tree: ovf at depth 2 with 300 siblings -> forces ovf_grow chain
 * (16 -> ... ) and the radix_visit worklist realloc (>256 siblings) */
struct wide_ctx { int leaves; int nodes; };

static int wide_leaf_cb(struct radix_leaf *lf, void *arg) {
    (void)lf;
    ((struct wide_ctx *)arg)->leaves++;
    return 0;
}

static int wide_node_cb(struct radix_node *node, void *arg) {
    (void)node;
    ((struct wide_ctx *)arg)->nodes++;
    return 0;
}

static void test_wide_deep_ovf(void) {
    struct radix_tree t;
    radix_init(&t);
    uint64_t h0[1] = { hash_mix64(888) };
    CHECK(radix_walk(&t, h0, 1, 1) != NULL);

    enum { N = 300 };
    for (int i = 0; i < N; i++) {
        uint64_t path[2] = { h0[0], hash_mix64(9000 + i) };
        struct radix_node *c = radix_walk(&t, path, 2, 1);
        CHECK(c != NULL);
        struct radix_leaf *lf = (struct radix_leaf *)calloc(1, sizeof(*lf));
        lf->ver.store((unsigned)i + 1);
        /* exercise destroy's stripe free path on some leaves */
        if (i % 7 == 0) {
            lf->stripe = (struct kv_stripe_addr *)calloc(
                1, sizeof(struct kv_stripe_addr) +
                       sizeof(struct kv_stripe_part));
            lf->stripe->n_parts = 1;
        }
        c->leaf = lf;
        t.n_leaves.fetch_add(1);
    }
    struct radix_node *p = radix_walk(&t, h0, 1, 0);
    CHECK(p->ovf_flag == 1);
    CHECK(p->n_edges == N);

    /* all lookups intact after migrate + grow */
    for (int i = 0; i < N; i++) {
        uint64_t path[2] = { h0[0], hash_mix64(9000 + i) };
        struct radix_node *c = radix_walk(&t, path, 2, 0);
        CHECK(c && leaf_ver(c) == i + 1);
    }
    /* n_leaves accounting */
    CHECK(t.n_leaves.load() == N);

    /* full traversal: 1 root + 1 depth-1 + N depth-2 nodes and N leaves;
     * the worklist must realloc while 300 siblings are stacked */
    struct wide_ctx wc = { 0, 0 };
    CHECK(radix_visit(&t, wide_node_cb, &wc) == 0);
    CHECK(radix_visit_leaves(&t, wide_leaf_cb, &wc) == 0);
    CHECK(wc.nodes == N + 2);
    CHECK(wc.leaves == N);
    radix_destroy(&t);
}

int main(void) {
    test_basic();
    test_deep();
    test_overflow();
    test_visit_leaves();
    test_hit_ascend();
    test_hit_decay();
    test_visit_early_stop();
    test_wide_deep_ovf();
    if (failures == 0) {
        printf("test_radix: OK\n");
        return 0;
    }
    printf("test_radix: %d failures\n", failures);
    return 1;
}
