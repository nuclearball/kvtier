#include "common.hpp"
#include "journal.hpp"
#include "device.hpp"
#include "radix.hpp"
#include "hash.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

#define TEST_IMG "/tmp/kv_journal_test.img"

struct replay_state { int puts, dels, trims; };

static int replay_cb(const journal_rec *rec, uint32_t data_crc,
                     const uint32_t *lens, uint16_t n_lens, uint32_t data_len,
                     const uint64_t *path, void *arg) {
    replay_state *st = (replay_state *)arg;
    (void)data_crc; (void)lens; (void)n_lens; (void)data_len; (void)path;
    if (rec->op == KV_JOP_PUT) st->puts++;
    else if (rec->op == KV_JOP_DEL) st->dels++;
    else if (rec->op == KV_JOP_TRIM) st->trims++;
    return 0;
}

static void test_journal_roundtrip(void) {
    /* create a small device file */
    int fd = open(TEST_IMG, O_RDWR | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0);
    ftruncate(fd, 16 * 1024 * 1024);
    close(fd);

    auto dev = dev_make_file();
    CHECK(dev_open(dev.get(), TEST_IMG, 0) == KV_EOK);

    uint64_t jpage = 1000;
    journal j;
    CHECK(journal_init(&j, dev.get(), jpage, 64) == KV_EOK);
    CHECK(journal_reset(&j) == KV_EOK);

    kv_addr a = { .dev_id = 1, .page_no = 500, .page_off = 0,
                  .len_pages = 2, .last_write_ts = 0,
                  .expire_ts = 999, .crc = 0xDEAD };
    for (int i = 0; i < 100; i++) {
        uint32_t l[1] = { 4096 };
        CHECK(journal_put(&j, 1000 + i, 0, (uint32_t)i, &a, 999, 0, 0, 0,
                      l, 1, 4096, NULL, 0) == KV_EOK);
    }
    CHECK(journal_del(&j, 42, 1) == KV_EOK);
    CHECK(journal_trim(&j, 3, 77) == KV_EOK);
    journal_flush(&j);

    /* replay */
    replay_state st = { 0, 0, 0 };
    int rc = journal_replay(&j, replay_cb, &st);
    CHECK(rc == KV_EOK);
    CHECK(st.puts == 100);
    CHECK(st.dels == 1);
    CHECK(st.trims == 1);

    journal_destroy(&j);
    dev->close();
    unlink(TEST_IMG);
}

static void load_cb(radix_node *node, const ckpt_leaf_snapshot *s,
                    void *arg) {
    (void)node; (void)s;
    (*(int *)arg)++;
}

static void test_ckpt_roundtrip(void) {
    radix_tree t;
    radix_init(&t);

    /* build a small tree */
    for (int i = 0; i < 8; i++) {
        uint64_t h[2] = { hash_mix64(10 + i), hash_mix64(20 + i) };
        radix_node *n = radix_walk(&t, h, 2, 1);
        auto *lf = (radix_leaf *)calloc(1, sizeof(radix_leaf));
        lf->ver.store((unsigned)i + 1);
        lf->prefix_id = 100 + i;
        lf->addr[0].dev_id = i;
        lf->addr[0].page_no = 1000 + i;
        lf->addr[0].len_pages = 1;
        lf->addr[0].expire_ts = 500;
        n->leaf = lf;
        t.n_leaves.fetch_add(1);
    }

    CHECK(ckpt_write("/tmp/kv_ckpt_test.ckpt", &t, 12345, 2, 6, 8192, NULL)
          == KV_EOK);

    /* reload */
    radix_tree t2;
    radix_init(&t2);
    uint64_t ver = 0;
    int leaves = 0;
    CHECK(ckpt_load("/tmp/kv_ckpt_test.ckpt", &t2, &ver, 2, 6, 8192, NULL,
                    load_cb, &leaves) == KV_EOK);
    CHECK(ver == 12345u);
    CHECK(leaves == 8);
    CHECK(t2.n_leaves.load() == 8);

    /* verify leaves are findable via hash path */
    for (int i = 0; i < 8; i++) {
        uint64_t h[2] = { hash_mix64(10 + i), hash_mix64(20 + i) };
        radix_node *n = radix_walk(&t2, h, 2, 0);
        CHECK(n && n->leaf);
        CHECK(n->leaf->ver.load() == (unsigned)i + 1);
        CHECK(n->leaf->prefix_id == (uint64_t)(100 + i));
    }

    radix_destroy(&t);
    radix_destroy(&t2);
    unlink("/tmp/kv_ckpt_test.ckpt");
}

int main(void) {
    test_journal_roundtrip();
    test_ckpt_roundtrip();
    if (failures == 0) {
        printf("test_journal: OK\n");
        return 0;
    }
    printf("test_journal: %d failures\n", failures);
    return 1;
}
