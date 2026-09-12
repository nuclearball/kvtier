/* unit tests for the cuckoo filter (tombstone alternative) */
#undef NDEBUG
#include "cuckoo.hpp"
#include "hash.hpp"
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace sc;

int main(void) {
    kv_cuckoo *cf = NULL;
    assert(kv_cuckoo_init(&cf, 4096) == 0);
    assert(kv_cuckoo_capacity(cf) >= 4096);

    /* insert + lookup */
    for (uint64_t k = 1; k <= 3000; k++)
        assert(kv_cuckoo_insert(cf, k * 0x9e3779b97f4a7c15ull) == 0);
    assert(kv_cuckoo_count(cf) >= 3000 - 100);  /* fp collisions may merge */
    for (uint64_t k = 1; k <= 3000; k++)
        assert(kv_cuckoo_lookup(cf, k * 0x9e3779b97f4a7c15ull));

    /* delete removes membership */
    for (uint64_t k = 1; k <= 1500; k++)
        kv_cuckoo_delete(cf, k * 0x9e3779b97f4a7c15ull);
    assert(kv_cuckoo_count(cf) <= 1500 + 100);
    /* remaining half still found */
    for (uint64_t k = 1501; k <= 3000; k++)
        assert(kv_cuckoo_lookup(cf, k * 0x9e3779b97f4a7c15ull));

    /* false-positive rate on absent keys (8-bit fp, 2 buckets -> ~2.3%) */
    uint32_t fp_hits = 0;
    for (uint64_t k = 0; k < 100000; k++) {
        uint64_t key = hash_mix64(k + 0xdeadbeef);
        if (kv_cuckoo_lookup(cf, key)) fp_hits++;
    }
    double fpr = (double)fp_hits / 100000.0;
    printf("cuckoo: count=%u fpr=%.4f\n", kv_cuckoo_count(cf), fpr);
    assert(fpr < 0.10);

    kv_cuckoo_destroy(cf);

    /* full-insert failure path: tiny filter */
    kv_cuckoo *tiny = NULL;
    assert(kv_cuckoo_init(&tiny, 16) == 0);
    int full_seen = 0, ins_ok = 0;
    for (uint64_t k = 0; k < 4096; k++) {
        int rc = kv_cuckoo_insert(tiny, hash_mix64(k));
        if (rc == 0) ins_ok++; else full_seen++;
    }
    assert(ins_ok > 0 && full_seen > 0);
    kv_cuckoo_destroy(tiny);

    printf("test_cuckoo: OK\n");
    return 0;
}
