#include "cuckoo.hpp"
#include "common.hpp"
#include "hash.hpp"

#include <cstdlib>
#include <mutex>
#include <new>

namespace sc {

inline constexpr int CK_BUCKET_SLOTS = 4;
inline constexpr int CK_FP_BITS = 8;
inline constexpr uint32_t CK_FP_MASK = (1u << CK_FP_BITS) - 1;
inline constexpr int CK_MAX_KICKS = 500;

struct ck_bucket {
    uint8_t fp[CK_BUCKET_SLOTS];
};

struct kv_cuckoo {
    std::mutex mtx;
    uint32_t n_buckets = 0;
    uint32_t count = 0;
    ck_bucket *buckets = nullptr;
};

static inline uint32_t fp_of(uint64_t key) {
    uint32_t fp =
        static_cast<uint32_t>(hash_mix64(key ^ 0x9e3779b97f4a7c15ull)) &
        CK_FP_MASK;
    return fp ? fp : 1;
}

static inline uint32_t alt_index(uint32_t i, uint32_t fp, uint32_t n_buckets) {
    return static_cast<uint32_t>((i ^ hash_mix64(fp)) & (n_buckets - 1));
}

int kv_cuckoo_init(kv_cuckoo **out, uint32_t n_entries) {
    if (!out || !n_entries)
        return -1;
    uint32_t nb = 16;
    while (nb < (n_entries / CK_BUCKET_SLOTS) * 100 / 85 + 1)
        nb <<= 1;
    auto *cf = new (std::nothrow) kv_cuckoo();
    if (!cf)
        return -1;
    cf->buckets = static_cast<ck_bucket *>(std::calloc(nb, sizeof(ck_bucket)));
    if (!cf->buckets) {
        delete cf;
        return -1;
    }
    cf->n_buckets = nb;
    *out = cf;
    return 0;
}

void kv_cuckoo_destroy(kv_cuckoo *cf) {
    if (!cf)
        return;
    std::free(cf->buckets);
    delete cf;
}

static bool bucket_has(const ck_bucket *b, uint32_t fp) {
    for (int i = 0; i < CK_BUCKET_SLOTS; i++)
        if (b->fp[i] == fp)
            return true;
    return false;
}

static int bucket_insert(ck_bucket *b, uint32_t fp) {
    for (int i = 0; i < CK_BUCKET_SLOTS; i++)
        if (!b->fp[i]) {
            b->fp[i] = static_cast<uint8_t>(fp);
            return 0;
        }
    return -1;
}

static bool bucket_remove(ck_bucket *b, uint32_t fp) {
    for (int i = 0; i < CK_BUCKET_SLOTS; i++)
        if (b->fp[i] == fp) {
            b->fp[i] = 0;
            return true;
        }
    return false;
}

int kv_cuckoo_insert(kv_cuckoo *cf, uint64_t key) {
    uint32_t fp = fp_of(key);
    uint32_t i1 = static_cast<uint32_t>(hash_mix64(key) & (cf->n_buckets - 1));
    uint32_t i2 = alt_index(i1, fp, cf->n_buckets);

    std::lock_guard<std::mutex> lk(cf->mtx);
    if (bucket_insert(&cf->buckets[i1], fp) == 0 ||
        bucket_insert(&cf->buckets[i2], fp) == 0) {
        cf->count++;
        return 0;
    }

    uint32_t cur = (now_ns() & 1) ? i1 : i2;
    for (int k = 0; k < CK_MAX_KICKS; k++) {
        ck_bucket *b = &cf->buckets[cur & (cf->n_buckets - 1)];
        int slot = static_cast<int>(hash_mix64(key + k) % CK_BUCKET_SLOTS);
        uint32_t victim = b->fp[slot];
        b->fp[slot] = static_cast<uint8_t>(fp);
        fp = victim;
        cur = alt_index(cur, fp, cf->n_buckets);
        if (bucket_insert(&cf->buckets[cur], fp) == 0) {
            cf->count++;
            return 0;
        }
    }
    return -1;
}

bool kv_cuckoo_lookup(const kv_cuckoo *cf, uint64_t key) {
    uint32_t fp = fp_of(key);
    uint32_t i1 = static_cast<uint32_t>(hash_mix64(key) & (cf->n_buckets - 1));
    uint32_t i2 = alt_index(i1, fp, cf->n_buckets);
    std::lock_guard<std::mutex> lk(const_cast<kv_cuckoo *>(cf)->mtx);
    return bucket_has(&cf->buckets[i1], fp) || bucket_has(&cf->buckets[i2], fp);
}

bool kv_cuckoo_delete(kv_cuckoo *cf, uint64_t key) {
    uint32_t fp = fp_of(key);
    uint32_t i1 = static_cast<uint32_t>(hash_mix64(key) & (cf->n_buckets - 1));
    uint32_t i2 = alt_index(i1, fp, cf->n_buckets);
    std::lock_guard<std::mutex> lk(cf->mtx);
    bool removed = bucket_remove(&cf->buckets[i1], fp) ||
                   bucket_remove(&cf->buckets[i2], fp);
    if (removed && cf->count)
        cf->count--;
    return removed;
}

uint32_t kv_cuckoo_count(const kv_cuckoo *cf) { return cf->count; }
uint32_t kv_cuckoo_capacity(const kv_cuckoo *cf) {
    return cf->n_buckets * CK_BUCKET_SLOTS;
}

} // namespace sc
