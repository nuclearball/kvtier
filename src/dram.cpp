#include "dram.hpp"
#include "hash.hpp"

#include <sys/mman.h>
#include <unistd.h>

namespace sc {

#define KV_DRAM_HUGEPAGE_SZ (2ull * 1024 * 1024)

static const uint32_t class_cap[KV_DRAM_N_CLASSES] = {
    4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 8 * 1024 * 1024,
};

#ifdef __linux__
static bool probe_hugetlb() {
    void *p = mmap(nullptr, KV_DRAM_HUGEPAGE_SZ, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p == MAP_FAILED)
        return false;
    munmap(p, KV_DRAM_HUGEPAGE_SZ);
    return true;
}
#endif

static uint8_t arena_map(dram_cache *d, uint8_t backend_req) {
#ifdef __linux__
    uint8_t order[3];
    int n = 0;
    if (backend_req == KV_DRAM_BACKEND_AUTO) {
        order[n++] = KV_DRAM_BACKEND_HUGETLB;
        order[n++] = KV_DRAM_BACKEND_THP;
    } else {
        order[n++] = backend_req;
    }
    for (int i = 0; i < n; i++) {
        if (order[i] == KV_DRAM_BACKEND_HUGETLB) {
            if (!probe_hugetlb())
                continue;
            d->mmap_len = align_up(d->budget, KV_DRAM_HUGEPAGE_SZ);
            d->mmap_base = mmap(nullptr, d->mmap_len, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (d->mmap_base == MAP_FAILED)
                continue;
            d->arena = d->mmap_base;
            return KV_DRAM_BACKEND_HUGETLB;
        }
        if (order[i] == KV_DRAM_BACKEND_THP) {
            d->mmap_len = align_up(d->budget, 4096) + KV_DRAM_HUGEPAGE_SZ;
            d->mmap_base = mmap(nullptr, d->mmap_len, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (d->mmap_base == MAP_FAILED)
                continue;
            d->arena = reinterpret_cast<void *>(
                align_up(reinterpret_cast<uint64_t>(d->mmap_base),
                         KV_DRAM_HUGEPAGE_SZ));
            uint64_t adv_len =
                d->mmap_len -
                (reinterpret_cast<uint64_t>(d->arena) -
                 reinterpret_cast<uint64_t>(d->mmap_base));
            if (madvise(d->arena, adv_len, MADV_HUGEPAGE) != 0) {
                munmap(d->mmap_base, d->mmap_len);
                d->mmap_base = nullptr;
                continue;
            }
            return KV_DRAM_BACKEND_THP;
        }
    }
#else
    (void)backend_req;
#endif
    d->mmap_len = align_up(d->budget, static_cast<uint64_t>(sysconf(_SC_PAGESIZE)));
    d->mmap_base = mmap(nullptr, d->mmap_len, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (d->mmap_base == MAP_FAILED)
        return 0;
    d->arena = d->mmap_base;
    return KV_DRAM_BACKEND_PLAIN;
}

static uint32_t entry_total(uint16_t n_records, uint32_t buf_len) {
    return static_cast<uint32_t>(sizeof(dram_entry) +
                                 static_cast<size_t>(n_records) *
                                     sizeof(dram_rec) +
                                 buf_len);
}

static uint8_t pick_class(uint32_t total) {
    for (uint8_t c = 0; c < KV_DRAM_N_CLASSES; c++)
        if (total <= class_cap[c])
            return c;
    return KV_DRAM_N_CLASSES;
}

static void *class_alloc(dram_cache *d, uint8_t cls, uint32_t total) {
    if (cls < KV_DRAM_N_CLASSES) {
        void *p = d->freelists[cls];
        if (p) {
            d->freelists[cls] = *static_cast<void **>(p);
            return p;
        }
        if (d->bump + class_cap[cls] <=
            d->mmap_len -
                (reinterpret_cast<uint64_t>(d->arena) -
                 reinterpret_cast<uint64_t>(d->mmap_base))) {
            void *q = static_cast<char *>(d->arena) + d->bump;
            d->bump += class_cap[cls];
            return q;
        }
        return malloc(class_cap[cls]);
    }
    return malloc(total);
}

static void class_free(dram_cache *d, void *p, uint8_t cls) {
    (void)d;
    if (cls < KV_DRAM_N_CLASSES) {
        *static_cast<void **>(p) = d->freelists[cls];
        d->freelists[cls] = p;
    } else {
        free(p);
    }
}

static void queue_unlink(dram_shard *sh, dram_entry *e) {
    if (e->sprev)
        e->sprev->snext = e->snext;
    else
        sh->s_head = e->snext;
    if (e->snext)
        e->snext->sprev = e->sprev;
    else
        sh->s_tail = e->sprev;
    if (sh->s_hand == e)
        sh->s_hand = e->snext;
    e->sprev = e->snext = nullptr;
}

static void queue_push_tail(dram_shard *sh, dram_entry *e) {
    e->snext = nullptr;
    e->sprev = sh->s_tail;
    if (sh->s_tail)
        sh->s_tail->snext = e;
    else
        sh->s_head = e;
    sh->s_tail = e;
}

static void entry_free(dram_cache *d, dram_shard *sh, dram_entry *e) {
    uint32_t total = entry_total(e->n_records, e->buf_len);
    sh->bytes -= total;
    d->bytes -= total;
    d->entries--;
    if (e->external)
        free(e);
    else
        class_free(d, e, e->size_class);
}

static void sieve_evict(dram_cache *d, dram_shard *sh, dram_entry *protect) {
    int guard = 0;
    while (sh->bytes > sh->budget && sh->s_head && guard++ < 4096) {
        dram_entry *e = sh->s_hand;
        if (!e)
            e = sh->s_head;
        if (e == protect) {
            sh->s_hand = e->snext;
            continue;
        }
        if (e->visited) {
            e->visited = 0;
            sh->s_hand = e->snext;
            continue;
        }
        uint32_t b = static_cast<uint32_t>(
                         hash_mix64(e->prefix_id ^
                                    (static_cast<uint64_t>(e->group_idx) << 32))) &
                     (KV_DRAM_SHARD_BUCKETS - 1);
        dram_entry **pp = &sh->buckets[b];
        while (*pp && *pp != e)
            pp = &(*pp)->hnext;
        if (*pp)
            *pp = e->hnext;
        dram_entry *next = e->snext;
        queue_unlink(sh, e);
        sh->s_hand = next;
        d->evictions.fetch_add(1);
        entry_free(d, sh, e);
    }
    if (!sh->s_head)
        sh->s_hand = nullptr;
}

int dram_init(dram_cache **out, uint64_t budget, uint64_t entry_max,
              uint8_t backend_req) {
    *out = nullptr;
    if (!budget)
        return KV_EOK;
    if (backend_req > KV_DRAM_BACKEND_MAX)
        return -KV_EINVAL;
    auto *d = new dram_cache();
    if (!d)
        return -KV_ENOMEM;
    d->budget = budget;
    d->entry_max =
        entry_max ? min_of(entry_max, budget) : KV_DRAM_DEFAULT_ENTRY_MAX;
    d->n_shards = KV_DRAM_N_SHARDS;
    while (d->n_shards > 1 && budget / static_cast<uint64_t>(d->n_shards) < 1024 * 1024)
        d->n_shards >>= 1;
    d->shard_budget = budget / static_cast<uint64_t>(d->n_shards);
    for (int i = 0; i < d->n_shards; i++) {
        d->shards[i].budget = d->shard_budget;
    }

    d->backend = arena_map(d, backend_req);
    if (!d->backend) {
        free(d);
        return -KV_ENOMEM;
    }
    *out = d;
    return KV_EOK;
}

void dram_destroy(dram_cache *d) {
    if (!d)
        return;
    for (int i = 0; i < d->n_shards; i++) {
        dram_shard *sh = &d->shards[i];
        std::lock_guard<std::mutex> lk(sh->mtx);
        for (uint32_t b = 0; b < KV_DRAM_SHARD_BUCKETS; b++) {
            dram_entry *e = sh->buckets[b];
            while (e) {
                dram_entry *n = e->hnext;
                if (e->external)
                    free(e);
                e = n;
            }
        }
    }
    if (d->mmap_base)
        munmap(d->mmap_base, d->mmap_len);
    delete d;
}

static dram_shard *shard_of(dram_cache *d, uint64_t prefix_id,
                            uint32_t group_idx) {
    uint64_t h = hash_mix64(prefix_id ^ (static_cast<uint64_t>(group_idx) << 32));
    return &d->shards[h % static_cast<uint32_t>(d->n_shards)];
}

static uint32_t bucket_of(uint64_t prefix_id, uint32_t group_idx) {
    return static_cast<uint32_t>(
               hash_mix64(prefix_id ^ (static_cast<uint64_t>(group_idx) << 32))) &
           (KV_DRAM_SHARD_BUCKETS - 1);
}

static dram_entry *entry_find(dram_shard *sh, uint32_t b, uint64_t prefix_id,
                              uint32_t group_idx) {
    for (dram_entry *e = sh->buckets[b]; e; e = e->hnext)
        if (e->prefix_id == prefix_id && e->group_idx == group_idx)
            return e;
    return nullptr;
}

int dram_get(dram_cache *d, uint64_t prefix_id, uint32_t group_idx,
             uint32_t expect_ver, uint8_t **buf, uint32_t *buf_len,
             dram_rec *recs, uint16_t rec_cap, uint16_t *n_recs) {
    dram_shard *sh = shard_of(d, prefix_id, group_idx);
    uint32_t b = bucket_of(prefix_id, group_idx);
    std::lock_guard<std::mutex> lk(sh->mtx);
    dram_entry *e = entry_find(sh, b, prefix_id, group_idx);
    if (!e || e->ver != expect_ver ||
        (e->expire_ts && e->expire_ts < now_s())) {
        d->misses.fetch_add(1);
        return -KV_ENOENT;
    }
    uint8_t *out = static_cast<uint8_t *>(aligned_alloc(e->buf_len));
    if (!out)
        return -KV_ENOMEM;
    std::memcpy(out, dram_entry_buf(e), e->buf_len);
    uint16_t n = e->n_records < rec_cap ? e->n_records : rec_cap;
    const dram_rec *er = dram_entry_recs(e);
    for (uint16_t i = 0; i < n; i++)
        recs[i] = er[i];
    *n_recs = n;
    *buf = out;
    *buf_len = e->buf_len;
    e->visited = 1;
    d->hits.fetch_add(1);
    return KV_EOK;
}

int dram_put(dram_cache *d, uint64_t prefix_id, uint32_t group_idx, uint32_t ver,
             uint32_t expire_ts, const uint8_t *buf, uint32_t buf_len,
             const dram_rec *recs, uint16_t n_recs) {
    uint32_t total = entry_total(n_recs, buf_len);
    if (!buf_len || total > d->entry_max || total > d->budget)
        return KV_EFULL;

    dram_shard *sh = shard_of(d, prefix_id, group_idx);
    uint32_t b = bucket_of(prefix_id, group_idx);
    std::lock_guard<std::mutex> lk(sh->mtx);

    dram_entry *old = entry_find(sh, b, prefix_id, group_idx);
    if (old && old->ver >= ver)
        return KV_EOK;
    if (old) {
        uint32_t ob = bucket_of(old->prefix_id, old->group_idx);
        dram_entry **pp = &sh->buckets[ob];
        while (*pp && *pp != old)
            pp = &(*pp)->hnext;
        if (*pp)
            *pp = old->hnext;
        queue_unlink(sh, old);
        entry_free(d, sh, old);
    }

    uint8_t cls = pick_class(total);
    dram_entry *e = static_cast<dram_entry *>(class_alloc(d, cls, total));
    if (!e)
        return KV_EFULL;
    std::memset(e, 0, sizeof(*e));
    e->prefix_id = prefix_id;
    e->group_idx = group_idx;
    e->ver = ver;
    e->expire_ts = expire_ts;
    e->buf_len = buf_len;
    e->n_records = n_recs;
    e->external = (cls >= KV_DRAM_N_CLASSES);
    e->size_class = cls;
    dram_rec *er = dram_entry_recs(e);
    for (uint16_t i = 0; i < n_recs; i++)
        er[i] = recs[i];
    std::memcpy(dram_entry_buf(e), buf, buf_len);

    e->hnext = sh->buckets[b];
    sh->buckets[b] = e;
    queue_push_tail(sh, e);
    sh->bytes += total;
    d->bytes += total;
    d->entries++;

    sieve_evict(d, sh, e);
    return KV_EOK;
}

void dram_invalidate(dram_cache *d, uint64_t prefix_id, uint32_t group_idx) {
    dram_shard *sh = shard_of(d, prefix_id, group_idx);
    uint32_t b = bucket_of(prefix_id, group_idx);
    std::lock_guard<std::mutex> lk(sh->mtx);
    dram_entry *e = entry_find(sh, b, prefix_id, group_idx);
    if (e) {
        dram_entry **pp = &sh->buckets[b];
        while (*pp && *pp != e)
            pp = &(*pp)->hnext;
        if (*pp)
            *pp = e->hnext;
        queue_unlink(sh, e);
        entry_free(d, sh, e);
    }
}

uint64_t dram_stat(dram_cache *d, const char *key) {
    if (!d)
        return 0;
    if (!std::strcmp(key, "dram_hits"))
        return d->hits.load();
    if (!std::strcmp(key, "dram_misses"))
        return d->misses.load();
    if (!std::strcmp(key, "dram_evictions"))
        return d->evictions.load();
    if (!std::strcmp(key, "dram_entries"))
        return d->entries;
    if (!std::strcmp(key, "dram_bytes"))
        return d->bytes;
    if (!std::strcmp(key, "dram_backend"))
        return d->backend;
    return 0;
}

} // namespace sc
