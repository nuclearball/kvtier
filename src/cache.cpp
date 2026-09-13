#include "cache.hpp"
#include "page.hpp"
#include "crc32.hpp"
#include "gc.hpp"
#include "hash.hpp"
#include "hot.hpp"
#ifdef KV_USE_CUCKOO_TOMB
#include "cuckoo.hpp"
#endif

#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sc {

static int cache_sync_read(cache *c, Device *dev, uint64_t page_no, void *buf,
                           uint32_t npages);
static uint32_t shape_total(const kv_shape *sh);

static void kv_layout_derive(kv_layout *L, const kv_config *cfg) {
    L->page_size = KV_PAGE_SIZE;
    L->alignment = KV_PAGE_SIZE;
    L->stripe_unit = cfg->stripe_unit ? cfg->stripe_unit : KV_STRIPE_UNIT;
    L->stripe_threshold =
        cfg->stripe_threshold ? cfg->stripe_threshold : KV_STRIPE_THRESHOLD;
    L->batch_min_bytes =
        cfg->batch_min_bytes ? cfg->batch_min_bytes : KV_BATCH_MIN_BYTES;
    L->dram_entry_max_bytes = cfg->dram_entry_max_bytes
                                  ? cfg->dram_entry_max_bytes
                                  : KV_DRAM_DEFAULT_ENTRY_MAX;
    L->region_align_bytes =
        cfg->region_align_bytes ? cfg->region_align_bytes : KV_PAGE_SIZE;
}

void side_init(side_tbl *t) {
    t->cap = KV_SIDE_INIT_CAP;
    t->count = 0;
    t->slots.assign(t->cap, side_slot{});
}

void side_destroy(side_tbl *t) { t->slots.clear(); }

static void side_grow(side_tbl *t) {
    uint32_t ncap = t->cap * 2;
    std::vector<side_slot> ns(ncap);
    uint32_t mask = ncap - 1;
    for (uint32_t i = 0; i < t->cap; i++) {
        if (t->slots[i].used) {
            uint64_t h = hash_mix64(
                t->slots[i].prefix_id ^
                (static_cast<uint64_t>(t->slots[i].group_idx) << 32));
            for (uint32_t j = static_cast<uint32_t>(h) & mask;; j = (j + 1) & mask) {
                if (!ns[j].used) {
                    ns[j] = t->slots[i];
                    break;
                }
            }
        }
    }
    t->slots.swap(ns);
    t->cap = ncap;
}

int side_put(side_tbl *t, uint64_t prefix_id, uint32_t group_idx,
             radix_node *node) {
    if (t->count * 10 >= t->cap * 7)
        side_grow(t);
    uint32_t mask = t->cap - 1;
    uint64_t h =
        hash_mix64(prefix_id ^ (static_cast<uint64_t>(group_idx) << 32));
    for (uint32_t j = static_cast<uint32_t>(h) & mask;; j = (j + 1) & mask) {
        side_slot *s = &t->slots[j];
        if (!s->used) {
            s->prefix_id = prefix_id;
            s->group_idx = group_idx;
            s->node = node;
            s->used = 1;
            t->count++;
            return KV_EOK;
        }
        if (s->prefix_id == prefix_id && s->group_idx == group_idx) {
            s->node = node;
            return KV_EOK;
        }
    }
}

static void tomb_insert(cache *c, uint64_t prefix_id) {
#ifdef KV_USE_CUCKOO_TOMB
    if (c->tomb_cf) {
        if (prefix_id)
            kv_cuckoo_insert(c->tomb_cf, prefix_id);
        return;
    }
#endif
    if (!c->tomb_keys || prefix_id == 0)
        return;
    uint32_t m = c->tomb_mask;
    uint32_t j = static_cast<uint32_t>(hash_mix64(prefix_id) & m);
    for (int k = 0; k < 4; k++) {
        uint64_t *slot = &c->tomb_keys[(j + static_cast<uint32_t>(k)) & m];
        if (*slot == 0 || *slot == prefix_id) {
            *slot = prefix_id;
            return;
        }
    }
    c->tomb_keys[(j + static_cast<uint32_t>(now_ns() >> 20)) & m] = prefix_id;
}

static bool tomb_lookup(cache *c, uint64_t prefix_id) {
#ifdef KV_USE_CUCKOO_TOMB
    if (c->tomb_cf)
        return prefix_id ? kv_cuckoo_lookup(c->tomb_cf, prefix_id) : false;
#endif
    if (!c->tomb_keys || prefix_id == 0)
        return false;
    uint32_t m = c->tomb_mask;
    uint32_t j = static_cast<uint32_t>(hash_mix64(prefix_id) & m);
    for (int k = 0; k < 4; k++)
        if (c->tomb_keys[(j + static_cast<uint32_t>(k)) & m] == prefix_id)
            return true;
    return false;
}

radix_node *side_get(const side_tbl *t, uint64_t prefix_id,
                     uint32_t group_idx) {
    if (t->cap == 0)
        return nullptr;
    uint32_t mask = t->cap - 1;
    uint64_t h =
        hash_mix64(prefix_id ^ (static_cast<uint64_t>(group_idx) << 32));
    for (uint32_t j = static_cast<uint32_t>(h) & mask;; j = (j + 1) & mask) {
        const side_slot *s = &t->slots[j];
        if (!s->used)
            return nullptr;
        if (s->prefix_id == prefix_id && s->group_idx == group_idx)
            return s->node;
    }
}

kv_shape *shape_pool_intern(shape_pool *sp, const uint32_t *lens,
                            uint16_t n_recs) {
    std::lock_guard<std::mutex> lk(sp->mtx);
    for (size_t i = 0; i < sp->items.size(); i++) {
        kv_shape *s = sp->items[i];
        if (s->n_recs == n_recs &&
            (n_recs == 0 ||
             std::memcmp(shape_lens(s), lens,
                         static_cast<size_t>(n_recs) * sizeof(uint32_t)) == 0)) {
            return s;
        }
    }
    auto *s = static_cast<kv_shape *>(
        malloc(sizeof(kv_shape) + static_cast<size_t>(n_recs) * sizeof(uint32_t)));
    if (!s)
        return nullptr;
    s->id = static_cast<uint32_t>(sp->items.size());
    s->n_recs = n_recs;
    if (n_recs)
        std::memcpy(shape_lens(s), lens,
                    static_cast<size_t>(n_recs) * sizeof(uint32_t));
    sp->items.push_back(s);
    return s;
}

void shape_pool_destroy(shape_pool *sp) {
    for (kv_shape *s : sp->items)
        free(s);
    sp->items.clear();
}

void dfree_push(leaf_dfree *d, radix_leaf *leaf) {
    if (leaf->refcnt.load() == 0) {
        free(leaf->stripe);
        free(leaf);
        return;
    }
    std::lock_guard<std::mutex> lk(d->mtx);
    d->items.push_back(leaf);
}

void dfree_reap(leaf_dfree *d) {
    std::lock_guard<std::mutex> lk(d->mtx);
    size_t w = 0;
    for (size_t i = 0; i < d->items.size(); i++) {
        if (d->items[i]->refcnt.load() == 0) {
            free(d->items[i]->stripe);
            free(d->items[i]);
        } else {
            d->items[w++] = d->items[i];
        }
    }
    d->items.resize(w);
}

void dfree_destroy(leaf_dfree *d) {
    dfree_reap(d);
    std::lock_guard<std::mutex> lk(d->mtx);
    for (radix_leaf *lf : d->items) {
        free(lf->stripe);
        free(lf);
    }
    d->items.clear();
}

void region_live_add(cache *c, uint32_t dev_id, uint64_t page_no,
                     uint32_t len_pages) {
    int idx = region_of_page(&c->rms[dev_id], page_no);
    if (idx < 0)
        return;
    std::atomic_ref<int64_t>(c->rms[dev_id].rg[idx].live_bytes)
        .fetch_add(static_cast<int64_t>(len_pages) * KV_PAGE_SIZE);
}

void region_live_sub(cache *c, uint32_t dev_id, uint64_t page_no,
                     uint32_t len_pages) {
    int idx = region_of_page(&c->rms[dev_id], page_no);
    if (idx < 0)
        return;
    std::atomic_ref<int64_t>(c->rms[dev_id].rg[idx].live_bytes)
        .fetch_sub(static_cast<int64_t>(len_pages) * KV_PAGE_SIZE);
}

static void leaf_sub_live(cache *c, radix_leaf *lf) {
    for (int r = 0; r < KV_REPLICA_CNT; r++) {
        if (lf->addr[r].dev_id != KV_INVALID_DEV)
            region_live_sub(c, lf->addr[r].dev_id, lf->addr[r].page_no,
                            lf->addr[r].len_pages);
    }
    if (lf->stripe) {
        kv_stripe_part *parts = stripe_parts(lf->stripe);
        for (uint32_t r = 0; r < lf->stripe->n_parts; r++)
            region_live_sub(c, parts[r].dev_id, parts[r].page_no,
                            parts[r].len_pages);
    }
}

static void leaf_journal(shard_writer *w, uint64_t prefix_id,
                         uint32_t group_idx, uint32_t ver, const kv_addr *addr,
                         uint32_t expire_ts, uint32_t stripe_idx,
                         uint32_t stripe_cnt, uint32_t flags, uint32_t data_len,
                         kv_shape *shape, const uint64_t *path,
                         uint32_t path_len) {
    if (!w)
        return;
    int rc;
    if (shape && shape->n_recs) {
        const uint32_t *lens = shape_lens(shape);
        bool uni = true;
        for (uint16_t i = 1; i < shape->n_recs; i++)
            if (lens[i] != lens[0]) {
                uni = false;
                break;
            }
        if (uni) {
            uint32_t one = lens[0];
            rc = journal_put(w->jour, prefix_id, group_idx, ver, addr,
                             expire_ts, stripe_idx, stripe_cnt,
                             flags | KV_JFLAG_UNIFORM, &one, shape->n_recs,
                             data_len, path, path_len);
        } else {
            rc = journal_put(w->jour, prefix_id, group_idx, ver, addr,
                             expire_ts, stripe_idx, stripe_cnt, flags, lens,
                             shape->n_recs, data_len, path, path_len);
        }
    } else {
        rc = journal_put(w->jour, prefix_id, group_idx, ver, addr, expire_ts,
                         stripe_idx, stripe_cnt, flags, nullptr, 0, data_len,
                         path, path_len);
    }
    if (rc == -KV_EFULL && w->c)
        w->c->ckpt_wanted.store(1);
}

void leaf_publish(cache *c, shard_writer *w, radix_node *node,
                  uint64_t prefix_id, uint32_t group_idx, uint32_t ver,
                  const kv_addr *addr, uint32_t slot, uint32_t flags,
                  uint32_t stripe_idx, uint32_t stripe_cnt,
                  uint32_t stripe_total, uint32_t data_len, kv_shape *shape,
                  const uint64_t *path, uint32_t path_len) {
    radix_leaf *old = node->leaf;
    uint32_t jflags = flags | (slot == 1 ? KV_JFLAG_SLOT1 : 0);
    uint64_t expire_ts = addr->expire_ts;

    if (flags & KV_CHUNK_F_STRIPE) {
        if (old && old->ver.load() > ver)
            return;
        if (old && old->ver.load() == ver && old->stripe) {
            kv_stripe_part *parts = stripe_parts(old->stripe);
            kv_addr p{
                .dev_id = parts[stripe_idx].dev_id,
                .page_no = parts[stripe_idx].page_no,
                .len_pages = parts[stripe_idx].len_pages,
            };
            if (p.dev_id != KV_INVALID_DEV)
                region_live_sub(c, p.dev_id, p.page_no, p.len_pages);
            parts[stripe_idx].dev_id = addr->dev_id;
            parts[stripe_idx].page_no = addr->page_no;
            parts[stripe_idx].len_pages = addr->len_pages;
            old->addr[0].crc = addr->crc;
            region_live_add(c, addr->dev_id, addr->page_no, addr->len_pages);
            if (!old->shape && shape)
                old->shape = shape;
            leaf_journal(w, prefix_id, group_idx, ver, addr,
                         static_cast<uint32_t>(expire_ts), stripe_idx,
                         stripe_cnt, jflags, data_len, shape, path, path_len);
            return;
        }
        if (old) {
            leaf_sub_live(c, old);
            dfree_push(&c->dfree, old);
        }
        if (c->dram)
            dram_invalidate(c->dram, prefix_id, group_idx);
        auto *lf = static_cast<radix_leaf *>(calloc(1, sizeof(radix_leaf)));
        if (!lf)
            fatal("OOM");
        lf->refcnt.store(0);
        lf->ver.store(ver);
        lf->prefix_id = prefix_id;
        lf->shape = shape;
        lf->stripe = static_cast<kv_stripe_addr *>(
            malloc(sizeof(kv_stripe_addr) +
                   static_cast<size_t>(stripe_cnt) * sizeof(kv_stripe_part)));
        if (!lf->stripe)
            fatal("OOM");
        lf->stripe->n_parts = stripe_cnt;
        lf->stripe->total_len = 0;
        kv_stripe_part *parts = stripe_parts(lf->stripe);
        for (uint32_t r = 0; r < stripe_cnt; r++) {
            parts[r].dev_id = KV_INVALID_DEV;
            parts[r].page_no = 0;
            parts[r].len_pages = 0;
        }
        parts[stripe_idx].dev_id = addr->dev_id;
        parts[stripe_idx].page_no = addr->page_no;
        parts[stripe_idx].len_pages = addr->len_pages;
        lf->stripe->total_len = stripe_total;
        lf->addr[0].crc = addr->crc;
        node->leaf = lf;
        if (!old)
            c->radix.n_leaves.fetch_add(1);
        region_live_add(c, addr->dev_id, addr->page_no, addr->len_pages);
        leaf_journal(w, prefix_id, group_idx, ver, addr,
                     static_cast<uint32_t>(expire_ts), stripe_idx, stripe_cnt,
                     jflags, data_len, shape, path, path_len);
        return;
    }

    if (old && old->ver.load() > ver)
        return;

    if (old && old->ver.load() == ver) {
        if (slot < KV_REPLICA_CNT) {
            kv_addr *a = &old->addr[slot];
            if (a->dev_id != KV_INVALID_DEV)
                region_live_sub(c, a->dev_id, a->page_no, a->len_pages);
            *a = *addr;
            region_live_add(c, addr->dev_id, addr->page_no, addr->len_pages);
            if (!old->shape && shape)
                old->shape = shape;
            leaf_journal(w, prefix_id, group_idx, ver, addr,
                         static_cast<uint32_t>(expire_ts), 0, 0, jflags,
                         data_len, shape, path, path_len);
        }
        return;
    }

    if (old) {
        leaf_sub_live(c, old);
        dfree_push(&c->dfree, old);
    }
    if (c->dram)
        dram_invalidate(c->dram, prefix_id, group_idx);
    auto *lf = static_cast<radix_leaf *>(calloc(1, sizeof(radix_leaf)));
    if (!lf)
        fatal("OOM");
    lf->refcnt.store(0);
    lf->ver.store(ver);
    lf->prefix_id = prefix_id;
    lf->shape = shape;
    lf->addr[0].dev_id = KV_INVALID_DEV;
    lf->addr[1].dev_id = KV_INVALID_DEV;
    lf->addr[slot < KV_REPLICA_CNT ? slot : 0] = *addr;
    node->leaf = lf;
    if (!old)
        c->radix.n_leaves.fetch_add(1);
    region_live_add(c, addr->dev_id, addr->page_no, addr->len_pages);
    leaf_journal(w, prefix_id, group_idx, ver, addr,
                 static_cast<uint32_t>(expire_ts), 0, 0, jflags, data_len,
                 shape, path, path_len);
}

void leaf_drop(cache *c, radix_node *node) {
    radix_leaf *lf = node->leaf;
    if (!lf)
        return;
    if (c->dram)
        dram_invalidate(c->dram, lf->prefix_id,
                        node->depth ? node->depth - 1 : 0);
    for (int r = 0; r < KV_REPLICA_CNT; r++) {
        if (lf->addr[r].dev_id != KV_INVALID_DEV)
            region_live_sub(c, lf->addr[r].dev_id, lf->addr[r].page_no,
                            lf->addr[r].len_pages);
    }
    if (lf->stripe) {
        kv_stripe_part *parts = stripe_parts(lf->stripe);
        for (uint32_t r = 0; r < lf->stripe->n_parts; r++)
            region_live_sub(c, parts[r].dev_id, parts[r].page_no,
                            parts[r].len_pages);
    }
    journal *j = &c->journals[lf->addr[0].dev_id == KV_INVALID_DEV
                                  ? 0
                                  : lf->addr[0].dev_id];
    if (j && !c->replaying)
        journal_del(j, lf->prefix_id, node->depth ? node->depth - 1 : 0);
    dfree_push(&c->dfree, lf);
    node->leaf = nullptr;
    c->radix.n_leaves.fetch_sub(1);
}

void trim_region(cache *c, int dev_id, int idx) {
    RegionMgr *rm = &c->rms[dev_id];
    Device *dev = c->devs[dev_id].get();
    uint64_t base = rm->page_base(idx);
    dev->trim(base + 1, static_cast<uint32_t>(rm->region_size_pages - 1));
    rm->rg[idx].state = KV_RG_FREE;
    rm->rg[idx].live_bytes = 0;
    rm->rg[idx].epoch = 0;
    rm->rg[idx].close_ts = 0;
    rm->rg[idx].watermark_page = 1;
    rm->write_header(idx);
}

int cache_drop_region(cache *c, shard_writer *w, int region_idx) {
    RegionMgr *rm = &c->rms[w->dev_id];
    uint64_t base = rm->page_base(region_idx);
    uint64_t end = base + rm->region_size_pages;

    std::unique_lock<std::shared_mutex> lk(c->radix_lock);
    for (uint32_t i = 0; i < c->side.cap; i++) {
        side_slot *s = &c->side.slots[i];
        if (!s->used || !s->node || !s->node->leaf)
            continue;
        radix_leaf *lf = s->node->leaf;
        bool in_region = false;
        for (int r = 0; r < KV_REPLICA_CNT; r++) {
            if (lf->addr[r].dev_id == static_cast<uint32_t>(w->dev_id) &&
                lf->addr[r].page_no >= base && lf->addr[r].page_no < end) {
                in_region = true;
                break;
            }
        }
        if (lf->stripe) {
            kv_stripe_part *parts = stripe_parts(lf->stripe);
            for (uint32_t r = 0; r < lf->stripe->n_parts; r++) {
                if (parts[r].dev_id == static_cast<uint32_t>(w->dev_id) &&
                    parts[r].page_no >= base && parts[r].page_no < end) {
                    in_region = true;
                    break;
                }
            }
        }
        if (in_region && lf->refcnt.load() == 0) {
            leaf_drop(c, s->node);
            if (s->prefix_id <= 100)
                tomb_insert(c, s->prefix_id);
            c->stat_gc_evicted_leaves.fetch_add(1);
        }
    }
    c->stat_drops.fetch_add(1);
    return KV_EOK;
}

struct recompute_arg {
    cache *c;
};
static int recompute_cb(radix_leaf *lf, void *arg) {
    auto *a = static_cast<recompute_arg *>(arg);
    cache *c = a->c;
    for (int r = 0; r < KV_REPLICA_CNT; r++) {
        if (lf->addr[r].dev_id != KV_INVALID_DEV)
            region_live_add(c, lf->addr[r].dev_id, lf->addr[r].page_no,
                            lf->addr[r].len_pages);
    }
    if (lf->stripe) {
        kv_stripe_part *parts = stripe_parts(lf->stripe);
        for (uint32_t r = 0; r < lf->stripe->n_parts; r++)
            region_live_add(c, parts[r].dev_id, parts[r].page_no,
                            parts[r].len_pages);
    }
    return 0;
}

void cache_recompute_live(cache *c) {
    for (int d = 0; d < c->n_devs; d++)
        for (uint32_t i = 0; i < c->cfg.region_cnt; i++)
            c->rms[d].rg[i].live_bytes = 0;
    recompute_arg arg{c};
    radix_visit_leaves(&c->radix, recompute_cb, &arg);
}

int gc_migrate_region(cache *c, int dev_id, int region_idx) {
    RegionMgr *rm = &c->rms[dev_id];
    shard_writer *w = &c->writers[dev_id];
    uint64_t base = rm->page_base(region_idx);
    uint64_t end = base + rm->region_size_pages;
    int migrated = 0;

    std::shared_lock<std::shared_mutex> lk(c->radix_lock);
    for (uint32_t i = 0; i < c->side.cap; i++) {
        side_slot *s = &c->side.slots[i];
        if (!s->used || !s->node || !s->node->leaf)
            continue;
        radix_leaf *lf = s->node->leaf;
        if (lf->stripe || lf->refcnt.load() != 0 || !lf->shape)
            continue;
        bool in_region = false;
        uint32_t slot = 0;
        for (int r = 0; r < KV_REPLICA_CNT; r++) {
            if (lf->addr[r].dev_id == static_cast<uint32_t>(dev_id) &&
                lf->addr[r].page_no >= base && lf->addr[r].page_no < end) {
                in_region = true;
                slot = static_cast<uint32_t>(r);
                break;
            }
        }
        if (!in_region || lf->addr[slot].len_pages == 0)
            continue;
        kv_addr a = lf->addr[slot];
        uint32_t ver = lf->ver.load();
        kv_shape *shape = lf->shape;
        uint32_t data_len = shape_total(shape);
        auto *buf = static_cast<uint8_t *>(
            aligned_alloc(static_cast<size_t>(a.len_pages) * KV_PAGE_SIZE));
        if (!buf)
            continue;
        if (cache_sync_read(c, c->devs[dev_id].get(), a.page_no, buf,
                            a.len_pages) != KV_EOK ||
            pages_validate(buf, a.len_pages, a.crc) != KV_EOK) {
            aligned_free(buf);
            continue;
        }
        auto *r = new put_req();
        auto *pc = new put_ctx();
        put_ctx_init(pc, nullptr, nullptr);
        pc->c = c;
        pc->refs.store(1);
        r->prefix_id = s->prefix_id;
        r->group_idx = s->group_idx;
        r->expire_ts = a.expire_ts;
        r->ver = ver;
        r->slot = slot;
        r->flags = 0;
        r->n_recs = shape->n_recs;
        r->data_len = data_len;
        r->shape = shape;
        r->own_buf = buf;
        const uint32_t *lens = shape_lens(shape);
        for (uint16_t k = 0; k < shape->n_recs && k < KV_MAX_LAYERS_CAP; k++) {
            r->recs[k].base = buf;
            r->recs[k].off = 0;
            r->recs[k].len = lens[k];
        }
        uint32_t off = 0;
        for (uint16_t k = 0; k < r->n_recs; k++) {
            r->recs[k].off = off;
            off += r->recs[k].len;
        }
        r->ctx = pc;
        r->gc_req = 1;
        r->mig_ver = ver;
        shard_writer_enqueue(w, r);
        migrated++;
        if (migrated >= 1024)
            break;
    }
    return KV_EOK;
}

struct TlsRing {
    IoRing ring;
    bool inited = false;
    ~TlsRing() {
        if (inited)
            ring.exit();
    }
};

static thread_local TlsRing tls_ring;

static IoRing *cache_tls_ring() {
    if (!tls_ring.inited) {
        tls_ring.ring.init(128);
        tls_ring.inited = true;
    }
    return &tls_ring.ring;
}

static int cache_sync_read(cache *c, Device *dev, uint64_t page_no, void *buf,
                           uint32_t npages) {
    IoRing *r = cache_tls_ring();
    IoCtx ictx;
    ictx.inflight.store(1);
    ictx.first_err.store(KV_EOK);
    ictx.done = nullptr;
    ictx.arg = nullptr;
    r->submit(IoOp::Read, dev, page_no, buf, npages, &ictx);
    while (ictx.inflight.load() > 0) {
        r->wait();
        r->reap();
    }
    (void)c;
    return ictx.first_err.load() ? -KV_EREAD : KV_EOK;
}

struct replay_part {
    uint32_t dev_id;
    uint64_t page_no;
    uint32_t len_pages;
    uint32_t data_len;
};

struct replay_stripe {
    uint64_t prefix_id = 0;
    uint32_t group_idx = 0;
    uint32_t ver = 0;
    uint32_t stripe_cnt = 0;
    uint32_t n_seen = 0;
    std::vector<replay_part> parts;
    int used = 0;
};

struct replay_state {
    cache *c = nullptr;
    replay_stripe stripes[256];
    int n_stripes = 0;
    uint32_t max_ver = 0;
};

static replay_stripe *replay_find_stripe(replay_state *rs, uint64_t prefix_id,
                                         uint32_t group_idx, uint32_t ver,
                                         uint32_t stripe_cnt) {
    for (int i = 0; i < rs->n_stripes; i++) {
        if (rs->stripes[i].prefix_id == prefix_id &&
            rs->stripes[i].group_idx == group_idx &&
            rs->stripes[i].ver == ver)
            return &rs->stripes[i];
    }
    if (rs->n_stripes >= 256 || stripe_cnt < 2)
        return nullptr;
    replay_stripe *s = &rs->stripes[rs->n_stripes++];
    s->prefix_id = prefix_id;
    s->group_idx = group_idx;
    s->ver = ver;
    s->stripe_cnt = stripe_cnt;
    s->parts.assign(stripe_cnt, replay_part{});
    s->used = 1;
    return s;
}

static void replay_state_free(replay_state *rs) {
    for (int i = 0; i < rs->n_stripes; i++)
        rs->stripes[i].parts.clear();
    rs->n_stripes = 0;
}

static int replay_apply(const journal_rec *rec, uint32_t data_crc,
                        const uint32_t *lens, uint16_t n_lens, uint32_t data_len,
                        const uint64_t *path, void *arg) {
    auto *rs = static_cast<replay_state *>(arg);
    cache *c = rs->c;
    (void)n_lens;

    if (rec->ver > rs->max_ver)
        rs->max_ver = rec->ver;

    if (rec->op == KV_JOP_DEL) {
        radix_node *node = side_get(&c->side, rec->prefix_id, rec->group_idx);
        if (node && node->leaf) {
            std::unique_lock<std::shared_mutex> lk(c->radix_lock);
            if (node->leaf->refcnt.load() == 0)
                leaf_drop(c, node);
        }
        return 0;
    }
    if (rec->op == KV_JOP_TRIM) {
        int dev = rec->dev_id;
        int idx = static_cast<int>(rec->addr.page);
        if (dev >= 0 && dev < c->n_devs && idx >= 0 &&
            idx < static_cast<int>(c->cfg.region_cnt)) {
            c->rms[dev].rg[idx].state = KV_RG_FREE;
            c->rms[dev].rg[idx].live_bytes = 0;
        }
        return 0;
    }
    if (rec->op != KV_JOP_PUT)
        return 0;

    radix_node *node = side_get(&c->side, rec->prefix_id, rec->group_idx);
    if (!node) {
        if (!path || rec->path_len == 0)
            return 0;
        std::unique_lock<std::shared_mutex> lk(c->radix_lock);
        node = radix_walk(&c->radix, path, static_cast<int>(rec->path_len), 1);
        if (node)
            side_put(&c->side, rec->prefix_id, rec->group_idx, node);
        if (!node)
            return 0;
    }

    kv_shape *shape = nullptr;
    if (rec->n_recs && lens) {
        uint32_t full[KV_MAX_LAYERS_CAP];
        uint16_t nrecs = rec->n_recs;
        if (nrecs > KV_MAX_LAYERS_CAP)
            nrecs = KV_MAX_LAYERS_CAP;
        if (rec->flags & KV_JFLAG_UNIFORM) {
            for (uint16_t i = 0; i < nrecs; i++)
                full[i] = lens[0];
        } else {
            std::memcpy(full, lens,
                        static_cast<size_t>(nrecs) * sizeof(uint32_t));
        }
        shape = shape_pool_intern(&c->shapes, full, nrecs);
    }

    uint32_t slot = (rec->flags & KV_JFLAG_SLOT1) ? 1 : 0;
    kv_addr addr{
        .dev_id = rec->addr.dev,
        .page_no = rec->addr.page,
        .page_off = 0,
        .len_pages = rec->addr.len,
        .last_write_ts = 0,
        .expire_ts = rec->expire_ts,
        .crc = data_crc,
    };

    if (rec->stripe_cnt > 1) {
        replay_stripe *s = replay_find_stripe(rs, rec->prefix_id,
                                              rec->group_idx, rec->ver,
                                              rec->stripe_cnt);
        if (!s)
            return 0;
        if (rec->stripe_idx < s->stripe_cnt) {
            s->parts[rec->stripe_idx].dev_id = rec->addr.dev;
            s->parts[rec->stripe_idx].page_no = rec->addr.page;
            s->parts[rec->stripe_idx].len_pages = rec->addr.len;
            s->parts[rec->stripe_idx].data_len = data_len;
            s->n_seen++;
        }
        if (s->n_seen >= s->stripe_cnt) {
            uint32_t stream_len = 0;
            for (uint32_t p = 0; p < s->stripe_cnt; p++)
                stream_len += s->parts[p].data_len;
            std::unique_lock<std::shared_mutex> lk(c->radix_lock);
            for (uint32_t p = 0; p < s->stripe_cnt; p++) {
                kv_addr pa{
                    .dev_id = s->parts[p].dev_id,
                    .page_no = s->parts[p].page_no,
                    .page_off = 0,
                    .len_pages = s->parts[p].len_pages,
                    .expire_ts = rec->expire_ts,
                    .crc = data_crc,
                };
                leaf_publish(c, nullptr, node, rec->prefix_id, rec->group_idx,
                             rec->ver, &pa, 0, KV_CHUNK_F_STRIPE, p,
                             s->stripe_cnt, stream_len, s->parts[p].data_len,
                             shape, path, rec->path_len);
            }
            s->used = 0;
        }
        return 0;
    }

    {
        std::unique_lock<std::shared_mutex> lk(c->radix_lock);
        leaf_publish(c, nullptr, node, rec->prefix_id, rec->group_idx, rec->ver,
                     &addr, slot, rec->flags, 0, 0, 0, data_len, shape, path,
                     rec->path_len);
    }
    return 0;
}

static void cache_tail_scan(cache *c) { (void)c; }

int cache_checkpoint(cache *c) {
    uint64_t vc = c->ver_counter.load();
    int rc;
    std::lock_guard<std::mutex> ck(c->ckpt_mtx);
    std::unique_lock<std::shared_mutex> lk(c->radix_lock);
    rc = ckpt_write(c->ckpt_path, &c->radix, vc, static_cast<uint32_t>(c->n_devs),
                    c->cfg.region_cnt, c->cfg.region_size_pages, &c->shapes);
    if (rc == KV_EOK) {
        for (int i = 0; i < c->n_devs; i++)
            journal_reset(&c->journals[i]);
    }
    return rc;
}

void cache_fire_ack(void (*ack)(void *, int), void *user, int rc) {
    if (ack)
        ack(user, rc);
}

static int cache_prep_device(const char *uri, uint64_t pages) {
    struct stat st;
    if (stat(uri, &st) != 0) {
        int fd = ::open(uri, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0)
            return -KV_EIO;
        if (ftruncate(fd, static_cast<off_t>(pages) * KV_PAGE_SIZE) != 0) {
            ::close(fd);
            return -KV_EIO;
        }
        ::close(fd);
        return KV_EOK;
    }
    if (!S_ISREG(st.st_mode))
        return -KV_EINVAL;
    uint64_t have = static_cast<uint64_t>(st.st_size) / KV_PAGE_SIZE;
    if (have < pages) {
        int fd = ::open(uri, O_RDWR | O_CLOEXEC);
        if (fd < 0)
            return -KV_EIO;
        if (ftruncate(fd, static_cast<off_t>(pages) * KV_PAGE_SIZE) != 0) {
            ::close(fd);
            return -KV_EIO;
        }
        ::close(fd);
    }
    return KV_EOK;
}

struct side_build_arg {
    cache *c;
};
static int side_build_cb(radix_node *node, void *arg) {
    auto *a = static_cast<side_build_arg *>(arg);
    if (node->leaf)
        side_put(&a->c->side, node->leaf->prefix_id,
                 node->depth ? node->depth - 1 : 0, node);
    return 0;
}
struct ver_cb_arg {
    uint64_t maxv;
};
static int ver_scan_cb(radix_leaf *lf, void *arg) {
    auto *a = static_cast<ver_cb_arg *>(arg);
    uint32_t v = lf->ver.load();
    if (v > a->maxv)
        a->maxv = v;
    return 0;
}

int cache_open_impl(cache **out, const char *const *dev_uris, int n_devs,
                    const kv_config *cfg_in) {
    if (!out || !dev_uris || n_devs < 1)
        return -KV_EINVAL;

    auto *c = new cache();
    if (cfg_in)
        c->cfg = *cfg_in;
    else
        ::kv_config_default(&c->cfg);
    if (c->cfg.n_groups_tokens == 0)
        ::kv_config_default(&c->cfg);
    if (c->cfg.abi_version > KV_CONFIG_ABI_VERSION) {
        delete c;
        return -KV_EINVAL;
    }
    c->n_devs = n_devs;
    c->cfg.region_cnt = c->cfg.region_cnt ? c->cfg.region_cnt : 6;
    c->cfg.journal_pages = c->cfg.journal_pages ? c->cfg.journal_pages : 512;
    c->cfg.max_layers =
        c->cfg.max_layers ? c->cfg.max_layers : KV_MAX_LAYERS_CAP;
    kv_layout_derive(&c->layout, &c->cfg);
    c->cfg.stripe_unit = c->layout.stripe_unit;
    c->cfg.stripe_threshold = c->layout.stripe_threshold;
    c->cfg.batch_min_bytes = c->layout.batch_min_bytes;
    c->cfg.dram_entry_max_bytes = c->layout.dram_entry_max_bytes;
    c->cfg.region_align_bytes = c->layout.region_align_bytes;
    if (c->cfg.max_layers > KV_MAX_LAYERS_CAP ||
        c->cfg.batch_min_bytes % KV_PAGE_SIZE != 0 ||
        c->cfg.batch_min_bytes > c->cfg.stripe_threshold) {
        delete c;
        return -KV_EINVAL;
    }
    if (c->cfg.dram_alloc_backend > KV_DRAM_BACKEND_MAX) {
        delete c;
        return -KV_EINVAL;
    }

    if (c->cfg.dram_cache_bytes &&
        dram_init(&c->dram, c->cfg.dram_cache_bytes,
                  c->cfg.dram_entry_max_bytes,
                  c->cfg.dram_alloc_backend) != KV_EOK)
        c->dram = nullptr;

    c->devs.reserve(n_devs);
    for (int i = 0; i < n_devs; i++)
        c->devs.push_back(dev_make_file());
    c->rms = std::make_unique<RegionMgr[]>(n_devs);
    c->journals = std::make_unique<journal[]>(n_devs);
    c->writers = std::make_unique<shard_writer[]>(n_devs);
    for (int i = 0; i < n_devs; i++)
        c->devs[i]->fd = -1;

    c->ver_counter.store(0);
    c->ckpt_wanted.store(0);
    side_init(&c->side);
    radix_init(&c->radix);
    c->stat_meta_bytes_written.store(0);
    kv_hist_init(&c->m.h_put_ack);
    kv_hist_init(&c->m.h_get);

    {
        uint32_t tcap = c->cfg.evict_tomb_cap;
        if (tcap && !(tcap & (tcap - 1))) {
#ifdef KV_USE_CUCKOO_TOMB
            if (kv_cuckoo_init(&c->tomb_cf, tcap) != 0)
                c->tomb_cf = nullptr;
#endif
            c->tomb_keys = std::make_unique<uint64_t[]>(tcap);
            c->tomb_mask = tcap - 1;
        }
    }

    uint64_t region_pages = c->cfg.region_size_pages;
    bool any_new = false;
    for (int i = 0; i < n_devs; i++) {
        struct stat st;
        if (stat(dev_uris[i], &st) != 0)
            any_new = true;
    }
    if (region_pages == 0) {
        if (!any_new) {
            struct stat st;
            if (stat(dev_uris[0], &st) == 0) {
                uint64_t total = static_cast<uint64_t>(st.st_size) / KV_PAGE_SIZE;
                uint64_t usable = total > c->cfg.journal_pages
                                      ? total - c->cfg.journal_pages
                                      : 0;
                region_pages = usable > kRegionBasePage
                                   ? (usable - kRegionBasePage) / c->cfg.region_cnt
                                   : 8192;
            } else {
                region_pages = 8192;
            }
        } else {
            region_pages = 8192;
        }
    }
    if (region_pages < 64)
        region_pages = 64;
    if (c->layout.region_align_bytes > KV_PAGE_SIZE) {
        uint64_t bytes = region_pages * KV_PAGE_SIZE;
        uint64_t ab = c->layout.region_align_bytes;
        uint64_t aligned = (bytes / ab) * ab;
        if (aligned >= 64ull * KV_PAGE_SIZE)
            region_pages = aligned / KV_PAGE_SIZE;
    }
    c->cfg.region_size_pages = region_pages;

    uint64_t dev_total =
        kRegionBasePage +
        static_cast<uint64_t>(c->cfg.region_cnt) * region_pages +
        c->cfg.journal_pages;
    uint64_t journal_page = dev_total - c->cfg.journal_pages;

    for (int i = 0; i < n_devs; i++) {
        int rc = cache_prep_device(dev_uris[i], dev_total);
        if (rc != KV_EOK) {
            cache_close_impl(c);
            return rc;
        }
        rc = dev_open(c->devs[i].get(), dev_uris[i], i);
        if (rc != KV_EOK) {
            cache_close_impl(c);
            return rc;
        }
        if (c->devs[i]->geom.logical_bs > KV_PAGE_SIZE) {
            cache_close_impl(c);
            return -KV_EINVAL;
        }
        if (i > 0 && c->devs[0]->geom.logical_bs &&
            c->devs[i]->geom.logical_bs &&
            c->devs[i]->geom.logical_bs != c->devs[0]->geom.logical_bs) {
            cache_close_impl(c);
            return -KV_EINVAL;
        }
        c->rms[i].init(c->devs[i].get(), c->cfg.region_cnt, region_pages);
        journal_init(&c->journals[i], c->devs[i].get(), journal_page,
                     c->cfg.journal_pages);
        c->writers[i].c = c;
        c->writers[i].dev_id = i;
        c->writers[i].dev = c->devs[i].get();
        c->writers[i].rm = &c->rms[i];
        c->writers[i].jour = &c->journals[i];
        c->writers[i].cursor.region = 0;
        c->writers[i].cursor.next_page = 1;
        int wrc = shard_writer_start(&c->writers[i]);
        if (wrc != KV_EOK) {
            cache_close_impl(c);
            return wrc;
        }
    }

    std::snprintf(c->ckpt_path, sizeof(c->ckpt_path), "%s.ckpt", dev_uris[0]);

    Superblock sb;
    int rc = c->rms[0].load(&sb);
    if (rc != KV_EOK) {
        for (int i = 0; i < n_devs; i++) {
            c->rms[i].format(0);
            journal_reset(&c->journals[i]);
        }
        for (int i = 0; i < n_devs; i++) {
            RegionMgr *rm = &c->rms[i];
            rm->epoch_counter++;
            rm->rg[0].epoch = rm->epoch_counter;
            rm->rg[0].base_ts = now_s();
            rm->rg[0].watermark_page = 1;
            rm->rg[0].state = KV_RG_OPEN;
            rm->write_header(0);
            rm->sb_write();
        }
        for (int i = 0; i < n_devs; i++) {
            c->writers[i].cursor.region = 0;
            c->writers[i].cursor.next_page = 1;
            c->writers[i].confirmed_page = c->rms[i].page_base(0) + 1;
        }
        if (gc_start(c) != KV_EOK) {
            cache_close_impl(c);
            return -KV_EIO;
        }
        cache_checkpoint(c);
        *out = c;
        return KV_EOK;
    }

    for (int i = 1; i < n_devs; i++) {
        Superblock sb_i;
        if (c->rms[i].load(&sb_i) != KV_EOK) {
            c->rms[i].format(sb.cur_epoch);
            journal_reset(&c->journals[i]);
        }
    }

    replay_state rs;
    rs.c = c;
    c->replaying = 1;

    if (ckpt_load(c->ckpt_path, &c->radix, nullptr, static_cast<uint32_t>(n_devs),
                  c->cfg.region_cnt, region_pages, &c->shapes, nullptr,
                  nullptr) == KV_EOK) {
    }

    side_build_arg sba{c};
    radix_visit(&c->radix, side_build_cb, &sba);

    for (int i = 0; i < n_devs; i++)
        journal_replay(&c->journals[i], replay_apply, &rs);

    cache_tail_scan(c);
    c->replaying = 0;
    replay_state_free(&rs);

    cache_recompute_live(c);
    for (int d = 0; d < n_devs; d++) {
        RegionMgr *rm = &c->rms[d];
        for (uint32_t i = 0; i < rm->region_cnt; i++) {
            if (rm->rg[i].live_bytes == 0 && rm->rg[i].state != KV_RG_OPEN)
                rm->rg[i].state = KV_RG_FREE;
            else if (rm->rg[i].state != KV_RG_OPEN)
                rm->rg[i].state = KV_RG_FROZEN;
            rm->write_header(static_cast<int>(i));
        }
    }

    for (int d = 0; d < n_devs; d++) {
        RegionMgr *rm = &c->rms[d];
        int opened = -1;
        for (uint32_t i = 0; i < rm->region_cnt; i++) {
            if (rm->rg[i].state == KV_RG_FREE) {
                opened = static_cast<int>(i);
                break;
            }
        }
        if (opened < 0) {
            uint64_t oldest = UINT64_MAX;
            for (uint32_t i = 0; i < rm->region_cnt; i++) {
                if (rm->rg[i].state == KV_RG_FROZEN && rm->rg[i].epoch &&
                    rm->rg[i].epoch < oldest) {
                    oldest = rm->rg[i].epoch;
                    opened = static_cast<int>(i);
                }
            }
            if (opened >= 0)
                cache_drop_region(c, &c->writers[d], opened);
        }
        if (opened < 0)
            opened = 0;

        rm->epoch_counter++;
        rm->rg[opened].epoch = rm->epoch_counter;
        rm->rg[opened].base_ts = now_s();
        rm->rg[opened].close_ts = 0;
        rm->rg[opened].watermark_page = 1;
        rm->rg[opened].live_bytes = 0;
        rm->rg[opened].state = KV_RG_OPEN;
        rm->write_header(opened);
        rm->sb_write();
        c->writers[d].cursor.region = opened;
        c->writers[d].cursor.next_page = 1;
        c->writers[d].confirmed_page = rm->page_base(opened) + 1;
    }

    uint64_t maxv = rs.max_ver;
    {
        ver_cb_arg vca{rs.max_ver};
        radix_visit_leaves(&c->radix, ver_scan_cb, &vca);
        maxv = vca.maxv;
    }
    c->ver_counter.store(maxv + 1);

    if (gc_start(c) != KV_EOK) {
        cache_close_impl(c);
        return -KV_EIO;
    }
    cache_checkpoint(c);

    *out = c;
    return KV_EOK;
}

static void cache_stop_writers(cache *c) {
    c->stop.store(1);
    gc_stop(c);
    for (int i = 0; i < c->n_devs; i++)
        shard_writer_stop(&c->writers[i]);
}

void cache_close_impl(cache *c) {
    if (!c)
        return;
    cache_stop_writers(c);
    if (c->dram)
        dram_destroy(c->dram);
    c->dram = nullptr;
    for (int i = 0; i < c->n_devs; i++) {
        if (c->journals[i].dev)
            journal_destroy(&c->journals[i]);
        c->rms[i].destroy();
        if (c->devs[i]->fd >= 0)
            c->devs[i]->close();
    }
    radix_destroy(&c->radix);
    side_destroy(&c->side);
    shape_pool_destroy(&c->shapes);
    dfree_destroy(&c->dfree);
    c->tomb_keys.reset();
#ifdef KV_USE_CUCKOO_TOMB
    if (c->tomb_cf) {
        kv_cuckoo_destroy(c->tomb_cf);
        c->tomb_cf = nullptr;
    }
#endif
    delete c;
}

void cache_signal_stop(cache *c) { c->stop.store(1); }

inline constexpr uint32_t KV_MAX_GROUPS = 256;

static uint32_t group_token_count(uint32_t n_tokens_total, uint32_t k) {
    uint32_t start = k * KV_TOKENS_PER_GROUP;
    if (n_tokens_total <= start)
        return 0;
    uint32_t cnt = n_tokens_total - start;
    return cnt > KV_TOKENS_PER_GROUP ? KV_TOKENS_PER_GROUP : cnt;
}

int cache_put_impl(cache *c, uint64_t prefix_id, uint32_t group_idx,
                   const uint32_t *tokens, uint32_t n_tokens_total,
                   uint32_t expire_ts, uint16_t n_layers,
                   const kv_data_ref *recs, void (*ack)(void *, int),
                   void *user) {
    if (!c || c->stop.load() || group_idx >= KV_MAX_GROUPS)
        return -KV_EINVAL;
    if (n_layers > c->cfg.max_layers)
        return -KV_EINVAL;

    uint64_t hashes[KV_MAX_GROUPS];
    for (uint32_t k = 0; k <= group_idx; k++) {
        uint32_t cnt = group_token_count(n_tokens_total, k);
        hashes[k] = hash_path(prefix_id, tokens + k * KV_TOKENS_PER_GROUP, cnt,
                              k);
    }

    uint64_t total_len = 0;
    for (int i = 0; i < n_layers; i++)
        total_len += recs[i].len;

    uint32_t ver = static_cast<uint32_t>(c->ver_counter.fetch_add(1)) + 1;

    bool hot = false;
    {
        std::shared_lock<std::shared_mutex> lk(c->radix_lock);
        hot = hot_check_path(&c->radix, hashes, static_cast<int>(group_idx),
                             c->cfg.hot_hit_threshold);
    }

    uint32_t base_dev =
        static_cast<uint32_t>(hash_mix64(prefix_id) % c->n_devs);
    c->stat_put_reqs.fetch_add(1);

    {
        uint64_t plen = 0;
        for (int i = 0; i < n_layers; i++)
            plen += recs[i].len;
        c->stat_payload_bytes.fetch_add(plen);
    }

    auto *pc = new put_ctx();
    put_ctx_init(pc, ack, user);
    pc->c = c;
    if (c->cfg.metrics_level >= KV_MLEVEL_FULL)
        pc->t0_ns = now_ns();

    uint32_t stripe_thr = static_cast<uint32_t>(c->cfg.stripe_threshold);
    uint32_t lens[KV_MAX_LAYERS_CAP];
    for (int i = 0; i < n_layers; i++)
        lens[i] = recs[i].len;
    kv_shape *shape = shape_pool_intern(&c->shapes, lens,
                                        static_cast<uint16_t>(n_layers));
    if (!shape) {
        put_ctx_unref(pc);
        return -KV_ENOMEM;
    }

    if (total_len > stripe_thr) {
        uint32_t stream_len = static_cast<uint32_t>(total_len);
        auto *stream = static_cast<uint8_t *>(malloc(stream_len));
        if (!stream) {
            put_ctx_unref(pc);
            return -KV_ENOMEM;
        }
        uint8_t *p = stream;
        for (int i = 0; i < n_layers; i++) {
            if (recs[i].len)
                std::memcpy(p,
                            static_cast<const uint8_t *>(recs[i].base) +
                                recs[i].off,
                            recs[i].len);
            p += recs[i].len;
        }
        pc->staging = stream;
        uint32_t stream_crc = pages_crc(stream, payload_npages(stream_len));

        uint32_t n_parts = 0;
        if (stripe_split(stream_len, c->layout.stripe_unit, &n_parts) !=
            KV_EOK) {
            put_ctx_unref(pc);
            return -KV_EINVAL;
        }
        if (n_parts > KV_MAX_STRIPE_PARTS) {
            put_ctx_unref(pc);
            return -KV_EINVAL;
        }
        pc->refs.store(static_cast<int>(n_parts));
        for (uint32_t part = 0; part < n_parts; part++) {
            uint32_t dev = (base_dev + part) % c->n_devs;
            auto *r = new put_req();
            r->prefix_id = prefix_id;
            r->group_idx = group_idx;
            r->n_tokens = group_token_count(n_tokens_total, group_idx);
            r->n_tokens_total = n_tokens_total;
            r->expire_ts = expire_ts;
            r->ver = ver;
            r->flags = KV_CHUNK_F_STRIPE;
            r->stripe_idx = part;
            r->stripe_cnt = n_parts;
            r->stripe_total = stream_len;
            r->slice.base = stream + static_cast<uint64_t>(part) *
                                         c->layout.stripe_unit;
            r->slice.len = static_cast<uint32_t>(min_of(
                c->layout.stripe_unit,
                static_cast<uint64_t>(stream_len) -
                    static_cast<uint64_t>(part) * c->layout.stripe_unit));
            r->data_len = r->slice.len;
            r->crc = stream_crc;
            r->shape = shape;
            r->ctx = pc;
            r->hashes = static_cast<uint64_t *>(
                malloc(static_cast<size_t>(group_idx + 1) * sizeof(uint64_t)));
            if (r->hashes)
                std::memcpy(r->hashes, hashes,
                            static_cast<size_t>(group_idx + 1) *
                                sizeof(uint64_t));
            shard_writer_enqueue(&c->writers[dev], r);
        }
        return KV_EOK;
    }

    if (hot) {
        uint32_t dev0 = base_dev;
        uint32_t dev1 = (base_dev + 1) % c->n_devs;
        c->stat_puts_replica.fetch_add(1);
        pc->refs.store(2);
        for (int slot = 0; slot < 2; slot++) {
            uint32_t dev = slot == 0 ? dev0 : dev1;
            auto *r = new put_req();
            r->prefix_id = prefix_id;
            r->group_idx = group_idx;
            r->n_tokens = group_token_count(n_tokens_total, group_idx);
            r->n_tokens_total = n_tokens_total;
            r->expire_ts = expire_ts;
            r->ver = ver;
            r->flags = KV_CHUNK_F_REPLICA | KV_CHUNK_F_HOT;
            r->slot = static_cast<uint32_t>(slot);
            r->n_recs = n_layers;
            r->data_len = static_cast<uint32_t>(total_len);
            r->shape = shape;
            std::memcpy(r->recs, recs,
                        sizeof(kv_data_ref) * n_layers);
            r->ctx = pc;
            r->hashes = static_cast<uint64_t *>(
                malloc(static_cast<size_t>(group_idx + 1) * sizeof(uint64_t)));
            if (r->hashes)
                std::memcpy(r->hashes, hashes,
                            static_cast<size_t>(group_idx + 1) *
                                sizeof(uint64_t));
            shard_writer_enqueue(&c->writers[dev], r);
        }
        return KV_EOK;
    }

    pc->refs.store(1);
    auto *r = new put_req();
    r->prefix_id = prefix_id;
    r->group_idx = group_idx;
    r->n_tokens = group_token_count(n_tokens_total, group_idx);
    r->n_tokens_total = n_tokens_total;
    r->expire_ts = expire_ts;
    r->ver = ver;
    r->n_recs = n_layers;
    r->data_len = static_cast<uint32_t>(total_len);
    r->shape = shape;
    std::memcpy(r->recs, recs, sizeof(kv_data_ref) * n_layers);
    r->ctx = pc;
    r->hashes = static_cast<uint64_t *>(
        malloc(static_cast<size_t>(group_idx + 1) * sizeof(uint64_t)));
    if (r->hashes)
        std::memcpy(r->hashes, hashes,
                    static_cast<size_t>(group_idx + 1) * sizeof(uint64_t));
    shard_writer_enqueue(&c->writers[base_dev], r);
    return KV_EOK;
}

static int cache_read_chunk(cache *c, uint32_t dev_id, uint64_t page_no,
                            uint32_t len_pages, uint32_t crc, uint8_t **out) {
    Device *dev = c->devs[dev_id].get();
    auto *buf = static_cast<uint8_t *>(
        aligned_alloc(static_cast<size_t>(len_pages) * KV_PAGE_SIZE));
    if (!buf)
        return -KV_ENOMEM;
    int rc = cache_sync_read(c, dev, page_no, buf, len_pages);
    if (rc != KV_EOK) {
        aligned_free(buf);
        return rc;
    }
    if (pages_validate(buf, len_pages, crc) != KV_EOK) {
        aligned_free(buf);
        return -KV_ECRC;
    }
    *out = buf;
    return KV_EOK;
}

static uint32_t shape_total(const kv_shape *sh) {
    uint32_t t = 0;
    if (sh) {
        const uint32_t *lens = shape_lens(sh);
        for (uint16_t i = 0; i < sh->n_recs; i++)
            t += lens[i];
    }
    return t;
}

static int shape_fill(const kv_shape *sh, cache_get_result *out) {
    if (!sh)
        return -KV_ECRC;
    uint16_t n = sh->n_recs;
    if (n > KV_MAX_LAYERS_CAP)
        n = KV_MAX_LAYERS_CAP;
    const uint32_t *lens = shape_lens(sh);
    uint32_t off = 0;
    for (uint16_t i = 0; i < n; i++) {
        out->recs[i].layer_id = i;
        out->recs[i].off = off;
        out->recs[i].len = lens[i];
        off += lens[i];
    }
    out->n_records = n;
    return KV_EOK;
}

static int cache_get_impl(cache *c, uint64_t prefix_id, const uint32_t *tokens,
                          uint32_t n_tokens_total, cache_get_result *out) {
    if (!c || !out)
        return -KV_EINVAL;
    std::memset(out, 0, sizeof(*out));

    uint32_t n_groups =
        (n_tokens_total + KV_TOKENS_PER_GROUP - 1) / KV_TOKENS_PER_GROUP;
    if (n_groups == 0 || n_groups > KV_MAX_GROUPS)
        return -KV_EINVAL;
    uint64_t hashes[KV_MAX_GROUPS];
    for (uint32_t k = 0; k < n_groups; k++) {
        uint32_t cnt = group_token_count(n_tokens_total, k);
        hashes[k] =
            hash_path(prefix_id, tokens + k * KV_TOKENS_PER_GROUP, cnt, k);
    }

    radix_leaf *lf;
    uint32_t ver;
    uint32_t expire;
    kv_stripe_addr *stripe;
    kv_shape *shape;
    kv_addr addr0, addr1;
    {
        std::shared_lock<std::shared_mutex> lk(c->radix_lock);
        radix_node *node =
            radix_walk(&c->radix, hashes, static_cast<int>(n_groups), 0);
        if (!node || !node->leaf) {
            c->stat_get_misses.fetch_add(1);
            if (tomb_lookup(c, prefix_id))
                return -KV_EVICTED;
            return -KV_ENOENT;
        }
        lf = node->leaf;
        ver = lf->ver.load();
        expire = lf->addr[0].expire_ts;
        uint64_t cur = now_s();
        if (ver == 0 || (expire != 0 && expire < cur)) {
            c->stat_get_misses.fetch_add(1);
            return -KV_ENOENT;
        }
        radix_hit_ascend(node, 1);
        lf->refcnt.fetch_add(1);
        stripe = lf->stripe;
        shape = lf->shape;
        addr0 = lf->addr[0];
        addr1 = lf->addr[1];
    }

    if (c->dram) {
        uint8_t *dbuf = nullptr;
        uint32_t dbuf_len = 0;
        dram_rec drecs[KV_MAX_LAYERS_CAP];
        uint16_t dn = 0;
        int drc = dram_get(c->dram, prefix_id, n_groups - 1, ver, &dbuf,
                           &dbuf_len, drecs, KV_MAX_LAYERS_CAP, &dn);
        if (drc == KV_EOK) {
            lf->refcnt.fetch_sub(1);
            out->buf = dbuf;
            out->buf_len = dbuf_len;
            out->n_records = dn;
            for (int i = 0; i < dn; i++) {
                out->recs[i].layer_id = drecs[i].layer_id;
                out->recs[i].off = drecs[i].off;
                out->recs[i].len = drecs[i].len;
            }
            out->ver = ver;
            out->prefix_id = prefix_id;
            out->group_idx = n_groups - 1;
            c->stat_get_hits.fetch_add(1);
            return KV_EOK;
        }
    }

    int rc;
    if (stripe) {
        uint32_t stream_len = stripe->total_len;
        auto *stream = static_cast<uint8_t *>(aligned_alloc(stream_len));
        if (!stream) {
            lf->refcnt.fetch_sub(1);
            return -KV_ENOMEM;
        }

        uint32_t max_pages = 0;
        kv_stripe_part *parts = stripe_parts(stripe);
        for (uint32_t p = 0; p < stripe->n_parts; p++)
            if (parts[p].len_pages > max_pages)
                max_pages = parts[p].len_pages;
        auto *tmp = static_cast<uint8_t *>(
            aligned_alloc(static_cast<size_t>(max_pages) * KV_PAGE_SIZE));
        if (!tmp) {
            aligned_free(stream);
            lf->refcnt.fetch_sub(1);
            return -KV_ENOMEM;
        }

        rc = KV_EOK;
        for (uint32_t p = 0; p < stripe->n_parts; p++) {
            uint32_t dev = parts[p].dev_id;
            uint32_t lenp = parts[p].len_pages;
            uint64_t page = parts[p].page_no;
            if (dev >= static_cast<uint32_t>(c->n_devs) || page == 0) {
                rc = -KV_EREAD;
                break;
            }
            if (cache_sync_read(c, c->devs[dev].get(), page, tmp, lenp) !=
                KV_EOK) {
                rc = -KV_EREAD;
                break;
            }
            uint32_t doff =
                static_cast<uint32_t>(static_cast<uint64_t>(p) *
                                      c->layout.stripe_unit);
            uint32_t part_len = static_cast<uint32_t>(
                min_of(c->layout.stripe_unit,
                       static_cast<uint64_t>(stream_len) - doff));
            std::memcpy(stream + doff, tmp,
                        min_of(part_len, stripe->total_len - doff));
        }
        aligned_free(tmp);
        if (rc == KV_EOK &&
            pages_validate(stream, payload_npages(stream_len), addr0.crc) !=
                KV_EOK)
            rc = -KV_ECRC;
        if (rc != KV_EOK) {
            aligned_free(stream);
            lf->refcnt.fetch_sub(1);
            return rc;
        }
        out->buf = stream;
        out->buf_len = stream_len;
        if (shape_fill(shape, out) != KV_EOK) {
            aligned_free(stream);
            out->buf = nullptr;
            lf->refcnt.fetch_sub(1);
            return -KV_ECRC;
        }
        out->ver = ver;
        out->prefix_id = prefix_id;
        out->group_idx = n_groups - 1;
        rc = KV_EOK;
    } else {
        uint8_t *chunk = nullptr;
        rc = cache_read_chunk(c, addr0.dev_id, addr0.page_no, addr0.len_pages,
                              addr0.crc, &chunk);
        if (rc != KV_EOK && addr1.dev_id != KV_INVALID_DEV) {
            rc = cache_read_chunk(c, addr1.dev_id, addr1.page_no,
                                  addr1.len_pages, addr1.crc, &chunk);
        }
        if (rc != KV_EOK) {
            lf->refcnt.fetch_sub(1);
            return rc;
        }
        out->buf = chunk;
        out->buf_len = shape_total(shape);
        if (shape_fill(shape, out) != KV_EOK) {
            aligned_free(chunk);
            out->buf = nullptr;
            lf->refcnt.fetch_sub(1);
            return -KV_ECRC;
        }
        out->ver = ver;
        out->prefix_id = prefix_id;
        out->group_idx = n_groups - 1;
        rc = KV_EOK;
    }
    c->stat_get_hits.fetch_add(1);
    if (c->dram && rc == KV_EOK) {
        dram_rec drecs[KV_MAX_LAYERS_CAP];
        for (int i = 0; i < out->n_records; i++) {
            drecs[i].layer_id = out->recs[i].layer_id;
            drecs[i].off = out->recs[i].off;
            drecs[i].len = out->recs[i].len;
        }
        dram_put(c->dram, prefix_id, n_groups - 1, ver, expire,
                 static_cast<const uint8_t *>(out->buf), out->buf_len, drecs,
                 out->n_records);
    }
    lf->refcnt.fetch_sub(1);
    return rc;
}

int cache_evict_impl(cache *c, uint64_t prefix_id, const uint32_t *tokens,
                     uint32_t n_tokens_total) {
    if (!c)
        return -KV_EINVAL;
    uint32_t n_groups =
        (n_tokens_total + KV_TOKENS_PER_GROUP - 1) / KV_TOKENS_PER_GROUP;
    if (n_groups == 0 || n_groups > KV_MAX_GROUPS)
        return -KV_EINVAL;
    uint64_t hashes[KV_MAX_GROUPS];
    for (uint32_t k = 0; k < n_groups; k++) {
        uint32_t cnt = group_token_count(n_tokens_total, k);
        hashes[k] =
            hash_path(prefix_id, tokens + k * KV_TOKENS_PER_GROUP, cnt, k);
    }
    std::unique_lock<std::shared_mutex> lk(c->radix_lock);
    radix_node *node =
        radix_walk(&c->radix, hashes, static_cast<int>(n_groups), 0);
    if (!node || !node->leaf)
        return -KV_ENOENT;
    radix_leaf *lf = node->leaf;
    if (lf->refcnt.load() > 0)
        return -KV_EBUSY;
    leaf_drop(c, node);
    tomb_insert(c, prefix_id);
    return KV_EOK;
}

uint64_t cache_stats_impl(cache *c, const char *key) {
    if (!c)
        return 0;
    if (!std::strcmp(key, "puts"))
        return c->stat_put_reqs.load();
    if (!std::strcmp(key, "hits"))
        return c->stat_get_hits.load();
    if (!std::strcmp(key, "misses"))
        return c->stat_get_misses.load();
    if (!std::strcmp(key, "batches"))
        return c->stat_batches.load();
    if (!std::strcmp(key, "bytes_written"))
        return c->stat_bytes_written.load();
    if (!std::strcmp(key, "bytes_migrated"))
        return c->stat_bytes_migrated.load();
    if (!std::strcmp(key, "rotations"))
        return c->stat_rotations.load();
    if (!std::strcmp(key, "drops"))
        return c->stat_drops.load();
    if (!std::strcmp(key, "leaves"))
        return c->radix.n_leaves.load();
    if (!std::strcmp(key, "stripe_unit"))
        return c->layout.stripe_unit;
    if (!std::strcmp(key, "stripe_threshold"))
        return c->layout.stripe_threshold;
    if (!std::strcmp(key, "batch_min_bytes"))
        return c->layout.batch_min_bytes;
    if (!std::strcmp(key, "region_align_bytes"))
        return c->layout.region_align_bytes;
    if (!std::strcmp(key, "live_bytes")) {
        uint64_t live = 0;
        for (int d = 0; d < c->n_devs; d++)
            for (uint32_t i = 0; i < c->rms[d].region_cnt; i++)
                live += static_cast<uint64_t>(c->rms[d].rg[i].live_bytes);
        return live;
    }
    if (!std::strcmp(key, "capacity_bytes")) {
        uint64_t cap = 0;
        for (int d = 0; d < c->n_devs; d++)
            cap += static_cast<uint64_t>(c->rms[d].region_cnt) *
                   c->rms[d].region_size_pages * KV_PAGE_SIZE;
        return cap;
    }
    if (!std::strcmp(key, "generations")) {
        uint64_t mask[64] = {0};
        int gens = 0;
        for (int d = 0; d < c->n_devs; d++)
            for (uint32_t i = 0; i < c->rms[d].region_cnt; i++) {
                uint64_t e = c->rms[d].rg[i].epoch;
                if (e == 0 || c->rms[d].rg[i].state == KV_RG_FREE)
                    continue;
                uint64_t bit = 1ull << (e & 63);
                if (!(mask[d] & bit)) {
                    mask[d] |= bit;
                    gens++;
                }
            }
        return static_cast<uint64_t>(gens);
    }
    if (!std::strcmp(key, "gc_triggers"))
        return c->stat_gc_triggers.load();
    if (!std::strcmp(key, "gc_evicted_leaves"))
        return c->stat_gc_evicted_leaves.load();
    if (!std::strcmp(key, "journal_dropped")) {
        uint64_t dropped = 0;
        for (int d = 0; d < c->n_devs; d++)
            dropped += c->journals[d].dropped;
        return dropped;
    }
    if (!std::strcmp(key, "ver"))
        return c->ver_counter.load();
    if (!std::strncmp(key, "dram_", 5))
        return dram_stat(c->dram, key);
    return 0;
}

} // namespace sc

// ------------------------------------------------------------------
// public C ABI
// ------------------------------------------------------------------

extern "C" void kv_config_default(struct kv_config *cfg) {
    std::memset(cfg, 0, sizeof(*cfg));
    KV_CONFIG_APPLY_DEFAULTS(cfg);
}

extern "C" int cache_open(cache_t **out, const char *const *dev_uris, int n_devs,
                          const struct kv_config *cfg) {
    return sc::cache_open_impl(out, dev_uris, n_devs, cfg);
}

extern "C" void cache_close(cache_t *c) { sc::cache_close_impl(c); }

extern "C" int cache_put(cache_t *c, uint64_t prefix_id, uint32_t group_idx,
                         const uint32_t *tokens, uint32_t n_tokens_total,
                         uint32_t expire_ts, uint16_t n_layers,
                         const struct kv_data_ref *recs,
                         void (*ack)(void *, int), void *user) {
    return sc::cache_put_impl(c, prefix_id, group_idx, tokens, n_tokens_total,
                              expire_ts, n_layers, recs, ack, user);
}

extern "C" int cache_get(cache_t *c, uint64_t prefix_id,
                         const uint32_t *tokens, uint32_t n_tokens_total,
                         struct cache_get_result *out) {
    uint64_t t0 =
        (c && c->cfg.metrics_level >= sc::KV_MLEVEL_FULL) ? sc::now_ns() : 0;
    int rc = sc::cache_get_impl(c, prefix_id, tokens, n_tokens_total, out);
    if (t0)
        sc::kv_hist_sample(&c->m.h_get, sc::now_ns() - t0);
    return rc;
}

extern "C" void cache_result_free(struct cache_get_result *res) {
    if (res && res->buf)
        sc::aligned_free(res->buf);
    if (res)
        std::memset(res, 0, sizeof(*res));
}

extern "C" int cache_evict(cache_t *c, uint64_t prefix_id,
                           const uint32_t *tokens, uint32_t n_tokens_total) {
    return sc::cache_evict_impl(c, prefix_id, tokens, n_tokens_total);
}

extern "C" uint64_t cache_stats(cache_t *c, const char *key) {
    return sc::cache_stats_impl(c, key);
}

extern "C" int cache_dev_count(cache_t *c) { return c ? c->n_devs : 0; }
