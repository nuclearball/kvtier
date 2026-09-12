#include "writer.hpp"
#include "cache.hpp"
#include "chunk.hpp"
#include "crc32.hpp"

#include <chrono>

namespace sc {

static uint32_t batch_buf_size(const shard_writer *w) {
    uint64_t m = w->c->cfg.stripe_threshold;
    if (w->c->layout.stripe_unit > m)
        m = w->c->layout.stripe_unit;
    return static_cast<uint32_t>(m + 4ull * KV_PAGE_SIZE);
}

int mpsc_init(kv_mpsc_ring *q, uint32_t power) {
    if (power < 4)
        power = 4;
    uint64_t cap = 1ull << power;
    q->slots = static_cast<kv_mpsc_slot *>(calloc(cap, sizeof(kv_mpsc_slot)));
    if (!q->slots)
        return -KV_ENOMEM;
    q->mask = cap - 1;
    for (uint64_t i = 0; i < cap; i++)
        q->slots[i].seq.store(i);
    q->tail.store(0);
    q->head.store(0);
    q->init = 1;
    return KV_EOK;
}

void mpsc_destroy(kv_mpsc_ring *q) {
    free(q->slots);
    q->slots = nullptr;
}

int mpsc_push(kv_mpsc_ring *q, void *ptr) {
    for (;;) {
        uint64_t tail = q->tail.load(std::memory_order_relaxed);
        uint64_t pos = tail & q->mask;
        kv_mpsc_slot *s = &q->slots[pos];
        uint64_t seq = s->seq.load(std::memory_order_acquire);
        int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(tail);
        if (diff == 0) {
            if (q->tail.compare_exchange_weak(tail, tail + 1)) {
                __atomic_store_n(&s->ptr, ptr, __ATOMIC_RELAXED);
                s->seq.store(tail + 1, std::memory_order_release);
                return KV_EOK;
            }
        } else if (diff < 0) {
            return -KV_EFULL;
        }
    }
}

void *mpsc_pop(kv_mpsc_ring *q) {
    uint64_t head = q->head.load(std::memory_order_relaxed);
    uint64_t pos = head & q->mask;
    kv_mpsc_slot *s = &q->slots[pos];
    uint64_t seq = s->seq.load(std::memory_order_acquire);
    if (static_cast<int64_t>(seq) - static_cast<int64_t>(head + 1) != 0)
        return nullptr;
    void *ptr = __atomic_load_n(&s->ptr, __ATOMIC_ACQUIRE);
    s->seq.store(head + q->mask + 1, std::memory_order_release);
    q->head.store(head + 1, std::memory_order_release);
    return ptr;
}

void put_ctx_init(put_ctx *pc, void (*ack)(void *, int), void *user) {
    pc->refs.store(0);
    pc->err.store(KV_EOK);
    pc->ack = ack;
    pc->user = user;
}

void put_ctx_unref(put_ctx *pc) {
    if (pc->refs.fetch_sub(1) == 1) {
        int err = pc->err.load();
        if (pc->c) {
            if (err != KV_EOK)
                pc->c->stat_puts_err.fetch_add(1);
            if (pc->c->cfg.metrics_level >= KV_MLEVEL_FULL && pc->t0_ns)
                kv_hist_sample(&pc->c->m.h_put_ack, now_ns() - pc->t0_ns);
        }
        if (pc->ack)
            pc->ack(pc->user, err);
        free(pc->staging);
        delete pc;
    }
}

static void put_ctx_set_err(put_ctx *pc, int err) {
    if (err != KV_EOK) {
        int e = KV_EOK;
        pc->err.compare_exchange_strong(e, err);
    }
}

static uint64_t region_abs_page(const RegionMgr *rm, int idx, uint64_t rel) {
    return rm->page_base(idx) + rel;
}

static void writer_flush_watermark(shard_writer *w) {
    RegionMgr *rm = w->rm;
    int idx = w->cursor.region;
    uint64_t base = rm->page_base(idx);
    if (w->confirmed_page < base)
        return;
    uint64_t rel = w->confirmed_page - base;
    if (rel > rm->region_size_pages)
        rel = rm->region_size_pages;
    rm->rg[idx].watermark_page = rel;
    rm->write_header(idx);
}

static int writer_freeze_and_rotate(shard_writer *w) {
    RegionMgr *rm = w->rm;
    cache *c = w->c;

    std::lock_guard<std::mutex> lk(rm->rotate_mtx);

    int old = w->cursor.region;
    uint64_t base = rm->page_base(old);
    uint64_t rel = w->confirmed_page > base ? w->confirmed_page - base : 1;
    rm->rg[old].watermark_page = rel;
    rm->rg[old].close_ts = now_s();
    rm->rg[old].state = KV_RG_FROZEN;
    rm->write_header(old);
    journal_trim(w->jour, static_cast<uint32_t>(old), rm->rg[old].epoch);
    journal_flush(w->jour);

    int next = (old + 1) % static_cast<int>(rm->region_cnt);
    if (rm->rg[next].state != KV_RG_FREE) {
        int freed = cache_force_free_region(c, w, old);
        if (freed < 0)
            return -KV_EFULL;
        next = freed;
    }

    rm->epoch_counter++;
    rm->rg[next].epoch = rm->epoch_counter;
    rm->rg[next].base_ts = now_s();
    rm->rg[next].close_ts = 0;
    rm->rg[next].watermark_page = 1;
    rm->rg[next].live_bytes = 0;
    rm->rg[next].state = KV_RG_OPEN;
    rm->write_header(next);
    rm->sb_write();

    w->cursor.region = next;
    w->cursor.next_page = 1;
    w->confirmed_page = region_abs_page(rm, next, 1);
    w->since_wm = 0;

    c->stat_rotations.fetch_add(1);
    return KV_EOK;
}

static void on_batch_io_done(IoCtx *ictx) {
    auto *bb = static_cast<batch_buf *>(ictx->arg);
    shard_writer *w = bb->w;
    cache *c = w->c;
    if (c->cfg.metrics_level >= KV_MLEVEL_FULL && bb->submit_ns)
        kv_hist_sample(&w->h_batch, now_ns() - bb->submit_ns);
    int err = ictx->first_err.load();

    if (err != KV_EOK) {
        for (int i = 0; i < bb->n_reqs; i++)
            put_ctx_set_err(bb->reqs[i]->ctx, -KV_EWRITE);
    }

    if (bb->region == w->cursor.region) {
        if (bb->last_page > w->confirmed_page)
            w->confirmed_page = bb->last_page;
        w->since_wm += bb->npages;
    }

    {
        std::lock_guard<std::shared_mutex> lk(c->radix_lock);
        for (int i = 0; i < bb->n_reqs; i++) {
            put_req *r = bb->reqs[i];
            if (err == KV_EOK) {
                radix_node *node =
                    side_get(&c->side, r->prefix_id, r->group_idx);
                if (!node && r->hashes) {
                    node = radix_walk(&c->radix, r->hashes, r->group_idx + 1, 1);
                    if (node)
                        side_put(&c->side, r->prefix_id, r->group_idx, node);
                }
                if (!node) {
                    put_ctx_set_err(r->ctx, -KV_ENOMEM);
                } else if (r->gc_req) {
                    radix_leaf *lf = node->leaf;
                    if (lf && lf->ver.load() == r->mig_ver &&
                        lf->refcnt.load() == 0) {
                        kv_addr addr{
                            .dev_id = static_cast<uint32_t>(w->dev_id),
                            .page_no = r->page_no,
                            .page_off = 0,
                            .len_pages = r->npages,
                            .last_write_ts = now_ms(),
                            .expire_ts = r->expire_ts,
                            .crc = r->crc,
                        };
                        leaf_publish(c, w, node, r->prefix_id, r->group_idx,
                                     r->ver, &addr, r->slot, r->flags,
                                     r->stripe_idx, r->stripe_cnt,
                                     r->stripe_total, r->data_len, r->shape,
                                     r->hashes,
                                     r->hashes ? r->group_idx + 1 : 0);
                        c->stat_bytes_migrated.fetch_add(
                            static_cast<uint64_t>(r->npages) * KV_PAGE_SIZE);
                    } else {
                        put_ctx_set_err(r->ctx, KV_EOK);
                    }
                } else {
                    kv_addr addr{
                        .dev_id = static_cast<uint32_t>(w->dev_id),
                        .page_no = r->page_no,
                        .page_off = 0,
                        .len_pages = r->npages,
                        .last_write_ts = now_ms(),
                        .expire_ts = r->expire_ts,
                        .crc = r->crc,
                    };
                    leaf_publish(c, w, node, r->prefix_id, r->group_idx, r->ver,
                                 &addr, r->slot, r->flags, r->stripe_idx,
                                 r->stripe_cnt, r->stripe_total, r->data_len,
                                 r->shape, r->hashes,
                                 r->hashes ? r->group_idx + 1 : 0);
                }
            }
        }
    }

    if (w->since_wm >= KV_WATERMARK_INTERVAL) {
        writer_flush_watermark(w);
        w->since_wm = 0;
    }
    journal_flush(w->jour);

    for (int i = 0; i < bb->n_reqs; i++) {
        put_req *r = bb->reqs[i];
        free(r->tokens);
        free(r->hashes);
        free(r->own_buf);
        put_ctx_unref(r->ctx);
        delete r;
    }

    bb->n_reqs = 0;
    bb->used = 0;
    bb->busy.store(0);
}

static int writer_submit_batch(shard_writer *w) {
    batch_buf *bb = &w->batch[w->cur_buf];
    RegionMgr *rm = w->rm;
    cache *c = w->c;

    if (bb->n_reqs == 0)
        return KV_EOK;

    while (bb->busy.load())
        w->iour.wait(), w->iour.reap();

    uint64_t need_pages = 0;
    for (int i = 0; i < bb->n_reqs; i++)
        need_pages += payload_npages(bb->reqs[i]->data_len);
    if (w->cursor.next_page + need_pages > rm->region_size_pages) {
        int rc = writer_freeze_and_rotate(w);
        if (rc != KV_EOK) {
            for (int i = 0; i < bb->n_reqs; i++) {
                put_ctx_set_err(bb->reqs[i]->ctx, -KV_EFULL);
                free(bb->reqs[i]->tokens);
                free(bb->reqs[i]->hashes);
                put_ctx_unref(bb->reqs[i]->ctx);
                delete bb->reqs[i];
            }
            bb->n_reqs = 0;
            bb->used = 0;
            bb->busy.store(0);
            return rc;
        }
    }

    bb->region = w->cursor.region;
    bb->start_page = region_abs_page(rm, w->cursor.region, w->cursor.next_page);

    uint8_t *dst = bb->buf;
    uint64_t abs_page = bb->start_page;
    uint32_t total_pages = 0;
    for (int i = 0; i < bb->n_reqs; i++) {
        put_req *r = bb->reqs[i];
        uint32_t npages;
        if (r->flags & KV_CHUNK_F_STRIPE) {
            npages = payload_npages(r->data_len);
            if (r->data_len)
                memcpy(dst, r->slice.base, r->data_len);
            if (static_cast<size_t>(npages) * KV_PAGE_SIZE > r->data_len)
                memset(dst + r->data_len, 0,
                       static_cast<size_t>(npages) * KV_PAGE_SIZE - r->data_len);
        } else {
            uint64_t off = 0;
            for (uint16_t k = 0; k < r->n_recs; k++) {
                const kv_data_ref *ref = &r->recs[k];
                if (ref->len)
                    memcpy(dst + off,
                           static_cast<const uint8_t *>(ref->base) + ref->off,
                           ref->len);
                off += ref->len;
            }
            npages = payload_npages(off);
            r->data_len = static_cast<uint32_t>(off);
            if (static_cast<size_t>(npages) * KV_PAGE_SIZE > off)
                memset(dst + off, 0,
                       static_cast<size_t>(npages) * KV_PAGE_SIZE - off);
            r->crc = pages_crc(dst, npages);
        }
        r->page_no = abs_page;
        r->npages = npages;
        abs_page += npages;
        dst += static_cast<size_t>(npages) * KV_PAGE_SIZE;
        total_pages += npages;
    }
    bb->npages = total_pages;
    bb->last_page = bb->start_page + total_pages;

    w->cursor.next_page += total_pages;
    c->stat_bytes_written.fetch_add(static_cast<uint64_t>(total_pages) *
                                    KV_PAGE_SIZE);
    c->stat_batches.fetch_add(1);

    w->stat_bytes_written.fetch_add(static_cast<uint64_t>(total_pages) *
                                    KV_PAGE_SIZE);
    for (int i = 0; i < bb->n_reqs; i++)
        if (bb->reqs[i]->flags & KV_CHUNK_F_STRIPE)
            w->stat_stripe_parts.fetch_add(1);
    if (w->c->cfg.metrics_level >= KV_MLEVEL_FULL)
        bb->submit_ns = now_ns();

    bb->busy.store(1);
    bb->ictx.ref();
    w->iour.submit(IoOp::Write, w->dev, bb->start_page, bb->buf, bb->npages,
                   &bb->ictx);

    w->cur_buf ^= 1;
    batch_buf *nb = &w->batch[w->cur_buf];
    while (nb->busy.load())
        w->iour.wait(), w->iour.reap();
    nb->used = 0;
    nb->n_reqs = 0;
    return KV_EOK;
}

static void writer_thread(shard_writer *w) {
    for (;;) {
        put_req *first = static_cast<put_req *>(mpsc_pop(&w->q));
        if (!first) {
            if (w->stop)
                break;
            w->iour.reap();
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        batch_buf *bb = &w->batch[w->cur_buf];

        for (;;) {
            if (bb->n_reqs >= KV_BATCH_MAX_REQS)
                break;
            if (bb->used >= w->c->cfg.batch_min_bytes)
                break;
            put_req *r = first;
            first = nullptr;
            if (!r)
                r = static_cast<put_req *>(mpsc_pop(&w->q));
            if (!r)
                break;
            uint32_t npages = payload_npages(r->data_len);
            if (bb->n_reqs > 0 &&
                bb->used + npages * KV_PAGE_SIZE >
                    static_cast<uint64_t>(batch_buf_size(w))) {
                mpsc_push(&w->q, r);
                break;
            }
            bb->reqs[bb->n_reqs++] = r;
            bb->used += npages * KV_PAGE_SIZE;
        }

        writer_submit_batch(w);
        w->iour.reap();
    }

    batch_buf *bb = &w->batch[w->cur_buf];
    for (;;) {
        put_req *r = static_cast<put_req *>(mpsc_pop(&w->q));
        if (!r)
            break;
        bb->reqs[bb->n_reqs++] = r;
        bb->used += KV_PAGE_SIZE;
    }
    if (bb->n_reqs)
        writer_submit_batch(w);
    w->iour.reap();
}

int shard_writer_start(shard_writer *w) {
    mpsc_init(&w->q, 12);
    w->iour.init(256);
    uint32_t bbuf = batch_buf_size(w);
    for (int i = 0; i < 2; i++) {
        batch_buf *bb = &w->batch[i];
        bb->buf = static_cast<uint8_t *>(aligned_alloc(bbuf));
        if (!bb->buf)
            return -KV_ENOMEM;
        bb->w = w;
        bb->n_reqs = 0;
        bb->used = 0;
        bb->busy.store(0);
        bb->ictx.inflight.store(0);
        bb->ictx.first_err.store(KV_EOK);
        bb->ictx.done = on_batch_io_done;
        bb->ictx.arg = bb;
    }
    w->cur_buf = 0;
    w->since_wm = 0;
    w->confirmed_page = region_abs_page(w->rm, w->cursor.region, 1);
    kv_hist_init(&w->h_batch);
    try {
        w->tid = std::thread(writer_thread, w);
    } catch (...) {
        return -KV_EIO;
    }
    w->started = 1;
    return KV_EOK;
}

void shard_writer_stop(shard_writer *w) {
    if (!w->started) {
        for (int i = 0; i < 2; i++)
            if (w->batch[i].buf)
                aligned_free(w->batch[i].buf);
        mpsc_destroy(&w->q);
        return;
    }
    w->stop = 1;
    if (w->tid.joinable())
        w->tid.join();
    w->iour.reap();
    w->iour.exit();
    for (int i = 0; i < 2; i++) {
        batch_buf *bb = &w->batch[i];
        if (bb->n_reqs) {
            for (int k = 0; k < bb->n_reqs; k++) {
                put_req *r = bb->reqs[k];
                put_ctx_set_err(r->ctx, -KV_EIO);
                free(r->tokens);
                free(r->hashes);
                put_ctx_unref(r->ctx);
                delete r;
            }
        }
        aligned_free(bb->buf);
    }
    mpsc_destroy(&w->q);
    w->started = 0;
}

int shard_writer_enqueue(shard_writer *w, put_req *r) {
    int rc;
    while ((rc = mpsc_push(&w->q, r)) == -KV_EFULL)
        std::this_thread::yield();
    uint64_t backlog = w->q.tail.load() - w->q.head.load();
    uint64_t cur = w->stat_q_max.load();
    while (backlog > cur && !w->stat_q_max.compare_exchange_weak(cur, backlog))
        ;
    return rc;
}

} // namespace sc
