#include "gc.hpp"
#include "hot.hpp"

#include <chrono>
#include <climits>

namespace sc {

static void gc_tick(cache *c) {
    uint64_t cur_s = now_s();
    uint64_t cur_ms = now_ms();
    kv_config *cfg = &c->cfg;

    if (c->last_decay_ms && cur_ms - c->last_decay_ms >= cfg->hot_decay_ms) {
        hot_decay(&c->radix);
        c->last_decay_ms = cur_ms;
    }

    if (cur_ms - c->last_ttl_ms >= 1000) {
        c->last_ttl_ms = cur_ms;
        {
            std::unique_lock<std::shared_mutex> lk(c->radix_lock);
            for (uint32_t i = 0; i < c->side.cap; i++) {
                side_slot *s = &c->side.slots[i];
                if (!s->used || !s->node || !s->node->leaf)
                    continue;
                radix_leaf *lf = s->node->leaf;
                uint32_t exp = lf->addr[0].expire_ts;
                if (exp != 0 && exp < cur_s) {
                    if (lf->refcnt.load() == 0)
                        leaf_drop(c, s->node);
                }
            }
        }
        dfree_reap(&c->dfree);
    }

    for (int d = 0; d < c->n_devs; d++) {
        RegionMgr *rm = &c->rms[d];
        for (uint32_t i = 0; i < rm->region_cnt; i++) {
            if (rm->rg[i].state == KV_RG_FROZEN &&
                rm->rg[i].live_bytes == 0 && rm->rg[i].close_ts != 0 &&
                cur_s > rm->rg[i].base_ts + cfg->epoch_secs) {
                trim_region(c, d, static_cast<int>(i));
            }
        }
    }

    if (cur_ms - c->last_ckpt_ms >= 2000) {
        for (int d = 0; d < c->n_devs; d++) {
            if (c->journals[d].cur_off > KV_PAGE_SIZE * 4 / 5 ||
                c->journals[d].cur_page >= c->cfg.journal_pages - 1 ||
                c->journals[d].dropped) {
                cache_checkpoint(c);
                break;
            }
        }
        c->last_ckpt_ms = cur_ms;
    }

    {
        uint64_t cap_bytes = 0, live_total = 0;
        for (int d = 0; d < c->n_devs; d++) {
            RegionMgr *rm = &c->rms[d];
            for (uint32_t i = 0; i < rm->region_cnt; i++) {
                cap_bytes += static_cast<uint64_t>(rm->region_size_pages) *
                             KV_PAGE_SIZE;
                if (rm->rg[i].state != KV_RG_FREE)
                    live_total += static_cast<uint64_t>(rm->rg[i].live_bytes);
            }
        }
        uint32_t start_pct = cfg->gc_start_pct ? cfg->gc_start_pct : 80;
        uint32_t stop_pct = cfg->gc_stop_pct ? cfg->gc_stop_pct : 70;
        uint32_t step = cfg->gc_step_gens ? cfg->gc_step_gens : 1;
        uint64_t now_s_gc = cur_s;

        bool over_start =
            cap_bytes && live_total * 100 > cap_bytes * start_pct;
        bool over_stop = cap_bytes && live_total * 100 > cap_bytes * stop_pct;

        if (over_start && !c->gc_draining) {
            c->gc_draining = 1;
            c->stat_gc_triggers.fetch_add(1);
        } else if (!over_stop && c->gc_draining) {
            c->gc_draining = 0;
        }

        if (c->gc_draining && over_stop) {
            for (uint32_t round = 0; round < step; round++) {
                uint64_t oldest = UINT64_MAX;
                for (int d = 0; d < c->n_devs; d++) {
                    RegionMgr *rm = &c->rms[d];
                    for (uint32_t i = 0; i < rm->region_cnt; i++) {
                        if (rm->rg[i].state != KV_RG_FROZEN || !rm->rg[i].epoch)
                            continue;
                        if (cfg->min_gen_age_secs &&
                            now_s_gc <
                                rm->rg[i].base_ts + cfg->min_gen_age_secs)
                            continue;
                        if (rm->rg[i].epoch < oldest)
                            oldest = rm->rg[i].epoch;
                    }
                }
                if (oldest == UINT64_MAX)
                    break;
                for (int d = 0; d < c->n_devs; d++) {
                    RegionMgr *rm = &c->rms[d];
                    for (uint32_t i = 0; i < rm->region_cnt; i++) {
                        if (rm->rg[i].state == KV_RG_FROZEN &&
                            rm->rg[i].epoch == oldest) {
                            cache_drop_region(c, &c->writers[d],
                                              static_cast<int>(i));
                            if (rm->rg[i].live_bytes == 0)
                                trim_region(c, d, static_cast<int>(i));
                        }
                    }
                }
            }
        }
    }
}

static void gc_thread_main(cache *c) {
    while (!c->gc.stop.load()) {
        if (c->ckpt_wanted.exchange(0))
            cache_checkpoint(c);
        if (now_ms() % 2000 < 210)
            gc_tick(c);
        std::this_thread::sleep_for(std::chrono::microseconds(200 * 1000));
    }
}

int gc_start(cache *c) {
    c->last_decay_ms = now_ms();
    c->last_ttl_ms = now_ms();
    c->last_ckpt_ms = now_ms();
    try {
        c->gc.tid = std::thread(gc_thread_main, c);
    } catch (...) {
        return -KV_EIO;
    }
    return KV_EOK;
}

void gc_stop(cache *c) {
    c->gc.stop.store(1);
    if (c->gc.tid.joinable())
        c->gc.tid.join();
}

int cache_force_free_region(cache *c, shard_writer *w, int avoid) {
    RegionMgr *rm = &c->rms[w->dev_id];

    for (uint32_t i = 0; i < rm->region_cnt; i++) {
        if (static_cast<int>(i) == avoid || rm->rg[i].state == KV_RG_FREE)
            continue;
        if (rm->rg[i].state == KV_RG_FROZEN && rm->rg[i].live_bytes == 0) {
            trim_region(c, w->dev_id, static_cast<int>(i));
            return static_cast<int>(i);
        }
    }

    int victim = -1;
    int64_t best = INT64_MAX;
    for (uint32_t i = 0; i < rm->region_cnt; i++) {
        if (static_cast<int>(i) == avoid || rm->rg[i].state == KV_RG_FREE)
            continue;
        if (rm->rg[i].state == KV_RG_FROZEN && rm->rg[i].live_bytes < best) {
            best = rm->rg[i].live_bytes;
            victim = static_cast<int>(i);
        }
    }
    if (victim < 0)
        return -1;
    cache_drop_region(c, w, victim);
    trim_region(c, w->dev_id, victim);
    return victim;
}

} // namespace sc
