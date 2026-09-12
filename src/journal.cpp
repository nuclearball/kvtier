#include "journal.hpp"
#include "crc32.hpp"
#include "hash.hpp"

#include <unistd.h>

namespace sc {

int journal_init(journal *j, Device *dev, uint64_t journal_page,
                 uint32_t journal_pages) {
    j->dev = dev;
    j->journal_page = journal_page;
    j->journal_pages = journal_pages;
    j->stage = static_cast<uint8_t *>(aligned_alloc(KV_PAGE_SIZE));
    if (!j->stage)
        return -KV_ENOMEM;
    std::memset(j->stage, 0, KV_PAGE_SIZE);
    return KV_EOK;
}

void journal_destroy(journal *j) {
    if (j->stage) {
        aligned_free(j->stage);
        j->stage = nullptr;
    }
}

static uint16_t jrec_n_lens(const journal_rec *r) {
    if (r->op != KV_JOP_PUT || r->n_recs == 0)
        return 0;
    return (r->flags & KV_JFLAG_UNIFORM) ? 1 : r->n_recs;
}

static uint32_t jrec_crc(const journal_rec *r, uint32_t data_crc,
                         const uint32_t *lens, const uint64_t *path) {
    uint32_t crc = crc32_partial(0, reinterpret_cast<const uint8_t *>(r), 56);
    crc = crc32_partial(crc, reinterpret_cast<const uint8_t *>(r) + 60, 4);
    if (r->op == KV_JOP_PUT)
        crc = crc32_partial(crc, &data_crc, sizeof(data_crc));
    uint16_t nl = jrec_n_lens(r);
    if (lens && nl)
        crc = crc32_partial(crc, lens, static_cast<size_t>(nl) * 4);
    if (path && r->path_len)
        crc = crc32_partial(crc, path, static_cast<size_t>(r->path_len) * 8);
    return crc;
}

static int journal_flush_page(journal *j) {
    if (j->cur_off == 0)
        return KV_EOK;
    uint64_t abs = j->journal_page + j->cur_page;
    return j->dev->write_pages(abs, j->stage, 1);
}

static int journal_next_page(journal *j) {
    int rc = journal_flush_page(j);
    if (rc != KV_EOK)
        return rc;
    j->cur_page++;
    j->cur_off = 0;
    std::memset(j->stage, 0, KV_PAGE_SIZE);
    return KV_EOK;
}

static int journal_append(journal *j, journal_rec *r, uint32_t data_crc,
                          const uint32_t *lens, const uint64_t *path) {
    r->dev_id = static_cast<uint16_t>(j->dev->dev_id);
    r->crc32 = 0;
    r->crc32 = jrec_crc(r, data_crc, lens, path);

    uint16_t nl = jrec_n_lens(r);
    uint32_t dlen = (r->op == KV_JOP_PUT) ? sizeof(uint32_t) : 0;
    uint32_t total = sizeof(*r) + dlen + static_cast<uint32_t>(nl) * 4 +
                     static_cast<uint32_t>(r->path_len) * 8;
    if (total > KV_PAGE_SIZE)
        return -KV_EINVAL;
    if (j->cur_page >= j->journal_pages) {
        j->dropped++;
        return KV_EFULL;
    }
    if (j->cur_off + total > KV_PAGE_SIZE) {
        int rc = journal_next_page(j);
        if (rc != KV_EOK)
            return rc;
        if (j->cur_page >= j->journal_pages) {
            j->dropped++;
            return KV_EFULL;
        }
    }
    uint8_t *dst = j->stage + j->cur_off;
    std::memcpy(dst, r, sizeof(*r));
    dst += sizeof(*r);
    if (dlen) {
        std::memcpy(dst, &data_crc, sizeof(data_crc));
        dst += sizeof(data_crc);
    }
    if (nl && lens) {
        std::memcpy(dst, lens, static_cast<size_t>(nl) * 4);
        dst += static_cast<size_t>(nl) * 4;
    }
    if (r->path_len && path)
        std::memcpy(dst, path, static_cast<size_t>(r->path_len) * 8);
    j->cur_off += total;
    j->bytes_written += total;
    return KV_EOK;
}

int journal_put(journal *j, uint64_t prefix_id, uint32_t group_idx,
                uint32_t ver, const kv_addr *addr, uint32_t expire_ts,
                uint32_t stripe_idx, uint32_t stripe_cnt, uint32_t flags,
                const uint32_t *rec_lens, uint16_t n_recs, uint32_t data_len,
                const uint64_t *path, uint32_t path_len) {
    journal_rec r;
    std::memset(&r, 0, sizeof(r));
    r.op = KV_JOP_PUT;
    r.prefix_id = prefix_id;
    r.group_idx = group_idx;
    r.ver = ver;
    r.expire_ts = expire_ts;
    if (addr) {
        r.addr.dev = addr->dev_id;
        r.addr.page = addr->page_no;
        r.addr.off = data_len;
        r.addr.len = addr->len_pages;
    }
    r.stripe_idx = stripe_idx;
    r.stripe_cnt = stripe_cnt;
    r.flags = flags;
    r.path_len = static_cast<uint16_t>(path_len);
    r.n_recs = n_recs;
    std::lock_guard<std::mutex> lk(j->mtx);
    return journal_append(j, &r, addr ? addr->crc : 0, rec_lens, path);
}

int journal_del(journal *j, uint64_t prefix_id, uint32_t group_idx) {
    journal_rec r;
    std::memset(&r, 0, sizeof(r));
    r.op = KV_JOP_DEL;
    r.prefix_id = prefix_id;
    r.group_idx = group_idx;
    std::lock_guard<std::mutex> lk(j->mtx);
    return journal_append(j, &r, 0, nullptr, nullptr);
}

int journal_trim(journal *j, uint32_t region_idx, uint64_t epoch) {
    journal_rec r;
    std::memset(&r, 0, sizeof(r));
    r.op = KV_JOP_TRIM;
    r.addr.page = region_idx;
    r.addr.len = static_cast<uint32_t>(epoch);
    std::lock_guard<std::mutex> lk(j->mtx);
    return journal_append(j, &r, 0, nullptr, nullptr);
}

int journal_replay(journal *j, journal_replay_fn cb, void *arg) {
    SC_ALIGNED_PAGE(page);
    for (uint32_t p = 0; p < j->journal_pages; p++) {
        int rc = j->dev->read_pages(j->journal_page + p, page, 1);
        if (rc != KV_EOK)
            return rc;
        uint32_t off = 0;
        while (off + sizeof(journal_rec) <= KV_PAGE_SIZE) {
            auto *r = reinterpret_cast<journal_rec *>(page + off);
            if (r->op == 0) {
                if (off == 0)
                    return KV_EOK;
                break;
            }
            if (r->op > KV_JOP_TRIM || r->path_len > 256 ||
                r->n_recs > KV_MAX_LAYERS_CAP)
                return KV_EOK;
            uint16_t nl = jrec_n_lens(r);
            uint32_t dlen = (r->op == KV_JOP_PUT) ? sizeof(uint32_t) : 0;
            uint32_t total = sizeof(*r) + dlen + static_cast<uint32_t>(nl) * 4 +
                             static_cast<uint32_t>(r->path_len) * 8;
            if (off + total > KV_PAGE_SIZE)
                return KV_EOK;
            uint32_t data_crc = 0;
            const uint8_t *tail = page + off + sizeof(*r);
            if (dlen) {
                std::memcpy(&data_crc, tail, sizeof(data_crc));
                tail += sizeof(data_crc);
            }
            const uint32_t *lens =
                nl ? reinterpret_cast<const uint32_t *>(tail) : nullptr;
            tail += static_cast<size_t>(nl) * 4;
            const uint64_t *path =
                r->path_len ? reinterpret_cast<const uint64_t *>(tail)
                            : nullptr;
            if (r->crc32 != jrec_crc(r, data_crc, lens, path))
                return KV_EOK;
            int done = cb(r, data_crc, lens, nl, r->addr.off, path, arg);
            if (done)
                return KV_EOK;
            off += total;
        }
    }
    return KV_EOK;
}

int journal_reset(journal *j) {
    SC_ALIGNED_PAGE(page);
    std::memset(page, 0, sizeof(page));
    std::lock_guard<std::mutex> lk(j->mtx);
    for (uint32_t p = 0; p < j->journal_pages; p++) {
        int rc = j->dev->write_pages(j->journal_page + p, page, 1);
        if (rc != KV_EOK)
            return rc;
    }
    j->cur_page = 0;
    j->cur_off = 0;
    std::memset(j->stage, 0, KV_PAGE_SIZE);
    return KV_EOK;
}

int journal_flush(journal *j) {
    std::lock_guard<std::mutex> lk(j->mtx);
    return journal_flush_page(j);
}

static int ckpt_write_rec(FILE *fp, radix_node *node, uint64_t child_hash,
                          uint64_t *count) {
    ckpt_node_rec rec;
    std::memset(&rec, 0, sizeof(rec));
    rec.type = CKPT_REC_INTERNAL;
    rec.depth = node->depth;
    rec.child_hash = child_hash;
    rec.crc32 = 0;
    rec.crc32 = crc32(reinterpret_cast<const uint8_t *>(&rec) + 4,
                      sizeof(rec) - 4 - 4);
    if (fwrite(&rec, sizeof(rec), 1, fp) != 1)
        return -KV_EIO;
    (*count)++;

    if (node->leaf) {
        radix_leaf *lf = node->leaf;
        std::memset(&rec, 0, sizeof(rec));
        if (lf->stripe) {
            rec.type = CKPT_REC_LEAF_STRIPE;
            rec.n_parts = lf->stripe->n_parts;
            rec.total_len = lf->stripe->total_len;
        } else {
            rec.type = CKPT_REC_LEAF;
        }
        rec.depth = node->depth;
        rec.child_hash = child_hash;
        rec.prefix_id = lf->prefix_id;
        rec.group_idx = node->depth ? node->depth - 1 : 0;
        rec.ver = lf->ver.load();
        rec.expire_ts = lf->addr[0].expire_ts;
        rec.last_write_ts = lf->addr[0].last_write_ts;
        rec.shape_id = lf->shape ? lf->shape->id : KV_SHAPE_NONE;
        std::memcpy(rec.addr, lf->addr, sizeof(rec.addr));
        rec.crc32 = 0;
        rec.crc32 = crc32(reinterpret_cast<const uint8_t *>(&rec) + 4,
                          sizeof(rec) - 4 - 4);
        if (fwrite(&rec, sizeof(rec), 1, fp) != 1)
            return -KV_EIO;
        (*count)++;

        if (rec.type == CKPT_REC_LEAF_STRIPE) {
            kv_stripe_part *parts = stripe_parts(lf->stripe);
            if (fwrite(parts, sizeof(kv_stripe_part), lf->stripe->n_parts,
                       fp) != lf->stripe->n_parts)
                return -KV_EIO;
        }
    }

    for (int i = 0; i < KV_INLINE_SLOTS; i++) {
        radix_edge *e = &node->slots[i];
        if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED)) {
            if (ckpt_write_rec(fp, e->child, e->child_hash, count) != KV_EOK)
                return -KV_EIO;
        }
    }
    if (node->ovf) {
        for (uint32_t i = 0; i < node->ovf->cap; i++) {
            radix_edge *e = &node->ovf->slots[i];
            if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED)) {
                if (ckpt_write_rec(fp, e->child, e->child_hash, count) !=
                    KV_EOK)
                    return -KV_EIO;
            }
        }
    }
    return KV_EOK;
}

int ckpt_write(const char *path, radix_tree *t, uint64_t ver_counter,
               uint32_t n_devs, uint32_t region_cnt, uint64_t region_size_pages,
               shape_pool *shapes) {
    char tmp[512];
    std::snprintf(tmp, sizeof(tmp), "%s.tmp", path);

    FILE *fp = fopen(tmp, "wb");
    if (!fp)
        return -KV_EIO;

    ckpt_hdr hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic = KV_CP_MAGIC;
    hdr.version = 2;
    hdr.n_devs = n_devs;
    hdr.region_cnt = region_cnt;
    hdr.region_size_pages = region_size_pages;
    hdr.ver_counter = ver_counter;
    hdr.n_recs = 0;
    hdr.crc32 = 0;
    hdr.pad = static_cast<uint32_t>(shapes ? shapes->items.size() : 0);
    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        fclose(fp);
        return -KV_EIO;
    }

    if (shapes) {
        for (size_t s = 0; s < shapes->items.size(); s++) {
            kv_shape *sh = shapes->items[s];
            uint32_t n = sh->n_recs;
            if (fwrite(&n, sizeof(n), 1, fp) != 1 ||
                (n && fwrite(shape_lens(sh), sizeof(uint32_t), n, fp) != n)) {
                fclose(fp);
                return -KV_EIO;
            }
        }
    }

    uint64_t count = 0;
    int rc = KV_EOK;
    for (int i = 0; i < KV_INLINE_SLOTS; i++) {
        radix_edge *e = &t->root->slots[i];
        if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED)) {
            if (ckpt_write_rec(fp, e->child, e->child_hash, &count) != KV_EOK) {
                rc = -KV_EIO;
                goto out;
            }
        }
    }
    if (t->root->ovf) {
        for (uint32_t i = 0; i < t->root->ovf->cap; i++) {
            radix_edge *e = &t->root->ovf->slots[i];
            if (e->child_hash && !(e->flags & KV_EDGE_F_DELETED)) {
                if (ckpt_write_rec(fp, e->child, e->child_hash, &count) !=
                    KV_EOK) {
                    rc = -KV_EIO;
                    goto out;
                }
            }
        }
    }
    hdr.n_recs = count;
    hdr.crc32 =
        crc32(reinterpret_cast<const uint8_t *>(&hdr) + 4, sizeof(hdr) - 4 - 8);
    if (fseek(fp, 0, SEEK_SET) != 0) {
        rc = -KV_EIO;
        goto out;
    }
    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        rc = -KV_EIO;
        goto out;
    }
    if (fflush(fp) != 0) {
        rc = -KV_EIO;
        goto out;
    }
    if (fsync(fileno(fp)) != 0) {
        rc = -KV_EIO;
        goto out;
    }
    if (fclose(fp) != 0) {
        rc = -KV_EIO;
        goto out;
    }
    fp = nullptr;

    if (rename(tmp, path) != 0) {
        rc = -KV_EIO;
        goto out;
    }
    return KV_EOK;
out:
    if (fp)
        fclose(fp);
    return rc;
}

int ckpt_load(const char *path, radix_tree *t, uint64_t *ver_counter,
              uint32_t n_devs, uint32_t region_cnt, uint64_t region_size_pages,
              shape_pool *shapes, ckpt_leaf_apply apply, void *arg) {
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return -KV_ENOENT;

    ckpt_hdr hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fclose(fp);
        return -KV_EIO;
    }
    if (hdr.magic != KV_CP_MAGIC ||
        hdr.crc32 !=
            crc32(reinterpret_cast<const uint8_t *>(&hdr) + 4,
                  sizeof(hdr) - 4 - 8)) {
        fclose(fp);
        return -KV_ECRC;
    }
    if (hdr.n_devs != n_devs || hdr.region_cnt != region_cnt ||
        hdr.region_size_pages != region_size_pages) {
        fclose(fp);
        return -KV_EINVAL;
    }
    if (ver_counter)
        *ver_counter = hdr.ver_counter;

    if (hdr.pad) {
        if (!shapes) {
            fclose(fp);
            return -KV_EINVAL;
        }
        uint32_t lens[KV_MAX_LAYERS_CAP];
        for (uint32_t s = 0; s < hdr.pad; s++) {
            uint32_t n = 0;
            if (fread(&n, sizeof(n), 1, fp) != 1 || n > KV_MAX_LAYERS_CAP) {
                fclose(fp);
                return -KV_EIO;
            }
            if (n && fread(lens, sizeof(uint32_t), n, fp) != n) {
                fclose(fp);
                return -KV_EIO;
            }
            shape_pool_intern(shapes, lens, static_cast<uint16_t>(n));
        }
    }

    radix_node **stack =
        static_cast<radix_node **>(calloc(128, sizeof(radix_node *)));
    if (!stack) {
        fclose(fp);
        return -KV_ENOMEM;
    }
    int sp = 0;
    stack[sp++] = t->root;

    for (uint64_t i = 0; i < hdr.n_recs; i++) {
        ckpt_node_rec rec;
        if (fread(&rec, sizeof(rec), 1, fp) != 1) {
            fclose(fp);
            free(stack);
            return -KV_EIO;
        }
        if (rec.crc32 !=
            crc32(reinterpret_cast<const uint8_t *>(&rec) + 4,
                  sizeof(rec) - 4 - 4)) {
            fclose(fp);
            free(stack);
            return -KV_ECRC;
        }

        if (rec.type == CKPT_REC_INTERNAL) {
            uint32_t d = rec.depth;
            while (sp > static_cast<int>(d))
                sp--;
            if (sp == 0) {
                fclose(fp);
                free(stack);
                return -KV_EIO;
            }
            radix_node *parent = stack[d - 1];
            auto *child = static_cast<radix_node *>(calloc(1, sizeof(radix_node)));
            if (!child) {
                fclose(fp);
                free(stack);
                return -KV_ENOMEM;
            }
            child->depth = d;
            child->parent = parent;
            radix_edge e{rec.child_hash, child, 0};
            if (!parent->ovf_flag) {
                int placed = 0;
                for (int k = 0; k < KV_INLINE_SLOTS; k++) {
                    if (parent->slots[k].child_hash == 0) {
                        parent->slots[k] = e;
                        parent->n_edges++;
                        placed = 1;
                        break;
                    }
                }
                if (!placed) {
                    auto *o = static_cast<radix_ovf *>(calloc(1, sizeof(radix_ovf)));
                    parent->ovf = o;
                    o->cap = 16;
                    o->slots =
                        static_cast<radix_edge *>(calloc(o->cap, sizeof(radix_edge)));
                    if (!o->slots) {
                        fclose(fp);
                        free(stack);
                        free(child);
                        return -KV_ENOMEM;
                    }
                    for (int k = 0; k < KV_INLINE_SLOTS; k++) {
                        if (parent->slots[k].child_hash) {
                            uint32_t m = o->cap - 1;
                            for (uint32_t j = hash_mix64(parent->slots[k].child_hash) & m;;
                                 j = (j + 1) & m) {
                                if (o->slots[j].child_hash == 0) {
                                    o->slots[j] = parent->slots[k];
                                    o->used++;
                                    break;
                                }
                            }
                        }
                    }
                    std::memset(parent->slots, 0, sizeof(parent->slots));
                    parent->ovf_flag = 1;
                    uint32_t m = o->cap - 1;
                    for (uint32_t j = hash_mix64(e.child_hash) & m;; j = (j + 1) & m) {
                        if (o->slots[j].child_hash == 0) {
                            o->slots[j] = e;
                            o->used++;
                            break;
                        }
                    }
                    parent->n_edges++;
                }
            } else {
                radix_ovf *o = parent->ovf;
                if ((o->used + o->n_del + 1) > o->cap * 7 / 10) {
                    uint32_t new_cap = o->cap * 2;
                    auto *slots =
                        static_cast<radix_edge *>(calloc(new_cap, sizeof(radix_edge)));
                    if (!slots) {
                        fclose(fp);
                        free(stack);
                        free(child);
                        return -KV_ENOMEM;
                    }
                    for (uint32_t k = 0; k < o->cap; k++) {
                        radix_edge *oe = &o->slots[k];
                        if (oe->child_hash && !(oe->flags & KV_EDGE_F_DELETED)) {
                            uint32_t m = new_cap - 1;
                            for (uint32_t j = hash_mix64(oe->child_hash) & m;;
                                 j = (j + 1) & m) {
                                if (slots[j].child_hash == 0) {
                                    slots[j] = *oe;
                                    break;
                                }
                            }
                        }
                    }
                    free(o->slots);
                    o->slots = slots;
                    o->cap = new_cap;
                    o->n_del = 0;
                }
                uint32_t m = o->cap - 1;
                for (uint32_t j = hash_mix64(e.child_hash) & m;; j = (j + 1) & m) {
                    if (o->slots[j].child_hash == 0 ||
                        (o->slots[j].flags & KV_EDGE_F_DELETED)) {
                        if (o->slots[j].child_hash == 0)
                            o->used++;
                        else
                            o->n_del--;
                        o->slots[j] = e;
                        break;
                    }
                }
                parent->n_edges++;
            }
            if (sp < 128)
                stack[sp++] = child;
        } else if (rec.type == CKPT_REC_LEAF ||
                   rec.type == CKPT_REC_LEAF_STRIPE) {
            uint32_t d = rec.depth;
            while (sp > static_cast<int>(d + 1))
                sp--;
            if (sp < static_cast<int>(d + 1)) {
                fclose(fp);
                free(stack);
                return -KV_EIO;
            }
            radix_node *node = stack[d];
            auto *lf = static_cast<radix_leaf *>(calloc(1, sizeof(radix_leaf)));
            if (!lf) {
                fclose(fp);
                free(stack);
                return -KV_ENOMEM;
            }
            lf->refcnt.store(0);
            lf->ver.store(rec.ver);
            lf->prefix_id = rec.prefix_id;
            std::memcpy(lf->addr, rec.addr, sizeof(lf->addr));
            if (shapes && rec.shape_id != KV_SHAPE_NONE &&
                rec.shape_id < shapes->items.size())
                lf->shape = shapes->items[rec.shape_id];
            if (rec.type == CKPT_REC_LEAF_STRIPE) {
                auto *st = static_cast<kv_stripe_addr *>(
                    malloc(sizeof(kv_stripe_addr) +
                           static_cast<size_t>(rec.n_parts) *
                               sizeof(kv_stripe_part)));
                if (!st) {
                    fclose(fp);
                    free(stack);
                    free(lf);
                    return -KV_ENOMEM;
                }
                st->n_parts = rec.n_parts;
                st->total_len = rec.total_len;
                if (fread(stripe_parts(st), sizeof(kv_stripe_part), rec.n_parts,
                          fp) != rec.n_parts) {
                    fclose(fp);
                    free(stack);
                    free(lf);
                    free(st);
                    return -KV_EIO;
                }
                lf->stripe = st;
            }
            node->leaf = lf;
            if (apply) {
                ckpt_leaf_snapshot snap;
                std::memset(&snap, 0, sizeof(snap));
                snap.prefix_id = rec.prefix_id;
                snap.group_idx = rec.group_idx;
                snap.ver = rec.ver;
                snap.expire_ts = rec.expire_ts;
                snap.last_write_ts = rec.last_write_ts;
                std::memcpy(snap.addr, rec.addr, sizeof(snap.addr));
                snap.n_parts = rec.n_parts;
                snap.total_len = rec.total_len;
                snap.parts = nullptr;
                apply(node, &snap, arg);
            }
            t->n_leaves.fetch_add(1);
        } else {
            fclose(fp);
            free(stack);
            return -KV_EINVAL;
        }
    }

    fclose(fp);
    free(stack);
    return KV_EOK;
}

} // namespace sc
