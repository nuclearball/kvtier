#include "chunk.hpp"
#include "crc32.hpp"

namespace sc {

uint32_t pages_crc(const uint8_t *buf, uint32_t npages) {
    return crc32(buf, static_cast<size_t>(npages) * KV_PAGE_SIZE);
}

int pages_validate(const uint8_t *buf, uint32_t npages, uint32_t expected_crc) {
    if (!buf || npages == 0)
        return -KV_EINVAL;
    return pages_crc(buf, npages) == expected_crc ? KV_EOK : -KV_ECRC;
}

uint32_t chunk_build_total_len(const ChunkBuild *b) {
    if (b->flags & KV_CHUNK_F_STRIPE)
        return b->slice_len;
    uint64_t t = 0;
    for (int i = 0; i < b->n_records; i++)
        t += sizeof(RecordHdr) + b->recs[i].len;
    return static_cast<uint32_t>(t);
}

uint32_t chunk_encode(uint8_t *out, const ChunkBuild *b) {
    auto *hdr = reinterpret_cast<ChunkHdr *>(out);
    uint32_t total_len = chunk_build_total_len(b);
    uint32_t npages = chunk_len_pages(total_len);

    std::memset(out, 0, static_cast<size_t>(npages) * KV_PAGE_SIZE);

    hdr->magic = KV_CK_MAGIC;
    hdr->crc32 = 0;
    hdr->prefix_id = b->prefix_id;
    hdr->epoch = b->epoch;
    hdr->group_idx = b->group_idx;
    hdr->n_tokens = b->n_tokens;
    hdr->ver = b->ver;
    hdr->expire_ts = b->expire_ts;
    hdr->flags = b->flags;
    hdr->total_len = total_len;
    hdr->stripe_idx = b->stripe_idx;
    hdr->stripe_cnt = b->stripe_cnt;

    uint8_t *dst = out + KV_CHUNK_HDR_SIZE;
    if (b->flags & KV_CHUNK_F_STRIPE) {
        if (b->slice && b->slice_len)
            std::memcpy(dst, b->slice, b->slice_len);
        hdr->n_records = 0;
    } else {
        hdr->n_records = b->n_records;
        for (int i = 0; i < b->n_records; i++) {
            const kv_data_ref *r = &b->recs[i];
            auto *rh = reinterpret_cast<RecordHdr *>(dst);
            rh->layer_id = static_cast<uint16_t>(i);
            rh->_pad = 0;
            rh->kv_len = r->len;
            if (r->len)
                std::memcpy(dst + sizeof(*rh),
                            static_cast<const uint8_t *>(r->base) + r->off,
                            r->len);
            dst += sizeof(*rh) + r->len;
        }
    }

    hdr->crc32 = crc32(out, static_cast<size_t>(npages) * KV_PAGE_SIZE);
    return npages;
}

int chunk_validate(const uint8_t *buf, uint32_t npages) {
    const auto *hdr = reinterpret_cast<const ChunkHdr *>(buf);
    if (hdr->magic != KV_CK_MAGIC)
        return -KV_ECRC;
    if (npages != chunk_len_pages(hdr->total_len))
        return -KV_ECRC;
    uint8_t tmp[64];
    std::memcpy(tmp, buf, sizeof(tmp));
    reinterpret_cast<ChunkHdr *>(tmp)->crc32 = 0;
    uint32_t crc = crc32_partial(0, tmp, sizeof(tmp));
    crc = crc32_partial(crc, buf + sizeof(tmp),
                        static_cast<size_t>(npages) * KV_PAGE_SIZE -
                            sizeof(tmp));
    return crc == hdr->crc32 ? KV_EOK : -KV_ECRC;
}

int chunk_parse_records(const uint8_t *buf, uint32_t npages, RecDesc *recs,
                        int cap) {
    const auto *hdr = reinterpret_cast<const ChunkHdr *>(buf);
    uint32_t avail = hdr->total_len;
    uint32_t off = KV_CHUNK_HDR_SIZE;
    int n = 0;
    for (uint32_t i = 0; i < hdr->n_records; i++) {
        if (off + sizeof(RecordHdr) > KV_CHUNK_HDR_SIZE + avail)
            return -KV_EINVAL;
        const auto *rh = reinterpret_cast<const RecordHdr *>(buf + off);
        uint16_t layer = rh->layer_id;
        uint32_t len = rh->kv_len;
        off += sizeof(*rh);
        if (off + len > KV_CHUNK_HDR_SIZE + avail)
            return -KV_EINVAL;
        if (n < cap) {
            recs[n].layer_id = layer;
            recs[n].off = off;
            recs[n].len = len;
        }
        n++;
        off += len;
    }
    (void)npages;
    return n;
}

int stripe_split(uint64_t total_len, uint32_t unit, uint32_t *n_parts_out) {
    if (unit == 0)
        return -KV_EINVAL;
    uint32_t n = static_cast<uint32_t>((total_len + unit - 1) / unit);
    if (n == 0)
        n = 1;
    *n_parts_out = n;
    return KV_EOK;
}

} // namespace sc
