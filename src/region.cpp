#include "region.hpp"
#include "crc32.hpp"

namespace sc {

int RegionMgr::init(Device *dev_in, uint32_t region_cnt_in,
                    uint64_t region_size_pages_in) {
    dev = dev_in;
    region_cnt = region_cnt_in;
    region_size_pages = region_size_pages_in;
    cur_open = -1;
    rg.resize(region_cnt);
    return KV_EOK;
}

void RegionMgr::destroy() {
    rg.clear();
}

static uint32_t sb_crc(const Superblock *sb) {
    uint32_t crc = crc32_partial(0, reinterpret_cast<const uint8_t *>(sb), 4);
    return crc32_partial(crc, reinterpret_cast<const uint8_t *>(sb) + 8,
                         sizeof(*sb) - 8);
}

static void sb_set_crc(Superblock *sb) { sb->crc32 = sb_crc(sb); }

int RegionMgr::sb_load(Device *dev, Superblock *sb_out) {
    SC_ALIGNED_PAGE(page0);
    SC_ALIGNED_PAGE(page1);
    auto *sb0 = reinterpret_cast<Superblock *>(page0);
    auto *sb1 = reinterpret_cast<Superblock *>(page1);

    int rc;
    int have = 0;
    Superblock *best = nullptr;

    if ((rc = dev->read_pages(0, page0, 1)) == KV_EOK &&
        sb0->magic == KV_SB_MAGIC && sb_crc(sb0) == sb0->crc32) {
        best = sb0;
        have = 1;
    }
    if ((rc = dev->read_pages(1, page1, 1)) == KV_EOK &&
        sb1->magic == KV_SB_MAGIC && sb_crc(sb1) == sb1->crc32) {
        if (!have || sb1->cur_epoch > best->cur_epoch ||
            (sb1->cur_epoch == best->cur_epoch && sb1->fmt_ts > best->fmt_ts))
            best = sb1;
        have = 1;
    }
    (void)rc;
    if (!have)
        return -KV_ENOENT;
    std::memcpy(sb_out, best, sizeof(*sb_out));
    return KV_EOK;
}

int RegionMgr::sb_write() {
    Superblock sb;
    std::memset(&sb, 0, sizeof(sb));
    sb.magic = KV_SB_MAGIC;
    sb.fmt_ts = rg[0].base_ts ? rg[0].base_ts : now_s();
    sb.dev_id = static_cast<uint16_t>(dev->dev_id);
    sb.version = 1;
    sb.region_cnt = region_cnt;
    sb.region_size_pages = region_size_pages;
    sb.region_base_page = kRegionBasePage;
    sb.journal_page = 0;
    sb.journal_pages = 0;
    sb.cur_epoch = epoch_counter;
    sb_set_crc(&sb);

    SC_ALIGNED_PAGE(page);
    std::memset(page, 0, sizeof(page));
    std::memcpy(page, &sb, sizeof(sb));

    uint64_t dst = (epoch_counter % 2 == 0) ? 0 : 1;
    return dev->write_pages(dst, page, 1);
}

static uint32_t rg_crc(const RegionHeader *h) {
    uint32_t crc = crc32_partial(0, reinterpret_cast<const uint8_t *>(h), 4);
    return crc32_partial(crc, reinterpret_cast<const uint8_t *>(h) + 8,
                         sizeof(*h) - 8);
}

int RegionMgr::write_header(int idx) {
    SC_ALIGNED_PAGE(page);
    std::memset(page, 0, sizeof(page));
    auto *h = reinterpret_cast<RegionHeader *>(page);
    const RegionHdrSh *sh = &rg[idx];
    h->magic = KV_RG_MAGIC;
    h->epoch = sh->epoch;
    h->base_ts = sh->base_ts;
    h->close_ts = sh->close_ts;
    h->watermark_page = sh->watermark_page;
    h->live_bytes_hint = static_cast<uint64_t>(sh->live_bytes < 0 ? 0
                                                                  : sh->live_bytes);
    h->state = sh->state;
    h->crc32 = rg_crc(h);
    return dev->write_pages(page_base(idx), page, 1);
}

int RegionMgr::load(Superblock *sb_out) {
    SC_ALIGNED_PAGE(buf);
    int rc = sb_load(dev, sb_out);
    if (rc != KV_EOK)
        return rc;
    if (sb_out->region_cnt != region_cnt ||
        sb_out->region_size_pages != region_size_pages)
        return -KV_EINVAL;

    for (uint32_t i = 0; i < region_cnt; i++) {
        rc = dev->read_pages(page_base(static_cast<int>(i)), buf, 1);
        if (rc != KV_EOK)
            return rc;
        auto *h = reinterpret_cast<RegionHeader *>(buf);
        if (h->magic != KV_RG_MAGIC || rg_crc(h) != h->crc32) {
            rg[i].state = KV_RG_FREE;
            rg[i].epoch = 0;
            rg[i].watermark_page = 1;
            continue;
        }
        rg[i].epoch = h->epoch;
        rg[i].base_ts = h->base_ts;
        rg[i].close_ts = h->close_ts;
        rg[i].watermark_page = h->watermark_page;
        rg[i].live_bytes = static_cast<int64_t>(h->live_bytes_hint);
        rg[i].state = h->state;
    }
    epoch_counter = sb_out->cur_epoch;
    return KV_EOK;
}

int RegionMgr::format(uint64_t cur_epoch) {
    epoch_counter = cur_epoch;
    for (uint32_t i = 0; i < region_cnt; i++) {
        rg[i].epoch = cur_epoch;
        rg[i].base_ts = 0;
        rg[i].close_ts = 0;
        rg[i].watermark_page = 1;
        rg[i].live_bytes = 0;
        rg[i].state = KV_RG_FREE;
        int rc = write_header(static_cast<int>(i));
        if (rc != KV_EOK)
            return rc;
    }
    return sb_write();
}

int RegionMgr::next_free() {
    for (uint32_t i = 0; i < region_cnt; i++)
        if (rg[i].state == KV_RG_FREE)
            return static_cast<int>(i);
    return -1;
}

} // namespace sc
