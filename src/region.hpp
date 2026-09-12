#ifndef SC_REGION_HPP
#define SC_REGION_HPP

#include "common.hpp"
#include "device.hpp"

#include <mutex>
#include <vector>

namespace sc {

#pragma pack(push, 1)

struct Superblock {
    uint32_t magic;
    uint32_t crc32;
    uint64_t fmt_ts;
    uint16_t dev_id;
    uint16_t version;
    uint32_t region_cnt;
    uint64_t region_size_pages;
    uint64_t region_base_page;
    uint64_t journal_page;
    uint64_t journal_pages;
    uint64_t cur_epoch;
    uint8_t pad[512 - 64];
};
static_assert(sizeof(Superblock) == 512, "superblock must be 512B");

struct RegionHeader {
    uint32_t magic;
    uint32_t crc32;
    uint64_t epoch;
    uint64_t base_ts;
    uint64_t close_ts;
    uint64_t watermark_page;
    uint64_t live_bytes_hint;
    uint8_t state;
    uint8_t pad[KV_PAGE_SIZE - 49];
};
static_assert(sizeof(RegionHeader) == KV_PAGE_SIZE,
              "region header must fill a 4K page");

#pragma pack(pop)

inline constexpr uint64_t kRegionBasePage = 2;

struct RegionHdrSh {
    uint64_t epoch = 0;
    uint64_t base_ts = 0;
    uint64_t close_ts = 0;
    uint64_t watermark_page = 1;
    int64_t live_bytes = 0;
    uint8_t state = KV_RG_FREE;
};

class RegionMgr {
public:
    Device *dev = nullptr;
    uint32_t region_cnt = 0;
    uint64_t region_size_pages = 0;
    std::vector<RegionHdrSh> rg;
    int cur_open = -1;
    uint64_t epoch_counter = 0;
    std::mutex rotate_mtx;

    ~RegionMgr() = default;

    int init(Device *dev, uint32_t region_cnt, uint64_t region_size_pages);
    void destroy();

    uint64_t page_base(int i) const {
        return kRegionBasePage + static_cast<uint64_t>(i) * region_size_pages;
    }

    int format(uint64_t cur_epoch);
    int load(Superblock *sb_out);
    int write_header(int idx);
    int sb_write();
    static int sb_load(Device *dev, Superblock *sb_out);
    int next_free();
};

} // namespace sc

#endif
