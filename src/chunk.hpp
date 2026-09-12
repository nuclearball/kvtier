#ifndef SC_CHUNK_HPP
#define SC_CHUNK_HPP

#include "common.hpp"

namespace sc {

#pragma pack(push, 1)

struct ChunkHdr {
    uint32_t magic;
    uint32_t crc32;
    uint64_t prefix_id;
    uint64_t epoch;
    uint32_t group_idx;
    uint32_t n_tokens;
    uint32_t ver;
    uint32_t expire_ts;
    uint16_t n_records;
    uint16_t flags;
    uint32_t total_len;
    uint32_t stripe_idx;
    uint32_t stripe_cnt;
    uint32_t pad[2];
};
static_assert(sizeof(ChunkHdr) == 64, "ChunkHdr must be 64B");

struct RecordHdr {
    uint16_t layer_id;
    uint16_t _pad;
    uint32_t kv_len;
};
static_assert(sizeof(RecordHdr) == 8, "RecordHdr must be 8B");

#pragma pack(pop)

inline uint32_t chunk_len_pages(uint32_t total_len) {
    return static_cast<uint32_t>(
        page_align(KV_CHUNK_HDR_SIZE + static_cast<uint64_t>(total_len)) /
        KV_PAGE_SIZE);
}

inline uint32_t payload_npages(uint64_t payload_len) {
    return static_cast<uint32_t>((payload_len + KV_PAGE_SIZE - 1) /
                                 KV_PAGE_SIZE);
}

uint32_t pages_crc(const uint8_t *buf, uint32_t npages);
int pages_validate(const uint8_t *buf, uint32_t npages, uint32_t expected_crc);

struct ChunkBuild {
    uint64_t prefix_id = 0;
    uint64_t epoch = 0;
    uint32_t group_idx = 0;
    uint32_t n_tokens = 0;
    uint32_t ver = 0;
    uint32_t expire_ts = 0;
    uint16_t flags = 0;
    uint32_t stripe_idx = 0;
    uint32_t stripe_cnt = 0;
    const kv_data_ref *recs = nullptr;
    uint16_t n_records = 0;
    const void *slice = nullptr;
    uint32_t slice_len = 0;
};

uint32_t chunk_encode(uint8_t *out, const ChunkBuild *b);
uint32_t chunk_build_total_len(const ChunkBuild *b);
int chunk_validate(const uint8_t *buf, uint32_t npages);

struct RecDesc {
    uint16_t layer_id;
    uint32_t off, len;
};
int chunk_parse_records(const uint8_t *buf, uint32_t npages, RecDesc *recs,
                        int cap);

int stripe_split(uint64_t total_len, uint32_t unit, uint32_t *n_parts_out);

} // namespace sc

#endif
