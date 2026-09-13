#ifndef SC_PAGE_HPP
#define SC_PAGE_HPP

#include "common.hpp"

namespace sc {

/* Page/payload helpers for the data plane.  Solidcacher stores raw,
 * page-aligned payloads on disk; framing metadata (address, per-layer
 * lengths, CRC, version) lives in the radix index and the journal, not in
 * an on-disk header. */

inline uint32_t payload_npages(uint64_t payload_len) {
    return static_cast<uint32_t>((payload_len + KV_PAGE_SIZE - 1) /
                                 KV_PAGE_SIZE);
}

uint32_t pages_crc(const uint8_t *buf, uint32_t npages);
int pages_validate(const uint8_t *buf, uint32_t npages, uint32_t expected_crc);

int stripe_split(uint64_t total_len, uint32_t unit, uint32_t *n_parts_out);

} // namespace sc

#endif
