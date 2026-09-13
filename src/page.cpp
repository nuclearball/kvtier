#include "page.hpp"
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
