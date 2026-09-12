#ifndef SC_CRC32_HPP
#define SC_CRC32_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace sc {

namespace detail {

constexpr std::array<uint32_t, 256> make_crc_table() {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        t[i] = c;
    }
    return t;
}

inline constexpr std::array<uint32_t, 256> crc_table = make_crc_table();

} // namespace detail

inline uint32_t crc32_partial(uint32_t crc, const void *data, size_t len) {
    const auto *p = static_cast<const uint8_t *>(data);
    crc ^= 0xFFFFFFFFu;
    while (len--)
        crc = (crc >> 8) ^ detail::crc_table[(crc ^ *p++) & 0xFF];
    return crc ^ 0xFFFFFFFFu;
}

inline uint32_t crc32(const void *data, size_t len) {
    return crc32_partial(0, data, len);
}

} // namespace sc

#endif
