#ifndef SC_HASH_HPP
#define SC_HASH_HPP

#include <cstddef>
#include <cstdint>

namespace sc {

inline uint64_t hash_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

inline uint64_t hash64(const void *data, size_t len, uint64_t seed) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xcbf29ce484222325ull ^ seed;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return hash_mix64(h);
}

inline uint64_t hash_path(uint64_t prefix_id, const uint32_t *tokens,
                          uint32_t n_tokens, uint32_t group_idx) {
    uint64_t h = 0xcbf29ce484222325ull ^ prefix_id;
    h = (h << 1) ^ static_cast<uint64_t>(group_idx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint64_t t = tokens[i];
        for (int b = 0; b < 32; b += 8) {
            h ^= (t >> b) & 0xFF;
            h *= 0x100000001b3ull;
        }
    }
    return hash_mix64(h);
}

} // namespace sc

#endif
