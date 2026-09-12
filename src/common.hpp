#ifndef SC_COMMON_HPP
#define SC_COMMON_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "kvtier.h"

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "kvtier requires a little-endian host"
#endif

namespace sc {

inline constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + (a - 1)) & ~(a - 1);
}
inline constexpr uint64_t page_align(uint64_t v) {
    return align_up(v, KV_PAGE_SIZE);
}

template <typename T>
inline constexpr T min_of(T a, T b) { return a < b ? a : b; }
template <typename T>
inline constexpr T max_of(T a, T b) { return a > b ? a : b; }

inline uint32_t byteswap32(uint32_t v) { return __builtin_bswap32(v); }
inline uint64_t byteswap64(uint64_t v) { return __builtin_bswap64(v); }

inline uint16_t le16(const void *p) {
    const auto *b = static_cast<const uint8_t *>(p);
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
}
inline uint32_t le32(const void *p) {
    const auto *b = static_cast<const uint8_t *>(p);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
}
inline uint64_t le64(const void *p) {
    const auto *b = static_cast<const uint8_t *>(p);
    return static_cast<uint64_t>(le32(b)) |
           (static_cast<uint64_t>(le32(b + 4)) << 32);
}

inline void put_le16(void *p, uint16_t v) {
    auto *b = static_cast<uint8_t *>(p);
    b[0] = static_cast<uint8_t>(v);
    b[1] = static_cast<uint8_t>(v >> 8);
}
inline void put_le32(void *p, uint32_t v) {
    auto *b = static_cast<uint8_t *>(p);
    b[0] = static_cast<uint8_t>(v);
    b[1] = static_cast<uint8_t>(v >> 8);
    b[2] = static_cast<uint8_t>(v >> 16);
    b[3] = static_cast<uint8_t>(v >> 24);
}
inline void put_le64(void *p, uint64_t v) {
    auto *b = static_cast<uint8_t *>(p);
    put_le32(b, static_cast<uint32_t>(v));
    put_le32(b + 4, static_cast<uint32_t>(v >> 32));
}

inline int clz(uint64_t x) { return __builtin_clzll(x); }
inline int ctz(uint32_t x) { return __builtin_ctz(x); }
inline bool pow2(uint64_t x) { return x && !(x & (x - 1)); }

void *aligned_alloc(size_t sz);
void aligned_free(void *p);

uint64_t now_ns();
uint64_t now_ms();
uint64_t now_s();

[[noreturn]] void fatal(const char *fmt, ...);

} // namespace sc

#define SC_ALIGNED_PAGE(name) \
    alignas(KV_PAGE_SIZE) uint8_t name[KV_PAGE_SIZE]

#endif
