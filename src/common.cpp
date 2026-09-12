#include "common.hpp"

#include <cstdarg>
#include <ctime>

namespace sc {

void fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fputs("kvtier: ", stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
    std::abort();
}

void *aligned_alloc(size_t sz) {
    void *p = nullptr;
    if (posix_memalign(&p, KV_PAGE_SIZE, sz) != 0)
        return nullptr;
    return p;
}

void aligned_free(void *p) {
    std::free(p);
}

uint64_t now_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000u +
           static_cast<uint64_t>(ts.tv_nsec) / 1000000u;
}

uint64_t now_s() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec);
}

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

} // namespace sc
