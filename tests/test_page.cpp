/* page/payload helpers: length<->pages, page CRC validation, stripe split.
 * Also covers the low-level CRC and hash helpers used by the data plane. */
#include "page.hpp"
#include "crc32.hpp"
#include "hash.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace sc;

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static void test_crc(void) {
    uint8_t data[256];
    for (int i = 0; i < 256; i++) data[i] = (uint8_t)i;
    uint32_t c1 = crc32(data, sizeof(data));
    uint32_t c2 = crc32_partial(0, data, sizeof(data));
    CHECK(c1 == c2);
    /* single-bit corruption must change the crc */
    data[100] ^= 0x01;
    CHECK(crc32(data, sizeof(data)) != c1);
}

static void test_hash_deterministic(void) {
    uint32_t tokens[32];
    for (int i = 0; i < 32; i++) tokens[i] = (uint32_t)(i * 2654435761u);
    uint64_t h1 = hash_path(42, tokens, 32, 3);
    uint64_t h2 = hash_path(42, tokens, 32, 3);
    uint64_t h3 = hash_path(43, tokens, 32, 3);
    uint64_t h4 = hash_path(42, tokens, 31, 3);
    CHECK(h1 == h2);
    CHECK(h1 != h3);
    CHECK(h1 != h4);
}

static void test_payload_npages(void) {
    CHECK(payload_npages(0) == 0);
    CHECK(payload_npages(1) == 1);
    CHECK(payload_npages(KV_PAGE_SIZE - 1) == 1);
    CHECK(payload_npages(KV_PAGE_SIZE) == 1);
    CHECK(payload_npages(KV_PAGE_SIZE + 1) == 2);
    CHECK(payload_npages(3 * KV_PAGE_SIZE) == 3);
}

static void test_pages_crc(void) {
    uint32_t npages = 3;
    uint8_t *buf = static_cast<uint8_t *>(aligned_alloc(npages * KV_PAGE_SIZE));
    CHECK(buf != nullptr);
    memset(buf, 0x5A, npages * KV_PAGE_SIZE);

    uint32_t crc = pages_crc(buf, npages);
    CHECK(pages_validate(buf, npages, crc) == KV_EOK);

    /* corruption must fail validation */
    buf[npages * KV_PAGE_SIZE - 1] ^= 0xFF;
    CHECK(pages_validate(buf, npages, crc) == -KV_ECRC);

    /* invalid arguments */
    CHECK(pages_validate(nullptr, npages, crc) == -KV_EINVAL);
    CHECK(pages_validate(buf, 0, crc) == -KV_EINVAL);
    aligned_free(buf);
}

static void test_stripe_split(void) {
    uint32_t n = 0;
    CHECK(stripe_split(0, KV_STRIPE_UNIT, &n) == KV_EOK && n == 1);
    CHECK(stripe_split(KV_STRIPE_UNIT, KV_STRIPE_UNIT, &n) == KV_EOK && n == 1);
    CHECK(stripe_split(KV_STRIPE_UNIT + 1, KV_STRIPE_UNIT, &n) == KV_EOK &&
          n == 2);
    CHECK(stripe_split(KV_STRIPE_UNIT * 3, KV_STRIPE_UNIT, &n) == KV_EOK &&
          n == 3);
    CHECK(stripe_split(KV_STRIPE_UNIT, 0, &n) == -KV_EINVAL);
}

int main(void) {
    test_crc();
    test_hash_deterministic();
    test_payload_npages();
    test_pages_crc();
    test_stripe_split();
    if (failures == 0) {
        printf("test_page: OK\n");
        return 0;
    }
    printf("test_page: %d failures\n", failures);
    return 1;
}
