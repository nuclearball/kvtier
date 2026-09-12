#include "common.hpp"
#include "chunk.hpp"
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

static void test_chunk_roundtrip(void) {
    uint8_t rec0[100], rec1[200];
    memset(rec0, 0xA5, sizeof(rec0));
    memset(rec1, 0x5A, sizeof(rec1));
    struct kv_data_ref recs[2] = {
        { .base = rec0, .off = 0, .len = sizeof(rec0) },
        { .base = rec1, .off = 0, .len = sizeof(rec1) },
    };
    ChunkBuild b;
    memset(&b, 0, sizeof(b));
    b.prefix_id = 7;
    b.epoch = 3;
    b.group_idx = 2;
    b.n_tokens = 32;
    b.ver = 9;
    b.expire_ts = 1000;
    b.recs = recs;
    b.n_records = 2;

    uint32_t npages = chunk_len_pages(chunk_build_total_len(&b));
    CHECK(npages >= 1);
    uint8_t *buf = (uint8_t *)calloc(1, (size_t)npages * KV_PAGE_SIZE);
    uint32_t pages = chunk_encode(buf, &b);
    CHECK(pages == npages);

    /* validate */
    CHECK(chunk_validate(buf, npages) == KV_EOK);
    /* corruption must fail */
    buf[2000] ^= 0xFF;
    CHECK(chunk_validate(buf, npages) != KV_EOK);
    buf[2000] ^= 0xFF;

    /* parse records */
    struct RecDesc descs[2];
    int n = chunk_parse_records(buf, npages, descs, 2);
    CHECK(n == 2);
    CHECK(descs[0].layer_id == 0 && descs[0].len == sizeof(rec0));
    CHECK(descs[1].layer_id == 1 && descs[1].len == sizeof(rec1));
    CHECK(memcmp(buf + descs[0].off, rec0, sizeof(rec0)) == 0);
    CHECK(memcmp(buf + descs[1].off, rec1, sizeof(rec1)) == 0);
    free(buf);
}

static void test_chunk_stripe(void) {
    /* a big payload split into parts */
    uint32_t big = KV_STRIPE_UNIT * 2 + 123;
    uint8_t *data = (uint8_t *)malloc(big);
    for (uint32_t i = 0; i < big; i++) data[i] = (uint8_t)(i * 7);

    uint32_t n_parts;
    CHECK(stripe_split(big, KV_STRIPE_UNIT, &n_parts) == KV_EOK);
    CHECK(n_parts == 3);

    /* encode parts as stripe chunks and re-assemble */
    uint32_t stream_len = big;
    uint8_t *assembled = (uint8_t *)malloc(stream_len);
    for (uint32_t p = 0; p < n_parts; p++) {
        ChunkBuild b;
        memset(&b, 0, sizeof(b));
        b.prefix_id = 1;
        b.epoch = 1;
        b.group_idx = 0;
        b.n_tokens = 32;
        b.ver = 1;
        b.flags = KV_CHUNK_F_STRIPE;
        b.stripe_idx = p;
        b.stripe_cnt = n_parts;
        b.slice = data + (uint64_t)p * KV_STRIPE_UNIT;
        b.slice_len = (uint32_t)min_of<uint64_t>(KV_STRIPE_UNIT,
                          (uint64_t)big - (uint64_t)p * KV_STRIPE_UNIT);
        uint32_t npages = chunk_len_pages(b.slice_len);
        uint8_t *buf = (uint8_t *)calloc(1, (size_t)npages * KV_PAGE_SIZE);
        CHECK(chunk_encode(buf, &b) == npages);
        CHECK(chunk_validate(buf, npages) == KV_EOK);
        /* extract the slice data */
        uint32_t slice_len = (uint32_t)min_of<uint64_t>(KV_STRIPE_UNIT,
                            (uint64_t)big - (uint64_t)p * KV_STRIPE_UNIT);
        memcpy(assembled + (uint64_t)p * KV_STRIPE_UNIT,
               buf + KV_CHUNK_HDR_SIZE, slice_len);
        free(buf);
    }
    /* verify stream (only the valid bytes are meaningful) */
    for (uint32_t i = 0; i < stream_len; i++)
        CHECK(assembled[i] == data[i]);
    free(assembled);
    free(data);
}

int main(void) {
    test_crc();
    test_hash_deterministic();
    test_chunk_roundtrip();
    test_chunk_stripe();
    if (failures == 0) {
        printf("test_chunk: OK\n");
        return 0;
    }
    printf("test_chunk: %d failures\n", failures);
    return 1;
}
