/* device geometry probe tests: file backend sanity + cache integration */
#include "device.hpp"
#include "solidcacher.h"
#include "cache.hpp"      /* white-box: struct device geom after cache_open */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

using namespace sc;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static bool bs_sane(uint32_t v) {
    return v == 0 || (v >= 512 && (v & (v - 1)) == 0);
}

static void test_probe_file(void) {
    const char *path = "/tmp/kvd_probe.img";
    unlink(path);
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    CHECK(ftruncate(fileno(f), 16 * 1024 * 1024) == 0);
    fclose(f);

    DevGeom g;
    CHECK(dev_probe(path, &g) == KV_EOK);
    /* file backend: device LBA/physical invisible under the FS */
    printf("[probe] file: lba=%u phys=%u fs=%u io_hint=%u\n",
           g.logical_bs, g.physical_bs, g.fs_bs, g.io_hint);
    CHECK(bs_sane(g.logical_bs));
    CHECK(bs_sane(g.physical_bs));
    CHECK(bs_sane(g.fs_bs));
    CHECK(g.io_hint >= 512);

    CHECK(dev_probe("/nonexistent/path/img", &g) == -KV_EIO);
    CHECK(dev_probe(NULL, &g) == -KV_EINVAL);
    unlink(path);
}

static void test_cache_integration(void) {
    const char *path = "/tmp/kvd_probe_cache.img";
    unlink(path);
    kv_config cfg;
    kv_config_default(&cfg);
    cfg.region_cnt = 2;
    cfg.region_size_pages = 512;
    cfg.dram_cache_bytes = 0;
    const char *uris[] = { path };
    cache_t *c = NULL;
    CHECK(cache_open(&c, uris, 1, &cfg) == KV_EOK);
    /* probed geometry is carried on the device handle */
    const Device *d = c->devs[0].get();
    printf("[probe] cache: lba=%u phys=%u fs=%u\n",
           d->geom.logical_bs, d->geom.physical_bs, d->geom.fs_bs);
    CHECK(bs_sane(d->geom.logical_bs));
    CHECK(bs_sane(d->geom.physical_bs));
    /* LBA (when visible) must fit the 4K page addressing model */
    CHECK(d->geom.logical_bs <= KV_PAGE_SIZE);
    cache_close(c);
    unlink(path);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    test_probe_file();
    test_cache_integration();
    if (failures == 0) {
        printf("test_device: OK\n");
        return 0;
    }
    printf("test_device: %d failures\n", failures);
    return 1;
}
