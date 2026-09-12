#ifndef SC_DEVICE_HPP
#define SC_DEVICE_HPP

#include "common.hpp"

#include <memory>

namespace sc {

struct DevStats {
    uint64_t bytes_written = 0;
    uint64_t bytes_read = 0;
    uint32_t io_errs = 0;
};

struct DevGeom {
    uint32_t logical_bs = 0;
    uint32_t physical_bs = 0;
    uint32_t fs_bs = 0;
    uint32_t io_hint = 0;
};

class Device {
public:
    virtual ~Device() = default;

    virtual int open(const char *uri) = 0;
    virtual int write_pages(uint64_t page_no, const void *buf,
                            uint32_t npages) = 0;
    virtual int read_pages(uint64_t page_no, void *buf, uint32_t npages) = 0;
    virtual int trim(uint64_t page_no, uint32_t npages) = 0;
    virtual int flush() = 0;
    virtual void close() = 0;

    int fd = -1;
    int dev_id = 0;
    bool o_direct = false;
    uint64_t total_pages = 0;
    DevStats stats;
    DevGeom geom;
    char uri[256] = {};
};

class FileDevice final : public Device {
public:
    int open(const char *uri) override;
    int write_pages(uint64_t page_no, const void *buf,
                    uint32_t npages) override;
    int read_pages(uint64_t page_no, void *buf, uint32_t npages) override;
    int trim(uint64_t page_no, uint32_t npages) override;
    int flush() override;
    void close() override;
};

int dev_probe(const char *uri, DevGeom *out);

std::unique_ptr<Device> dev_make_file();
int dev_open(Device *d, const char *uri, int dev_id);

} // namespace sc

#endif
