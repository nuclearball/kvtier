#include "device.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/falloc.h>
#include <linux/fs.h>
#endif
#ifdef __APPLE__
#include <sys/disk.h>
#endif

namespace sc {

int FileDevice::open(const char *uri) {
    int flags = O_RDWR | O_CLOEXEC;
#ifdef O_DIRECT
    int fd = ::open(uri, flags | O_DIRECT);
    o_direct = (fd >= 0);
    if (fd < 0)
        fd = ::open(uri, flags);
#else
    int fd = ::open(uri, flags);
    o_direct = false;
#endif
    if (fd < 0)
        return -errno;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        int err = -errno;
        ::close(fd);
        return err;
    }
    if (!S_ISREG(st.st_mode)) {
        ::close(fd);
        return -KV_EINVAL;
    }
    this->fd = fd;
    total_pages = static_cast<uint64_t>(st.st_size) / KV_PAGE_SIZE;
    if (total_pages < 8) {
        ::close(fd);
        this->fd = -1;
        return -KV_EINVAL;
    }
    return KV_EOK;
}

int FileDevice::write_pages(uint64_t page_no, const void *buf,
                            uint32_t npages) {
    size_t len = static_cast<size_t>(npages) * KV_PAGE_SIZE;
    uint64_t off = page_no * KV_PAGE_SIZE;
    const auto *p = static_cast<const uint8_t *>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, p + done, len - done,
                           static_cast<off_t>(off + done));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            stats.io_errs++;
            return -errno;
        }
        done += static_cast<size_t>(n);
    }
    stats.bytes_written += len;
    return KV_EOK;
}

int FileDevice::read_pages(uint64_t page_no, void *buf, uint32_t npages) {
    size_t len = static_cast<size_t>(npages) * KV_PAGE_SIZE;
    uint64_t off = page_no * KV_PAGE_SIZE;
    auto *p = static_cast<uint8_t *>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, p + done, len - done,
                          static_cast<off_t>(off + done));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            stats.io_errs++;
            return -errno;
        }
        if (n == 0)
            break;
        done += static_cast<size_t>(n);
    }
    stats.bytes_read += done;
    return done == len ? KV_EOK : -KV_EREAD;
}

int FileDevice::trim(uint64_t page_no, uint32_t npages) {
    int rc = KV_EOK;
#ifdef __linux__
    off_t off = static_cast<off_t>(page_no) * KV_PAGE_SIZE;
    off_t len = static_cast<off_t>(npages) * KV_PAGE_SIZE;
    if (fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, off, len) < 0)
        rc = -errno;
#else
    (void)page_no;
    (void)npages;
#endif
    return rc;
}

int FileDevice::flush() {
#ifdef __APPLE__
    if (fcntl(fd, F_FULLFSYNC) < 0)
        return -errno;
#else
    if (fsync(fd) < 0)
        return -errno;
#endif
    return KV_EOK;
}

void FileDevice::close() {
    if (fd >= 0)
        ::close(fd);
    fd = -1;
}

static bool bs_valid(uint32_t v) {
    return v == 0 || (v >= 512 && (v & (v - 1)) == 0);
}

int dev_probe(const char *uri, DevGeom *out) {
    if (!uri || !out)
        return -KV_EINVAL;
    std::memset(out, 0, sizeof(*out));

    struct stat st;
    if (stat(uri, &st) != 0)
        return -KV_EIO;
    out->io_hint = static_cast<uint32_t>(st.st_blksize);
    if (!bs_valid(out->io_hint))
        out->io_hint = 0;

    struct statvfs vfs;
    if (statvfs(uri, &vfs) == 0) {
        out->fs_bs = static_cast<uint32_t>(vfs.f_bsize);
        if (!bs_valid(out->fs_bs))
            out->fs_bs = 0;
    }

    if (S_ISBLK(st.st_mode) || S_ISCHR(st.st_mode)) {
#ifdef __linux__
        int fd = ::open(uri, O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            int lba = 0, phys = 0;
            if (ioctl(fd, BLKSSZGET, &lba) == 0 && lba > 0)
                out->logical_bs = static_cast<uint32_t>(lba);
            if (ioctl(fd, BLKPBSZGET, &phys) == 0 && phys > 0)
                out->physical_bs = static_cast<uint32_t>(phys);
            ::close(fd);
        }
#endif
#ifdef __APPLE__
        int fd = ::open(uri, O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            uint32_t bs = 0;
            if (ioctl(fd, DKIOCGETBLOCKSIZE, &bs) == 0 && bs > 0) {
                out->logical_bs = bs;
                uint64_t cnt = 0;
                if (ioctl(fd, DKIOCGETBLOCKCOUNT, &cnt) == 0 && cnt)
                    out->physical_bs = bs;
            }
            ::close(fd);
        }
#endif
    }
    if (!bs_valid(out->logical_bs) || !bs_valid(out->physical_bs))
        return -KV_EIO;
    return KV_EOK;
}

std::unique_ptr<Device> dev_make_file() {
    return std::make_unique<FileDevice>();
}

int dev_open(Device *d, const char *uri, int dev_id) {
    d->fd = -1;
    d->dev_id = dev_id;
    std::snprintf(d->uri, sizeof(d->uri), "%s", uri);
    int rc = d->open(uri);
    if (rc != KV_EOK)
        return rc;
    DevGeom g;
    if (dev_probe(uri, &g) == KV_EOK)
        d->geom = g;
    return KV_EOK;
}

} // namespace sc
