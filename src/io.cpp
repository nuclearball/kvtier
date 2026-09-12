#include "io.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef __linux__
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#endif

namespace sc {

#define IORING_OP_READ 22
#define IORING_OP_WRITE 23
#define IORING_ENTER_GETEVENTS (1U << 0)

struct io_sqring_offsets {
    uint32_t head, tail, ring_mask, ring_entries, flags, dropped, array, resv1;
    uint64_t user_addr;
};
struct io_cqring_offsets {
    uint32_t head, tail, ring_mask, ring_entries, overflow, cqes, flags, resv1;
    uint64_t user_addr;
};
struct io_uring_params {
    uint32_t sq_entries, cq_entries, flags, sq_thread_cpu, sq_thread_idle,
        features, wq_fd, resv[3];
    struct io_sqring_offsets sq_off;
    struct io_cqring_offsets cq_off;
};

#define IORING_OFF_SQ_RING 0ULL
#define IORING_OFF_CQ_RING 0x8000000ULL
#define IORING_OFF_SQES 0x10000000ULL

struct io_uring_sqe {
    uint8_t opcode;
    uint8_t flags;
    uint16_t ioprio;
    int32_t fd;
    uint64_t off;
    uint64_t addr;
    uint32_t len;
    uint32_t rw_flags;
    uint64_t user_data;
    uint16_t buf_index;
    uint16_t personality;
    int32_t splice_fd_in;
    uint64_t pad2[2];
};
struct io_uring_cqe {
    uint64_t user_data;
    int32_t res;
    uint32_t flags;
};

#ifdef __linux__

static int uring_setup(unsigned entries, struct io_uring_params *p) {
    return static_cast<int>(syscall(__NR_io_uring_setup, entries, p));
}
static int uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                       unsigned flags) {
    return static_cast<int>(syscall(__NR_io_uring_enter, fd, to_submit,
                                    min_complete, flags, nullptr, 0));
}

static void *mmap_sz(size_t sz, int fd, off_t off) {
    void *p = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, fd, off);
    return p == MAP_FAILED ? nullptr : p;
}

int IoRing::uring_init(int qd_in) {
    struct io_uring_params p;
    std::memset(&p, 0, sizeof(p));
    int fd = uring_setup(static_cast<unsigned>(qd_in), &p);
    if (fd < 0)
        return -1;

    ring_fd_ = fd;

    size_t sq_ring_sz = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    size_t cq_ring_sz =
        p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    size_t sqes_sz = p.sq_entries * sizeof(struct io_uring_sqe);
    if (sq_ring_sz < 4096)
        sq_ring_sz = 4096;
    if (cq_ring_sz < 4096)
        cq_ring_sz = 4096;

    sq_ring_ptr_ = mmap_sz(sq_ring_sz, fd, IORING_OFF_SQ_RING);
    cq_ring_ptr_ = mmap_sz(cq_ring_sz, fd, IORING_OFF_CQ_RING);
    sqes_ptr_ = mmap_sz(sqes_sz, fd, IORING_OFF_SQES);
    if (!sq_ring_ptr_ || !cq_ring_ptr_ || !sqes_ptr_) {
        if (sq_ring_ptr_)
            munmap(sq_ring_ptr_, sq_ring_sz);
        if (cq_ring_ptr_)
            munmap(cq_ring_ptr_, cq_ring_sz);
        if (sqes_ptr_)
            munmap(sqes_ptr_, sqes_sz);
        ::close(fd);
        return -1;
    }
    sq_ring_size_ = sq_ring_sz;
    cq_ring_size_ = cq_ring_sz;
    sqes_size_ = sqes_sz;

    auto *sqr = static_cast<uint8_t *>(sq_ring_ptr_);
    auto *cqr = static_cast<uint8_t *>(cq_ring_ptr_);
    sq_head_ = reinterpret_cast<uint32_t *>(sqr + p.sq_off.head);
    sq_tail_ = reinterpret_cast<uint32_t *>(sqr + p.sq_off.tail);
    sq_mask_ = reinterpret_cast<uint32_t *>(sqr + p.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<uint32_t *>(sqr + p.sq_off.array);
    cq_head_ = reinterpret_cast<uint32_t *>(cqr + p.cq_off.head);
    cq_tail_ = reinterpret_cast<uint32_t *>(cqr + p.cq_off.tail);
    cq_mask_ = reinterpret_cast<uint32_t *>(cqr + p.cq_off.ring_mask);
    sqes_ = static_cast<struct io_uring_sqe *>(sqes_ptr_);
    cqes_ = reinterpret_cast<struct io_uring_cqe *>(cqr + p.cq_off.cqes);
    qd = static_cast<int>(p.sq_entries);
    ring_ready_ = 1;
    return 0;
}

void IoRing::uring_exit() {
    if (!ring_ready_)
        return;
    munmap(sq_ring_ptr_, sq_ring_size_);
    munmap(cq_ring_ptr_, cq_ring_size_);
    munmap(sqes_ptr_, sqes_size_);
    ::close(ring_fd_);
    ring_ready_ = 0;
}

int IoRing::uring_submit(IoOp op, Device *d, uint64_t page_no, void *buf,
                         uint32_t npages, IoCtx *ictx) {
    uint32_t head, tail;
    for (;;) {
        head = *sq_head_;
        tail = *sq_tail_;
        if (tail - head < static_cast<uint32_t>(qd))
            break;
        if (uring_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS) < 0 &&
            errno != EINTR)
            return -1;
        uring_reap();
    }
    uint32_t idx = tail & *sq_mask_;
    struct io_uring_sqe *sqe = &sqes_[idx];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = (op == IoOp::Write) ? IORING_OP_WRITE : IORING_OP_READ;
    sqe->flags = 0;
    sqe->ioprio = 0;
    sqe->fd = d->fd;
    sqe->off = static_cast<int64_t>(page_no) * KV_PAGE_SIZE;
    sqe->addr = reinterpret_cast<uintptr_t>(buf);
    sqe->len = npages * KV_PAGE_SIZE;
    sqe->user_data = reinterpret_cast<uint64_t>(ictx);
    sq_array_[idx] = idx;
    __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
    int n = uring_enter(ring_fd_, 1, 0, 0);
    if (n < 0) {
        ictx->set_err(-errno);
        return -1;
    }
    return 0;
}

int IoRing::uring_reap() {
    int completed = 0;
    uint32_t head = *cq_head_;
    uint32_t mask = *cq_mask_;
    while (head != *cq_tail_) {
        struct io_uring_cqe *cqe = &cqes_[head & mask];
        auto *ictx = reinterpret_cast<IoCtx *>(cqe->user_data);
        if (cqe->res < 0)
            ictx->set_err(cqe->res);
        ictx->unref();
        completed++;
        head++;
        __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
    }
    return completed;
}

int IoRing::uring_wait() {
    int rc = uring_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS);
    if (rc < 0 && errno == EINTR)
        return 0;
    return rc < 0 ? -errno : 0;
}

#else

int IoRing::uring_init(int) { return -1; }
void IoRing::uring_exit() {}
int IoRing::uring_submit(IoOp, Device *, uint64_t, void *, uint32_t, IoCtx *) {
    return -1;
}
int IoRing::uring_reap() { return 0; }
int IoRing::uring_wait() { return 0; }

#endif

static void sync_submit(IoOp op, Device *d, uint64_t page_no, void *buf,
                        uint32_t npages, IoCtx *ictx) {
    int rc = (op == IoOp::Write) ? d->write_pages(page_no, buf, npages)
                                 : d->read_pages(page_no, buf, npages);
    if (rc != KV_EOK)
        ictx->set_err(rc);
    ictx->unref();
}

int IoRing::init(int qd_in) {
    if (qd_in < 8)
        qd_in = 8;
    if (qd_in > 4096)
        qd_in = 4096;
    mode = kIoSync;
    qd = qd_in;
    if (uring_init(qd_in) == 0)
        mode = kIoUring;
    else
        mode = kIoSync;
    return KV_EOK;
}

void IoRing::exit() {
    if (mode == kIoUring)
        uring_exit();
}

int IoRing::submit(IoOp op, Device *d, uint64_t page_no, void *buf,
                   uint32_t npages, IoCtx *ictx) {
    if (mode == kIoUring)
        return uring_submit(op, d, page_no, buf, npages, ictx);
    sync_submit(op, d, page_no, buf, npages, ictx);
    return 0;
}

int IoRing::reap() {
    if (mode == kIoUring)
        return uring_reap();
    return 0;
}

int IoRing::wait() {
    if (mode == kIoUring)
        return uring_wait();
    return 0;
}

int IoRing::wait_all() { return 0; }

} // namespace sc
