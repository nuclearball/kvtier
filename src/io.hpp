#ifndef SC_IO_HPP
#define SC_IO_HPP

#include "common.hpp"
#include "device.hpp"

#include <atomic>

namespace sc {

struct IoCtx {
    std::atomic<int> inflight{0};
    std::atomic<int> first_err{0};
    void (*done)(IoCtx *) = nullptr;
    void *arg = nullptr;

    void ref() { inflight.fetch_add(1, std::memory_order_relaxed); }

    void unref() {
        if (inflight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (done)
                done(this);
        }
    }

    void set_err(int err) {
        int old = first_err.load(std::memory_order_relaxed);
        while (old == KV_EOK || old > err) {
            if (first_err.compare_exchange_weak(old, err,
                                                std::memory_order_relaxed))
                break;
        }
    }
};

enum class IoOp { Read, Write };

inline constexpr int kIoUring = 1;
inline constexpr int kIoSync = 2;

class IoRing {
public:
    int init(int qd);
    void exit();
    int submit(IoOp op, Device *d, uint64_t page_no, void *buf, uint32_t npages,
               IoCtx *ictx);
    int reap();
    int wait();
    int wait_all();

    int mode = kIoSync;
    int qd = 0;

private:
#ifdef __linux__
    int ring_fd_ = -1;
    uint32_t *sq_head_ = nullptr, *sq_tail_ = nullptr, *sq_mask_ = nullptr,
             *sq_array_ = nullptr;
    uint32_t *cq_head_ = nullptr, *cq_tail_ = nullptr, *cq_mask_ = nullptr;
    struct io_uring_sqe *sqes_ = nullptr;
    struct io_uring_cqe *cqes_ = nullptr;
    void *sq_ring_ptr_ = nullptr, *cq_ring_ptr_ = nullptr, *sqes_ptr_ = nullptr;
    size_t sq_ring_size_ = 0, cq_ring_size_ = 0, sqes_size_ = 0;
    int ring_ready_ = 0;
#endif

    int uring_init(int qd);
    void uring_exit();
    int uring_submit(IoOp op, Device *d, uint64_t page_no, void *buf,
                     uint32_t npages, IoCtx *ictx);
    int uring_reap();
    int uring_wait();
};

} // namespace sc

#endif
