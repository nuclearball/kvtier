# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- DRAM read cache: free size-class blocks allocated by the arena-exhaustion
  fallback so `dram_destroy` no longer leaks them (fixes the LeakSanitizer
  failure in `test_dram`).

## [0.1.0] - 2026-09-12

Initial release.

### Added

- C++20 core with a stable C ABI (`include/kvtier.h`).
- Multi-device region management with alternated superblock writes and CRC.
- Append-only chunk format (header + per-layer records + CRC32) and striping.
- Per-shard writer threads with MPSC queues and group-commit batching; async
  `put` with ack-after-durability.
- Position-hash radix index with journal + checkpoint crash recovery.
- Generational GC: watermark hysteresis, FRU oldest-generation discard, TTL
  expiry, whole-region TRIM, reader pins, and evicted-key tombstones.
- DRAM read cache with SIEVE eviction and huge-page degradation
  (`hugetlbfs` → THP → plain).
- Optional cuckoo-filter tombstones (`-DSC_CUCKOO=ON`).
- Device abstraction (`FileDevice`, `O_DIRECT`) with an `io_uring` I/O ring on
  Linux and a synchronous fallback elsewhere.
- Model-aware layout module (`model_profile`) with a `kv-profile` CLI.
- SGLang (Python ctypes) and llama.cpp adapters.
- `demo` and `bench` tools, and 12 CTest suites.
- CMake install rules and `find_package(kvtier)` package config.

[Unreleased]: https://github.com/nuclearball/kvtier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nuclearball/kvtier/releases/tag/v0.1.0
