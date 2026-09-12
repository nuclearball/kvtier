# solidcacher

LLM 推理 KV-cache 的 SSD offload 库：把按 token 字符串索引的 KVCache 落到
NVMe/SSD，采用世代化 region + 零拷贝 GC（稳态 WAF ≈ 1.02），支持多盘散列、
大 chunk 条带、hot 前缀双副本与 DRAM 读缓存。核心为 **C++20**，对外提供稳定的
**C ABI**（`include/solidcacher.h`），可被 C/C++、Python(ctypes)、llama.cpp 等
直接消费。

## 特性

- 写路径：哈希路由 → MPSC 队列 → 攒批顺序写 → radix 发布 + journal。
- 读路径：radix 下行 + CRC 校验；条带并行读重组，双副本 CRC 失败回退。
- GC：TTL 过期、时驱动整区 TRIM、空间水位滞回的世代化丢弃。
- 崩溃恢复：superblock 双写 → checkpoint → journal 重放 → live_bytes 重算。
- DRAM 读缓存：SIEVE 淘汰，huge page（hugetlbfs→THP→plain）自动降级。
- 设备层：`Device` 抽象 + `FileDevice`（O_DIRECT）；`IoRing` 在 Linux 用
  io_uring，其它平台自动回退同步 IO。

## 目录结构

```
include/              公共 C ABI（solidcacher.h / model_profile.h / kv_config_*）
src/
├── common / crc32 / hash      工具
├── device / io               设备抽象 + io_uring/同步 IO
├── chunk / region            chunk 编解码、region 状态机与 superblock
├── radix / journal / cuckoo   位置哈希 radix、index journal + checkpoint、tombstone
├── hot / metrics / dram       热度衰减、指标、DRAM 读缓存
├── writer / gc               per-shard writer 线程 + GC
├── cache                     API 层组装、崩溃恢复（+ C ABI 包装）
└── model_profile.cpp         adapter 侧、模型感知的布局推导（独立库）
tests/               12 个测试，接入 CTest
examples/            demo、bench
tools/               kv_profile + 配置生成 schema/脚本
adapters/
├── sglang/          Python ctypes 绑定 + SGLang 后端 + pytest
└── llama/           llama.cpp save/restore 集成（C++，CMake）
```

## 构建与测试

依赖：C++20 编译器、CMake ≥ 3.20、pthread。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

可选开关：

- `-DSC_BUILD_TESTS=OFF` / `-DSC_BUILD_EXAMPLES=OFF` / `-DSC_BUILD_SHARED=OFF`
- `-DSC_CUCKOO=ON`：用 cuckoo filter 取代直接映射 tombstone 环
- `-DSC_ENABLE_ASAN=ON`：ASan/UBSan 构建

产物：`build/libsolidcacher.a`、`build/libsolidcacher.dylib`（Linux 为 `.so`）、
`build/libsolidcacher_mp.a`、`build/demo`、`build/bench`、`build/kv-profile`
以及各 `build/test_*`。

## 快速开始

```c
#include "solidcacher.h"

struct kv_config cfg;
kv_config_default(&cfg);

cache_t *c = NULL;
const char *uris[] = { "dev0.img", "dev1.img" };
cache_open(&c, uris, 2, &cfg);

/* async put of one 32-token group; ack fires after durability */
uint32_t tokens[32] = { /* ... */ };
struct kv_data_ref recs[1] = { { .base = payload, .off = 0, .len = len } };
cache_put(c, prefix_id, 0, tokens, 32, 0, 1, recs, ack_fn, user);

cache_close(c);
```

## 适配器

- **sglang**：`solidcacher_py` 通过 ctypes 加载 `libsolidcacher`
  （优先 `build/`，可用 `SOLIDCACHER_LIBRARY` 覆盖路径）。`pytest tests/` 运行测试。
- **llama**：`adapters/llama` 用 `-DSOLIDCACHER_ROOT=` 指向本仓库根
  （默认 `../..`），依赖已安装的 llama.cpp。

## License

MIT，见 `LICENSE`。
