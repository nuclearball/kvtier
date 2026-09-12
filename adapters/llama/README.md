# llama adapter

llama.cpp KV cache ↔ SSD KV cache 适配层（C++ 版）。将 llama.cpp 的 per-sequence KV cache 序列化后存入 solidcacher SSD cache，支持 save / restore / evict 操作。

## 原理

```
llama_state_seq_get_data()  →  序列化的 KV state (opaque blob)
                                      ↓
                          solidcacher cache_put()
                                      ↓
                               SSD 持久化缓存

solidcacher cache_get()     →  序列化的 KV state
                                      ↓
llama_state_seq_set_data()  →  恢复到 llama context
```

不依赖 llama.cpp 内部 KV cache 布局，兼容任意模型架构和量化格式。

## 构建

```bash
# 前置依赖:
# 1. llama.cpp 已安装到 /usr/local
# 2. solidcacher 核心已编译 (在 solidcacher-cpp/build/)

cd solidcacher-cpp/adapters/llama
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

适配器默认把同级的 `solidcacher-cpp` 目录作为核心库根目录
（即 `${CMAKE_CURRENT_SOURCE_DIR}/../..`），使用其 `include/` 与
`build/libsolidcacher.*`。如果核心库在别处，可显式指定：

```bash
cmake .. -DSOLIDCACHER_ROOT=/path/to/solidcacher-cpp
```

如果本适配器通过父项目的 `add_subdirectory` 引入，且已存在 CMake
目标 `solidcacher`，则直接链接该目标。

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_INCLUDE_DIR` | 自动查找 | llama.h 路径 |
| `LLAMA_LIBRARY` | 自动查找 | libllama 路径 |
| `SOLIDCACHER_ROOT` | `../..` | solidcacher-cpp 核心根目录 |
| `SOLIDCACHER_INCLUDE_DIR` | `${SOLIDCACHER_ROOT}/include` | solidcacher.h 路径 |
| `SOLIDCACHER_LIBRARY` | `${SOLIDCACHER_ROOT}/build` | libsolidcacher 路径 |
| `LLAMAKVCACHE_BUILD_EXAMPLES` | ON | 编译示例 |
| `LLAMAKVCACHE_SHARED` | OFF | 编译为动态库 |

## API

```c
#include "llama-kvcache.h"

// 初始化
struct llama_kvcache_cfg cfg;
llama_kvcache_cfg_default(&cfg);
cfg.metrics_level = 1;

struct llama_kvcache_ctx ctx;
llama_kvcache_ctx_init(&ctx, llama_ctx,
    (const char *[]){ "/dev/nvme0n1" }, 1, &cfg);

// 保存 (异步)
llama_kvcache_save(&ctx, seq_id, tokens, n_tokens, ack_fn, user);

// 恢复 (同步)
size_t bytes = llama_kvcache_restore(&ctx, seq_id, dest_seq_id,
                                     tokens_out, cap, &n_out);

// 检查是否存在
bool exists = llama_kvcache_has(&ctx, seq_id);

// 驱逐
llama_kvcache_evict(&ctx, seq_id);

// 打印统计
llama_kvcache_stats(&ctx);

// 销毁
llama_kvcache_ctx_destroy(&ctx);
```

## 示例

```bash
# 保存并恢复 KV cache
./example_save_restore -m /path/to/model.gguf -n 99

# 带后台噪声 IO 的 save/restore 压测
./example_noise_mixed -m /path/to/model.gguf -r 20 --noise 1
```

## 与 llama-server 集成

llama-server 已有 slot save/restore 机制 (`--slot-save-path`)。本适配器可以替代文件存储：

```
llama-server slot save:
  llama_state_seq_get_data() → 序列化到文件 → 写磁盘

本适配器:
  llama_state_seq_get_data() → cache_put() → SSD (更快, 带 GC)
  llama_state_seq_set_data() ← cache_get() ← SSD
```

集成点: `llama-kv-cache.cpp` 的 `state_write` / `state_read` 方法。

## 目录结构

```
llama adapter（本目录）
├── CMakeLists.txt
├── README.md
├── include/
│   └── llama-kvcache.h          # 公共 API
├── src/
│   └── llama-kvcache.cpp        # 实现
└── examples/
    ├── example_save_restore.cpp
    └── example_noise_mixed.cpp
```
