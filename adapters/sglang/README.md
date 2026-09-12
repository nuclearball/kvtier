# sglang adapter

sglang HiCache storage backend shim，底层接 solidcacher SSD KV cache。
当前阶段定位：**用 sglang 的分页语义验证 solidcacher**（radix 多组寻址、
longest-prefix 查询、内容寻址去重、GC 驱逐、崩溃恢复），而非生产集成。

- 接口 pin：**sglang 0.5.18**（`python/sglang/srt/mem_cache/hicache_storage.py` v1 合约）
- 与 `adapters/llama` 对称：llama 侧验证"整段 opaque blob"接入，
  本项目验证"每页多 record、页粒度 key"接入——后者才真正覆盖 solidcacher
  的 radix 多组 / stripe / per-layer 解析路径

## 构建

```bash
# 1. solidcacher 共享库（ctypes 需要 .dylib/.so，静态库不行）
cd ../solidcacher && make shared

# 2. python 环境 + 测试
python3 -m venv .venv
.venv/bin/pip install pytest numpy
.venv/bin/python -m pytest tests/ -q

# 3. 端到端 demo（不需要安装 sglang/torch）
.venv/bin/python demo_sglang_shim.py
```

库查找顺序：`Solidcacher(lib_path=...)` > `$SOLIDCACHER_LIBRARY` >
`../solidcacher/libsolidcacher.{dylib,so}` > 系统路径。

## 目录结构

```
sglang adapter（本目录）
├── solidcacher_py/            # 通用 ctypes 绑定（无 sglang 依赖）
│   ├── _binding.py            #   solidcacher.h 结构体/原型镜像、错误码
│   └── cache.py               #   Solidcacher: put(async ack)/get/evict/stats
├── sglang_backend/
│   ├── keycodec.py            #   BlobCodec（生产形态）/ TokenCodec（内容寻址）
│   └── backend.py             #   SolidcacherStorage（HiCacheStorage v1 duck-type）
├── tests/                     # 30 个用例，全部可在 macOS 跑
├── demo_sglang_shim.py        # 三阶段端到端演示
└── README.md
```

## sglang 接口映射（v0.5.18 HiCacheStorage v1）

| sglang 方法 | 语义 | 实现 |
|---|---|---|
| `set(key, value)` | 一页 flat data page | `cache_put`（1 group + 1 record），已存在则跳过（对齐 HiCacheFile fast path） |
| `get(key, target_location)` | 读进 host tensor | `cache_get` → memcpy 进 target |
| `exists` / `batch_exists` | 精确 hit/miss；**最长连续前缀语义在 sglang 调用方** | 逐 key 精确查询 |
| `clear()` | 清空 | 关闭→删设备文件→重开 |
| 驱逐 | 接口无 delete，LRU 属后端内部事务 | **交给 solidcacher GC**（被验证对象） |
| `batch_*_v2` / PoolTransfer | 新接口 | 本期不实现 |

TP/PP 分片用 `storage_config` 生成 key 后缀（对齐 HiCacheFile 的
`config_suffix`），避免跨 rank 冲突。

## 两种 key 编码

- **BlobCodec**（生产形态）：sglang 页 key 是不可逆内容 hash →
  `prefix_id = hash(key)`、合成 32 token、永远 group 0。同构于 llama
  adapter；可直接接真实 sglang。
- **TokenCodec**（验证形态）：真实 token 按页内容寻址
  `prefix_id = hash(tokens[: (k+1)*32])`、`group_idx = k`。相同前缀页在
  两个序列间自然去重（同地址、后写覆盖），分叉页完全隔离。

## 验证结论（本次发现，写给 solidcacher 作者）

1. **逻辑地址是 `(prefix_id, group_idx)`，不含 token 内容**。
   writer 的 publish 路径先查 side table（key 恰为 `(prefix_id,
   group_idx)`，writer.c:213），命中即跳过 token-hash 路径。两个内容不同
   但 prefix_id 相同的流、同 group_idx 必然互相覆盖（实测复现）。TokenCodec
   因此必须把整段前缀内容编进 prefix_id。
2. **内容寻址天然去重，代价是共享页"后写者赢"**。两序列共享前 4 页时，
   后写序列的字节成为唯一物理副本，双方读到的都是它。
3. **精确寻址，无部分命中**。leaf 必须恰好在请求深度（cache_get 的
   radix_walk 要求 `node->leaf`）；sglang 的 longest-prefix 由调用方
   `batch_exists` 逐页数出来，与 solidcacher 兼容良好。
4. **rotation 可先于 GC 触发回收**。写游标耗尽 region 空间时 writer 直接
   force-free 最老 region（实测 8 MiB 容量塞 8 MiB 数据：`drops=1` 且
   `gc_triggers=0`，前 3 页被回收）。容量规划必须留余量。
5. **错误码符号**。C 层实际返回正数幅值（枚举定义为负、返回时取负，
   如 miss 返回 3 而非 -3）；Python 绑定统一归一化为 solidcacher.h 枚举的负
   形式。
6. **put 不拷贝缓冲区**（单写/副本路径只存描述符，writer 线程稍后编码）。
   Python 侧用 pending-put 注册表持有每个 buffer 直到 ack 触发。
7. **恢复**：close/reopen 与 SIGKILL（子进程不 close）后 journal replay
   均恢复全部已 ack 的 put（tests/test_recovery.py）。统计计数器
   （puts/hits/...）是 DRAM 态，reopen 后归零属预期。

## 环境矩阵

| 环境 | 能跑 | 说明 |
|------|------|------|
| macOS（本机） | 全部测试 + demo | device.c 无 O_DIRECT、io.c 同步 fallback；设备用普通文件（sparse） |
| Linux NVMe 测试机 | 全部 + 真实性能/GC/崩溃 | io_uring + O_DIRECT 生效；性能基线（p50/p99、WAF）对照 `solidcacher/examples/bench.c`，报告入 `test-reports/` |

## 接入真实 sglang（0.5.18 调度路径已核对）

`srt/managers/cache_controller.py` 对 file 类后端的默认路径恰好是 v1 抽象
方法（`batch_get/batch_set/batch_exists`，:984/:1142/:1073），本 shim 的
签名与返回约定（get 返回 target_location 本体、每页一个 key、flat page
字节流）与之对齐；v2（`batch_*_v2`/PoolTransfer）只有 mooncake/hf3fs 等
零拷贝后端和 draft v2 注册路径才走，无需实现。

动态挂载（不改 sglang 源码）：

```bash
--hicache-storage-backend dynamic \
--hicache-storage-backend-extra-config '{
    "backend_name": "solidcacher",
    "module_path": "sglang_backend.backend",
    "class_name": "SolidcacherStorage",
    "dev_uris": ["/dev/nvme0n1"],
    "region_cnt": 6
}'
```

`SolidcacherStorage.from_config(storage_config, kwargs)` 匹配
`StorageBackendFactory._create_dynamic_backend` 的调用约定
（backend_factory.py），设置来源优先级：extra_kwargs >
storage_config.extra_config > `$SOLIDCACHER_DEV_URIS` / `$SOLIDCACHER_LIBRARY`；
非 kv_config 的键（如 `interface_v1`）被忽略。

**尚未验证**：以上只在 pytest 层核对了调用形状，还没在真实 sglang 进程
（storage IO 线程 + prefetch 并发）里运行过——这是接实机前的第一件事。

## 已知边界

- TokenCodec 页数上限 256（`KV_MAX_GROUPS`，cache.c 私有常量）→ 单
  (prefix_id, token 链) 最多 8192 token；更长会话需分链（BlobCodec 不受限）。
- `SolidcacherStorage` 是 duck-type；要塞进 sglang 的 storage-backend
  工厂时用 `as_hicache_storage()`（sglang 可导入时自动混入 ABC），或直接
  走上面的 dynamic 配置。
- 一个设备文件同时只应打开一个 `cache_t`（两个实例共写同一文件是未定义行为）。
