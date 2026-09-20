# 07 参考：配置、接口、上限与构建

## 1. 编译与安装

PGXS 构建，混合三种语言：C（PostgreSQL 胶水）、C++17（引擎）、可选 CUDA（GPU 算子）。

```bash
cd extension/paves
make             # 默认目标为 all
make install
```

Makefile 显式设置默认目标为 `all`，裸 `make` 会构建完整扩展。

**GPU 支持自动探测**：`command -v nvcc` 找得到（或 `/usr/local/cuda*/bin/nvcc`
匹配得上）就把 `gpu_engine.o` 加入目标、给引擎打上 `-DFV_GPU`、链接 `cudart_static`。
找不到就编出纯 CPU 版本，所有 GPU 入口返回 NULL / -1。
GPU 架构默认 `sm_89`，可用 `make CUDA_ARCH=sm_80` 等参数调整。

编译选项要点：引擎用 `-O3 -march=native -fno-strict-aliasing`；
`SHLIB_LINK` 里的 `-rdynamic` 是为了让 worker 崩溃回溯里的符号名可解析。

**改动共享内存布局后必须 `make clean`**（见 [05](05-gpu-arbiter.md) §1）。

安装后：

```
# postgresql.conf
shared_preload_libraries = 'paves'
```

```sql
CREATE EXTENSION vector;   -- 前置依赖
CREATE EXTENSION paves;
SELECT paves_build('items');
```

worker 是在 `shared_preload_libraries` 阶段注册的后台进程，所以 GPU 路径必须重启实例
才能启用。worker 启动时要探测显存，大约需要几秒才会置 `worker_ready`。

**算子级压测工具**（不依赖 PostgreSQL）：

```bash
make gpu_micro    # tools/gpu_micro.cpp，直接 mmap 索引 + 查询文件驱动 GPU 批接口
```

该工具接受查询向量二进制文件；SQL 端到端复现请用 `bench/benchmark.py`。查询向量不能复用索引里的行
（图搜索会精确命中并立即收敛，QPS 虚高），`max_visited` 必须显式传
`max(200000, ef × 400)`。

## 2. GUC

| GUC | 类型 | 默认 | 范围 | 生效级别 | 说明 |
|---|---|---|---|---|---|
| `paves.enable` | bool | on | | USERSET | 关掉后不再注入自定义路径 |
| `paves.force_strategy` | enum | auto | auto/brute/hnsw/gpu_brute/gpu_hnsw | USERSET | 强制单条路径，用于隔离算子性能 |
| `paves.gpu_enable` | bool | on | | USERSET | off 时 auto 只在两条 CPU 路径间选；被强制的 gpu_* 不受影响 |
| `paves.ef_search` | int | 100 | 10 – 10000 | USERSET | 图搜索名义束宽，执行期按选择率下调 |
| `paves.threads` | int | 8 | 1 – 256 | USERSET | 每查询的引擎工作线程数（暴力扫描） |
| `paves.cost_scale` | real | 0.01 | 1e-6 – 1e6 | USERSET | 整体压低四条路径的代价，四者相对关系不变 |
| `paves.cpu_cores` | int | 384 | 1 – 4096 | USERSET | 代价模型饱和项假设的核数 |
| `paves.gpu_mode` | enum | worker | worker/direct | USERSET | direct 让后端自己持有 CUDA 上下文，仅适合测试 |
| `paves.gpu_device` | int | -1 | -1 – 15 | USERSET | -1 = 按剩余显存自动选卡 |
| `paves.gpu_precision` | enum | sq8 | fp32/sq8 | SIGHUP | 上传时生效；已驻留索引保持原精度 |
| `paves.gpu_batch_max` | int | 64 | 1 – 512 | SIGHUP | 单微批最大请求数 |
| `paves.gpu_batch_wait_us` | int | 200 | 0 – 20000 | SIGHUP | 聚集窗口时长 |
| `paves.gpu_batch_idle_us` | int | 40 | 0 – 20000 | SIGHUP | 连续无新到达即提前发车；0 = 固定窗口 |
| `paves.build_m` | int | 16 | 4 – 64 | USERSET | HNSW 图度数 M（第 0 层为 2M） |
| `paves.build_ef_construction` | int | 200 | 8 – 2000 | USERSET | HNSW 构建束宽 |
| `paves.build_threads` | int | 64 | 1 – 384 | USERSET | 构建线程数 |

`paves.gpu_precision = fp32` 用于未量化距离计算；CPU/GPU 浮点归约顺序可能不同，应按 recall 和距离容差验证，不保证逐位一致。

## 3. SQL 接口

### `paves_build(rel regclass) → bigint`

扫描全表，构建索引文件并原子替换，返回索引进去的行数。
详见 [02-index-build.md](02-index-build.md)。构建期间不要重启实例（会杀掉扫描）。

`DROP TABLE` 不会删除索引文件，孤儿文件需要手工清理。

### `paves_capacity(rel regclass) → TABLE(metric, value, ok, note)`

只读预检。建索引和上传显存都是分钟级、几十 GB 的操作，而若干快速路径在触到上限时是
**静默降级**的（维度超过槽位容量 → 走 CPU；维度没有 tile 变体 → 走 legacy kernel）。
这个函数把"试了才知道"变成"查一下就知道"。输出行：

| metric | 含义 |
|---|---|
| `rows` / `row_limit` | `reltuples` 估计与 2^32 上限 |
| `dim` | 向量列维度（取自 typmod） |
| `cached_scalar_columns` | 可缓存标量列数，超过 32 的部分会被标 `PARTIAL` |
| `index_size_est` | 索引文件大小估计，须远小于内存（mmap 走页缓存） |
| `build_peak_mem` | 构建期峰值内存（约索引的 2.5 倍） |
| `build_disk_need` | 重建期磁盘需求（新旧两份索引短暂共存） |
| `build_time_est` | 构建耗时估计（按 SIFT1M 标定，随行数、维度、线程数缩放） |
| `per_backend_visited` | CPU 图搜索的每后端访问标记数组（`count × 4` 字节），要乘连接数 |
| `gpu_resident` | 显存占用估计，与 worker 发布的剩余显存比对 |
| `gpu_fast_brute` | 该维度是否有 tile 变体（`tiled (QT=n)` 或 `legacy`） |
| `gpu_arbiter_dim` | 维度是否在仲裁槽位容量内 |

显存数字来自共享内存里 worker 发布的值——后端不允许自己建 CUDA 上下文。

### `paves_feedback() → SETOF record`

导出代价模型的运行时反馈表，每 `(索引 × 策略侧 × 选择率桶)` 一行：

```
dbid, relid, strat, bucket, bucket_mid,
gpu_q_us, gpu_batch_us, gpu_batch_n,
cpu_q_us, cpu_ref_active,
gpu_inflight, cpu_active
```

排查路由行为时先看这张表。字段含义见
[03-planner-routing.md](03-planner-routing.md) §4。

## 4. 容量上限一览

| 项 | 上限 | 常量 | 超限行为 |
|---|---|---|---|
| 索引行数 | 2^32 − 1 | 引擎行号为 uint32 | 构建报错 |
| 缓存标量列 | 32 | `FV_MAX_COLS` | 多余列不进索引，谓词只走执行器重查 |
| GPU 路径维度 | 2000 | `FV_ARB_MAX_DIM` | 退回 CPU，每后端记一次 LOG |
| GPU 快速暴力维度 | 2048（sq8）；fp32 另需整除 4/4/8/16 | tile 变体表 | 退回 legacy kernel |
| GPU 单次 top-B | 4096 | `FV_GPU_MAX_TOPB` | 退回 CPU 游标 |
| tiled kernel 的 top-B | 64 | `FV_TILE_TOPB` | 退回 legacy kernel |
| GPU 图搜索 ef | 512 | `FV_GPU_MAX_EF` | 退回 CPU HNSW 游标 |
| 单微批查询数 | 1024 | `FV_GPU_MAX_BATCH` | 批接口报错，整组回退 |
| 仲裁槽位 | 512 | `FV_ARB_SLOTS` | 抢不到槽位的后端自己在 CPU 上算 |
| 每请求谓词节点 / 子节点 / IN 值 / span | 48 / 96 / 128 / 128 | `FV_ARB_MAX_*` | 序列化失败，走 CPU |
| 每请求结果 | 1024 | `FV_ARB_MAX_RESULTS` | 走 CPU |

## 5. 功能边界

- 距离只支持 L2（pgvector 的 `<->`）。
- 索引是静态快照，没有增量更新；表变化后需要重新 `paves_build()`。
  期间被删改的行由回表可见性检查滤掉，只影响召回。
- 只接管带 LIMIT 的单键距离排序查询；其余查询不受影响。
- 多 GPU 分片没有实现：一个索引常驻一张卡。

## 6. 回归用例

`extension/paves/test/smoke.py --dsn "$DSN" --gpu` 对六类过滤、取模残余谓词、空结果和无过滤执行只读检查。真值来自关闭 PAVES 与 ANN 索引的 PostgreSQL 扫描；CPU brute 比较距离并允许并列近邻；ANN 路径检查足额、无重复、谓词满足并记录 recall。

正式 recall 和吞吐验收由 `bench/benchmark.py` 在独立确认集上完成，不能用少量 smoke 查询代替。

## 7. 从旧名 `pgfavor` 迁移

代码内的标识符已整体改名，运行时状态需要跟着改：

- `postgresql.conf` 里 `shared_preload_libraries` 改成 `'paves'`，
  所有 `pgfavor.*` GUC 改成 `paves.*`。
- 索引文件目录从 `$PGDATA/pgfavor/` 改成 `$PGDATA/paves/`；
  文件名格式不变（`<dbid>_<relid>.idx`），文件格式与 magic 也不变，
  直接把目录改名即可复用已有索引。
- 数据库里 `DROP EXTENSION pgfavor; CREATE EXTENSION paves;`，
  函数名相应变成 `paves_build` / `paves_capacity` / `paves_feedback`。
