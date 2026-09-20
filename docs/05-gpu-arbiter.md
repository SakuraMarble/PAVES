# 05 GPU 微批仲裁

PostgreSQL 是多进程的，而 GPU 需要来自同一个上下文的大批量请求。PAVES 的做法是：
后端进程在 worker 模式下**从不触碰 CUDA**，而是把查询序列化进共享内存槽位，唤醒一个
常驻后台进程；这个 **GPU worker** 拥有整个实例里唯一的 CUDA 上下文，把提交上来的槽位
聚成微批，一次 kernel 服务一组，再逐槽回答。

代码：`src/arbiter.h`（结构与协议）、`src/arbiter.c`（共享内存与后端侧）、
`src/worker.c`（worker 进程）。

## 1. 共享内存布局

`FvArbShared` 在 `shmem_startup_hook` 里一次性分配（大小由
`paves_arbiter_shmem_size()` 声明，`RequestAddinShmemSpace` 预留）：

```
worker_ready         worker 正在服务时为 1
worker_latch         worker 的 latch，后端用它唤醒 worker
fb[16]               按 (dbid, relid) 认领的代价模型反馈表
gpu_inflight         已提交未回答的请求数
cpu_active           正在跑 CPU 游标的查询数
gpu_free/total_bytes 由 worker 发布的显存状态
gpu_device_plus1     worker 选中的设备号 + 1（0 = 尚未选定）
slots[512]           请求槽位
```

槽位是定长的，容量写死：

| 字段 | 上限 | 常量 |
|---|---|---|
| 查询向量维度 | 2000 | `FV_ARB_MAX_DIM` |
| 谓词节点 | 48 | `FV_ARB_MAX_NODES` |
| 子节点索引 | 96 | `FV_ARB_MAX_CHILDREN` |
| IN 取值 | 128 | `FV_ARB_MAX_INVALS` |
| 驱动 span | 128 | `FV_ARB_MAX_SPANS` |
| 结果 | 1024 | `FV_ARB_MAX_RESULTS` |
| 槽位数 | 512 | `FV_ARB_SLOTS` |

维度上限对齐 pgvector 自己的类型上限，代价是每槽约 8 KB 的向量空间；
整个槽位约 30 KB，512 个约 16 MB。槽位数选 512 是因为图 kernel 一条查询占一个
thread block，在 wiki 规模数据上的饱和点大约就是 512 条并发查询
（用 `tools/gpu_micro` 测得）。

超出任何一项容量的请求不会报错，后端直接改走 CPU。

**共享内存布局一旦改动必须 `make clean`**：PGXS 不追踪 `.c → .h` 依赖，
`arbiter.h` 改了槽位数而 `arbiter.o` 没重编，会造出尺寸不一致的段，worker 启动即崩。
Makefile 里已经显式声明了这几个头依赖，但改布局时仍建议全量重编。

## 2. 槽位状态机

无锁，全部靠 `pg_atomic` 的 CAS：

```
FREE ──后端 CAS──▶ FILLING ──后端填完──▶ SUBMITTED
                                            │ worker CAS
                                            ▼
                                         RUNNING ──worker──▶ DONE / ERROR
                                                                │ 后端读走
                                                                ▼
                                                              FREE
```

回答里携带的是**打包 TID**（`block << 16 | offset`），不是引擎行号。翻译在 worker 侧
用 worker 自己打开的索引文件完成，因此并发重建最多损失召回，
不会把行号按错误的文件版本翻译。

## 3. 后端侧提交

`paves_arbiter_submit()`：

1. 先序列化谓词（`fv_pred_serialize`）到栈上缓冲。容量不够就直接返回 0，
   查询留在 CPU——这一步放在抢槽位之前，避免白占槽位。
2. 遍历槽位，CAS 抢一个 `FREE`。抢不到返回 0（队列满，后端自己算）。
3. 填入 dbid / relid / 索引 inode / 策略 / 维度 / 向量 / 谓词数组 / `want` /
   `max_visited` / 选择率千分比，写内存屏障，置 `SUBMITTED`，`gpu_inflight` 加一，
   唤醒 worker 的 latch。
4. 在自己的 latch 上等待，超时 10 ms 轮询一次状态，每轮 `CHECK_FOR_INTERRUPTS`。
   整段包在 `PG_TRY/PG_CATCH` 里：中途出错时先释放槽位再 re-throw。
5. `DONE` → 拷出结果与 `more` 标志，返回 1。`ERROR` → 返回 -1。

worker 中途死掉的处理：`worker_ready` 归零后，后端若发现槽位仍是 `SUBMITTED`
（worker 没接手）就 CAS 回 `FREE`；若是 `RUNNING` 则直接置 `FREE` 回收。
错误清理路径 `arbiter_slot_release()` 会等 worker 结束该槽位再释放，
上限 10 秒——正常情况下 kernel 批是毫秒级。

**请求宽度**：`gpu_brute` 首次要 `max(4k, 64)`，再请求 ×4；
`gpu_hnsw` 首次要 `max(ef, k)`，再请求 ×2，并附带
`max_visited = max(200000, want × 400)`。GPU 批接口没有"默认值"语义，
`max_visited = 0` 会让图 kernel 首轮就停，所以这个值必须显式给。

## 4. worker 主循环

worker 由 `RegisterBackgroundWorker` 在 `shared_preload_libraries` 阶段注册，
`bgw_restart_time = 5`。它**不连接任何数据库**：索引文件按
`$PGDATA/paves/<dbid>_<relid>.idx` 直接寻址并 mmap。

```
collect_submitted()          遍历槽位，CAS SUBMITTED→RUNNING，最多 gpu_batch_max 个
  n == 0 → WaitLatch 100 ms，继续
  n < gpu_batch_max → 进入聚集窗口：
      每 20 µs 收一次，总时长上限 paves.gpu_batch_wait_us
      连续 paves.gpu_batch_idle_us 没有新到达就提前发车
run_batch(batch, n)
```

**空闲发车**（`gpu_batch_idle_us`，默认 40 µs）解决的是：负载填不满批时，
死等满窗口只是在两次 kernel 之间插入空转。饱和时新到达不断重置空闲计时器，
批照样能填满；中等负载下到达流一停批就走。设成 0 恢复固定窗口。

每约 2 秒打一条统计日志：批数、请求数、每批平均请求数、每请求执行微秒数、
每批等待微秒数，并顺带刷新显存状态。

worker 装了 `SIGSEGV` / `SIGBUS` 处理器，用 `write(2)` 打裸栈帧到服务器日志再重抛。
worker 崩溃会通过 postmaster 重启拖垮整个实例，而平台的 core 处理器可能吞掉转储，
所以这条回溯是必需的（Makefile 里的 `-rdynamic` 保证符号名可解析；
用 addr2line 分析前务必留一份当时的 `.so`，重编译后偏移就失效了）。

## 5. 批的分组与执行

`run_batch()` 把收集到的槽位按 `(dbid, relid, strategy)` 分组，逐组调 `run_group()`。

`run_group()` 的步骤：

1. 打开该组的索引，确保它已上传到 GPU（`fv_index_gpu_ensure`）。失败则整组回 ERROR。
2. **逐槽校验**维度与索引文件 inode，不符的回 ERROR 并从组里剔除，然后压紧数组。
   注意组描述符要在压紧后**重新锚定**到存活的槽位上：原来的 `slots[0]` 可能已经被
   回答 ERROR，其后端随即释放槽位，另一个后端可以立刻填进一个完全不相干的请求。
3. **组内 LPT 排序**（仅图搜索组且 n > 2）：一条查询占一个 block，block 按索引顺序
   启动，所以按预计工作量降序排会让各波 block 一起结束，而不是拖一条长尾。
   排序键取 `want × 1000 / 选择率千分比`——图查询的访问量大致正比于此，
   每次扩展触碰的邻居数是索引常数，不影响排序。暴力组内各查询近似等量，不排。
4. 拼接查询向量、谓词描述符数组；图搜索还要在 CPU 上为每条查询算第 0 层入口点
   （`fv_index_graph_entry`，上层不带过滤的贪心下降）。
5. 用组内最大的 `want` / `max_visited` 调一次批接口
   （`fv_gpu_exec_graph_batch` 或 `fv_gpu_exec_brute_batch`），每个槽位取自己那一段前缀。

## 6. 反馈归因

批的墙钟由**最慢的那个 block** 决定，直接按批时间除以请求数会让快查询继承慢查询的成本。
所以 worker 按解析权重归因：

```
w_i   = want_i * 1000 / sel_i    图搜索
      = 1                        暴力（tiled kernel 对各查询等量）
c     = T / max(w_i)             单位权重成本
桶 b 的样本 = c * mean{ w_i : i 在桶 b }
EWMA α      = n_b / (n_b + 64)
```

`T` 包含入口点计算与 kernel 执行两段。同时写入整批墙钟 `gpu_batch_us` 与批规模
`gpu_batch_n16`（α 均为 0.125），供计划器按"批"给排队定价。

## 7. 回答

对组内每个槽位：

- 取 `min(cnt_i, want_i)` 条结果，用 `fv_index_tid_checked()` 把行号翻成 TID。
  **越界行号被丢弃**并限量告警——设备返回的行号不能被信任地拿去索引 mmap，
  一次越界读会带走整个实例。
- 置 `more` 标志：图搜索是"遍历未穷尽，或结果多于 `want`"；
  暴力是"相对本槽 `want` 被截断，或结果多于 `want`"。后端据此决定是否加宽再问一次。
- 写内存屏障，置 `DONE`，`SetLatch` 唤醒等待的后端。
