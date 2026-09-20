# 02 构建期：索引文件

索引是离线构建的静态快照：一次 `paves_build()` 产出一个文件，之后所有查询路径
（CPU 与 GPU、扫描与图搜索）都从这一个文件读数据。没有增量更新——表变了就重建。

## 1. `paves_build(regclass)` 的扫描过程

代码在 `src/build.c`。

**选列**。打开表（`AccessShareLock`）后遍历属性：

- 第一个类型为 `vector`（pgvector）的列成为向量列。若它带 typmod，维度直接取
  typmod；否则维度由第一行决定，此时建造器会被重建一次。
- 其余类型属于 `bool / int2 / int4 / int8 / float4 / float8` 的列成为**缓存标量列**，
  最多 32 个（`FV_MAX_COLS`）。超出的列不进索引，相关谓词只能由执行器重查处理。
- 没有向量列则报错。

**扫行**。在活动快照下做一次 `table_beginscan` 顺序扫描，每行：

- 向量为 NULL 的行跳过（这类行在距离排序里本来就排最后）。
- 维度与既定维度不符则报错。
- 每个缓存标量列取值转成 `double`；NULL 存成 NaN。
- TID 压成 64 位整数 `block << 16 | offset` 一并记下。

每 8192 行检查一次中断。零行会报错退出。

**落盘**。`fv_builder_finish()` 构建 HNSW 图与倒排表，序列化到 `<path>.tmp`，
`fflush` + `fsync` 之后 `rename(2)` 到目标路径。同一文件系统内的 rename 是原子的，
因此不存在"半个索引"的中间状态。

目标路径：`$PGDATA/paves/<dbid>_<relid>.idx`，目录在首次构建时创建。
（若把 `$PGDATA/paves` 做成指向别的目录的符号链接，需要保证 tmp 与目标同在一个
文件系统，否则原子替换的前提不成立。）

相关 GUC：`paves.build_m`（默认 16）、`paves.build_ef_construction`（默认 200）、
`paves.build_threads`（默认 64）。

## 2. 文件格式

头部是定长的 `FileHeader`（`engine.cpp`），后面是若干 64 字节对齐的区段。
打开时整个文件被 `mmap(PROT_READ, MAP_SHARED)`，各区段只是头部记录的偏移量上的指针，
不做任何解析或拷贝。

```
FileHeader
  magic "PGFAVR01" | version | dim | count | vec_attnum
  ncols | col_attnums[32]
  M | ef_construction | max_level | entry_node
  off_vectors / off_tids / off_cols / off_post_vals / off_post_rows
  off_levels / off_l0 / off_upper_offs / off_upper | file_size

vectors      float32[count * dim]        行主序，原始向量
tids         uint64 [count]              行号 → 打包 TID
cols         float64[ncols * count]      列主序标量缓存，NULL = NaN
post_vals    float64[ncols * count]      每列：排序后的取值
post_rows    uint32 [ncols * count]      每列：按该列取值排序的行号
levels       uint8  [count]              每行的 HNSW 最高层
l0           uint32 [count * (2M+1)]     第 0 层邻接：[度数, 邻居...]
upper_offs   uint64 [count + 1]          每行在 upper 区的槽位偏移
upper        uint32 []                   上层邻接，每行每层一块 (M+1)
```

打开文件时（`fv_index_open`）除了校验 magic / version / 文件大小，还会对每个缓存列
二分出 `post_vals` 里第一个 NaN 的位置（NaN 排在最后），记作 `nan_start[col]`。
后续所有倒排表二分都只在 `[0, nan_start)` 这段有限值前缀里进行。

**列主序**是刻意的：谓词求值按列访问，一次只关心一列的一段连续内存；GPU 侧的
`d_cols` 也照搬这个布局，kernel 里 `cols[col * count + row]` 的访问模式与主机端一致。

## 3. 倒排表（posting list）

对每个缓存标量列，把全部行号按该列的取值排序，得到一对数组：排序后的取值
`post_vals` 与对应行号 `post_rows`。NaN（即 NULL）排在末尾，取值相同的行按行号排序，
所以构建结果是确定的。

这个结构的用处是把范围谓词变成**位置区间**：`col < v` 就是
`post_rows[col][0 .. lower_bound(v))`，`col = v` 就是
`[lower_bound(v), upper_bound(v))`。这些区间就是[04](04-executor-cpu.md) §3 里的
**驱动 span**，也是暴力扫描不必读全表的原因。

排序按列并行（线程池，任务数 = 列数）。

## 4. HNSW 图构建

标准 HNSW，实现在 `engine.cpp` 的 `HnswBuild`。

**层分配**是确定的：用 `splitmix64(i * 常数 + 12345)` 从行号导出 [0,1) 均匀数 u，
层 = `floor(-ln(u) / ln(M))`，上限 30。同一批数据重建两次得到同样的层分布。

**度数上限**：第 0 层 `M0 = 2M`，上层 `M`。这解释了 `l0` 的步长 `2M+1`
（一个计数槽 + 2M 个邻居槽）与 `upper` 的每层块大小 `M+1`。

**插入**（`insert`）：

1. 从当前入口点开始，在高于本行层数的每一层做贪心下降（`greedy`）。
2. 从 `min(入口层, 本行层)` 到 0，每层跑一次 `search_layer`（束宽
   `ef_construction`），得到候选集。
3. 候选集经 `select_neighbors` 裁剪到度数上限，写成本行在该层的邻接表。
4. 反向 `link` 每个被选中的邻居；邻居表满时，把"新目标 + 现有邻居"一起重新裁剪。

`select_neighbors` 是启发式裁剪：按距离升序遍历候选 c，只有当已保留集合里不存在
比 `dist(q, c)` 更靠近 c 的元素时才保留 c。它抑制的是"一堆候选挤在同一方向"，
让邻接表在方向上更分散。

**并发**：每行一个自旋锁保护它的邻接表；入口点与入口层用一个 mutex 加原子变量维护。
行 0 先单线程插入以确立入口点，其余行由线程池抢占式领取（`std::atomic` 计数器）。
每个工作线程持有自己的 `visit_tags` 数组（`count` 个 uint32）与 epoch 计数器，
所以每次 `search_layer` 不需要清零访问标记。

线程池是刻意泄漏的单例：`static ThreadPool *pool = new ThreadPool()`。用静态对象的话，
后端退出时析构函数会对仍处于 joinable 状态的工作线程调用 `~thread()`，
直接 `std::terminate`。

## 5. 索引的打开与失效

后端侧（`planner.c` 的 `paves_get_index`）与 worker 侧（`worker.c` 的
`worker_get_index`）各有一份索引缓存，键分别是 `relid` 与 `(dbid, relid)`。
两者都用 `stat()` 的 `(ino, size, mtime)` 三元组判断文件是否被重建过；变了就重新
`fv_index_open`。

**旧映射不 munmap**，是刻意的：已持有的游标（例如未关闭的 portal）可能还在引用旧映射，
解映射会导致段错误。泄漏量以"会话内重建次数"为界。

GPU 侧同理：`FvIndex` 里的 `gpu` 指针挂着显存副本，旧 `FvIndex` 不关闭，
显存副本也就不释放。

**版本一致性**：后端提交 GPU 请求时把自己看到的索引文件 inode 一并写进槽位，
worker 用自己打开的文件的 inode 比对；不一致直接回 ERROR，后端退回 CPU 路径。
worker 返回的是 TID 而不是行号，翻译在 worker 侧、用 worker 自己的索引完成，
所以并发重建最多损失召回，不可能把行号按错误的文件版本翻译成 TID。
