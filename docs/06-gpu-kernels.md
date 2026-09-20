# 06 GPU 算子

代码在 `src/engine/gpu_engine.cu`，接口是 `src/engine/gpu_engine.h`。
`engine.cpp` 通过 `fv_gpu_exec_brute_batch` / `fv_gpu_exec_graph_batch` 两个封装调用它们，
封装负责把每条查询的谓词切片**拼接**成一组连续数组并重定基（`GpuConcat`）。
编译时没有 nvcc 就整个跳过，相关入口一律返回 -1。

## 1. 索引上传与显存布局

`fv_gpu_index_upload()`。上传是惰性的：第一条走 GPU 的查询触发，之后常驻。
失败会被记住（`gpu_failed`），不会每条查询重试。

**选卡**（`device = -1` 时）：遍历所有设备，选剩余显存最多、且满足
`剩余 > 需求 + 512 MB` 的那张。探测过程会在每张卡上建出上下文（几百 MB），
所以之后要把没选中的卡 `cudaDeviceReset()` 清掉。

这里有一个必须遵守的约束：`dev_refcount[]` 按卡记录本进程还有多少个常驻索引，
`pick_device` **跳过引用计数非零的卡**。否则第二个索引上传时的探测清理会销毁第一张卡的
primary context，连同它上面的显存和页锁定主机缓冲一起失效，之后表现为
偶发的、位置飘忽的段错误（有时崩在主机侧的锁页缓冲写入，有时崩在驱动内部的
`cuMemcpyHtoD` 深处），且只在两个数据集都驻留、策略交替时触发。

**显存内容**：

| 数组 | fp32 模式 | sq8 模式 |
|---|---|---|
| 向量 | `d_vecs` float32[count × dim] | `d_vq` packed int8[count × dq4]、`d_vh` fp16[count × dpad]、`d_vnorm` int32[count] |
| 标量列 | `d_cols` float64[ncols × count]，列主序 | 同左 |
| 倒排行号 | `d_prows` uint32[ncols × count] | 同左 |
| 第 0 层邻接 | `d_l0` uint32[count × (2M+1)] | 同左 |

`dpad` 是维度向上取整到 16 的倍数，`dq4 = dpad / 4`（一个 uint32 装 4 个 int8）。
sq8 每维 3 字节（int8 扫描副本 + fp16 重排副本）对比 fp32 的 4 字节。

其余是按需增长的暂存缓冲（`DevBuf`）。`DevBuf::ensure` 禁止一块缓冲在
device 内存与页锁定主机内存之间改变类型：用 `cudaFreeHost` 释放设备内存
（或反之）会破坏分配器，症状要到之后某次无关的 `cudaMemcpy` 里才暴露。

`h2d_check()` 在每次关键的 H2D 拷贝前校验两端：主动触碰主机源的首尾字节
（源若已失效，就在**本函数的栈帧**里 fault，日志说得清是什么问题），
并用 `cudaPointerGetAttributes` 确认目的地确实是当前设备的设备内存
（陈旧或跨设备的指针会让驱动把这次拷贝当成主机到主机，写到任意地址）。

### 1.1 SQ8 量化

`sq8_build()`，两遍分块 PCIe 传输，峰值瞬时显存只有一个暂存块（256 MB）。

```
量化式：x_q[d] = clamp(round((x[d] - mean[d]) / s), -127, 127)
        mean[]  逐维均值
        s       单一全局尺度 = max_d(max(|max_d - mean_d|, |mean_d - min_d|)) / 127
```

- 第 1 遍：`fv_stats_kernel` 在 GPU 上归约逐维的 sum / min / max。
- 主机据此算出 `mean[]` 与 `s`。
- 第 2 遍：`fv_quant_kernel`（一个 warp 处理一行）打包 int8、写 fp16 副本、
  算量化域下的整数模长 `|x_q|²`。

逐维中心化对两侧距离的平移是等量的，完全不扭曲 L2；单一全局尺度保证
量化域距离与真实距离成正比（`dist ≈ s² · dist_q`），所以扫描阶段可以完全用整数算术排序：

```
dist_q(x, y) = |x_q|² + |y_q|² - 2·(x_q · y_q)     点积用 dp4a
```

查询向量在主机侧用同一套 `mean[]` 和 `s` 量化。
`paves.gpu_precision` 只影响**后续的上传**；已驻留的索引保持它上传时的精度。

## 2. 暴力批接口的路由

`fv_gpu_brute_batch()` 收到一批查询后：

1. 校验 `nq ≤ 1024`、`topb ≤ 4096`。
2. 按维度和精度选 tile 变体。fp32 的变体表要求维度能整除 4/4/8/16 且共享内存
   ≤ 96 KB；sq8 因为零填充到 16 的倍数，`dim ≤ 2048` 都有变体。
   `topb > 64`（`FV_TILE_TOPB`）或没有可用变体时整批走 legacy kernel。
3. 否则**逐查询按驱动集大小路由**：`nspans > 0 && total < count / QT` 的查询进
   span 子集（legacy kernel），其余进分块全扫子集（tiled kernel）。
   `QT` 是该变体的查询 tile 宽度。分界点的道理是：span kernel 每条查询读
   `span_rows` 行，tiled kernel 读 `count` 行但摊给 QT 条查询，所以盈亏点大致在
   `count / QT`。
4. 混合批把两个子集各自 gather 出来跑，再 scatter 回原位
   （查询描述符里指向拼接数组的索引仍然有效）。

### 2.1 tiled kernel（分块全扫）

`fv_brute_tile_kernel`（fp32）/ `fv_brute_tile8_kernel`（sq8）。

grid 是二维的：`(rowblocks, ceil(nq / QT))`，256 线程。每个 block 拥有一段连续行区间
和一个查询 tile。查询被转置进共享内存（`qT[dim][QT]`），行分批读进共享行缓冲
（`rowbuf[8][RT][dim]`），warp 内每个 lane 负责 QPL 条查询、LPQ 分之一的维度，
维度分区之间用 shuffle 蝶形归约合并。

要点：

- **向量数组每个查询 tile 只被读一遍**，而不是每条查询读一遍。
- **谓词惰性求值**：只有能进候选堆的行才求值谓词——堆未满时每行都要检查，
  堆满后只检查能击败当前最差的行。
- **候选用打包键** `dist_bits << 32 | row`。键的大小顺序等同 `(dist, node)` 字典序，
  所以堆淘汰与最终排序口径一致，选出的 top-B 是规范的：重算之间一致，
  也与 CPU 游标按 `(dist, node)` 的排序一致。sq8 版本打包的是整数距离。
- **每 lane 的堆直接开在全局暂存槽里**（L2 缓存住，热了之后操作很少）。
  放成局部数组会和谓词求值器的递归栈帧一起击穿 1 KB 的设备栈。
- `rowblocks` 由行数推导（64 起步，`rb × 8192 < count` 时翻倍，上限 256），
  保证每条 lane 流足够长，把约 topb 次的堆填充成本摊掉。

### 2.2 merge kernel

`fv_brute_merge_kernel`，一个 block 处理一条查询，把
`rowblocks × 8 warps × 64` 个填充过的键归约成最终的升序 top-B：

- 阶段 1：64 个线程各自在跨步切片上维护一个本地 top-64（全局 top-64 的任一元素
  必然在它所属切片的 top-64 里）。热路径上用一个寄存器 `w0` 镜像堆顶，
  让"不具竞争力"这一常见分支不必碰局部内存。
- 阶段 2：4096 个幸存者在共享内存里做双调排序。

只有 `nq × topb` 条结果需要过 PCIe。

### 2.3 rerank kernel（仅 sq8）

`fv_rerank_kernel`。merge 输出的是 `topb_r = min(2 × topb, 128)` 条**近似**候选；
本 kernel 一个 block 一条查询，每个 warp 重算一条候选对 fp16 行的 fp32 距离，
128 个键双调排序后输出最终的 top-B，并覆写计数与截断标志。
这一步把量化噪声挡在消费者的 top-k 之外。

### 2.4 legacy kernel（按 span 枚举）

`fv_brute_kernel`。每条查询 8 个 block × 256 线程 = 64 个 warp；
一个 warp 一次处理一行：

- 枚举位置 `e = wq, wq + 64, ...` 走遍 `qd.total`。有 span 时二分 `spanoff`
  找到所在 span，再经 `prows` 取行号；没有 span 时行号就是 `e`。
- lane 0 求值谓词，`__shfl_sync` 广播结果。
- 全 warp 用 float4（或 half2）加载算 L2，shuffle 归约。
- lane 0 维护本 warp 在全局暂存里的 top-B 最大堆。

主机把 64 个分堆合并、按 `(dist, node)` 排序、必要时去重、截断——
与 CPU `BruteCursor::compute()` 的合并逻辑逐步对应。

这条路径读的行数正比于驱动集大小，所以高选择性谓词下它比分块全扫划算；
它也是没有 tile 变体的维度、以及 `topb > 64` 时的唯一选择。

## 3. 图搜索 kernel

`fv_graph_kernel`，一个 thread block 一条查询，512 线程。入口点由 worker 在 CPU 上
算好传进来。

**共享内存状态**：前沿（`fr_d` / `fr_n`，容量 1024）、结果表（`re_d` / `re_n`，
容量 ef ≤ 512，只装通过谓词的节点）、邻居缓冲（256）、若干标量。

**访问标记用每查询的全局显存位图**（`count/32` 个字，`atomicOr` 测试并置位），
不是共享内存哈希表。低选择率谓词合法地会访问上万个节点，共享哈希一旦填满，
每次探查都报"新节点"，遍历会退化成重访循环直到 `max_visited`，
表现为双峰的、不确定的性能。

一次迭代：

1. **warp 0 一趟 shuffle 同时选出**前沿里最优的 W 个未扩展项**和**前沿最大值
   （打包键让 dist+index 一次比较搞定，不需要块级树形归约和额外同步）。
   一次扩展 W 个节点把串行迭代次数——以及每次迭代的固定选择/同步开销——除以 W。
   `W` 是动态的：`clamp(fr_cnt / 64, 4, wsel)`。前沿很大意味着这是一条探索型查询
   （谓词稀疏），宽扩展摊掉开销；前沿很小说明查询在收敛，窄扩展避免浪费投机距离计算。
   `wsel = min(8, 256 / 最大度数)`，`deg_clamp = 256 / wsel`，二者共同保证
   邻居缓冲不溢出。
2. **弹出选中项**（按索引降序，使 swap-remove 的下标始终有效；前沿最大值的下标
   在其条目被搬动时同步修正）。
3. **协作收集**所有选中节点的邻居。
4. **访问过滤 + 压紧**成稠密工作表。重叠邻居表带来的重复项靠位图去重
   （只有第一次 test-and-set 成功）。压紧让后面的距离 warp 满载，
   而不是空转跳过已访问的槽。**这一步不求值谓词**。
5. **算距离**，稠密表上一个 warp 一条候选。
6. **接纳**：先做**距离门控**（结果表已满且距离不优于当前最差就直接跳过），
   通过门控的才求值谓词。前沿追加是并行的（`atomicAdd` 抢槽）；
   只有结果表插入和前沿溢出替换这两种少见情形留在 0 号线程里串行。

   谓词延后到距离门控之后是有实际意义的：早求值意味着为每个新节点付一次随机全局读，
   而其中大部分节点根本没有机会进结果表。

7. **停止**：前沿为空（穷尽）、访问数达到 `max_visited`（穷尽）、
   或结果表已满且前沿最优已劣于结果最差（自然停止）。

**sq8 模式**下遍历跑在 int8 行上（dp4a，每次邻居取数的带宽是 fp32 的四分之一），
最终结果表在**核内**对 fp16 行重排成 fp32 距离再输出，量化噪声不会传给消费者。
fp32 模式直接在 fp32 行上遍历。

主机对每条查询的 ≤ ef 条结果按 `(dist, node)` 排序后返回。

## 4. 诊断开关

| 环境变量 | 作用 |
|---|---|
| `FV_GPU_PROF=<秒>` | 每隔若干秒向 stderr（即 PG 日志）打一行 `FVPROF {...}`，分解 H2D / kernel / merge+rerank / D2H / host 各相位耗时。关闭时不插入任何额外同步，生产计时不受影响。上传时另打一行含量化耗时与尺度。 |
| `FV_GPU_GDBG=1` | 每批打一行 `FVGDBG {...}`：图 kernel 的每查询迭代数分位、平均访问节点数、平均谓词求值次数、平均结果数。 |
| `FV_TILE8_V=<n>` | 覆盖 sq8 的 tile 变体选择，用于变体对比实验。 |

`FVPROF` 行可用于算子阶段诊断；发布基准 `bench/benchmark.py` 测量 SQL 端到端耗时，不使用这些内部阶段计时计算加速比。
