# 04 执行期：CustomScan 执行器与 CPU 算子

执行器代码在 `src/exec.c`，CPU 引擎算子在 `src/engine/engine.cpp`。

## 1. Begin：换扫描槽

`BeginCustomScan` 里做的第一件正事是替换扫描槽：

```c
node->ss.ss_ScanTupleSlot = table_slot_create(rel, &estate->es_tupleTable);
node->ss.ps.scanops       = table_slot_callbacks(rel);
node->ss.ps.scanopsfixed  = true;
node->ss.ps.scanopsset    = true;
node->ss.ps.qual = ExecInitQual(cscan->scan.plan.qual, &node->ss.ps);
ExecAssignScanProjectionInfoWithVarno(&node->ss, cscan->scan.scanrelid);
```

原因：PG14 的 `ExecInitCustomScan` 把扫描槽硬编码成固定的虚拟槽
（`CustomScanState.slotOps` 要到 PG15 才有），而表达式编译器对固定虚拟槽会**整体省略
元组 deform 步骤**——虚拟槽被假定永远已填充。本节点返回的却是回表得到的
buffer-heap 元组，qual 与投影会直接读到陈旧的 `tts_values`。
症状很有迷惑性：谓词莫名全假、投影读到野指针在 detoast 里段错误。
所以必须换成真正的表槽并**重新编译** qual 与投影。

接着初始化查询向量的 `ExprState`，取索引句柄（拿不到就报错，提示重建或关闭
`paves.enable`），并在每查询内存上下文上注册一个 `MemoryContextCallback`。
引擎对象活在 malloc 内存里，这个回调保证任何退出路径——包括 `elog(ERROR)` 的
longjmp——都会释放游标与谓词。

## 2. 第一次取元组：启动搜索

`favor_start_search()`：

1. 求值查询向量表达式，detoast，校验维度与索引一致，拷进一块本地 `float[dim]`。
2. 逐条编译下推子句。**编译失败的顶层合取项直接跳过**——谓词变成超集，
   由 `ExecScan` 的 qual 重查滤掉多余的行。多个子句合成一个 AND 根节点。
3. `fv_pred_compile()` 把谓词树深拷进引擎内部形式，同时推导驱动 span（§3）。
   `fv_pred_estimate()` 返回驱动集大小：0 是权威的"不可能有匹配行"，
   `UINT64_MAX` 表示未知。
4. 计算束宽。`ef = max(k, paves.ef_search × scale)`，
   `scale = sel ≥ 0.4 ? 1.0 : sel ≥ 0.15 ? 0.8 : 0.4`。
   之所以低选择率反而用更小的名义 ef：内联过滤下结果堆只接纳通过谓词的节点，
   搜索本身已经隐式加宽了。
5. 按策略开游标（§5 是各种回退路径）。

匹配估计为 0 时不开任何游标，直接返回空结果。

## 3. 驱动 span

代码是 `engine.cpp` 的 `make_driver()`。目标是把谓词转成"倒排表上的若干位置区间"，
这样暴力扫描只需要遍历这些区间指向的行，而不是全表。

| 节点 | 驱动 |
|---|---|
| `CMP`（`<` `<=` `=` `>=` `>`） | 该列上一个区间；`covers = true` |
| `CMP`（`<>`） | 无驱动 |
| `IN` | 每个取值一个区间；`covers = true` |
| `AND` | 取子节点里**最小**的驱动；另外把同一列上的 ≥2 个 CMP 子项**求交**成单个区间，与前者比大小 |
| `OR` | 各子节点驱动的并集；任一子节点无驱动则整体无驱动；子节点多于一个时置 `may_dup` |
| `NOT` | 无驱动 |

三个标志决定后续行为：

- `covers`：驱动集恰好等于匹配集，扫描时不必再求值谓词。
  `AND` 有多个子项时通常为假；只有当同列交集吸收了**全部**子项时才保持为真。
- `may_dup`：同一行可能出现在多个 span 里（`OR`、多值 `IN`），结果需要去重。
- `driver_size`：驱动集基数，既是 `fv_pred_estimate()` 的返回值，
  也是 GPU 侧 span/全扫路由的判据。

同列求交这一条是必要的：`c >= a AND c < b` 若只取"最小子驱动"会退化成
`c >= a` 或 `c < b` 中较小的那个单边区间，而两者的交往往小一到两个数量级。

区间边界用 `post_lower` / `post_upper` 在 `post_vals` 的有限值前缀里二分得到。
`CMP` 的值为 NaN（来自 NULL 常量）时是恒假叶子，驱动集大小为 0。

## 4. 谓词求值语义

标量列的 NULL 存成 NaN，任何与 NaN 的比较都为假（`<>` 特意写成
`v == v && v != value`，保证 NaN 上也返回假）。这与 SQL 三值逻辑的差异只在
`NOT` 之下体现，且方向单一：只可能产生**假阳性**。执行器对每条元组重查全部原始 qual，
所以下推永远不影响正确性，只影响候选数量。

## 5. CPU 游标

### 5.1 `BruteCursor`（精确扫描）

`fv_brute_begin()`。把驱动 span 按 65536 行切成任务；没有驱动时任务就是整个
`[0, count)` 区间。`need_eval` 在"无驱动 / 不 covers / may_dup"时为真。

线程池并行跑所有任务，每个任务维护一个容量 B 的最大堆：

```
for row in task:
    if need_eval and not pred.eval(row): continue
    d = l2sq(query, vec(row))
    堆插入 / 替换堆顶
```

距离计算按编译目标选 AVX-512 / AVX2 / 标量三个实现之一。

主线程合并所有分堆，按 `(dist, node)` 排序，`may_dup` 时去重，截断到 B。
B 起始值是 `max(4 × want, 64)`；消费者要的比现有结果多且结果被截断过时，
用 `B × 4` **重算**。重算是确定的（同样的输入给出同样的前缀顺序），
所以已经吐出去的前缀不会错乱。

### 5.2 `HnswCursor`（过滤图搜索）

`fv_hnsw_begin()`。先在上层做不带过滤的贪心下降找到第 0 层入口点，
然后在第 0 层跑束搜索：

- `cand`（最小堆）是待扩展前沿，`res`（最大堆，容量 ef）只装**通过谓词**的节点。
- 遍历本身不受谓词限制：任何节点都可以被走过，只是不一定有资格进结果。
  这是过滤图搜索保持连通性的前提。
- 新邻居先算距离，`res` 未满或距离优于 `res` 堆顶时入前沿；通过谓词才入 `res`。
- 停止条件：前沿为空、前沿最优已劣于 `res` 堆顶且 `res` 已满、
  或访问节点数达到 `max_visited`（默认 `max(200000, ef × 400)`）。

**加宽**：消费者要的结果比当前 `res` 多时，`ef *= 2` 并继续跑同一次搜索
（`cand`、`res`、访问标记都保留），再把已经吐出去的节点过滤掉。

**访问标记**用 epoch 标记法：每个 `FvIndex` 挂一份 `count` 长的 uint32 数组和一个
epoch 计数器，新游标只需 epoch 自增，不必清零。同一后端同一索引上已有游标在用时
（`scratch_busy`），第二个游标退化为自己分配一份私有数组。

## 6. 流式取元组

```
buf 用尽 → fv_cursor_next(cursor, buf, want)
        want = min(16, k - 已发出数)     LIMIT 满足后降为 4
候选行号 → fv_index_tid() → ItemPointer
        → table_tuple_fetch_row_version(rel, tid, snapshot, slot)
        可见 → 返回槽（ExecScan 再跑一遍 qual）
        不可见 → 计入 invisible，取下一个候选
```

`want` 按 LIMIT 剩余量裁剪是有意义的：多要候选会迫使 HNSW 游标加宽束宽、
重跑搜索，去找这条查询根本不会消费的结果。只有被可见性判负的候选才需要补第二小批。

`favor_recheck` 恒返回 true，因为 qual 由 `ExecScan` 对每条元组求值，
无需在 recheck 回调里重复。

`ReScanCustomScan` 释放引擎对象、清空所有游标状态与 `seen_tids`，重新开始。

## 7. GPU 策略的降级

GPU 策略在 worker 模式下走仲裁（[05](05-gpu-arbiter.md)）。以下情况会退回 CPU：

| 情况 | 处理 |
|---|---|
| 维度超过槽位容量 | 记一次 LOG（每后端一次），走 CPU |
| worker 未就绪 | 记一次 LOG（每后端一次），走 CPU |
| 谓词序列化超出槽位容量 | 静默走 CPU |
| 槽位池满 | 静默走 CPU（后端自己算） |
| worker 回 ERROR | 走 CPU |
| 加宽后的请求超出仲裁上限 | 中途切到 CPU 游标，靠 `seen_tids` 跳过已消费的候选 |

`paves.gpu_mode = direct` 时后端自己持有 CUDA 上下文，直接调
`fv_gpu_brute_begin` / `fv_gpu_hnsw_begin`；失败再退 CPU。这个模式只适合测试或
少量后端：worker 模式下**绝不允许**后端自行上传索引，几十个后端各传一份
几百 MB 的索引会立刻打爆显存。

## 8. CPU 侧的反馈采样

`FV_ENGINE_TIMED` 宏把 `*_begin` 与每次 `fv_cursor_next` 的墙钟累加到 `engine_us`。
开始跑 CPU 游标时 `cpu_active` 加一并记下当时的水位；释放时减一，
并把 `(engine_us, 起始水位)` 写进反馈表。

采样口径包含 LIMIT 语义——记的是"这种形态的查询实际花的功夫"，
而不是把游标抽干的代价。只有 `strategy` 本身是 `BRUTE` / `HNSW` 的查询才写反馈，
理由见 [03-planner-routing.md](03-planner-routing.md) §4.2。
