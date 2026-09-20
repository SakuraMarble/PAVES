# 01 总览

## 1. 处理的查询形态

PAVES 接管形如下面这种查询：

```sql
SELECT id
FROM   items
WHERE  price < 100 AND category IN (3, 7)     -- 标量谓词
ORDER  BY embedding <-> '[0.1, 0.2, ...]'     -- pgvector 的 L2 距离算子
LIMIT  10;
```

即"在满足 SQL 谓词的行里取向量距离最近的 top-k"。识别条件是硬性的（见
[03-planner-routing.md](03-planner-routing.md) §1）：单一 ORDER BY 排序键、该键是
`vector <-> 常量表达式`、有可用的 LIMIT、目标表存在 PAVES 索引文件。不满足任何一条，
计划器就不插入自己的路径，查询走 PostgreSQL 原有计划。

当前只支持 L2 距离（`<->`），且索引是离线构建的静态快照，没有增量更新链路。

## 2. 组件划分

系统分为三层，边界清晰：

```
┌─ PostgreSQL 后端进程 ─────────────────────────────────────┐
│  planner.c   识别查询、判定谓词可下推、给四条路径定价       │
│  quals.c     把 SQL 谓词树编译成引擎谓词树                  │
│  exec.c      CustomScan 执行器：取候选 → 按 TID 回表 → 重查 │
│  build.c     paves_build()：扫表并交给引擎建索引            │
└───────────────┬───────────────────────┬───────────────────┘
                │ engine_api.h          │ arbiter.h（共享内存）
┌───────────────▼──────────────┐  ┌─────▼─────────────────────┐
│ engine.cpp（C++，无 PG 依赖）│  │ worker.c（GPU worker 进程）│
│  索引文件 mmap / HNSW 构建   │  │  收集槽位组成微批          │
│  brute 游标 / hnsw 游标      │  │  调 GPU 批接口、回填 TID   │
└──────────────────────────────┘  └─────┬─────────────────────┘
                                        │ gpu_engine.h
                                  ┌─────▼─────────────────────┐
                                  │ gpu_engine.cu（CUDA）      │
                                  │  索引上传 / SQ8 量化       │
                                  │  暴力 kernel / 图搜索 kernel│
                                  └───────────────────────────┘
```

- **PG 胶水层**（`src/*.c`）只做 PostgreSQL 侧的事：hook、代价、执行器协议、共享内存。
  它不包含任何距离计算或图遍历代码。
- **引擎层**（`src/engine/engine.cpp`）是纯 C++，不含任何 PostgreSQL 头文件，
  通过 `engine_api.h` 这一个头文件对外暴露。它拥有索引文件格式、HNSW 构建、
  两个 CPU 游标，以及把编译后谓词序列化成可进共享内存的扁平结构的能力。
- **CUDA 层**（`src/engine/gpu_engine.cu`）只被引擎层调用，接口是 `gpu_engine.h`。
  没有 nvcc 时整个文件不参与编译，扩展退化为纯 CPU 版本。

## 3. 四条执行路径

同一个索引文件同时支撑四条路径。它们只在"怎么找候选"上不同，输出契约一致：
按距离升序给出通过谓词的候选行。

| 路径 | 设备 | 候选来源 | 结果性质 |
|---|---|---|---|
| `brute` | CPU 多线程 | 倒排表驱动 span，无 span 时全扫 | 精确 |
| `hnsw` | CPU 单线程 | HNSW 第 0 层束搜索 | 近似（`paves.ef_search` 控制） |
| `gpu_brute` | GPU | 分块全扫，或按 span 枚举 | `sq8` 模式近似，`fp32` 模式精确 |
| `gpu_hnsw` | GPU | HNSW 第 0 层束搜索，一个 block 一条查询 | 近似 |

"精确"指结果与对全体满足谓词的行做穷举排序一致。近似路径的误差来源是束宽（图搜索）
和标量量化（`sq8`），两者都不影响正确性——执行器对每条返回的元组重新求值全部原始谓词。

## 4. 一次查询的端到端流程

### 构建期（一次性，见 [02](02-index-build.md)）

`SELECT paves_build('items')` 顺序扫描堆表，把向量、可缓存的标量列、堆元组 TID 收进
内存，构建 HNSW 图与每列的倒排表，写成单一文件
`$PGDATA/paves/<dbid>_<relid>.idx`，用 `rename(2)` 原子替换旧文件。

### 计划期（见 [03](03-planner-routing.md)）

1. `set_rel_pathlist_hook` 检查查询形态与索引文件是否存在。
2. 逐条检查 `baserestrictinfo`，判定哪些约束条件能下推给引擎；不能下推的留给执行器。
3. 用 PostgreSQL 自己的选择率估计 `sel = rel->rows / rel->tuples`，
   为四条路径各算一个"预测毫秒数"，乘统一系数换成代价单位，加进 `pathlist`。
4. 优化器按代价选一条。被选中的路径在 `PlanCustomPath` 里落成 `CustomScan` 计划节点，
   携带策略号、k、选择率估计（千分比）、查询向量表达式和下推子句。

### 执行期（第一次取元组时启动）

1. `BeginCustomScan` 换掉 PG14 默认的虚拟扫描槽，改用真正的表槽并重新编译 qual 与
   投影——因为本节点返回的是回表得到的 heap 元组（见 [04](04-executor-cpu.md) §1）。
2. 求值查询向量；把下推子句编译成引擎谓词树；`fv_pred_compile` 顺带从倒排表推导
   **驱动 span**（谓词在按值排序的行号数组上对应的连续区间）。
3. 按策略开游标：
   - CPU 策略 → 直接调 `fv_brute_begin` / `fv_hnsw_begin`。
   - GPU 策略（worker 模式，默认）→ 把查询序列化进共享内存槽位，唤醒 GPU worker，
     等待回填（见 [05](05-gpu-arbiter.md)）。任何容量不足或 worker 不可用的情况都
     退回 CPU 游标。
4. 每次执行器要元组时，从游标（或槽位结果）取一批候选行的 TID，
   `table_tuple_fetch_row_version` 在查询快照下回表；不可见的候选跳过。
5. `ExecScan` 对每条元组重新求值该节点上的全部 qual。下推只可能多给候选，不可能少给，
   因此不影响结果正确性。

### 反馈回写

CPU 路径在扫描结束时把实测引擎耗时（以及测量当时的 CPU 并发水位）写入共享内存的
反馈表；GPU worker 每跑完一个微批把摊销后的每请求服务时间、整批墙钟和批规模写入同一张表。
下一次计划期的定价直接读这张表。这条回路是 auto 路由能跟上负载变化的原因，
细节见 [03](03-planner-routing.md) §4。

## 5. 正确性边界

- **可见性**：候选只是行号，最终元组一律经 `table_tuple_fetch_row_version` 在查询快照下
  取得。索引构建之后被删除或更新的行会在这一步被过滤掉（`EXPLAIN ANALYZE` 里计为
  `Invisible Skipped`）。
- **谓词语义**：引擎把 NULL 存成 NaN，任何与 NaN 的比较都为假。经过 `NOT` 之后这与
  SQL 三值逻辑不同，只可能产生**假阳性**（多给候选）。执行器的 qual 重查兜底。
- **编译失败**：任何一个顶层合取项编译不出来就直接跳过，谓词位图变成超集，同样由重查兜底。
- **GPU 结果**：worker 把 kernel 返回的行号翻译成 TID 时用带边界检查的接口，
  越界行号被丢弃并告警，不会越界读 mmap。

## 6. 关键限制

| 限制 | 数值 | 出处 |
|---|---|---|
| 索引行数 | < 2^32 | 引擎内部行号是 uint32 |
| 向量维度 | ≤ 2000（GPU 路径），快速 kernel ≤ 2048 | `FV_ARB_MAX_DIM`、tile 变体表 |
| 缓存标量列 | ≤ 32 | `FV_MAX_COLS` |
| GPU 单次 top-B | ≤ 4096 | `FV_GPU_MAX_TOPB` |
| GPU 图搜索 ef | ≤ 512 | `FV_GPU_MAX_EF`（共享内存约束） |
| 单微批查询数 | ≤ 1024 | `FV_GPU_MAX_BATCH` |
| 仲裁槽位数 | 512 | `FV_ARB_SLOTS` |

超出这些上限的请求不会报错，而是退回 CPU 路径。`paves_capacity(regclass)` 在建索引前
就能把这些数字算出来，见 [07-reference.md](07-reference.md) §3。
