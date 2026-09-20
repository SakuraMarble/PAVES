# 03 计划期：识别、下推判定与路由

计划期做三件事：判断这条查询能不能接管、判断哪些谓词能下推给引擎、给四条路径定价。
代码在 `src/planner.c`（识别与定价）和 `src/quals.c`（下推判定）。

## 1. 查询识别

挂在 `set_rel_pathlist_hook` 上，逐条检查，任何一条不满足就直接返回：

1. `paves.enable` 为 on。
2. 目标是普通基表（`RTE_RELATION` + `RELKIND_RELATION` + `RELOPT_BASEREL`）。
3. `root->query_pathkeys` 恰好一条，方向为升序，且排序键的等价类不含 volatile 表达式。
4. `root->limit_tuples` 落在 `(0, 1000000]`，取整作为 k。
5. pgvector 的 `<->`（vector,vector）操作符能查到 OID。
6. 该关系存在索引文件。
7. 排序键的等价类里存在一个成员，形如 `Var <-> expr` 或 `expr <-> Var`，其中 Var 是
   本关系的、属性号等于索引记录的向量列属性号的变量；另一侧不含变量、不含 volatile 函数。
   这一侧就是查询向量表达式，允许是 Const、Param 或稳定函数调用——它在执行器 Begin 时求值。

编译时定义 `PAVES_DEBUG_PLAN` 会让每次放弃都打一条 NOTICE 说明原因。

## 2. 谓词下推判定

对 `rel->baserestrictinfo` 的每个 `RestrictInfo` 调 `paves_clause_pushdownable()`。
通过的进 `pushdown_clauses`（后续序列化进计划节点），不通过的只留给执行器重查。

可下推的形态是 AND / OR / NOT 组成的树，叶子是以下三种之一：

- `Var op 常量侧`：op 能映射到 btree 策略号（`<`、`<=`、`=`、`>=`、`>`、`<>`），
  Var 在右侧时策略号做交换处理。
- `Var = ANY (array)`（即 `IN (...)`，要求 `useOr` 且操作符是等值）。
- 裸的布尔列引用。

附加约束：

- Var 必须是本关系的**缓存标量列**（`fv_index_col_slot()` 查得到槽位）。
- 常量侧不含变量、不含 volatile 函数。
- 类型限于 `bool / int2 / int4 / int8 / float4 / float8`。`int8` 超过 2^52 时转
  `double` 会失真，直接判为不可下推。

判定与编译共用同一段递归代码（`compile_rec`），靠 `econtext` 是否为 NULL 区分两种模式：
计划期传 NULL，只做形态检查；执行期传真实的表达式上下文，产出谓词节点。
这样"能不能下推"和"下推成什么"永远不会分叉。

编译在**执行器 Begin 时**而不是计划期完成，因为 Param 的值那时才可用。

## 3. 加入候选路径

选择率取 PostgreSQL 自己的估计：`sel = rel->rows / rel->tuples`（无统计信息时为 1）。

- `brute` 与 `hnsw` 在 auto 模式或被显式强制时加入。
- `gpu_brute` 与 `gpu_hnsw` 只有在**全部**满足下列条件时才加入 auto 的竞价：
  `paves.gpu_enable = on`、`paves.gpu_mode = worker`、共享内存已初始化、
  worker 已就绪、`dim ≤ 2000`、`max(4k, 64) ≤ 1024`。
  被 `paves.force_strategy` 显式强制时则无条件加入（执行期发现不可用会自行退回 CPU）。

每条路径是一个 `CustomPath`，`custom_private` 里放
`(策略号, k, 查询向量表达式, 下推子句列表, 选择率千分比)`。
`pathkeys` 直接设为 `root->query_pathkeys`——路径自称已按距离有序，
所以上层不会再加 Sort 节点。`rows` 设为 `min(k, 匹配行数估计)`。

`PlanCustomPath` 把表达式类（查询向量、下推子句）放进 `custom_exprs`，把整数类
（策略号、k、选择率千分比）放进 `custom_private`。这个划分是必须的：
`setrefs` 阶段只对 `custom_exprs` 做 Var 编号修正。

## 4. 代价模型

四条路径都折算成**预测毫秒数**，再乘同一个系数 `FV_COST_UNITS_PER_MS = 7.5` 变成
代价单位，最后乘 `paves.cost_scale`（默认 0.01）。乘 `cost_scale` 只是为了让这四条路径
整体压过 PostgreSQL 的其他计划，四者之间的相对关系不受影响。

每条路径的预测由三部分合成：**解析先验**给出形状，**运行时反馈**锚定水平，
**实时负载**做外推。

### 4.1 解析先验

`cpu_prior_ms()`，描述一台空载机器上的服务时间。常数标定自 SIFT1M（128 维、100 万行、
EPYC 9654），实测 brute/HNSW 交叉点约在 1.5% 选择率，模型给出约 1.9%。

```
dist_unit = dim * cpu_operator_cost * 0.02          每维 SIMD 距离计算

brute:  enumerated = 有驱动 ? max(sel*N, 1) : N
        units = enumerated * (dist_unit + cpu_operator_cost * 0.2) / threads

hnsw:   ef_eff  = max(paves.ef_search, 2k)
        visited = min(N, ef_eff / max(sel, 0.0005) + ef_eff * 4)
        units   = visited * (dist_unit + cpu_operator_cost * 2.0)
```

brute 的形状是"枚举量 × 每行成本 / 线程数"；有下推驱动时枚举量是 `sel*N`，
没有时是全表。hnsw 的形状是"访问节点数 × 每跳成本"，单线程；由于结果堆只接纳通过
谓词的节点，束宽要在 `ef/选择率` 量级的节点里才能填满，这就是 `ef_eff / sel` 项。

GPU 路径没有对应的解析先验，只有冷启动种子值：暴力 1.0 ms、图搜索 0.5 ms。

### 4.2 运行时反馈表

共享内存里有 16 个反馈条目（`FvFeedbackEntry`），按 `(dbid, relid)` 认领——
一个 8M×1024d 索引的延迟不该污染一个 1M×128d 索引的路由。条目内再按
**策略侧**（0 = 暴力，1 = 图）和 **5 档选择率桶**细分：

| 桶 | 选择率 | 锚点中值 |
|---|---|---|
| 0 | ≥ 20% | 0.40 |
| 1 | 5% ~ 20% | 0.10 |
| 2 | 1% ~ 5% | 0.022 |
| 3 | 0.2% ~ 1% | 0.0045 |
| 4 | < 0.2% | 0.001 |

所有字段是微秒单位的 EWMA，0 表示还没有数据。分桶是必要的：单一均值无法表达
"GPU 在 3% 选择率上很好、在 0.1% 上很差"。

谁写什么：

- **GPU worker** 每跑完一个组写 `gpu_q_us[策略][桶]`（摊销后的每请求服务时间）、
  `gpu_batch_us[策略]`（整批墙钟）、`gpu_batch_n16[策略]`（平均批规模 ×16）。
  归因方式见 [05-gpu-arbiter.md](05-gpu-arbiter.md) §5。
- **后端**在 CPU 游标结束时写 `cpu_q_us[策略][桶]`（实测引擎耗时，带负载原值）与
  `cpu_ref_active16[策略]`（测量当时的 CPU 并发水位 ×16）。
  只有**计划为 CPU 策略**的查询才写；GPU 请求降级出来的 CPU 游标不写。

最后一点很关键：降级路径跑的是加宽 ef、带 seen_tids 去重的搜索，比正常规划的 CPU 搜索
贵得多，把它当作 CPU 样本会把 CPU 报价顶死。

### 4.3 GPU 定价

```
gpu_ms = 0.4                             提交 / 唤醒 / 回答的固定往返
       + q_ms                            自己那一份摊销服务时间
       + batch_ms * (inflight / navg)    前面排队的"批"数 × 每批墙钟
```

- `q_ms` 取本桶的观测；本桶为空时向两侧找最近的非空桶，图策略再按
  `mid(源桶)/mid(本桶)` 缩放（图的工作量大致正比于 1/选择率，暴力则跨桶基本平坦）。
- 排队按**批**而不是按请求计价。kernel 一次并行服务整个微批，
  按请求计价会把等待高估大约一个批规模，正是这一点会在高并发下把 auto 推离 GPU。
- 自己那份按**摊销（边际）**费率而不是整批时间计价：这样流量增长能把批做大，
  而不是自我抑制。
- 反馈表为空时 `batch_ms` 退化为 `q_ms`（乐观），让流量先流起来把表填上。

### 4.4 CPU 定价

```
base   = 观测毫秒 * (prior_now / prior_src)      用先验形状把邻桶观测插值到本桶
base   = max(base, prior_now * 0.125)            下限：只锚不封顶
cpu_ms = base * clamp(now / max(ref, 1), 0.25, 4)
```

`now` 是实时 `cpu_active` 计数，`ref` 是观测被采集时的水位。这个比值是**即时负反馈**：
一开始灌 CPU，后续每一次定价立刻变贵，不必等 EWMA 吸收新样本。

两个细节是踩过坑之后定下来的：

- **不设上限**。高并发下真实的带负载成本可以超过空载先验的任意倍数，
  一旦封顶就会在高客户端数下失衡。下限（先验的 1/8）只用来防止被取消查询的
  残片样本把估计压到荒谬的低位。
- **双向 clamp**。比值必须能小于 1：单向 `max(1, now/ref)` 会让洪峰过后的观测
  永远停在洪峰水平，把已经空闲的设备饿死。

没有任何观测时退回 `解析先验 × 饱和因子`，饱和因子 =
`max(1, cpu_active * 每查询线程数 / paves.cpu_cores)`。

### 4.5 反馈写入的 EWMA 系数

| 字段 | α | 理由 |
|---|---|---|
| `cpu_q_us` | 下降 0.25 / 上升 1/32 | 启动期一波首次触碰缺页不该把路由赶下 CPU；真实的负载上移靠样本量进来 |
| `cpu_ref_active16` | 1/16 | |
| `gpu_q_us` | n_bucket / (n_bucket + 64) | 按桶内请求数加权：小批不该污染估计 |
| `gpu_batch_us` / `gpu_batch_n16` | 0.125 | |

EWMA 更新是无锁的读-改-写，存在竞争；对反馈数据而言逐字段原子性已经足够。

## 5. 观察与调试

- `EXPLAIN` 显示所选策略与下推子句数量；`EXPLAIN ANALYZE` 追加 GPU 模式、
  谓词匹配估计、取回候选数、跳过的不可见候选数。
- `paves_feedback()` 导出反馈表的全部 EWMA 与实时负载计数器，
  每 `(索引 × 策略侧 × 桶)` 一行。这是判断"路由为什么这么选"的第一手工具。
- `paves.force_strategy` 可以强制单条路径，用于隔离算子性能。
- `paves.gpu_enable = off` 让 auto 只在两条 CPU 路径之间选。
