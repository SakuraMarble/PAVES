# 发布验证：2026-09-20

**验收通过。在下述 SIFT1M 负载上，双方每类平均 recall@10 均 ≥ 0.9；PAVES 相对 pgvector 的混合 SQL 吞吐为 6.00 倍，六类分别为 5.53–9.63 倍。**

这是从干净源码、全新数据库和重新构建的两个索引得到的结果，不是历史实验数字的整理。校准与确认查询集互不重叠。全部 42 组正式计时均为零查询错误。

## 结果

64 并发，预热 2 秒、测量 10 秒，共 3 轮；QPS 为三轮中位数。每类 167 条确认查询，混合为同一组 1,002 条等权交错查询。

| 负载 | pgvector recall | PAVES recall | pgvector QPS | PAVES QPS | 加速比 |
|---|---:|---:|---:|---:|---:|
| Equality_bool | 0.9246 | 0.9240 | 45,321 | 265,668 | 5.86× |
| Equality_int | 0.9533 | 0.9096 | 18,657 | 179,648 | 9.63× |
| Inclusion | 0.9114 | 0.9240 | 40,543 | 226,857 | 5.60× |
| Range_10 | 0.9545 | 0.9090 | 18,218 | 163,821 | 8.99× |
| Range_50 | 0.9234 | 0.9156 | 45,621 | 252,142 | 5.53× |
| Logic | 0.9743 | 0.9563 | 11,298 | 98,327 | 8.70× |
| **mixed** | **0.9402** | **0.9231** | **23,352** | **140,217** | **6.00×** |

混合负载的三轮 QPS 也有波动：PAVES 为 119,518 / 140,217 / 142,415，结果没有丢弃较慢轮次。用 **PAVES 最低轮 / pgvector 最高轮** 计算，混合仍为 **5.09×**，六类最低为 **5.33×**。

首次召回使用 16 个连接；计时后以 64 个连接、相同固定配置重新执行全部确认查询，得到相同的逐类 recall，所有查询都返回 10 个不重复结果。recall 是每类均值，不代表每条查询至少命中 9 个。

## 参数与环境

- SIFT1M：1,000,000 × 128，L2、top-10；标量属性独立随机生成，种子 `20240617`。六类固定谓词约覆盖 5%–50% 通过率。
- 双方建图 `M=16, ef_construction=200`；pgvector 保留标量 B-tree；PAVES 建图 32 线程。pgvector 建图约 95.02 秒，PAVES 约 17.88 秒，这些时间不计入查询吞吐。
- pgvector：`paves.enable=off; hnsw.ef_search=40; hnsw.iterative_scan=relaxed_order; hnsw.max_scan_tuples=200000`。
- PAVES：`paves.enable=on; paves.force_strategy=auto; paves.ef_search=25`。
- Ubuntu 24.04.4；双路 AMD EPYC 9654，共 192 核 / 384 线程；约 1 TiB 内存；一张可见 RTX 4090 24 GiB（物理 GPU 2）。
- PostgreSQL 14.23、pgvector 0.8.5（源码提交 `159b79aaad5983fb7459c1e3df2897fbb2d11788`）、GCC/G++ 12.4、CUDA 12.6.85、NVIDIA 驱动 595.71.05、Python 3.12.3。
- PostgreSQL 共享缓冲 8 GiB，`work_mem=64MB`、`maintenance_work_mem=4GB`，并行查询关闭，JIT 关闭；其他设置及二进制哈希保存在环境文件和数据库配置快照中。

**六类采样执行计划均为 PAVES CPU HNSW。这个结果证明上述负载的 PAVES 系统吞吐优势，不能作为 GPU 相对 CPU 的消融结论。** GPU 算子已经编译并运行 smoke 检查，完整 GPU 性能评估不属于本次 5 倍验收结论。

## 验证与证据

- [完整复现步骤与公平性口径](reproduce.md)。
- [最终汇总](../results/2026-09-20/results/summary.json)、[原始校准与三轮计时](../results/2026-09-20/results/events.jsonl)。
- [逐查询 SQL 与独立真值](../results/2026-09-20/results/workload.jsonl)、[PAVES 逐查询 recall](../results/2026-09-20/results/paves-recall.json)、[pgvector 逐查询 recall](../results/2026-09-20/results/pgvector-recall.json)。
- [64 并发复核](../results/2026-09-20/results/concurrent-recall.json)、[执行计划](../results/2026-09-20/results/plans.json)、[独立审计](../results/2026-09-20/audit.json)。
- [环境与实测源码哈希](../results/2026-09-20/environment.json)、[数据校验](../results/2026-09-20/data/manifest.json)、[CPU/GPU 正确性检查](../results/2026-09-20/smoke.json)。

无需数据库即可重算已提交结果：

```bash
python3 bench/verify_results.py results/2026-09-20/results
python3 -m unittest discover -s bench -p 'test_*.py' -v
```

验收脚本覆盖数字一致性、任何查询错误、缺失轮次、重复 ID 和虚假 recall 检测。源代码与测量绑定由文件哈希确认；发布没有修改实测的检索引擎、压测驱动、数据生成或主基准脚本。

108 项 CPU/GPU smoke 检查通过：brute 与 gpu_brute 在这些查询上的 recall 都为 1.0；两条图路径的单查询最低值均为 0.9。另验证无效 SET 和零客户端会使压测驱动非零退出；7 项离线审计回归测试全部通过。Smoke 样本只用于正确性排查，不承担主性能结论。

局限：标量和谓词为合成负载；未测试生产轨迹、动态更新、高维数据或所有选择率；比较的是单一通用 pgvector HNSW 索引与公开参数网格，不是所有可选索引设计的最优组合。服务器与其他用户共享，未人为干扰基线。重新建图和硬件差异可能改变结果；复现脚本必须重新验收，不能把本报告数字视为所有环境的保证。
