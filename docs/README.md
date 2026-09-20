# PAVES 文档

PAVES 是一个 PostgreSQL 14 扩展，把"向量近邻排序 + 标量条件过滤"的混合查询接管到自己的
执行路径上，在 CPU 与 GPU、扫描与图搜索之间逐查询选择算子。

本目录按**执行流程分层**组织：先看总览，再按需要深入某一层。

| 文档 | 覆盖范围 |
|---|---|
| [01-overview.md](01-overview.md) | 总览：查询形态、组件划分、一次查询的端到端流程 |
| [02-index-build.md](02-index-build.md) | 构建期：`paves_build()` 的扫描过程、索引文件格式、HNSW 与倒排表的构建 |
| [03-planner-routing.md](03-planner-routing.md) | 计划期：查询识别、谓词下推判定、代价模型与四路路由 |
| [04-executor-cpu.md](04-executor-cpu.md) | 执行期（CPU）：CustomScan 执行器、谓词编译、驱动 span 扫描与过滤图搜索 |
| [05-gpu-arbiter.md](05-gpu-arbiter.md) | 执行期（GPU 前半）：共享内存槽位协议、GPU worker 的微批组装与结果回填 |
| [06-gpu-kernels.md](06-gpu-kernels.md) | 执行期（GPU 后半）：索引上传与 SQ8 量化、三个暴力 kernel、图搜索 kernel |
| [07-reference.md](07-reference.md) | 参考：GUC 列表、SQL 接口、容量上限、编译方式、诊断开关 |

## 代码位置

```
extension/paves/
  paves.control          扩展元数据
  sql/paves--0.1.sql     SQL 函数定义
  Makefile               PGXS 构建（C + C++17 + 可选 CUDA）
  src/
    paves.c              入口：GUC 定义、hook 与 CustomScan 方法注册
    planner.c            计划期：路径识别、代价模型、四条 CustomPath
    quals.c              谓词形态判定与编译
    exec.c               执行期：CustomScan 执行器
    build.c              paves_build()
    capacity.c           paves_capacity() / paves_feedback()
    arbiter.h/.c         GPU 微批仲裁：共享内存结构与后端提交
    worker.c             GPU worker 后台进程
    engine/
      engine_api.h       引擎 C 接口（PG 与引擎之间的唯一边界）
      engine.cpp         CPU 引擎：索引文件、HNSW 构建、两个 CPU 游标
      gpu_engine.h/.cu   CUDA 引擎：索引上传、暴力与图搜索 kernel
  test/                  独立 SQL 真值的正确性检查
  tools/gpu_micro.cpp    脱离 PostgreSQL 的算子级压测工具

bench/                   基准与数据集装载脚本（见 bench 各脚本头部说明）
results/                 发布验证的原始测量和环境记录
```

复现入口见 [reproduce.md](reproduce.md)。本发布只保留当前实现、复现工具及相关说明。
