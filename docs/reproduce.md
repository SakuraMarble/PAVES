# 复现

## 环境

验证机器为 Ubuntu 24.04.4、双路 AMD EPYC 9654（共 192 物理核 / 384 线程）、约 1 TiB 内存、RTX 4090 24 GiB。程序只暴露一张卡。依赖固定为 PostgreSQL 14.23、pgvector 0.8.5、GCC/G++ 12.4、CUDA 12.6、Python 3.12；Python 包版本见 `bench/requirements.txt`。

主基准为完整 SQL 往返吞吐，64 个并发连接，两边使用相同客户端、数据库实例、表与查询。PAVES 使用自动路由；包含 GPU 源码不等于这组加速来自 GPU。执行路径以结果中的 `plans.json` 为准。

Linux 构建依赖：`build-essential git curl libreadline-dev zlib1g-dev libssl-dev bison flex python3.12-venv`。GPU 构建另需 NVIDIA 驱动和 CUDA Toolkit；RTX 4090 对应 `CUDA_ARCH=sm_89`。建议至少 32 GiB 内存和 15 GiB 可用磁盘。小机器可减小缓冲区及构建线程，但不能据此保证达到验证机器上的吞吐比。

## 一键运行

```bash
git clone git@github.com:SakuraMarble/PAVES.git
cd PAVES
export PAVES_RUN="$PWD/runtime"
export CUDA_VISIBLE_DEVICES=0  # 选择一张空闲卡
bash bench/reproduce.sh
```

脚本在用户目录安装工具链，不需要 root；数据库只监听独立 Unix socket，端口默认 55420。启动目录必须是全新目录，防止覆盖旧实验；退出时停止本次数据库。可通过 `PAVES_RUN`、`PG_PREFIX`、`PAVES_PORT`、`SIFT_SOURCE`、`JOBS` 和 `CUDA_ARCH` 覆盖默认值。路径请不要包含空格或单引号。

可用已有 PostgreSQL 14.23 + pgvector 0.8.5 安装作为 `PG_PREFIX`，但 PAVES 会安装到该前缀，应使用专用副本。本次验证复用了原服务器的 PostgreSQL/pgvector 工具链副本；PAVES、压测驱动、数据、两个索引和数据库均为本次重新生成，未复用历史索引或历史性能数字。

公开数据来自 [ANN-Benchmarks SIFT HDF5](https://ann-benchmarks.com/sift-128-euclidean.hdf5)，原始数据介绍见 [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)。脚本检查固定 SHA-256：

```text
dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984
```

## 分步运行

一键脚本就是以下步骤的组合；需要排查某一步时可单独执行：

```bash
export PAVES_RUN="$PWD/runtime"
export PG_PREFIX="$PAVES_RUN/pg14"
python3.12 -m venv "$PAVES_RUN/venv"
"$PAVES_RUN/venv/bin/pip" install -r bench/requirements.txt
bash bench/bootstrap.sh
"$PAVES_RUN/venv/bin/python" bench/prepare.py \
  --source /path/to/sift-128-euclidean.hdf5 --out "$PAVES_RUN/data"
CUDA_VISIBLE_DEVICES=0 bash bench/cluster.sh start
"$PG_PREFIX/bin/createdb" -h "$PAVES_RUN/socket" -p 55420 paves_bench
export DSN="host=$PAVES_RUN/socket port=55420 dbname=paves_bench"
"$PAVES_RUN/venv/bin/python" bench/load.py --dsn "$DSN" --data "$PAVES_RUN/data"
"$PAVES_RUN/venv/bin/python" bench/environment.py \
  --pg-prefix "$PG_PREFIX" --out "$PAVES_RUN/environment.json"
"$PAVES_RUN/venv/bin/python" bench/benchmark.py --dsn "$DSN" --out "$PAVES_RUN/results"
"$PAVES_RUN/venv/bin/python" bench/recheck.py --dsn "$DSN" --results "$PAVES_RUN/results"
"$PAVES_RUN/venv/bin/python" bench/verify_results.py "$PAVES_RUN/results"
"$PAVES_RUN/venv/bin/python" extension/paves/test/smoke.py --dsn "$DSN" --gpu
bash bench/cluster.sh stop
```

## 实验口径

- 数据：1,000,000 个 128 维 SIFT 向量，L2、top-10。三个标量属性由 NumPy `default_rng(20240617)` 合成，分别为 0/1、0–9 整数和 [0,100) 浮点数，与向量独立。
- 谓词：布尔等值、整数等值、IN、10% 范围、50% 范围、等值与范围的 AND；实际通过率约 5%–50%。这不是生产谓词轨迹，也没有验证极低通过率、高维数据或任意谓词都能达到 5 倍。
- 查询：原始 test 向量的前 2,004 条；前 1,002 条用于校准，后 1,002 条用于确认，每类各 167 条。类别按 query ID 循环分配；混合负载六类等权交错。
- 真值：PAVES 关闭、索引扫描关闭、并行查询关闭的 PostgreSQL 精确扫描；按距离与 ID 打破并列，保存逐查询 top-10。ANN 的 recall 按 ID 集合交集计算，并列边界可能导致相同距离的另一个合法 ID 不计命中。
- 索引：双方 `M=16, ef_construction=200`，同一份浮点向量；pgvector 保留三个标量 B-tree 索引。没有为固定谓词额外建立专用图或 partial index。
- 校准：pgvector 扫描 `ef_search=20/30/40/60/80/120/200` 与 `off/strict_order/relaxed_order`，`max_scan_tuples=200000`；PAVES 扫描 `ef_search=10/15/20/25/30/40/60`。每组满足全部类别平均 recall ≥ 0.9 后，用两次 3 秒混合查询吞吐选最快配置。两边各使用一组配置，不按确认类别调参；这只是公开网格中的最快配置，不宣称全局最优。
- pgvector 的 relaxed-order 模式也参与竞争，允许返回集合内部顺序略有变化；评估为 top-10 集合 recall，不额外对它收费重排序。模式语义见 [pgvector 0.8.5 文档](https://github.com/pgvector/pgvector/tree/v0.8.5#iterative-index-scans)。
- 测量：每组预热 2 秒、测量 10 秒，3 轮交替系统顺序。SQL 执行、计划、结果回传与必要 GPU 通信都在端到端计时内；不包含一次性编译、加载、建索引或真值计算。QPS 取三轮中位数之比，另保留 PAVES 最低轮 / pgvector 最高轮之比。
- 验收：双方确认集的每类平均 recall ≥ 0.9、返回足额结果、压测零错误；六类与混合的中位数吞吐比全部 ≥ 5 才返回成功。这里的 recall 门槛是每类均值，不是每条查询都 ≥ 0.9。

真值与首次 recall 使用 16 个连接；计时结束后 `recheck.py` 固定配置，用与压测相同的 64 个连接再次检查确认集 recall 和足额率，不重新调参。`verify_results.py` 则无需数据库，从原始 ID 和计时事件独立重算报告数字。确认集未参与配置选择。

`summary.json` 的 `passed=false` 或脚本非零退出就是未达标，不能拿历史数字替代。新机器、不同负载、重新并行建图和共享服务器噪声均可能改变结果。运行前应选空闲资源；本次未停止其他用户进程，也没有人为给 pgvector 添加背景负载。

## 输出

`data/manifest.json` 保存数据与种子校验；`data/build.json` 保存建索引时间；`environment.json` 保存硬件、软件与源文件哈希。`results/` 保存网格搜索和计时事件、SQL 工作负载、逐查询真值与召回、配置、执行计划及最终汇总。源码仓库不包含大型数据、数据库、编译产物或研究草稿。
