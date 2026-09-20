# PAVES

PostgreSQL 14 的过滤向量检索扩展。直接执行 `WHERE 标量谓词 + ORDER BY 向量距离 + LIMIT k`，在 CPU/GPU 的扫描与图搜索路径之间选择；支持 GPU worker 微批处理。

```sql
CREATE EXTENSION vector;
CREATE EXTENSION paves;
SELECT paves_build('items');

SELECT id FROM items
WHERE price < 100 AND category IN (3, 7)
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;
```

`items` 需包含维度匹配的 `vector` 列。安装前设置 `shared_preload_libraries = 'paves'` 并重启实例。

## 验证与复现

本发布的验收范围为 **SIFT1M、128 维 L2、六类合成标量过滤、top-10、64 并发 SQL 查询**。双方均须在独立确认集的每类平均 recall@10 ≥ 0.9，六类及混合吞吐的三轮中位数比均须 ≥ 5。具体实测值、环境及限制见 [验证报告](docs/validation.md)，原始证据保存在 [results/2026-09-20](results/2026-09-20)。

2026-09-20 实测通过：**混合吞吐 6.00×，六类 5.53–9.63×；PAVES 每类最低 recall 为 0.909，pgvector 为 0.911。** 本组主要走 CPU HNSW，不能将加速归因于 GPU。

```bash
export CUDA_VISIBLE_DEVICES=0  # 一张空闲 GPU
bash bench/reproduce.sh
```

脚本下载公开数据、检查校验和、构建依赖和扩展、生成属性与独立查询集、构建两个索引、计算精确真值、校准参数并进行三轮测试。未达到验收门槛时非零退出。完整环境要求、分步命令、计时边界与结果解释见 [复现说明](docs/reproduce.md)。

只安装扩展：

```bash
make -C extension/paves PG_CONFIG=/path/to/pg14/bin/pg_config
make -C extension/paves install PG_CONFIG=/path/to/pg14/bin/pg_config
```

需要 PostgreSQL 14、pgvector、C++17；检测到 nvcc 时启用 GPU，架构默认 `sm_89`，可通过 `CUDA_ARCH` 修改。无 CUDA 时可编译 CPU 版本。

## 目录与边界

| 路径 | 内容 |
|---|---|
| `extension/paves/` | PostgreSQL 胶水、CPU 引擎、CUDA 算子及正确性检查 |
| `bench/` | 从公开输入到验收结果的复现脚本 |
| `docs/` | 复现说明、验证报告、[实现文档](docs/README.md) |
| `results/` | 本次发布的原始结果、逐查询召回、环境与源码哈希 |

这是静态索引研究实现：只支持 L2 和带 LIMIT 的单距离排序；表更新后需重新构建索引，回表可见性检查不能代替索引维护。性能结论限于报告中的环境和负载，不保证任意数据、谓词或硬件都有相同加速。自动路由的系统收益不能直接归因于 GPU；实际路径见执行计划。
