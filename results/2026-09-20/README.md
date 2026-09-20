# 2026-09-20 发布证据

这些文件来自本次独立重建和实测，保留原始数字。阅读入口：[验证报告](../../docs/validation.md)。

- `results/`：主基准与并发召回复核，`events.jsonl` 包含全部校准候选与 42 组正式计时。
- `environment.json`：测量开始时记录的硬件、依赖、二进制和源码 SHA-256。服务器使用源码导出目录而非 Git checkout，所以其中 `git rev-parse` 报错是预期的；用源码哈希对应发布文件。`._*` 是首次传输的 macOS 元数据，未编译，不包含在发布源码中；旧 SQL 测试已由独立真值的 `smoke.py` 替代。
- `data/`：原始输入、生成 CSV 校验和、构建耗时。
- `audit.json`：服务端离线审计结果；可用仓库脚本在本地重算。
- `smoke.json`：最终 CPU/GPU 正确性检查输出。
- `build.log`、`prepare.log`、`load.log`：构建与装载日志；环境中路径属于实测服务器，不是复现脚本的依赖。

完整性清单 `SHA256SUMS` 覆盖本目录内的原始证据文件。在本目录执行 `sha256sum -c SHA256SUMS` 校验。
