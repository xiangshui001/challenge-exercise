# 09｜最新实施交接

当前路线见 [12](12_TRAINING_ROADMAP.md)：成熟 Flickr30K，SigLIP-base 未微调基线、简单部分微调、至多两档学习率、统一评估与错误分析。原 CLIP/SigLIP 2→Qwen 方案不再默认推进。

PR #2 已合并。exp5 审计/指标/缓存/统计有 24 项 CPU 测试；其 prepare 仅 val/test，B0/B1 仅 CLIP/SigLIP 2。新 SigLIP-base 封装与短小试已实现，真实权重尚未运行；train 数据层/checkpoint 适配还没有。

| 阶段 | 下一步 | 验收证据 |
|---|---|---|
| P0 | 核心环境已通过；服务器 checkout 仍待同步 | Python 3.11.16、torch 2.8.0+cu126、Transformers 4.57.3、pip check、A100 张量计算 |
| P1 | 用户运行 [15](15_SIGLIP_SMOKE.md) 的 SigLIP-base 小试 | revision、真实特征/相似度、参数名、显存/依赖 |
| P2 | train/val/test 审计、未微调基线 | 正确划分、完整图库、逐查询预测 |
| P3 | 部分微调、InfoNCE/AdamW、验证/checkpoint | loss、解冻层、验证结果、恢复状态 |
| P4 | 至多两档学习率，同预算验证 | 不用 test 选参，冻结配置 |
| P5 | 独立 test、修复/退化、人工分析/报告 | 可复算结果和版本/成本记录 |

## 交互式任务说明

```text
读取 AGENTS、README、docs/12、13、14 和本交接。
按最新阶段记录推进，主模型为 google/siglip-base-patch16-224。
保持统一目录，复用 exp5 指标；不把旧 B1 当作新模型，不写空跑训练脚本。
先检查实际模型层名，再定部分解冻范围。
云端只改仓库/做 CPU 检查，给用户具体服务器命令并接收日志继续修正。
不索要 SSH、不直接连接服务器、不自动下载或运行 GPU。
按需使用空闲 GPU，用完释放；卡 2 只是当前示例，不是固定限制。
每个可复现阶段提交文件、配置、实际证据和进度，推送 PR。
```

当前训练 JSON 与原 YAML 都是规划，不能执行。revision、层范围、batch 和指标空值保持待定，不自动置 0。CPU 合成测试不等于真实推理/训练完成，允许微调没有提升。

同步见 [13](13_GITHUB_SERVER_SYNC.md)。本地数据/权重/checkpoint 不上传；服务器修改先提交/推送再接续。每次记录起始 commit、模型/数据/依赖、实际命令、输出相对路径与成功/失败/未运行部分。
