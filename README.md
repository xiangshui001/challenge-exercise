# challenge-exercise

**当前项目：基于 SigLIP 微调的跨模态检索优化实验。**

实验五的第一次训练项目：成熟 Flickr30K → SigLIP-base 未微调基线 → 简单部分微调 → 至多两档学习率 → 冻结、测试和错误分析。先理解并跑通完整链路，不自建数据集、不从零预训练、不默认加入 hard negative、LoRA、重排或收费 API。

主模型：`google/siglip-base-patch16-224`，其封装、训练 Dataset/程序尚未实现。已有 `exp5` 是此前 CLIP/SigLIP 2 评测基础代码，审计/指标/统计可以复用；24 项 CPU 测试通过，真实数据/GPU 实验未运行。不能把旧 B1 的 SigLIP 2 命令当作新 SigLIP-base。

## 统一目录

| 相对路径 | 内容 |
|---|---|
| [data/](data/README.md) | 数据处理；原始数据放 data/local/ |
| [models/](models/README.md) | 模型封装；预训练权重不提交 |
| [train/](train/README.md) | 待实现的部分微调训练循环 |
| [evaluation/](evaluation/README.md) | 待接入的训练前/后评估入口 |
| [configs/](configs/siglip_finetune.plan.json) | 当前训练计划，尚不可执行 |
| [checkpoints/](checkpoints/README.md) | 本地训练输出，只提交说明 |
| exp5/ | 现有审计、评测、缓存、统计 |
| scripts/ / tests/ | 单卡启动、状态导出、测试 |
| docs/ / references/ / templates/ | 阶段记录、来源和报告模板 |

服务器以本仓库 checkout 为项目根目录，相对结构保持一致。数据/权重/特征/未经审查的原始日志留在服务器；代码、配置、文档和审查后的轻量结果进入 GitHub。

## 当前入口

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m exp5 --help
python scripts/project_state.py --out runs/sync/current.json
```

这些命令不自动运行模型。新训练入口待 P1/P3 实现，不提供空跑的 train 占位程序。

## 从这里继续

- [12 当前训练路线和阶段状态](docs/12_TRAINING_ROADMAP.md)
- [13 GitHub—服务器同步约定](docs/13_GITHUB_SERVER_SYNC.md)
- [14 阶段进度日志](docs/14_PROGRESS_LOG.md)
- [09 最新交接](docs/09_HANDOFF.md)
- [11 已有 CPU 测试证据](docs/11_CPU_VALIDATION.md)

每完成一个可复现阶段，就更新代码/配置/进度并推送 PR；用户审阅合并，服务器在干净工作区同步对应 commit。云端不连接公共服务器；用户执行命令，提供日志或服务器侧代码提交。

## 原方案与可复用资料

此前的 [文献调研](docs/01_LITERATURE_REVIEW.md)、[开题报告](docs/02_PROJECT_PROPOSAL.md)、[评测协议](docs/03_EXPERIMENT_PROTOCOL.md)、[实施方案](docs/04_IMPLEMENTATION_PLAN.md)、[预算](docs/05_TIMELINE_AND_BUDGET.md)、[错误分析](docs/06_ERROR_ANALYSIS_AND_API.md)、[风险](docs/07_RISKS_AND_DECISIONS.md)、[报告模板](docs/08_REPORT_AND_DEFENSE_TEMPLATE.md)保留参考。其中 CLIP→SigLIP 2→Qwen 不再是主路线。[10](docs/10_RUNNING_BASELINE.md)是已实现的原 B0/B1 CLI 手册；[原 YAML](configs/experiment5.plan.yaml)仍是历史规划，不能直接执行。

指标继续使用固定图库、命中式双向 R@1/5/10 和二值 NDCG@10；保留五描述/逐查询预测，验证选择并冻结后才评测试集。85% 是待实测目标；不预设微调一定有效、不填论文或示例成绩。来源见 [SOURCES](references/SOURCES.md) 与 [BibTeX](references/references.bib)。

公开仓库不上传原课程 PDF、图片、权重、密钥或未审查日志。目录框架更新不表示服务器已同步或训练已运行。
