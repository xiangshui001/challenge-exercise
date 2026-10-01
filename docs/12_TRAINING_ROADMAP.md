# 12｜当前主路线：第一次 SigLIP 微调项目

2026-09-30 用户已确认：希望学习模型训练、做好课程项目并用于 GitHub/简历，同时保持第一版简单。因此当前路线调整为**SigLIP-base 基线 → 简单部分微调 → 至多两档学习率 → 重新评估与错误分析**。

原 docs/01–10 和 experiment5.plan.yaml 保留为先前的评测/可选重排方案，作为技术参考，不继续默认推进 Qwen/门控/API。当前路线以本文、根 README 与 AGENTS 为准；项目仍对应实验五，加入微调作为优化方法，不额外展开其他课程实验。

## 范围和技术选择

成熟 Flickr30K 固定 Karpathy 划分；每图五描述，训练只用 train，验证用于参数/checkpoint 选择，测试用于冻结后的评测。训练数量以下载后的划分审计为准，不写死聊天中的 25000。不构造新领域数据集，不从头预训练。

首个模型为 [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)。用户已接受这个入门版本，第一版跑通后再决定升级 SigLIP 2。模型实际 revision、可训练层、micro batch 和累积步数尚未固定，不能把预计 batch/时长当作保证。

SigLIP 原预训练使用 sigmoid loss，参考[原论文](https://arxiv.org/abs/2303.15343)。本项目拟用对称 InfoNCE 做下游检索微调，报告应明确这是选择的新训练目标。先检查该具体模型的真实层名，再决定部分解冻；不机械假定存在与 CLIP 相同的 projection。每训练 batch 图像 ID 唯一，每图选一条描述；同图其他配对描述不当负例。梯度累积不自动增加批内对比负例数量。

## 分阶段记录

| 阶段 | 交付 | 当前状态 |
|---|---|---|
| P0 环境 | conda/Python/torch/CUDA、所选卡 UUID、依赖记录 | 核心环境已通过：Python 3.11.16、torch 2.8.0+cu126、Transformers 4.57.3、pip check、A100 CUDA 计算；完整锁文件/UUID 待 P1 记录 |
| P1 推理小试 | 一张图、两段文本、真实权重特征/相似度 | 封装与 [小试脚本](15_SIGLIP_SMOKE.md) 已实现，待服务器运行；默认合成图仅验收接口，不验收语义质量 |
| P2 基线 | 固定 val/test manifest、未微调 SigLIP-base 双向指标 | 现有审计/指标可复用，新模型入口待接 |
| P3 微调 | train Dataset、部分微调、真实损失曲线、验证 checkpoint | 待实现/待运行 |
| P4 小优化 | 至多两档学习率，同训练预算比较 | 待 P3 跑通 |
| P5 最终评测 | 配置冻结、独立 test、修复/退化、人工错误分类与报告 | 待执行 |

服务器事实来自用户提供的输出，云端没有连接权限，不直接操作服务器；所有 GPU/下载/训练命令由用户在其可用环境里执行。用户可按空闲情况和任务需要选择卡数，GPU 2 仅是当前选择；先用单卡跑通，多卡按实际收益再选实现方案。下载/CPU 准备时不提前把模型放到 GPU，结束后退出本项目进程释放显存，没有任务时不长期预占。GPU 2 的早前快照已有 33295 MiB 占用，0% 利用率不表示无人使用；正式运行前看实际剩余显存。

## 现有代码如何复用

PR #2 已合并；exp5 的数据审计、指标、缓存、统计有 24 项 CPU 测试。其 B0/B1 模型是 CLIP 和 SigLIP 2，与新 SigLIP-base 路线不同。当前 `exp5 prepare` 只准备 val/test，并未实现训练 Dataset。现有 CLI 尚不支持新模型/训练 checkpoint；不为了目录相同就复制另一套指标。

第一版目录已经建好，SigLIP-base 特征封装已实现，训练程序待实现。配置 [siglip_finetune.plan.json](../configs/siglip_finetune.plan.json) 标为 planning_only，不是已可执行的 train 配置。

## 统一项目结构

| GitHub 与服务器相对路径 | 用途 | 是否提交 Git |
|---|---|---|
| README.md / AGENTS.md | 入口、约束和进度 | 是 |
| data/ | 数据处理代码与说明 | 是 |
| data/local/ | 图片、划分与生成 manifest | 否 |
| models/ | 模型 Python 封装与说明 | 是 |
| models/pretrained/ 或外部 HF 缓存 | 原始权重 | 否 |
| train/ | 训练循环与 checkpoint 管理代码 | 是 |
| evaluation/ | 基线/微调评估入口 | 是 |
| exp5/ | 现有审计、指标、缓存与统计实现 | 是 |
| configs/ | 实验计划；将来接入经验证的运行配置 | 是 |
| checkpoints/ | 训练权重、optimizer state | 只提交说明 |
| cache/ / runs/ | 特征和未经审查的日志、预测 | 否 |
| results/ | 审查后的轻量汇总和曲线 | 是 |
| docs/ / scripts/ / tests/ | 阶段文档、脚本与测试 | 是 |

以仓库 checkout 为服务器项目根目录；目录叫 challenge-exercise 或 siglip-retrieval 均可，相对结构必须一致。不要再另建一份独立代码树。同步方法见 [13](13_GITHUB_SERVER_SYNC.md)。
