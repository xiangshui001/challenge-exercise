# 项目接手约定

## 当前路线和任务边界

用户于 2026-09-30 确认：第一次训练项目，成熟 Flickr30K + google/siglip-base-patch16-224，未微调基线 → 简单部分微调 → 至多两档学习率 → 评估与错误分析。训练代码是已确认路线的一部分；原“不训练”方案已被本轮更新。仍只服务实验五，不补做其他课程实验。

第一版不自建数据集、不从零预训练、不加 hard negative/LoRA/Qwen 重排/收费 API。无前后端、ANN、压测、TensorRT、分布式训练。用户获准的单张 GPU 2 为 A100 80GB，有现有占用；实际 batch 由启动时小试决定。

## 必读和真实状态

README → docs/12_TRAINING_ROADMAP.md → docs/13_GITHUB_SERVER_SYNC.md → docs/14_PROGRESS_LOG.md → docs/09_HANDOFF.md。原指标口径参考 docs/03_EXPERIMENT_PROTOCOL.md，来源见 references/SOURCES.md。

已有 exp5 的 CLIP/SigLIP 2 审计/指标/缓存/统计和 24 项 CPU 测试。新 SigLIP-base、训练 Dataset 和训练入口尚未实现；目录与 planning_only 配置不能描述为已运行程序。原 B1 与主模型不同，禁止静默替换缓存方法身份。

## 实验纪律

固定划分/完整图库/五描述和 ID/版本。训练仅用 train；验证选配置/checkpoint，冻结后才评 test。不把示例或论文成绩写成结果，保留未提升/退化。R@K 为命中式，不能与集合召回混淆。

拟用下游对称 InfoNCE，不冒称原 sigmoid 预训练。每批图像 ID 唯一、每图选一条描述；重复图片需多正例处理，同图描述不得互作负例。梯度累积不自动扩大批内负例池。先检查具体模型层名再选择解冻层，不假定 projection 与 CLIP 相同。

## 目录与阶段同步

服务器项目根目录为仓库 checkout，data/models/train/evaluation/configs/checkpoints 相对路径与 GitHub 一致。复用 exp5 指标，不复制两套实现。图片/manifest 放 data/local；下载权重放外部 HF 缓存或 models/pretrained；checkpoint 放 checkpoints；特征/log 放 cache/runs。这些大文件不提交。

每完成一个可复现阶段，更新实际文件、配置、docs/14_PROGRESS_LOG.md 和实际证据；推送工作分支并建立/更新 PR，由用户决定合并。开工先核对最新 commit 与未提交改动；服务器修改先推送再由云端接手，不覆盖未提交文件，不强推。

## 执行边界

云端没有且用户不授予公共服务器连接权限，不索要 SSH 信息。云端处理仓库与 CPU 检查；GPU/下载/训练由用户在获准环境中显式执行并反馈日志。提交不自动启动服务器任务，不改共享驱动或清理其他进程。

密钥只用环境变量，不入库；公开轻量结果前检查内容/使用条件。不要发自动执行评论，不创建未经请求的定时同步任务。
