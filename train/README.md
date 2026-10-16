# train：简单微调

第一版只跑 SigLIP-base 的一次完整部分微调，再比较至多两档学习率。成熟 Flickr30K 数据；不自建领域数据、不加 hard negative/LoRA/复杂重排。

计划实现 `train/train.py`，包含 forward、对称 InfoNCE、backward、AdamW、验证和 checkpoint 保存。训练目标是本项目选定的下游微调目标，不冒称复现 SigLIP 原始 sigmoid 预训练。

批内图像 ID 唯一，每图随机选一条原始描述；训练跨 epoch 轮换描述，评估始终保留五条。同一图的其他描述不能当负例；若以后批内重复图片，需改多正例 mask。梯度累积改变优化更新批量，不自动扩大 InfoNCE 同步负例池。

冻结范围、batch 和实际损失实现通过小试后定稿，当前训练程序未实现。不要只为创建框架写一个会空跑并输出“训练成功”的占位程序。
